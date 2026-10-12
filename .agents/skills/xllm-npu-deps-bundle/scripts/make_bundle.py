#!/usr/bin/env python3
"""Assemble the unified xLLM deps self-extracting run package.

Produces a makeself-style script (bash header + tar.gz payload):

    bash xllm-deps-{soc}-{arch}.run

which installs, in one shot: operator run packages + python wheels +
per-component verify + vendors/load_priority merge + metadata install +
environment setup (/etc/profile.d/xllm-deps.sh).

The bundle is installed inside the target container alongside xllm itself.

Usage:
    make_bundle.py --deps <depends.yaml|json> --meta-dir <dir> --artifacts-dir <dir> \
                   --platform <soc>-<arch> [--build-env <tag>] [--accp-seed <path>] \
                   --scripts-dir <dir> --out <bundle.run>

Design notes (all inlined into the generated install.sh):
  * components install in depends order; each ascend-op-run package's vendor
    directory is detected after install (never hardcoded);
  * ASCEND_CUSTOM_OPP_PATH / load_priority: our vendors first (later installs
    rank higher), baseline pre-registered vendors (e.g. custom_xllm_math)
    are kept last and never removed;
  * LD_LIBRARY_PATH is never touched: CANN loads op packages dynamically
    through ACCP/vendors registration;
  * wheels install with --no-deps; declared runtime deps are installed
    offline and reconciled by check_wheel_deps.py;
  * env setup goes to /etc/profile.d/xllm-deps.sh (idempotent prepend;
    applies to login and interactive shells; non-interactive callers must
    export it themselves -- install.sh prints the final value).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time

MARKER = "__XLLM_DEPS_PAYLOAD__"

# Private-address leak scan: the bundle may be distributed externally,
# generated text must not contain intranet addresses.
LEAK_RE = re.compile(
    r"(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.|100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.|169\.254\.)"
    r"[0-9]{1,3}\.[0-9]{1,3}"
)


def _die(msg):
    print(f"make_bundle: {msg}", file=sys.stderr)
    return sys.exit(1)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sh_quote(s):
    return "'" + str(s).replace("'", "'\\''") + "'"


def _load_cfg(path):
    with open(path, encoding="utf-8") as fh:
        if path.endswith(".json"):
            return json.load(fh)
        import yaml

        return yaml.safe_load(fh)


def _expand(tpl, mapping):
    out = str(tpl)
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _leak_scan(files):
    patterns = [LEAK_RE] + [re.compile(p) for p in re.split(r"\s+", os.environ.get("LEAK_PATTERNS", "")) if p]
    bad = 0
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        for pat in patterns:
            m = pat.search(text)
            if m:
                bad += 1
                print(f"make_bundle: leak scan hit {f}: {m.group(0)}", file=sys.stderr)
    if bad:
        _die(
            f"leak scan failed ({bad} private-address hits; the bundle may be "
            f"distributed externally, check metadata/script sources)"
        )


def _host_path_scan(files, needles):
    """Host paths ($HOME, work dirs, config dirs) must not appear in bundle
    text -- they leak personal information."""
    bad = 0
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        for n in needles:
            if n and len(n) > 3 and n in text:
                bad += 1
                print(f"make_bundle: host path leak in {f}: {n}", file=sys.stderr)
    if bad:
        _die(f"host path leak check failed ({bad} hits; the bundle may be distributed externally)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deps", required=True)
    ap.add_argument("--meta-dir", required=True)
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--platform", required=True, help="soc-arch (platform isolation key)")
    ap.add_argument("--build-env", default="local", help="build environment tag recorded in the manifest")
    ap.add_argument("--accp-seed", default="", help="baseline ASCEND_CUSTOM_OPP_PATH of the target environment")
    ap.add_argument("--scripts-dir", required=True, help="directory containing check_wheel_deps.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--soc", default="", help="filter out components whose supported_socs excludes this soc")
    args = ap.parse_args()

    cfg = _load_cfg(args.deps)
    comps = cfg.get("components") or []
    if not comps:
        return _die(f"no components found in {args.deps}")
    # filter out components that don't support the current soc
    if args.soc:
        all_count = len(comps)
        comps = [c for c in comps if not c.get("supported_socs") or args.soc in c["supported_socs"]]
        skipped = all_count - len(comps)
        if skipped:
            print(f"make_bundle: skipped {skipped} component(s) not supporting soc={args.soc}")

    accp_seed = args.accp_seed.strip()
    staging = tempfile.mkdtemp(prefix="xllm-bundle-")
    try:
        payload = os.path.join(staging, "payload")
        comp_root = os.path.join(payload, "components")
        meta_ctx = os.path.join(payload, "_meta")
        check_ctx = os.path.join(payload, "_check")
        os.makedirs(comp_root, exist_ok=True)
        os.makedirs(meta_ctx, exist_ok=True)
        os.makedirs(check_ctx, exist_ok=True)

        checker_src = os.path.join(args.scripts_dir, "check_wheel_deps.py")
        if not os.path.isfile(checker_src):
            return _die(f"missing {checker_src}")
        shutil.copy2(checker_src, os.path.join(check_ctx, "check_wheel_deps.py"))

        provenance = []
        metas = {}
        install_blocks = []
        idx_of = 0

        for idx_of, comp in enumerate(comps, start=1):
            name = comp.get("name") or _die("a component entry is missing 'name'")
            ctype = comp.get("type") or _die(f"{name}: missing type")
            meta_path = os.path.join(args.meta_dir, f"{name}.json")
            if not os.path.exists(meta_path):
                return _die(f"{name}: missing artifact metadata {meta_path} (build the component first)")
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            metas[name] = meta

            files = meta.get("files") or []
            files = [f for f in files if f.get("name") and f["name"] != "SHA256SUMS" and not f["name"].startswith(".")]
            if not files:
                return _die(f"{name}: metadata lists no installable artifact files")
            soc = meta.get("soc", "")
            arch = meta.get("arch", "")
            platform = f"{soc}-{arch}" if arch else soc
            if platform != args.platform:
                return _die(
                    f"{name}: metadata platform {platform} != target platform {args.platform} "
                    f"(cross-platform artifact mix-in?)"
                )
            vendor_dir = (meta.get("vendor_dir") or "").strip()
            provenance.append(
                {
                    "name": name,
                    "type": ctype,
                    "repo": meta.get("repo"),
                    "ref": meta.get("ref"),
                    "commit": meta.get("commit"),
                    "soc": soc,
                    "cann": meta.get("target_cann"),
                    "built_with": meta.get("built_with"),
                    "files": [f["name"] for f in files],
                    "vendor_dir": vendor_dir or None,
                }
            )

            cdir = os.path.join(comp_root, name)
            os.makedirs(cdir, exist_ok=True)
            for f in files:
                src = os.path.join(args.artifacts_dir, platform, name, f["name"])
                if not os.path.exists(src):
                    return _die(f"{name}: artifact missing {src}")
                # integrity check: files truncated/replaced between build and
                # packaging fail here when sha256 no longer matches metadata
                if f.get("sha256") and _sha256(src) != f["sha256"]:
                    return _die(
                        f"{name}: artifact checksum mismatch {src} (sha256 differs from "
                        f"metadata, rebuild the component)"
                    )
                shutil.copy2(src, os.path.join(cdir, f["name"]))

            # per-component verify script: placeholders expand at packaging
            # time ({vendor_dir} stays a runtime-detected variable)
            vmapping = {
                "name": name,
                "soc": soc,
                "commit": meta.get("commit", ""),
                "artifacts_dir": f"$SRC/components/{name}",
                "vendor_dir": "${XLLM_VENDOR_DIR}",
            }
            for vk, vv in ((comp.get("build") or {}).get("vars") or {}).items():
                vmapping.setdefault(vk, " ".join(str(vv).split()) if isinstance(vv, str) else vv)
            vlines = [
                "#!/bin/bash",
                f"# {name} component verify (device-free; XLLM_VENDOR_DIR detected and exported by install.sh)",
                "set -e",
                "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1",
                'SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"',
                f'echo "### verify: {name}"',
            ]
            for vc in comp.get("verify") or []:
                vlines.append(_expand(str(vc), vmapping))
            with open(os.path.join(cdir, "verify.sh"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(vlines) + "\n")
            os.chmod(os.path.join(cdir, "verify.sh"), 0o755)

            # install.sh component block (fully inlined; no config parsing at runtime)
            blk = [f'echo "=== [{idx_of}/{len(comps)}] {name} ({ctype}) ==="']
            if ctype == "ascend-op-run":
                if not vendor_dir:
                    return _die(f"{name}: metadata missing vendor_dir (not produced during install dry-run)")
                runs = [f["name"] for f in files if f["name"].endswith(".run")]
                if len(runs) != 1:
                    return _die(f"{name}: expected exactly 1 .run artifact, got {[f['name'] for f in files]}")
                inst = str(((comp.get("install") or {}).get("cmd")) or "bash {artifact}")
                if "\n" in inst:
                    return _die(f"{name}: install.cmd must be a single line (put multi-line logic in verify)")
                inst = inst.replace("{artifact}", f"$SRC/components/{name}/{runs[0]}")
                vbasename = os.path.basename(vendor_dir.rstrip("/"))
                blk += [
                    '_before="$(_list_vendors)"',
                    inst,
                    '_after="$(_list_vendors)"',
                    "_new=\"$(comm -13 <(printf '%s\\n' \"$_before\") <(printf '%s\\n' \"$_after\") | grep -v '^$' | head -1)\"",
                    # idempotent reinstall: when the vendor dir already exists
                    # the before/after diff is empty; fall back to the vendor
                    # name recorded in metadata (still requires it to exist
                    # and be non-empty, so a failed install is never misread
                    # as a successful reinstall)
                    f'if [[ -z "$_new" ]]; then _new="{vbasename}"; '
                    f'[[ -n "$_new" && -d "$_v/$_new" ]] || {{ echo "error: {name} installed but no new vendor '
                    f'directory detected (reinstall fallback {vbasename} missing too)"; exit 4; }}; fi',
                    # record full paths: ASCEND_CUSTOM_OPP_PATH entries must
                    # be absolute (vendor-priority semantics)
                    'XLLM_VENDORS+=("$_v/$_new")',
                    'export XLLM_VENDOR_DIR="$_v/$_new"',
                    'grep -q load_priority "$_v/config.ini" || { echo "error: vendors/config.ini has no load_priority"; exit 5; }',
                    'grep -q "$_new" "$_v/config.ini" || { echo "error: vendors/config.ini does not list $_new"; exit 5; }',
                    'chmod -R a+rX "$XLLM_VENDOR_DIR"',
                    f'bash "$SRC/components/{name}/verify.sh"',
                ]
            elif ctype == "python-wheel":
                whls = [f["name"] for f in files if f["name"].endswith(".whl")]
                if not whls:
                    return _die(f"{name}: metadata lists no .whl artifact (got {[f['name'] for f in files]})")
                dists = {}
                for w in whls:
                    dists.setdefault(w.split("-")[0], []).append(w)
                dups = {d: v for d, v in dists.items() if len(v) > 1}
                if dups:
                    return _die(
                        f"{name}: multiple files for the same wheel distribution {dups} "
                        f"(cross-SoC/platform residue? rebuild with --force)"
                    )
                pipdeps = ((comp.get("install") or {}).get("pip_deps")) or []
                if pipdeps:
                    src_pdir = os.path.join(args.artifacts_dir, platform, name, "pip_deps")
                    if not os.path.isdir(src_pdir) or not glob.glob(os.path.join(src_pdir, "*.whl")):
                        return _die(
                            f"{name}: install.pip_deps={pipdeps} declared but {src_pdir} has no "
                            f"installable wheel (re-run the build)"
                        )
                    shutil.copytree(src_pdir, os.path.join(cdir, "pip_deps"), dirs_exist_ok=True)
                    specs = " ".join(_sh_quote(str(s)) for s in pipdeps)
                    blk += [
                        f'echo "### declared runtime deps: {specs}"',
                        f'pip3 install --no-deps --no-index --progress-bar off --find-links "$SRC/components/{name}/pip_deps" {specs}',
                    ]
                for wf in whls:
                    blk.append(
                        f'pip3 install --force-reinstall --no-deps --progress-bar off "$SRC/components/{name}/{wf}"'
                    )
                wq = " ".join(f'"$SRC/components/{name}/{wf}"' for wf in whls)
                blk += [
                    # single-threaded BLAS during checks: old-docker seccomp can
                    # make numpy/OpenBLAS thread creation fail with EPERM
                    f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 "$SRC/_check/check_wheel_deps.py" {wq}',
                    f'bash "$SRC/components/{name}/verify.sh"',
                ]
            else:
                return _die(f"{name}: unknown type '{ctype}' (supported: ascend-op-run / python-wheel)")
            install_blocks.append("\n".join(blk))

        # manifest + metadata (shipped with the bundle for traceability)
        our_vendors = [os.path.basename(m["vendor_dir"].rstrip("/")) for m in metas.values() if m.get("vendor_dir")]
        predeclared = []
        for seed_path in accp_seed.split(":"):
            seed_path = seed_path.strip().rstrip("/")
            if not seed_path:
                continue
            base = os.path.basename(seed_path)
            if base and base not in our_vendors and base not in predeclared:
                predeclared.append(base)
        with open(args.deps, "rb") as fh:
            deps_sha = hashlib.sha256(fh.read()).hexdigest()
        deps_name = "depends.json" if args.deps.endswith(".json") else "depends.yaml"
        shutil.copy2(args.deps, os.path.join(meta_ctx, deps_name))
        for cname, cmeta in metas.items():
            with open(os.path.join(meta_ctx, f"{cname}.json"), "w", encoding="utf-8") as fh:
                json.dump(cmeta, fh, ensure_ascii=False, indent=2)
        manifest = {
            "bundle_format": 1,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "platform": args.platform,
            "arch": metas and next(iter(metas.values())).get("arch", ""),
            "build_env": args.build_env,
            "depends_config": {
                # filename + content hash only: absolute host paths are
                # personal information and do not ship with the bundle
                "sha256": deps_sha,
                "schema_version": cfg.get("schema_version"),
                "file": deps_name,
            },
            "components": provenance,
            "op_vendors": list(reversed(our_vendors)),
            "predeclared_vendors": predeclared,
            "install": "bash <this-file>.run (root required; see the xllm-npu-deps-bundle skill)",
        }
        with open(os.path.join(meta_ctx, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)

        # install.sh (generated; the only runtime unknown = vendor dir detection)
        seed_q = accp_seed.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
        install_lines = [
            "#!/bin/bash",
            "# xllm-deps unified bundle installer (generated by make_bundle.py, do not edit)",
            "# one-shot install: operator run packages + python wheels + metadata + env setup",
            "# prerequisites: target container provides a matching CANN stack; run as root.",
            "set -e",
            'SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            "",
            '[[ $(id -u) -eq 0 ]] || { echo "error: root required (writes to $ASCEND_OPP_PATH and /etc)"; exit 13; }',
            "",
            "# CANN runtime env (vendor detection, OPP paths and ldd checks depend on it)",
            '_cann_env=""',
            'if [[ -n "${ASCEND_HOME_PATH:-}" && -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then _cann_env="${ASCEND_HOME_PATH}/set_env.sh"',
            'else _cann_env="$(ls -1 /usr/local/Ascend/*/set_env.sh /usr/local/Ascend/ascend-toolkit/*/set_env.sh 2>/dev/null | head -1 || true)"; fi',
            '[[ -n "$_cann_env" ]] && source "$_cann_env"',
            "",
            "# baseline ASCEND_CUSTOM_OPP_PATH: prefer the current process env (interactive",
            "# install), fall back to the baseline probed at packaging time otherwise.",
            '_accp_base="${ASCEND_CUSTOM_OPP_PATH:-}"',
            f'[[ -n "$_accp_base" ]] || _accp_base="{seed_q}"',
            "unset ASCEND_CUSTOM_OPP_PATH",
            "",
            '_v="${ASCEND_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest/opp}/vendors"',
            '_list_vendors() { [[ -d "$_v" ]] && find "$_v" -mindepth 1 -maxdepth 1 -type d -printf \'%f\\n\' | sort; }',
            "declare -a XLLM_VENDORS=()",
            "",
            "\n\n".join(install_blocks),
            "",
            "# drop baseline entries already provided by this install: reinstalling an",
            "# updated bundle over a previous one must not accumulate duplicate",
            "# vendor paths in ASCEND_CUSTOM_OPP_PATH",
            'if [[ -n "$_accp_base" ]]; then',
            '  _filtered=""',
            "  for _p in ${_accp_base//:/ }; do",
            "    _skip=0",
            '    for _vd in "${XLLM_VENDORS[@]}"; do',
            '      if [[ "$(basename "${_p%/}")" == "$(basename "${_vd%/}")" ]]; then _skip=1; break; fi',
            "    done",
            '    if [[ $_skip -eq 0 ]]; then _filtered="${_filtered:+$_filtered:}$_p"; fi',
            "  done",
            '  _accp_base="$_filtered"',
            "fi",
            "",
            "# merge vendors/config.ini load_priority (non-destructive)",
            "# our components first (later installs rank higher); baseline pre-registered",
            "# vendors not installed yet (e.g. custom_xllm_math, installed later by the",
            "# xllm build) stay last (lowest priority) and are never removed; entries",
            "# written by the installers are preserved too (deduplicated).",
            '_ini="$_v/config.ini"',
            '_cur="$(sed -n \'s/^load_priority[[:space:]]*=[[:space:]]*//p\' "$_ini" 2>/dev/null || true)"',
            '_want=""',
            'for (( _i=${#XLLM_VENDORS[@]}-1; _i>=0; _i-- )); do _want="${_want:+$_want,}$(basename "${XLLM_VENDORS[$_i]}")"; done',
            f'_want="${{_want:+$_want,}}{",".join(predeclared)}"',
            'printf \'%s,%s\' "$_want" "$_cur" '
            '| awk -F, \'{o=""; for(i=1;i<=NF;i++) if($i!="" && !(seen[$i]++)) o=(o==""?$i:o","$i); print o}\' | { read _np; sed -i "s|^load_priority=.*|load_priority=${_np}|" "$_ini"; }',
            "",
            "# env setup: /etc/profile.d/xllm-deps.sh (idempotent prepend of our vendors,",
            "# existing value kept after); login shells get it via /etc/profile,",
            "# interactive shells via /etc/bashrc; non-interactive callers export it themselves.",
            '_accp=""',
            'for (( _i=${#XLLM_VENDORS[@]}-1; _i>=0; _i-- )); do _accp="${_accp:+$_accp:}${XLLM_VENDORS[$_i]}"; done',
            '_accp="$_accp${_accp_base:+:$_accp_base}"',
            "{",
            "    printf '%s\\n' \\",
            "      '# xllm-deps bundle (idempotent): prepend our vendors, keep any existing value after' \\",
            '      "_xllm_vendors=\\"$_accp\\"" \\',
            "      'case \":${ASCEND_CUSTOM_OPP_PATH:-}:\" in' \\",
            "      '    *\":${_xllm_vendors}:\"*) ;;' \\",
            "      '    *) export ASCEND_CUSTOM_OPP_PATH=\"${_xllm_vendors}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}\" ;;' \\",
            "      'esac' \\",
            "      'unset _xllm_vendors'",
            "} > /etc/profile.d/xllm-deps.sh",
            "chmod a+r /etc/profile.d/xllm-deps.sh",
            "",
            "# metadata ships with the bundle: stored next to the op vendors",
            '_dst="$_v/xllm_deps_metadata"',
            'mkdir -p "$_dst"',
            'cp "$SRC"/_meta/*.json "$_dst"/',
            f'cp "$SRC/_meta/{deps_name}" "$_dst"/',
            'chmod -R a+rX "$_dst"',
            'ln -sfn "$_dst" /etc/xllm-deps-metadata',
            "",
            'echo "xllm-deps install done: vendors=[${XLLM_VENDORS[*]}] (load_priority merged)"',
            'echo "ASCEND_CUSTOM_OPP_PATH (new shells): $_accp"',
            'echo "metadata: $_dst (/etc/xllm-deps-metadata)"',
            'echo "XLLM_DEPS_INSTALL_OK"',
        ]
        install_sh = os.path.join(payload, "install.sh")
        with open(install_sh, "w", encoding="utf-8") as fh:
            fh.write("\n".join(install_lines) + "\n")
        os.chmod(install_sh, 0o755)

        # leak scans on generated text (the bundle may be distributed externally):
        # 1) private addresses; 2) host paths ($HOME / work dirs / config paths)
        scan_files = (
            [install_sh] + glob.glob(os.path.join(comp_root, "*", "verify.sh")) + glob.glob(os.path.join(meta_ctx, "*"))
        )
        _leak_scan(scan_files)
        _host_path_scan(
            scan_files,
            [os.path.expanduser("~"), args.deps, args.meta_dir, args.artifacts_dir, args.scripts_dir],
        )

        # self-extracting wrapper: bash header + marker line + tar.gz payload
        wrapper = (
            "#!/bin/bash\n"
            "# ==========================================================================\n"
            f"# xllm-deps unified self-extracting bundle (generated by make_bundle.py; platform={args.platform})\n"
            "# contents: operator run packages + python wheels + component metadata + env setup\n"
            "# usage: bash <this-file> (root required; target container must provide a\n"
            "# matching CANN stack)\n"
            "# ==========================================================================\n"
            "set -e\n"
            '_self="$(readlink -f "$0")"\n'
            f'_skip="$(awk \'/^{MARKER}$/{{print NR + 1; exit}}\' "$_self")"\n'
            '[[ -n "$_skip" ]] || { echo "error: bundle corrupted (payload marker not found)"; exit 1; }\n'
            '_dir="$(mktemp -d /tmp/xllm-deps.XXXXXX)"\n'
            "trap 'rm -rf \"$_dir\"' EXIT\n"
            'echo "extracting bundle to $_dir ..."\n'
            'tail -n "+$_skip" "$_self" | tar -xzf - -C "$_dir"\n'
            'bash "$_dir/payload/install.sh"\n'
            "exit $?\n"
            f"{MARKER}\n"
        )

        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        tgz = os.path.join(staging, "payload.tgz")
        with tarfile.open(tgz, "w:gz") as tf:
            tf.add(payload, arcname="payload")
        with open(args.out, "wb") as out_fh:
            out_fh.write(wrapper.encode("utf-8"))
            with open(tgz, "rb") as pf:
                shutil.copyfileobj(pf, out_fh)
        os.chmod(args.out, 0o755)

        sha = _sha256(args.out)
        with open(args.out + ".sha256", "w", encoding="utf-8") as fh:
            fh.write(f"{sha}  {os.path.basename(args.out)}\n")
        size_mb = os.path.getsize(args.out) / 1024 / 1024
        print(f"make_bundle: {args.out} ({size_mb:.1f} MiB, {len(provenance)} components, sha256 {sha[:16]}...)")
        print(
            f"make_bundle: expected vendors (later installs rank higher): "
            f"{list(reversed(our_vendors)) or '<none>'}" + (f", baseline kept: {predeclared}" if predeclared else "")
        )
        return 0
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
