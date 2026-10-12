#!/usr/bin/env bash
# Build the unified xLLM NPU deps bundle in the current environment.
#
# Reads the component list from docker/npu_docker_depends.yaml (single source
# of truth), fetches each component's source, clean-builds it, installs it
# locally to detect the vendor directory, runs the per-component verify
# commands, then assembles everything into one self-extracting run package:
#
#     bash xllm-deps-{arm|x86}.run
#
# The bundle supports repeated installation: rebuilding after an operator fix
# and installing over a previous bundle is the normal workflow.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# locate the repo root: works for both .agents/skills (real path) and
# .claude/skills (symlink) invocation, and for non-git checkouts
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  _p="$SCRIPT_DIR"
  while [[ "$_p" != "/" ]]; do
    if [[ -f "$_p/docker/npu_docker_depends.yaml" ]]; then REPO_ROOT="$_p"; break; fi
    _p="$(dirname "$_p")"
  done
fi

usage() {
  cat <<'EOF'
Usage: build_deps_bundle.sh --soc <soc_version> [options]

Build the unified deps bundle from docker/npu_docker_depends.yaml, entirely
in the current environment (needs a local CANN stack, no docker daemon).
One bundle per --soc (e.g. ascend910_93), one per CANN soc_version.

Options:
  --soc SOC        CANN soc_version, e.g. ascend910_93 (cross-checked with
                   the local toolkit)
  --deps PATH      depends config (default: <repo>/docker/npu_docker_depends.yaml)
  --work-dir DIR   sources/artifacts/metadata workspace (default: ~/xllm-deps-work)
  --jobs N         compile parallelism (default: nproc/2)
  --only NAME      build a single component (skips bundle assembly)
  --force          rebuild even if artifacts already exist
  --out PATH       bundle output path (default: <work>/bundle/xllm-deps-<arm|x86>.run)
  -h, --help       show this help

Environment:
  GH_INSTEAD_OF    github clone acceleration prefix(es), comma-separated
                   (write as https://github without .com)
EOF
}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "error: $*" >&2; exit 1; }

DEPS=""; WORK=""; JOBS=""; ONLY=""; OUT=""; FORCE=0
while (( $# )); do
  case "$1" in
    --soc|--deps|--work-dir|--jobs|--only|--out)
      if [[ $# -lt 2 ]]; then
        usage
        die "missing value for $1"
      fi
      case "$1" in
        --soc)      SOC="$2" ;;
        --deps)     DEPS="$2" ;;
        --work-dir) WORK="$2" ;;
        --jobs)     JOBS="$2" ;;
        --only)     ONLY="$2" ;;
        --out)      OUT="$2" ;;
      esac
      shift ;;
    --force)    FORCE=1 ;;
    -h|--help)  usage; exit 0 ;;
    *)          usage; die "unknown argument: $1" ;;
  esac
  shift
done

if [[ -z "${SOC:-}" ]]; then
  usage
  die "--soc is required (e.g. --soc ascend910_93)"
fi

DEPS="${DEPS:-$REPO_ROOT/docker/npu_docker_depends.yaml}"
if [[ ! -f "$DEPS" ]]; then die "depends config not found: $DEPS (use --deps)"; fi
WORK="${WORK:-$HOME/xllm-deps-work}"
WORK="$(realpath -m "$WORK")"   # absolute: build subshells cd into source dirs
JOBS="${JOBS:-$(( $(nproc 2>/dev/null || echo 8) / 2 ))}"
(( JOBS > 64 )) && JOBS=64
(( JOBS < 1 )) && JOBS=1
ARCH="$(uname -m)"

mkdir -p "$WORK"/{src,artifacts,meta,bundle,logs}

# ---------- config reader (python3 + yaml) ----------
cfg() { # cfg <dotted.path> [json]  -> value on stdout, exit 3 when absent
  python3 - "$DEPS" "$1" "${2:-}" <<'PY'
import json, sys, yaml
path, expr, as_json = sys.argv[1], sys.argv[2], sys.argv[3]
cur = yaml.safe_load(open(path))
for seg in expr.split("."):
    if seg == "list":
        cur = len(cur) if isinstance(cur, (list, dict)) else None
    elif isinstance(cur, list):
        try:
            cur = cur[int(seg)]
        except (ValueError, IndexError):
            cur = None
    elif isinstance(cur, dict):
        cur = cur.get(seg)
    else:
        cur = None
    if cur is None and seg != "list":
        sys.exit(3)
if cur is None:
    sys.exit(3)
if isinstance(cur, bool):
    print("true" if cur else "false")
elif isinstance(cur, (dict, list)) or as_json:
    print(json.dumps(cur, ensure_ascii=False))
else:
    print(cur)
PY
}

# ---------- soc: cross-checked with the local toolkit ----------
TOOLKIT_CFG="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/opp/built-in/op_impl/ai_core/tbe/kernel/config"
if [[ ! -d "$TOOLKIT_CFG" ]]; then
  die "local CANN toolkit not found ($TOOLKIT_CFG); run inside a dev environment with CANN"
fi
SUPPORTED="$(ls "$TOOLKIT_CFG" 2>/dev/null | tr '\n' ' ')"
case " $SUPPORTED " in
  *" $SOC "*) ;;
  *) die "soc=$SOC not supported by the local toolkit (supported: $SUPPORTED)" ;;
esac
PLATFORM="${SOC}-${ARCH}"
mkdir -p "$WORK/meta/$PLATFORM" "$WORK/artifacts/$PLATFORM"

N_COMP="$(cfg components.list)" || die "depends config has no components"
log "soc=$SOC arch=$ARCH platform=$PLATFORM components=$N_COMP work=$WORK"

# ---------- source fetch ----------
declare -A _FETCHED=()
declare -A _FETCHED_REFS=()

ensure_source() { # $1=idx -> sets SRC_DIR / SRC_COMMIT
  local idx="$1" repo ref sdir name
  name="$(cfg "components.$idx.name")"
  repo="$(cfg "components.$idx.repo")"
  ref="$(cfg "components.$idx.ref")"
  sdir="$(cfg "components.$idx.source_dir" 2>/dev/null)" || sdir=""
  if [[ -z "$sdir" ]]; then sdir="$(basename "$repo" .git)"; fi
  SRC_DIR="$WORK/src/$sdir"
  if [[ -n "${_FETCHED[$sdir]:-}" ]]; then
    _cached_ref="${_FETCHED_REFS[$sdir]:-}"
    if [[ "$_cached_ref" == "$ref" ]]; then
      SRC_COMMIT="${_FETCHED[$sdir]}"
      return 0
    fi
    log "  WARN: source_dir $sdir cached at ref=$_cached_ref but this component uses ref=$ref; re-fetching"
    unset "_FETCHED[$sdir]" "_FETCHED_REFS[$sdir]"
  fi
  # GH_INSTEAD_OF: comma-separated mirror prefixes, tried in order, then one
  # final attempt with no url rewrite (direct clone)
  local pref ok=0 flog="$WORK/logs/fetch-$name.log"
  local -a prefs=()
  if [[ -n "${GH_INSTEAD_OF:-}" ]]; then
    IFS=' ' read -ra prefs <<< "${GH_INSTEAD_OF//,/ }"
  fi
  prefs+=("")
  : > "$flog"
  for pref in "${prefs[@]}"; do
    local -a gconf=(-c core.askPass= -c credential.helper=)
    if [[ -n "$pref" ]]; then gconf+=(-c "url.${pref}.insteadOf=https://github"); fi
    local -a grun=(env GIT_TERMINAL_PROMPT=0 git "${gconf[@]}")
    if [[ -d "$SRC_DIR/.git" ]]; then
      "${grun[@]}" -C "$SRC_DIR" remote set-url origin "$repo" >>"$flog" 2>&1 || true
      if { "${grun[@]}" -C "$SRC_DIR" fetch --depth 1 origin "$ref" \
           || "${grun[@]}" -C "$SRC_DIR" fetch origin "$ref"; } >>"$flog" 2>&1 \
         && "${grun[@]}" -C "$SRC_DIR" checkout -f FETCH_HEAD >>"$flog" 2>&1; then
        ok=1
      fi
    else
      rm -rf "$SRC_DIR"
      if { "${grun[@]}" clone --depth 1 --branch "$ref" --single-branch "$repo" "$SRC_DIR" \
           || "${grun[@]}" clone "$repo" "$SRC_DIR"; } >>"$flog" 2>&1; then
        "${grun[@]}" -C "$SRC_DIR" checkout -f "$ref" >>"$flog" 2>&1 || true
        ok=1
      fi
    fi
    if [[ $ok -eq 1 ]]; then break; fi
  done
  if [[ $ok -ne 1 ]]; then
    if [[ -d "$SRC_DIR/.git" ]]; then
      log "  WARN: all fetch attempts failed, reusing existing checkout (log: $flog)"
    else
      die "source fetch failed: $repo@${ref} (log: $flog)"
    fi
  fi
  SRC_COMMIT="$(git -C "$SRC_DIR" rev-parse HEAD)"
  _FETCHED["$sdir"]="$SRC_COMMIT"
  _FETCHED_REFS["$sdir"]="$ref"
}

# ---------- build ----------
build_component() { # $1=idx
  local idx="$1"
  local name type scmd soc_env pats syspkgs vars
  name="$(cfg "components.$idx.name")"
  type="$(cfg "components.$idx.type")"
  scmd="$(cfg "components.$idx.build.cmd")" || die "$name: missing build.cmd"
  soc_env="$(cfg "components.$idx.build.soc_env" 2>/dev/null)" || soc_env=""
  pats="$(cfg "components.$idx.artifacts" json)" || die "$name: missing artifacts"
  syspkgs="$(python3 -c 'import json,sys; print(" ".join(json.loads(sys.argv[1] or "[]")))' \
              "$(cfg "components.$idx.build.system_packages" json 2>/dev/null || echo '[]')")"
  vars="$(cfg "components.$idx.build.vars" json 2>/dev/null)" || vars='{}'

  local arel="artifacts/$PLATFORM/$name"
  local adir="$WORK/$arel"

  # expand placeholders in build.cmd
  local cmd_exp="$scmd" k v
  cmd_exp="${cmd_exp//\{soc\}/$SOC}"
  cmd_exp="${cmd_exp//\{jobs\}/$JOBS}"
  cmd_exp="${cmd_exp//\{name\}/$name}"
  cmd_exp="${cmd_exp//\{arch\}/$ARCH}"
  cmd_exp="${cmd_exp//\{commit\}/$SRC_COMMIT}"
  cmd_exp="${cmd_exp//\{ref\}/$(cfg "components.$idx.ref")}"
  while IFS=$'\t' read -r k v; do
    if [[ -n "$k" ]]; then cmd_exp="${cmd_exp//\{$k\}/$v}"; fi
  done < <(python3 - "$vars" <<'PY'
import json, sys
raw = sys.argv[1] if sys.argv[1] else "{}"
try:
    d = json.loads(raw)
except Exception:
    d = {}
for k, v in (d or {}).items():
    if isinstance(v, str):
        v = " ".join(v.split())
    print(f"{k}\t{v}")
PY
)

  # reuse existing artifacts unless --force
  if [[ $FORCE -eq 0 ]] && [[ -n "$(ls -A "$adir" 2>/dev/null | grep -vE '^(SHA256SUMS|\.built-commit)$' || true)" ]]; then
    local built
    built="$(cat "$adir/.built-commit" 2>/dev/null || echo "$SRC_COMMIT")"
    if [[ "$built" != "$SRC_COMMIT" ]]; then
      log "  $name: WARN artifacts built from ${built:0:12} but source HEAD is ${SRC_COMMIT:0:12} (moving ref); use --force to rebuild"
    fi
    log "  $name: reuse existing artifacts (--force to rebuild)"
    return 0
  fi

  rm -rf "$adir"
  mkdir -p "$adir"
  log "  $name: build ($cmd_exp)"

  if ! (
    cd "$SRC_DIR"
    # CANN env (usually already set; source for safety)
    if [[ -n "${ASCEND_HOME_PATH:-}" && -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
      source "${ASCEND_HOME_PATH}/set_env.sh"
    fi
    export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
    # optional system packages (additive, safe for the local environment)
    if [[ -n "$syspkgs" ]]; then
      for _pm in dnf yum apt-get zypper; do
        if command -v "$_pm" >/dev/null 2>&1; then
          "$_pm" install -y $syspkgs >/dev/null 2>&1 || echo "  WARN: $_pm install failed (continuing)"
          break
        fi
      done
    fi
    if [[ -n "$soc_env" ]]; then
      export "$soc_env=$SOC"
    fi
    # NOTE: pip build requirements from the depends config are NOT installed:
    # their loose ranges (e.g. numpy<2) fight the pinned NPU stack; the current
    # environment is expected to provide python build deps.
    # delete artifacts matching the patterns from previous builds
    while IFS= read -r _pat; do
      for _m in $_pat; do
        if [[ -f "$_m" ]]; then rm -fv -- "$_m"; fi
      done
    done < <(python3 -c 'import json,sys
for p in json.loads(sys.argv[1]): print(p)' "$pats")
    # clean CMake build dirs: incremental state from a previous parameter set
    # leaks stale kernels into the package (observed: undeclared ops in
    # binary_info_config.json shadowing correct implementations)
    find . -name CMakeCache.txt 2>/dev/null | while IFS= read -r _cc; do
      _d="$(dirname "$_cc")"
      if [[ -d "$_d/CMakeFiles" && ! -f "$_d/CMakeLists.txt" ]]; then
        echo "  clean build: removing CMake build dir $_d"
        rm -rf "$_d"
      fi
    done
    # build
    bash -c "$cmd_exp" 2>&1
    # collect artifacts
    mkdir -p "$adir"
    _hit=0
    while IFS= read -r _pat; do
      for _f in $_pat; do
        if [[ -e "$_f" ]]; then cp -v "$_f" "$adir/"; _hit=1; fi
      done
    done < <(python3 -c 'import json,sys
for p in json.loads(sys.argv[1]): print(p)' "$pats")
    if [[ $_hit -ne 1 ]]; then
      echo "no artifacts matched (bad artifacts glob?)"; exit 3
    fi
    (
      cd "$adir"
      rm -f SHA256SUMS
      find . -mindepth 1 -maxdepth 1 -type f ! -name '.*' -printf '%f\0' \
        | xargs -0 --no-run-if-empty sha256sum > SHA256SUMS
    )
    echo BUILD_OK
  ) >> "$WORK/logs/build-$name.log" 2>&1; then
    tail -6 "$WORK/logs/build-$name.log"
    die "$name: build failed (log: $WORK/logs/build-$name.log)"
  fi
  printf '%s' "$SRC_COMMIT" > "$adir/.built-commit"
}

# ---------- local install + verify ----------
verify_component() { # $1=idx
  local idx="$1"
  local name type arel adir
  name="$(cfg "components.$idx.name")"
  type="$(cfg "components.$idx.type")"
  arel="artifacts/$PLATFORM/$name"
  adir="$WORK/$arel"

  local vendor=""
  if [[ "$type" == "ascend-op-run" ]]; then
    local instcmd
    instcmd="$(cfg "components.$idx.install.cmd" 2>/dev/null)" || instcmd="bash {artifact}"
    local meta_f="$WORK/meta/$PLATFORM/$name.json"
    if ! (
      export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
      if [[ -n "${ASCEND_HOME_PATH:-}" && -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
        source "${ASCEND_HOME_PATH}/set_env.sh"
      fi
      _v="${ASCEND_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest/opp}/vendors"
      _list_vendors() { [[ -d "$_v" ]] && find "$_v" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort; }
      _before="$(_list_vendors || true)"
      unset ASCEND_CUSTOM_OPP_PATH
      _marker="/tmp/.xllm_vendor_marker_$$"
      touch "$_marker"
      shopt -s nullglob
      _runs=("$adir"/*.run)
      if [[ ${#_runs[@]} -eq 0 ]]; then echo "no .run artifact in $adir"; exit 3; fi
      for _r in "${_runs[@]}"; do
        echo "### install: $_r"
        eval "${instcmd//\{artifact\}/$_r}"
      done
      shopt -u nullglob
      _after="$(_list_vendors || true)"
      # fresh install: a new vendor dir appeared
      _new="$(comm -13 <(printf '%s\n' "$_before") <(printf '%s\n' "$_after") | grep -v '^$' | head -1)"
      if [[ -z "$_new" ]]; then
        # reinstall: files inside the overwritten vendor dir are newer than the
        # pre-install marker (the top-level dir mtime may not change)
        for _d in "$_v"/*/; do
          [[ -d "$_d" ]] || continue
          if [[ -n "$(find "$_d" -newer "$_marker" -print -quit 2>/dev/null)" ]]; then
            _new="$(basename "$_d")"
            echo "### reinstall: detected updated vendor $_new"
            break
          fi
        done
      fi
      rm -f "$_marker"
      if [[ -z "$_new" ]]; then
        echo "no vendor directory detected after install"; ls -la "$_v" 2>/dev/null; exit 4
      fi
      echo "META_VENDOR=$_v/$_new"
      grep -q load_priority "$_v/config.ini" || { echo "vendors/config.ini missing load_priority"; exit 5; }
      grep -q "$_new" "$_v/config.ini" || { echo "vendors/config.ini does not list $_new"; exit 5; }
      chmod -R a+rX "$_v/$_new"
      # ABI probe: undefined symbols reconciled against locally providable ones
      _prov=/tmp/.xllm_runtime_syms
      : > "$_prov"; _nprov=0
      if command -v nm >/dev/null 2>&1; then
        for _l in "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"/lib64/*.so* \
                  /usr/lib64/libboundscheck.so* /lib64/libc.so.6 /usr/lib64/libstdc++.so.6; do
          if [[ -e "$_l" ]]; then nm -D --defined-only "$_l" 2>/dev/null | awk '{print $NF}' >> "$_prov"; fi
        done
        sort -u -o "$_prov" "$_prov"
        _nprov=$(wc -l < "$_prov")
      fi
      _bad=0
      for _so in $(find "$_v/$_new" -name '*.so*' -type f 2>/dev/null | head -20); do
        _undef="$(ldd -r "$_so" 2>/dev/null | grep -i 'undefined symbol' | awk '{print $3}' | tr -d '(),' | grep -v '^$' | sort -u || true)"
        if [[ -z "$_undef" ]]; then continue; fi
        if (( _nprov > 0 )); then
          _miss="$(comm -23 <(printf '%s\n' "$_undef") "$_prov" || true)"
        else
          _miss="$_undef"
        fi
        if [[ -n "$_miss" ]]; then
          _bad=1
          echo "META_WARN=unresolved symbols in $(basename "$_so"): $(printf '%s,' $_miss | cut -c1-240)"
        fi
      done
      if (( _bad == 0 )); then
        echo "  ABI check ok: undefined symbols resolvable at runtime ($_nprov symbols)"
      else
        echo "ABI check failed: unresolved symbols in vendor libraries"
        exit 6
      fi
    ) > "$WORK/logs/verify-$name.log" 2>&1; then
      tail -20 "$WORK/logs/verify-$name.log"
      die "$name: install/verify failed (log: $WORK/logs/verify-$name.log)"
    fi
    vendor="$(sed -n 's/^META_VENDOR=//p' "$WORK/logs/verify-$name.log" | head -1)"
  else
    # wheel: download declared pip deps, install offline, reconcile
    local specs
    specs="$(python3 -c 'import json,sys; print(" ".join(json.loads(sys.argv[1] or "[]")))' \
              "$(cfg "components.$idx.install.pip_deps" json 2>/dev/null || echo '[]')")"
    if [[ -n "$specs" ]]; then
      mkdir -p "$adir/pip_deps"
      rm -f "$adir"/pip_deps/*.whl 2>/dev/null || true
      log "  $name: download declared deps: $specs"
      pip3 download --no-deps --progress-bar off -d "$adir/pip_deps" $specs \
        >> "$WORK/logs/verify-$name.log" 2>&1 \
        || die "$name: pip download failed for $specs"
    fi
    if ! (
      export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLLM_VENDOR_DIR=""
      shopt -s nullglob
      _pd=("$adir"/pip_deps/*.whl)
      if [[ ${#_pd[@]} -gt 0 ]]; then
        for _p in "${_pd[@]}"; do pip3 install --no-deps --no-index --progress-bar off "$_p"; done
      fi
      _whls=("$adir"/*.whl)
      if [[ ${#_whls[@]} -eq 0 ]]; then echo "no .whl artifact in $adir"; exit 3; fi
      for _w in "${_whls[@]}"; do
        echo "### install: $_w"
        pip3 install --force-reinstall --no-deps --progress-bar off "$_w"
      done
      python3 "$SCRIPT_DIR/check_wheel_deps.py" "${_whls[@]}"
    ) >> "$WORK/logs/verify-$name.log" 2>&1; then
      tail -20 "$WORK/logs/verify-$name.log"
      die "$name: install/verify failed (log: $WORK/logs/verify-$name.log)"
    fi
  fi

  # run verify commands (verify + verify_device when an NPU is attached)
  run_verify_cmds "$idx" "$name" "$vendor" "$adir" "$WORK/logs/verify-$name.log" \
    || die "$name: verify commands failed (log: $WORK/logs/verify-$name.log)"

  # rebuild checksums and collect file metadata
  ( cd "$adir" && rm -f SHA256SUMS \
      && find . -mindepth 1 -maxdepth 1 -type f ! -name '.*' -printf '%f\0' \
      | xargs -0 --no-run-if-empty sha256sum > SHA256SUMS ) \
    || die "$name: sha256sums failed"
  local files_json
  files_json="$(cd "$adir" && python3 - <<'PY'
import json, os
out = []
if os.path.exists("SHA256SUMS"):
    for line in open("SHA256SUMS", encoding="utf-8"):
        sha, _, fname = line.strip().partition("  ")
        if fname and fname != "SHA256SUMS" and not fname.startswith(".") and os.path.exists(fname):
            out.append({"name": fname, "sha256": sha, "bytes": os.path.getsize(fname)})
print(json.dumps(out))
PY
)"
  if [[ "$files_json" == "[]" ]]; then die "$name: metadata collection failed"; fi
  # same-name wheel appearing multiple times = cross-platform residue, fail fast
  python3 - "$files_json" <<'PY' || die "$name: duplicate wheel distribution (cross-platform residue? rebuild with --force)"
import json, sys
from collections import Counter
names = [f["name"] for f in json.loads(sys.argv[1]) if f["name"].endswith(".whl")]
dups = [n for n, c in Counter(f.split("-")[0] for f in names).items() if c > 1]
if dups:
    print("duplicate distributions: %s" % dups, file=sys.stderr)
    raise SystemExit(1)
PY

  local built_commit
  built_commit="$(cat "$adir/.built-commit" 2>/dev/null || echo "$SRC_COMMIT")"
  python3 - "$WORK/meta/$PLATFORM/$name.json" "$name" "$type" \
    "$(cfg "components.$idx.repo")" "$(cfg "components.$idx.ref")" "$built_commit" \
    "$SOC" "$ARCH" "$(basename "${ASCEND_HOME_PATH:-local}")" "$vendor" "$files_json" <<'PY'
import json, sys, time
path, name, ctype, repo, ref, commit, soc, arch, cann, vendor_dir, files_json = sys.argv[1:12]
meta = {
    "name": name, "type": ctype, "repo": repo, "ref": ref, "commit": commit,
    "soc": soc, "arch": arch, "built_with": "local",
    "target_cann": cann,
    "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "files": json.loads(files_json or "[]"),
    "warnings": [],
}
if vendor_dir:
    meta["vendor_dir"] = vendor_dir
with open(path, "w", encoding="utf-8") as fh:
    json.dump(meta, fh, ensure_ascii=False, indent=2)
PY
  log "  $name: verified, metadata written"
}

run_verify_cmds() { # idx name vendor adir logfile
  local idx="$1" name="$2" vendor="$3" adir="$4" logf="$5"
  local vcmds
  vcmds="$(python3 - "$DEPS" "$idx" <<'PY'
import json, os, sys, yaml
idx = int(sys.argv[2])
comp = yaml.safe_load(open(sys.argv[1]))["components"][idx]
out = list(comp.get("verify") or [])
if os.path.exists("/dev/davinci_manager"):
    out += list(comp.get("verify_device") or [])
print(json.dumps(out))
PY
)" || vcmds='[]'
  if [[ "$vcmds" == "[]" ]]; then return 0; fi
  local built_commit
  built_commit="$(cat "$adir/.built-commit" 2>/dev/null || echo "")"
  python3 - "$DEPS" "$idx" "$vcmds" "$vendor" "$SOC" "$name" "$built_commit" "$adir" > "$WORK/logs/verify-cmds-$name.sh" <<'PY'
import json, sys, yaml
deps, idx_s, vcmds, vendor, soc, name, commit, adir = sys.argv[1:9]
idx = int(idx_s)
comp = yaml.safe_load(open(deps))["components"][idx]
mapping = {
    "vendor_dir": vendor, "soc": soc, "name": name,
    "commit": commit, "artifacts_dir": adir,
}
for vk, vv in ((comp.get("build") or {}).get("vars") or {}).items():
    if isinstance(vv, str):
        vv = " ".join(vv.split())
    mapping.setdefault(vk, vv)
print("#!/bin/bash")
print("set -e")
print("export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1")
for c in json.loads(vcmds):
    c = str(c)
    for k, v in mapping.items():
        c = c.replace("{%s}" % k, str(v))
    print('echo "### verify: %s"' % c.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n"))
    print(c)
PY
  if ! bash "$WORK/logs/verify-cmds-$name.sh" >> "$logf" 2>&1; then
    return 1
  fi
  return 0
}

# ---------- assemble ----------
assemble() {
  local out="${OUT:-$WORK/bundle/xllm-deps-$PLATFORM.run}"
  local accp_base
  # baseline comes from the current process environment (same as install.sh);
  # probing with `bash -ic` would source this bundle's own /etc/profile.d/
  # xllm-deps.sh into the "baseline" and can hang without a tty
  accp_base="${ASCEND_CUSTOM_OPP_PATH:-}"
  accp_base="${accp_base%:}"
  python3 "$SCRIPT_DIR/make_bundle.py" \
    --deps "$DEPS" \
    --meta-dir "$WORK/meta/$PLATFORM" \
    --artifacts-dir "$WORK/artifacts" \
    --platform "$PLATFORM" \
    --soc "$SOC" \
    --accp-seed "$accp_base" \
    --scripts-dir "$SCRIPT_DIR" \
    --out "$out" || die "bundle assembly failed"
  log "bundle: $out (+ .sha256)"
  log "install: bash $out  (root; supports reinstall over a previous bundle)"
}

# ---------- main ----------
for (( i=0; i<N_COMP; i++ )); do
  name="$(cfg "components.$i.name")"
  if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then continue; fi
  # skip components that don't support the current soc (declared in depends config)
  SUPPORTED_SOCS="$(cfg "components.$i.supported_socs" json 2>/dev/null)" || SUPPORTED_SOCS=""
  if [[ -n "$SUPPORTED_SOCS" ]] && ! python3 -c "import json,sys; sys.exit(0 if sys.argv[1] in json.loads(sys.argv[2]) else 1)" "$SOC" "$SUPPORTED_SOCS" 2>/dev/null; then
    log "component [$((i+1))/$N_COMP] $name: skipped (soc=$SOC not in supported_socs)"
    continue
  fi
  log "component [$((i+1))/$N_COMP] $name ($(cfg "components.$i.type"), ref=$(cfg "components.$i.ref"))"
  ensure_source "$i"
  build_component "$i"
  verify_component "$i"
done

if [[ -n "$ONLY" ]]; then
  log "single-component mode: skipping bundle assembly"
else
  assemble
fi
