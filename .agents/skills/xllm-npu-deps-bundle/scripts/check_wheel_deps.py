#!/usr/bin/env python3
"""Wheel dependency reconciliation: wheels install with --no-deps everywhere,
so their Requires-Dist must be checked explicitly.

Why pip dependency resolution is disabled:
  * the NPU stack (torch/torch_npu/triton-ascend/numpy/CANN) is pinned as a
    matched set; the pip resolver silently upgrades/downgrades them to
    satisfy loose wheel declarations, breaking the match;
  * resolution needs network access to an index, while bundle installation
    is offline by design (all payloads are local files).
Instead: declare anything missing in the depends config (downloaded at build
time, installed offline at install time) and reconcile with this script.

Usage: check_wheel_deps.py <a.whl> [b.whl ...]
Reads Requires-Dist from each wheel's METADATA:
  * entries with an extra marker or a false environment marker are skipped
  * everything else must be installed with a satisfying version, otherwise
    it is reported as MISSING / MISMATCH
Exit code 1 means unsatisfied entries (the list is printed).
"""

from __future__ import annotations

import email
import sys
import zipfile

try:  # packaging may not be installed standalone, fall back to pip's vendored copy
    from packaging.markers import Marker
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
except ImportError:  # pragma: no cover
    from pip._vendor.packaging.markers import Marker  # type: ignore
    from pip._vendor.packaging.requirements import Requirement  # type: ignore
    from pip._vendor.packaging.specifiers import SpecifierSet  # type: ignore
    from pip._vendor.packaging.version import Version  # type: ignore

try:
    from importlib import metadata as md
except ImportError:  # py<3.8
    import importlib_metadata as md  # type: ignore


def requires_of(wheel_path):
    with zipfile.ZipFile(wheel_path) as zf:
        meta_names = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
        if not meta_names:
            raise SystemExit(f"check_wheel_deps: no METADATA found in {wheel_path}")
        msg = email.message_from_string(zf.read(meta_names[0]).decode("utf-8", "replace"))
    return msg.get_all("Requires-Dist") or [], msg.get("Name") or ""


def installed_version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def main(argv):
    wheels = [a for a in argv[1:] if a.endswith(".whl")]
    if not wheels:
        print("check_wheel_deps: no .whl files provided", file=sys.stderr)
        return 2

    bad = 0
    for w in wheels:
        reqs, dist = requires_of(w)
        print(f"=== {dist or w.rsplit('/', 1)[-1]}: Requires-Dist {len(reqs)} entries ===")
        if not reqs:
            print("  (no declared dependencies)")
            continue
        for raw in reqs:
            try:
                req = Requirement(raw)
            except Exception as exc:
                print(f"  SKIP unparsable: {raw} ({exc})")
                continue
            if req.marker is not None and not req.marker.evaluate():
                print(f"  SKIP marker not applicable: {raw}")
                continue
            have = installed_version(req.name)
            if have is None:
                bad += 1
                print(f"  MISSING {req.name}  <- {raw}")
                continue
            spec = req.specifier if req.specifier is not None else SpecifierSet("")
            if not spec.contains(have, prereleases=True):
                bad += 1
                print(f"  MISMATCH {req.name}: installed {have}, requires {spec or 'any'}")
            else:
                print(f"  ok      {req.name}=={have} ({spec or 'any'})")

    if bad:
        print(
            f"\n{bad} unsatisfied dependencies. Fix: declare install.pip_deps for the "
            f"component in the depends config (downloaded at build time, installed "
            f"offline at install time), or add it to the base dev image; do not use "
            f"pip install with dependency resolution.",
            file=sys.stderr,
        )
        return 1
    print("\ndependency check passed: all declared Requires-Dist entries satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
