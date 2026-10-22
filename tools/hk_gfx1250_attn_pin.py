###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pin the gfx1250 HipKittens attention code objects: compile, read their metadata, rewrite manifest.json.

    python tools/hk_gfx1250_attn_pin.py --kernels a,b --against DIR[:DIR...] [--record FILE] [--cxx PATH]
                                        [--allow-spill] [--no-bin] [--package-dir DIR]
    python tools/hk_gfx1250_attn_pin.py --check [--kernels a,b] [--against DIR[:DIR...]]
    python tools/hk_gfx1250_attn_pin.py --status

Writing pins needs the kernels named (``--kernels``) and the code objects the GPU validation ran
(``--against``, below); ``--unvalidated`` replaces ``--against`` (toy kernels, tests: never for kernels a
production run may load). For every selected kernel entry of ``<package>/manifest.json`` (``--check``
without ``--kernels``: every entry whose source exists), compiles the source exactly as setup.py does
(``hsaco.compile_kernel``: the manifest's flags, -cuid, the entry's extra flags, device only) with the
current toolchain, then writes into the entry

    hsaco_sha256   of the code object,
    block          the workgroup size the entry declares (checked against max_flat_workgroup_size),
    kernarg_size   the kernarg segment (no hidden arguments allowed),
    args           [{name, kind, offset, size}] from the entry's [name, kind] list and the code object's
                   offsets and sizes (``hiprt.kernel_meta``: count, sizes and pointer-ness must agree),
    lds, vgpr, sgpr, wavefront_size

and the manifest's ``toolchain``; the code objects are copied into ``<package>/bin`` (``--no-bin``: not).
An entry may list several kernels of one code object under ``symbols`` ({symbol: {block, args}}); each
gets the same fields there. Refused (nothing written): a compile error, a symbol missing from the code
object, arguments that disagree with it, scratch / VGPR or SGPR spills / a dynamic stack (unless
``--allow-spill``), and re-pinning a subset with another toolchain than the other pinned entries were
pinned with (re-pin all of them then).

Pin only sources that were validated on a GPU: the pin is what makes the HIPKITTENS backend load a code
object. ``--against DIR`` (``hk/lab/cc.sh`` output directories, the code objects the GPU runs used; several
joined by ``os.pathsep`` or given repeatedly, each entry taken from the first one holding
``<entry>.hsaco``) refuses to pin unless every compiled code object is byte-identical to that file, so the
pins are the validated bytes (same sources, flags and toolchain). Before compiling it also compares the
entry's ``extra_flags`` with the flags cc.sh recorded (``<entry>.meta.json`` / ``<entry>.flags``) and
refuses a cc.sh build whose meta.json says it failed or names other bytes. ``--record FILE`` (the JSON
lines the GPU tests append with ``HK_VALIDATION_RECORD``) further refuses any code object without a
record of its direction's full and production-shape tests passing on those bytes. ``--check`` compiles
and compares with the pins without writing; ``--status`` only reports.
Standard library only (runs in a device-less compile container).
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

DEFAULT_PACKAGE = Path(__file__).resolve().parents[1] / "primus_turbo" / "hipkittens" / "gfx1250_attn"
_TAG = "_hk_pin_pkg"
# cc.sh's own flags after -cuid=<name>, before the extra flags it was given (warnings: no code change)
_CC_SH_WARNINGS = ("-Wuninitialized", "-Wsometimes-uninitialized")
# what a validation record must show for a code object's direction (test_attention_hk_gfx1250.py)
RECORD_STAGES = ("full", "large")


class PinError(RuntimeError):
    pass


def load_package(pkg_dir):
    """The package at ``pkg_dir`` under a private name (its relative imports work, its __init__ runs),
    with its hsaco, codeobj and hiprt modules (all standard library at import)."""
    pkg_dir = Path(pkg_dir).resolve()
    for m in [m for m in sys.modules if m == _TAG or m.startswith(_TAG + ".")]:
        del sys.modules[m]
    init = pkg_dir / "__init__.py"
    if init.is_file():
        spec = importlib.util.spec_from_file_location(_TAG, init, submodule_search_locations=[str(pkg_dir)])
    else:
        spec = importlib.util.spec_from_loader(_TAG, loader=None, is_package=True)
        spec.submodule_search_locations.append(str(pkg_dir))
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[_TAG] = pkg
    if spec.loader is not None:
        spec.loader.exec_module(pkg)
    return {m: importlib.import_module(f"{_TAG}.{m}") for m in ("hsaco", "codeobj", "hiprt")}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _arg_spec(args, where):
    """[(name, kind)] of an entry's args: [name, kind] pairs or {name, kind, ...} dicts."""
    if not args:
        raise PinError(f"{where}: no args (fill [name, kind] per kernel argument, in kernarg order)")
    out = []
    for a in args:
        if isinstance(a, dict):
            out.append((a["name"], a["kind"]))
        elif isinstance(a, (list, tuple)) and len(a) == 2:
            out.append((a[0], a[1]))
        else:
            raise PinError(f"{where}: bad argument spec {a!r}")
    return out


def _symbol_specs(name, entry):
    """{symbol: (block, [(arg, kind)])} of one entry: ``symbols`` or the entry itself (symbol = entry name)."""
    syms = entry.get("symbols") or {name: {"block": entry.get("block"), "args": entry.get("args")}}
    out = {}
    for sym, spec in syms.items():
        if not isinstance(spec.get("block"), int):
            raise PinError(f"{name}/{sym}: no block (the workgroup size the module launches with)")
        out[sym] = (spec["block"], _arg_spec(spec.get("args"), f"{name}/{sym}"))
    return out


def inspect_code_object(mods, name, entry, path, allow_spill):
    """{symbol: manifest fields} of a freshly built code object; PinError on any disagreement."""
    table = mods["codeobj"].kernels(path)
    fields = {}
    for sym, (block, spec) in _symbol_specs(name, entry).items():
        if sym not in table:
            raise PinError(f"{name}: kernel {sym} is not in the code object ({sorted(table)})")
        co = table[sym]
        try:
            meta = mods["hiprt"].kernel_meta(co, sym, {"block": block, "args": spec})
        except mods["hiprt"].HipError as e:
            raise PinError(str(e)) from e
        bad = []
        if co["scratch"]:
            bad.append(f"scratch {co['scratch']} B")
        if co["vgpr_spill"] or co["sgpr_spill"]:
            bad.append(f"spills vgpr {co['vgpr_spill']} / sgpr {co['sgpr_spill']}")
        if co.get("dynamic_stack"):
            bad.append("dynamic stack")
        if bad and not allow_spill:
            raise PinError(f"{name}/{sym}: {', '.join(bad)} (--allow-spill to pin anyway)")
        fields[sym] = {
            "block": meta["block"],
            "kernarg_size": meta["kernarg_size"],
            "lds": co["lds"],
            "vgpr": co["vgpr"],
            "sgpr": co["sgpr"],
            "wavefront_size": co["wavefront_size"],
            "args": meta["args"],
        }
    return fields


def status(mods, out=print):
    hs = mods["hsaco"]
    m = json.loads(hs.MANIFEST_PATH.read_text())
    out(f"manifest {hs.MANIFEST_PATH} schema {m.get('schema')} toolchain {m.get('toolchain')}")
    for name, e in m["kernels"].items():
        src = (hs.HIP_DIR / e["src"]).is_file()
        pinned = hs.is_pinned(e)
        b = hs.binary_path(name)
        built = b.is_file() and pinned and _sha256(b) == e["hsaco_sha256"]
        out(
            f"  {name:24s} module={e.get('module')!s:8s} src={'yes' if src else 'MISSING'} "
            f"pinned={'yes' if pinned else 'no'} bin={'ok' if built else '-'}"
        )
    return 0


def search_dirs(against):
    """[Path] of ``--against`` values: each one a directory or several joined by ``os.pathsep``."""
    if against is None:
        return []
    values = [against] if isinstance(against, (str, Path)) else list(against)
    return [Path(d) for v in values for d in str(v).split(os.pathsep) if d]


def find_reference(hs, name, dirs):
    """The first ``<dir>/<name>.hsaco`` that exists, or None."""
    for d in dirs:
        ref = hs.binary_path(name, d)
        if ref.is_file():
            return ref
    return None


def lab_extra_flags(ref):
    """The extra flags cc.sh compiled ``ref`` (``<dir>/<name>.hsaco``) with, from ``<name>.meta.json``
    ("flags") or ``<name>.flags``; None when neither records them."""
    name = ref.stem
    meta, flags_file = ref.with_suffix(".meta.json"), ref.with_suffix(".flags")
    cmd = None
    if meta.is_file():
        cmd = json.loads(meta.read_text()).get("flags")
    if cmd is None and flags_file.is_file():
        cmd = [line for line in flags_file.read_text().splitlines() if line]
    if not cmd or f"-cuid={name}" not in cmd or "-c" not in cmd:
        return None
    start = cmd.index(f"-cuid={name}") + 1
    end = len(cmd) - 1 - cmd[::-1].index("-c")
    extra = cmd[start:end]
    while extra and extra[0] in _CC_SH_WARNINGS:
        extra = extra[1:]
    return extra


def _check_lab_builds(hs, entries, names, dirs):
    """PinError unless each kernel's reference build exists, its cc.sh meta.json (when present) says it
    succeeded and names these bytes, and the flags cc.sh used are the entry's extra_flags."""
    bad = []
    for name in names:
        ref = find_reference(hs, name, dirs)
        if ref is None:
            bad.append(f"{name}: no {name}.hsaco in {os.pathsep.join(map(str, dirs))}")
            continue
        meta = ref.with_suffix(".meta.json")
        if meta.is_file():
            m = json.loads(meta.read_text())
            if not m.get("ok", False) or m.get("hsaco_sha256") not in (None, _sha256(ref)):
                bad.append(f"{name}: {meta} says ok={m.get('ok')} sha256 {str(m.get('hsaco_sha256'))[:16]}")
                continue
        lab = lab_extra_flags(ref)
        want = list(entries[name].get("extra_flags", []))
        if lab is not None and lab != want:
            bad.append(f"{name}: manifest extra_flags {want} but {ref.parent} was built with {lab}")
    if bad:
        raise PinError("not the validated builds: " + "; ".join(bad))


def load_records(path):
    """The JSON lines of a validation record file (unreadable lines are skipped)."""
    out = []
    for line in Path(path).read_text().splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def record_reason(records, name, sha, stages=RECORD_STAGES):
    """None when the records of these bytes together show ``stages`` passed for the code object's direction
    (the toy, full and production-shape runs are separate test sessions, one line each), else why not."""
    got, found = set(), False
    for rec in records:
        co = (rec.get("code_objects") or {}).get(name) or {}
        if co.get("sha256") != sha:
            continue
        found = True
        passed = (rec.get("stages") or {}).get(co.get("direction")) or {}
        got |= {s for s in stages if passed.get(s)}
    if not found:
        return f"{name} {sha[:16]}: no validation record"
    missing = [s for s in stages if s not in got]
    return f"{name} {sha[:16]}: stages {missing} not recorded (need {list(stages)})" if missing else None


def pin(
    mods,
    kernels=None,
    cxx=None,
    check=False,
    allow_spill=False,
    copy_bin=True,
    out=print,
    against=None,
    unvalidated=False,
    record=None,
):
    """Compile and (unless ``check``) pin; returns the process exit status. Writing needs ``kernels`` and
    ``against`` (or ``unvalidated``)."""
    hs = mods["hsaco"]
    path = hs.MANIFEST_PATH
    m = json.loads(path.read_text())
    entries = m["kernels"]
    dirs = search_dirs(against)
    if not check:
        if not kernels:
            raise PinError(
                "name the kernels to pin (--kernels a,b): only kernels validated on a GPU are pinned"
            )
        if not dirs and not unvalidated:
            raise PinError(
                "--against <cc.sh dir of the validated build> is required to write pins "
                "(--unvalidated only for toy kernels no production run loads)"
            )
    if kernels:
        unknown = [n for n in kernels if n not in entries]
        if unknown:
            raise PinError(f"not in {path.name}: {unknown}")
        names = list(kernels)
        missing = [n for n in names if not (hs.HIP_DIR / entries[n]["src"]).is_file()]
        if missing:
            raise PinError(f"sources missing for {missing}")
    else:
        names = [n for n, e in entries.items() if (hs.HIP_DIR / e["src"]).is_file()]
        skipped = [n for n in entries if n not in names]
        if skipped:
            out(f"no source yet, not pinned: {', '.join(skipped)}")
    if not names:
        raise PinError("no kernel with a source to pin")
    if dirs:
        _check_lab_builds(hs, entries, names, dirs)
    cxx = hs.find_compiler(cxx)
    if cxx is None:
        raise PinError(f"amdclang++ not found (set {hs.CXX_ENV} or --cxx)")
    toolchain = hs.compiler_version(cxx)
    others = [n for n, e in entries.items() if n not in names and hs.is_pinned(e)]
    if not check and others and m.get("toolchain") not in (None, toolchain):
        raise PinError(
            f"{others} are pinned with {m.get('toolchain')!r}, this compiler is {toolchain!r}: "
            "re-pin every kernel (--kernels with all of them, or no --kernels)"
        )
    hs.reload_manifest()  # compile_kernel reads the manifest from disk
    built = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            obj = Path(tmp) / f"{name}.hsaco"
            try:
                hs.compile_kernel(cxx, name, obj)
            except RuntimeError as e:
                raise PinError(str(e)) from e
            built[name] = (_sha256(obj), inspect_code_object(mods, name, entries[name], obj, allow_spill))
        if dirs:
            _check_against(hs, built, dirs, out)
        if record is not None:
            records = load_records(record)
            bad = [r for r in (record_reason(records, n, sha) for n, (sha, _) in built.items()) if r]
            if bad:
                raise PinError(f"not validated per {record}: " + "; ".join(bad))
            out(f"every code object has a passing validation record in {record}")
        if not check and not dirs:
            out("WARNING: --unvalidated: pinned without comparing with a validated build")
        return _report_and_write(mods, m, built, toolchain, check, copy_bin, Path(tmp), out)


def _check_against(hs, built, dirs, out):
    """PinError unless every built code object equals its reference ``<dir>/<entry>.hsaco`` byte for byte."""
    bad, refs = [], []
    for name, (sha, _) in built.items():
        ref = find_reference(hs, name, dirs)
        got = _sha256(ref) if ref is not None else None
        if got != sha:
            where = "missing" if got is None else f"{got[:16]} in {ref.parent}"
            bad.append(f"{name}: {where}, built {sha[:16]}")
        else:
            refs.append(f"{name} = {ref}")
    if bad:
        raise PinError(
            "not the validated code objects (other sources, extra_flags or toolchain?): " + "; ".join(bad)
        )
    out("byte-identical to the validated builds: " + ", ".join(refs))


def _report_and_write(mods, m, built, toolchain, check, copy_bin, tmp, out):
    """Print the pins; unless ``check``, write the manifest, then copy the code objects into bin/ (only
    once every kernel compiled and passed, so a failed run leaves bin/ and the pins as they were)."""
    hs = mods["hsaco"]
    path, entries = hs.MANIFEST_PATH, m["kernels"]
    out(f"toolchain: {toolchain}")
    drift = 0
    for name, (sha, fields) in built.items():
        old = entries[name].get("hsaco_sha256")
        mark = "" if old == sha else (" (new pin)" if old is None else f" (was {str(old)[:16]})")
        drift += old != sha
        for sym, f in fields.items():
            out(
                f"  {name:20s} {sym:22s} sha256 {sha[:16]}{mark} vgpr {f['vgpr']} sgpr {f['sgpr']} "
                f"lds {f['lds']} kernarg {f['kernarg_size']} block {f['block']}"
            )
    if check:
        out(
            "check: every pin matches"
            if not drift
            else f"check: {drift} code object(s) differ from their pin"
        )
        return 1 if drift else 0
    for name, (sha, fields) in built.items():
        e = entries[name]
        e["hsaco_sha256"] = sha
        if e.get("symbols"):
            for sym, f in fields.items():
                e["symbols"][sym].update(f)
        else:
            e.update(fields[name])
    m["toolchain"] = toolchain
    path.write_text(json.dumps(m, indent=2) + "\n")
    if copy_bin:
        for name in built:
            hs.BIN_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tmp / f"{name}.hsaco", hs.binary_path(name))
    hs.reload_manifest()
    out(
        f"wrote {path} ({len(built)} pinned{'; code objects copied to ' + str(hs.BIN_DIR) if copy_bin else ''})"
    )
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--package-dir", default=str(DEFAULT_PACKAGE), help="HK package (default: %(default)s)")
    ap.add_argument("--kernels", default=None, help="comma-separated entries (required to write pins)")
    ap.add_argument("--cxx", default=None, help="compiler (default: $PRIMUS_TURBO_HK_CXX, PATH, $ROCM_PATH)")
    ap.add_argument("--check", action="store_true", help="compile and compare with the pins; write nothing")
    ap.add_argument("--status", action="store_true", help="report sources, pins and bin/ only")
    ap.add_argument("--allow-spill", action="store_true", help="pin despite scratch / spills")
    ap.add_argument("--no-bin", action="store_true", help="do not copy the code objects into bin/")
    ap.add_argument(
        "--against",
        action="append",
        default=None,
        help="validated cc.sh dirs (os.pathsep-joined or repeated): refuse unless byte-identical (required to pin)",
    )
    ap.add_argument("--unvalidated", action="store_true", help="pin without --against (toy kernels, tests)")
    ap.add_argument(
        "--record", default=None, help="validation record (JSON lines) the code objects must pass"
    )
    a = ap.parse_args(argv)
    mods = load_package(a.package_dir)
    if a.status:
        return status(mods)
    kernels = [k for k in a.kernels.split(",") if k] if a.kernels else None
    try:
        return pin(
            mods,
            kernels,
            a.cxx,
            a.check,
            a.allow_spill,
            not a.no_bin,
            against=a.against,
            unvalidated=a.unvalidated,
            record=a.record,
        )
    except PinError as e:
        print(f"hk_gfx1250_attn_pin: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
