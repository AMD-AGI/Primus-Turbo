#!/usr/bin/env python3
"""Read the provenance of an op-evolve refcache/*.pt WITHOUT torch (stdlib only, no GPU).

    python3 refcache_prov.py <job_context/op> [<eager impl.py to hash instead of op/eager/impl.py>]

torch.save writes a zip whose data.pkl references storages by persistent id; a stub
unpickler records tensor metadata instead of rebuilding tensors. Prints, per cache file:
the recorded provenance, the tensor dtypes/shapes, the sha256 of each storage blob, and
whether eager_sha / common_sha / dims match the files currently on disk.
"""
import ast
import hashlib
import pickle
import sys
import zipfile
from pathlib import Path


class _Storage:
    def __init__(self, pid):
        self.pid = pid


class _Tensor:
    def __init__(self, storage, offset, size, stride, *rest):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)

    def __repr__(self):
        st = self.storage
        return f"Tensor(dtype={st.pid[1]}, key={st.pid[2]}, numel_storage={st.pid[4]}, size={self.size})"


class _Stub:
    def __init__(self, name):
        self.name = name

    def __call__(self, *a, **k):
        return self

    def __repr__(self):
        return self.name


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in (("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_tensor")):
            return _Tensor
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        if module.startswith("torch"):
            return _Stub(f"{module}.{name}")
        return super().find_class(module, name)

    def persistent_load(self, pid):
        # ('storage', <StorageType>, key, location, numel)
        return _Storage(tuple(pid))


def _sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def _shapes(common_py):
    tree = ast.parse(Path(common_py).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SHAPES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("SHAPES not found")


def main():
    op = Path(sys.argv[1]).resolve()
    eager = Path(sys.argv[2]) if len(sys.argv) > 2 else op / "eager" / "impl.py"   # optional: another eager file
    eager_sha, common_sha = _sha16(eager), _sha16(op / "ut" / "common.py")
    shapes = _shapes(op / "ut" / "common.py")
    print(f"on disk: eager/impl.py sha16={eager_sha}  ut/common.py sha16={common_sha}")
    ok_all = True
    for pt in sorted((op / "refcache").glob("*.pt")):
        with zipfile.ZipFile(pt) as z:
            pkl = next(n for n in z.namelist() if n.endswith("/data.pkl"))
            root = pkl[: -len("data.pkl")]
            blob = _Unpickler(z.open(pkl)).load()
            blobs = {n[len(root) + 5:]: hashlib.sha256(z.read(n)).hexdigest()[:16]
                     for n in z.namelist() if n.startswith(root + "data/")}
        prov = blob.get("provenance", {})
        name = pt.stem
        want = {"shape": name, "dims": tuple(shapes[name]), "seed": 0,
                "eager_sha": eager_sha, "common_sha": common_sha}
        bad = [k for k in want if (tuple(prov.get(k, ())) if k == "dims" else prov.get(k)) != want[k]]
        ok_all &= not bad
        print(f"{pt.name}: provenance {prov}")
        print(f"  o   {blob.get('o')!r}\n  lse {blob.get('lse')!r}")
        print(f"  storages sha16 {blobs}")
        print(f"  check vs disk: {'MATCH (cache will be used)' if not bad else 'MISMATCH on ' + ','.join(bad)}")
    print("RESULT:", "ALL MATCH" if ok_all else "MISMATCH")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
