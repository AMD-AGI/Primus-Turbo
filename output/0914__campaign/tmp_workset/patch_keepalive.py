import io, sys
OLD = '''        sys.path.remove(_VENDOR)
        for k in [k for k in sys.modules if _is_pt(k)]:
            del sys.modules[k]
        sys.modules.update(saved)
'''
NEW = '''        sys.path.remove(_VENDOR)
        # Hold a reference to EVERY module the import created, not just the
        # three named above. The vendored package defines its forward as a
        # `torch.library.custom_op` in a module nothing else keeps alive; once
        # that module is dropped from `sys.modules` its `Library` destructor
        # deregisters the schema, and the next call dies at a whim of the
        # collector with `schema_.has_value() INTERNAL ASSERT FAILED ... Tried
        # to access the schema for .` -- which is exactly what it did.
        _KEEPALIVE.append({k: v for k, v in sys.modules.items() if _is_pt(k)})
        for k in [k for k in sys.modules if _is_pt(k)]:
            del sys.modules[k]
        sys.modules.update(saved)
'''
OLD2 = '''_CACHE = builtins.__dict__.setdefault("_op_evolve_vendor_cache", {})'''
NEW2 = '''_CACHE = builtins.__dict__.setdefault("_op_evolve_vendor_cache", {})
_KEEPALIVE = builtins.__dict__.setdefault("_op_evolve_vendor_keepalive", [])'''
for path in sys.argv[1:]:
    s = io.open(path).read()
    for o, n in ((OLD, NEW), (OLD2, NEW2)):
        assert s.count(o) == 1, (path, o[:30])
        s = s.replace(o, n)
    io.open(path, "w").write(s)
    print("patched", path)
