import io, sys
OLD = '''        _KEEPALIVE.append({k: v for k, v in sys.modules.items() if _is_pt(k)})
        for k in [k for k in sys.modules if _is_pt(k)]:
            del sys.modules[k]
        sys.modules.update(saved)
'''
NEW = '''        # Move, do not delete. The tree stays registered in `sys.modules` under a
        # private prefix, so no module object is ever torn down and no
        # `torch.library` registration it owns is ever destroyed; the name
        # `primus_turbo` is simply free again, which is all `op/baseline`'s guard
        # looks at. Deleting the modules and holding them in a list instead was
        # not enough: two runs died later with
        # `schema_.has_value() INTERNAL ASSERT FAILED ... Tried to access the
        # schema for`, the dispatcher having lost an operator whose defining
        # module Python had begun to tear down.
        _prefix = "_op_evolve_%d." % len(_CACHE)
        for k in [k for k in sys.modules if _is_pt(k)]:
            sys.modules[_prefix + k] = sys.modules.pop(k)
        sys.modules.update(saved)
'''
for path in sys.argv[1:]:
    s = io.open(path).read()
    assert s.count(OLD) == 1, path
    s = s.replace(OLD, NEW)
    s = s.replace('_KEEPALIVE = builtins.__dict__.setdefault("_op_evolve_vendor_keepalive", [])\n', '')
    io.open(path, "w").write(s)
    print("patched", path)
