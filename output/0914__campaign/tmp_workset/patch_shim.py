import io, sys
OLD = '''    from torch._library import opaque_object as _oo

    opaque_before = set(_oo._OPAQUE_TYPES_BY_NAME)
    sys.path.insert(0, _VENDOR)'''
NEW = '''    _allow_reregistering_opaque_types()
    sys.path.insert(0, _VENDOR)'''

OLD2 = '''        sys.modules.update(saved)
        # Second half of the same harness fix. `primus_turbo.pytorch.core.
        # low_precision` calls `register_opaque_type` at import, and that
        # registry is process-global in torch, so the NEXT genuine import of a
        # primus_turbo tree in this process -- op/baseline's -- dies with
        # "already registered as an opaque type". Undoing our registrations
        # leaves the process as found, which is the whole contract of this
        # function. The types are fp8 quantisation configs; nothing in the bf16
        # attention path under measurement touches them.
        for _name in set(_oo._OPAQUE_TYPES_BY_NAME) - opaque_before:
            _info = _oo._OPAQUE_TYPES_BY_NAME.pop(_name)
            for _cls, _i in list(_oo._OPAQUE_TYPES.items()):
                if _i is _info:
                    del _oo._OPAQUE_TYPES[_cls]
            torch._C._unregister_opaque_type(_name)
'''
NEW2 = '''        sys.modules.update(saved)
'''

HELPER = '''def _allow_reregistering_opaque_types():
    """Second half of the same harness fix.

    `primus_turbo.pytorch.core.low_precision` calls `register_opaque_type` at
    import, and torch's opaque-type registry is process-global and keyed by
    qualname. So the SECOND primus_turbo tree imported into this process --
    ours and then `op/baseline`'s -- dies with "Type '...Float8QuantConfig' is
    already registered as an opaque type", whichever order they come in.

    Make the C-level registration idempotent instead of unregistering ours
    afterwards: unregistering perturbs a registry that the dispatcher also
    reads, and it cost this round two runs that died later, elsewhere, with
    `schema_.has_value() INTERNAL ASSERT FAILED`. The types are fp8
    quantisation configs; the two registrations are the same class from two
    identical copies of the same file, and nothing in the bf16 attention path
    under measurement touches them.
    """
    if getattr(torch._C._register_opaque_type, "_op_evolve_shim", False):
        return
    _orig = torch._C._register_opaque_type

    def _shim(name):
        if torch._C._is_opaque_type_registered(name):
            return
        return _orig(name)

    _shim._op_evolve_shim = True
    torch._C._register_opaque_type = _shim


'''
for path in sys.argv[1:]:
    s = io.open(path).read()
    for o, n in ((OLD, NEW), (OLD2, NEW2)):
        assert s.count(o) == 1, (path, o[:40])
        s = s.replace(o, n)
    anchor = "def _import_vendored():"
    assert s.count(anchor) == 1
    s = s.replace(anchor, HELPER + anchor)
    io.open(path, "w").write(s)
    print("patched", path)
