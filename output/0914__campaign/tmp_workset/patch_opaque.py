import io, sys

OLD = '''    if _VENDOR in _CACHE:
        return _CACHE[_VENDOR]
    saved = {k: v for k, v in sys.modules.items() if _is_pt(k)}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, _VENDOR)
    try:
        import primus_turbo.pytorch.kernels.attention.attention_fused_bwd_impl as fused
        import primus_turbo.pytorch.kernels.attention.attention_triton_impl as tri
        import primus_turbo.triton.attention.fused_mha_bwd_kernel as bwd_kernel
        for mod in (fused, tri, bwd_kernel):
            assert mod.__file__.startswith(_VENDOR), mod.__file__
    finally:
        sys.path.remove(_VENDOR)
        for k in [k for k in sys.modules if _is_pt(k)]:
            del sys.modules[k]
        sys.modules.update(saved)
'''

NEW = '''    if _VENDOR in _CACHE:
        return _CACHE[_VENDOR]
    saved = {k: v for k, v in sys.modules.items() if _is_pt(k)}
    for k in saved:
        del sys.modules[k]
    from torch._library import opaque_object as _oo

    opaque_before = set(_oo._OPAQUE_TYPES_BY_NAME)
    sys.path.insert(0, _VENDOR)
    try:
        import primus_turbo.pytorch.kernels.attention.attention_fused_bwd_impl as fused
        import primus_turbo.pytorch.kernels.attention.attention_triton_impl as tri
        import primus_turbo.triton.attention.fused_mha_bwd_kernel as bwd_kernel
        for mod in (fused, tri, bwd_kernel):
            assert mod.__file__.startswith(_VENDOR), mod.__file__
    finally:
        sys.path.remove(_VENDOR)
        for k in [k for k in sys.modules if _is_pt(k)]:
            del sys.modules[k]
        sys.modules.update(saved)
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

for path in sys.argv[1:]:
    s = io.open(path).read()
    assert s.count(OLD) == 1, path
    io.open(path, "w").write(s.replace(OLD, NEW))
    print("patched", path)
