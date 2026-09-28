"""NaN-prefill of the caching allocator, so an output element the kernel never writes reads
as non-finite instead of as a plausible zero.

Adapted from the backward job's op/poison_util.py (same pattern and sizing logic; only
the docstrings changed). Two properties matter and both are kept:

  1. SIZE. A cached block only satisfies a request no larger than itself, so the poison
     must cover the largest output. Here that is prod `o` = [4, 8192, 32, 128] bf16 =
     256 MiB; callers pass it explicitly from the shape under test.
  2. BIT PATTERN. 0x7FC07FC0 is NaN as fp32 and as each bf16/fp16 half. A plain fp32 NaN
     (0x7FC00000) reads as a bf16 ZERO in one half.
"""
import torch

# 0x7FC07FC0: as float32 this is a quiet NaN; as two bf16 halves both are 0x7FC0, also a
# quiet NaN. So the same bytes poison an fp32, a bf16 or an fp16 output -- 0x7FC0 is NaN in
# fp16 too. Written through an int32 view because torch.full with a NaN float would be
# re-encoded per dtype and lose this property.
_POISON_WORD = 0x7FC07FC0   # 2143322048 < 2**31, so this IS already a valid int32.
# The first draft wrote `0x7FC07FC0 - (1 << 32)` on the assumption that the constant
# needed wrapping into signed range. It does not: subtracting 2**32 gives -2151645248,
# which is OUTSIDE int32, and Tensor.fill_ raises before a single byte is poisoned.


def poison_allocator(device="cuda", max_bytes=None):
    """Fill the caching allocator's free blocks with a dtype-agnostic NaN pattern.

    An implementation allocates its own outputs, so the test cannot prefill them. Freeing
    poisoned blocks of the sizes about to be requested makes `torch.empty` hand back NaN
    instead of zeros, so an element the kernel never writes shows up as non-finite rather
    than as a plausible zero. Best-effort by construction -- the isfinite coverage assert
    is the gate; this only makes it bite.

    `max_bytes` should be the largest single output the next call will allocate. Pass it
    from the shape under test rather than relying on the default, which is sized for this
    job's production shape (o at [4, 8192, 32, 128] bf16 = 256 MiB).
    """
    if max_bytes is None:
        max_bytes = 256 << 20            # prod o; see docstring

    # Cover the largest request and every power-of-two bin below it down to 256 KiB. The
    # caching allocator bins by size, so a block only satisfies a later request if it is at
    # least as large -- which is precisely what the shipped version got wrong.
    sizes, n = [], max(max_bytes, 1 << 18)
    while n >= (1 << 18):
        sizes.append(n)
        n >>= 1

    blocks = []
    for nbytes in sizes:
        try:
            b = torch.empty(nbytes // 4, device=device, dtype=torch.int32)
        except torch.cuda.OutOfMemoryError:
            break                        # poison what fits; the coverage assert still gates
        b.fill_(_POISON_WORD)
        blocks.append(b)
    del blocks
    torch.cuda.synchronize()


def _self_test(device="cpu"):
    """Prove the pattern is NaN in all three dtypes. """
    b = torch.empty(4, device=device, dtype=torch.int32).fill_(_POISON_WORD)
    assert b[0].item() == 0x7FC07FC0, f"bit pattern wrong: {b[0].item():#x}"
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        v = b.view(dt)
        assert not torch.isfinite(v).any(), f"{dt} reads finite: {v}"
        print(f"  {str(dt):16s} -> all non-finite over {v.numel()} elements  OK")
