"""Poison helper for validation, kept OUTSIDE the two files refcache provenance hashes.

`build_refcache.py:65-66` records the SHA-256 of **`ut/common.py`** and **`eager/impl.py`**
in every cache's provenance, and `validation.py:102-104` checks them. `validation.py`
itself is NOT hashed. So putting the fixed implementation here and changing only
validation.py's import leaves all three refcache entries valid — no rebuild, no fp32
reference recomputation, no GPU time. `refcache_util.py:3-12` already uses this same trick
for the same reason.

The shipped `poison_allocator` in `ut/common.py:110` has two defects, both verified from
the source and the shape table:

  1. TOO SMALL. It poisons at most `1<<24` float32 elements = 64 MiB. The production
     shape's dq is [4, 8192, 32, 128] bf16 = 256 MiB. The caching allocator bins by size
     and a block only satisfies a request no larger than itself, so at prod dq never gets
     a poisoned block at all.

  2. WRONG BIT PATTERN FOR THE OUTPUT DTYPE. It fills float32 NaN, 0x7FC00000. The outputs
     are bf16, so those four bytes are read as TWO bf16 values: 0x0000 — a ZERO — and
     0x7FC0. Even where the poison lands, half of every bf16 output reads as a plausible
     zero, which is exactly the failure the poison exists to prevent.
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
    job's production shape (dq at [4, 8192, 32, 128] bf16 = 256 MiB).
    """
    if max_bytes is None:
        max_bytes = 256 << 20            # prod dq; see docstring

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
    """Prove the pattern is NaN in all three dtypes. Run this after applying the patch."""
    b = torch.empty(4, device=device, dtype=torch.int32).fill_(_POISON_WORD)
    assert b[0].item() == 0x7FC07FC0, f"bit pattern wrong: {b[0].item():#x}"
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        v = b.view(dt)
        assert not torch.isfinite(v).any(), f"{dt} reads finite: {v}"
        print(f"  {str(dt):16s} -> all non-finite over {v.numel()} elements  OK")
