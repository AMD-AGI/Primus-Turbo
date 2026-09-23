"""Replacement for `poison_allocator` in the job's op/ut/common.py.

NOT APPLIED. Applying it invalidates every refcache entry -- see README.md beside this
file -- so it belongs in a deliberate maintenance window, not mid-round.

Two independent defects in the shipped version, both found on 2026-09-23 after a candidate
kernel's out-of-bounds write cost a power cycle:

  1. TOO SMALL. It poisons blocks of 1<<24, 1<<22, 1<<20, 1<<18 and 1<<16 float32
     elements, so the largest is 64 MiB. The production shape's dq is
     [4, 8192, 32, 128] bf16 = 256 MiB, and dk/dv are 64 MiB each. At prod the caching
     allocator has no poisoned block big enough for dq, so `torch.empty` hands back a
     fresh segment and an unwritten element reads as whatever was there -- typically zero.

  2. WRONG BIT PATTERN FOR THE OUTPUT DTYPE. It fills float32, whose NaN is 0x7FC00000.
     The outputs are bf16, so those four bytes are read as TWO bf16 values: 0x7FC0, a NaN,
     followed by 0x0000, a ZERO. Even where the poison does land, half of every bf16
     output reads as a plausible zero -- exactly the failure the poison exists to prevent.

The fix addresses both: poison with a word that is NaN when read as float32 AND as either
bf16 half, and size the blocks from the largest tensor actually under test.
"""
import torch

# 0x7FC07FC0: as float32 this is a quiet NaN; as two bf16 halves both are 0x7FC0, also a
# quiet NaN. So the same bytes poison an fp32, a bf16 or an fp16 output -- 0x7FC0 is NaN in
# fp16 too. Written through an int32 view because torch.full with a NaN float would be
# re-encoded per dtype and lose this property.
_POISON_WORD = 0x7FC07FC0 - (1 << 32)   # as a signed int32


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


def _self_test(device="cuda"):
    """Prove the pattern is NaN in all three dtypes. Run this after applying the patch."""
    b = torch.empty(4, device=device, dtype=torch.int32).fill_(_POISON_WORD)
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        v = b.view(dt)
        assert not torch.isfinite(v).any(), f"{dt} reads finite: {v}"
        print(f"  {str(dt):16s} -> all non-finite over {v.numel()} elements  OK")
