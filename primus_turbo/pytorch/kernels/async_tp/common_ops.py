###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import triton
import triton.language as tl


@triton.jit
def wait_signal_acq_sys(rank, barrier_ptr, n_elements: tl.constexpr):
    barrier_ptr = barrier_ptr.to(tl.pointer_type(tl.int32))
    for i in range(n_elements):
        while i != rank and tl.atomic_cas(barrier_ptr + i, 1, 0, scope="sys", sem="acquire") != 1:
            pass

    tl.debug_barrier()


@triton.jit
def put_signal_rel_sys(barrier_ptr):
    barrier_ptr = barrier_ptr.to(tl.pointer_type(tl.int32))
    while tl.atomic_cas(barrier_ptr, 0, 1, scope="sys", sem="release") != 0:
        pass

    tl.debug_barrier()
