/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// hk_attn/prims.h -- the gfx1250 (MI455X, wave32) flash-attention primitive layer: HK-style thin
// wrappers, each lowering to the instruction(s) named in its comment (checked in the disassembly of
// hip/tests/t_prims.hip; catalogue and lane tables in hk/notes/prims.md).
//
//   layout.h  lane maps (host-compilable constexpr; mirrored by hk/lab/test_prims.py)
//   types.h   register types (opnd_t bf16x16, acc_t f32x8), static_for, cvt_pk_bf16, lane / wave id
//   wmma.h    v_wmma_f32_16x16x32_bf16 with reuse hints
//   lds.h     ds_load_b128 / ds_load_tr16_b128 operand loaders (any 16-B pitch), C^T ds_store_b128
//   frag.h    C -> operand pack, causal masks, row / column max and sum, exp2 / softmax argument
//   gmem.h    buffer resources with true extents, buffer loads / stores, fp32 buffer atomics
//   tdm.h     tensor_load_to_lds descriptors (padded 2-D tiles, NULL descriptor), s_wait_tensorcnt
//   sync.h    s_wait_*cnt, split barrier with LDS-only fences, sched masks, lock_simd, setprio,
//             scheduling mode
//
// Build: amdclang++ -x hip --offload-arch=gfx1250 --cuda-device-only -O3 -std=c++20
// -DKITTENS_CDNA5 -I<pkg>/hip/include (the vendored HK cdna5 subset provides sched.cuh).
#pragma once
#include "frag.h"
#include "gmem.h"
#include "lds.h"
#include "layout.h"
#include "sync.h"
#include "tdm.h"
#include "types.h"
#include "wmma.h"
