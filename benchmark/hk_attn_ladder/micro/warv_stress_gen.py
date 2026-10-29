#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Generate micro/warv_stress.cpp: the GPU test of the WAR-V-by-class wait rule (expert scheduling mode).

The rule (tools/warv_regen.py, abcgen's asmkit "warv"): a memory instruction that overwrites registers an in-flight
WMMA X reads may issue after `s_wait_alu depctr_va_vdst(N)` with N = the number of WMMAs issued after X, even while
other VALU classes are in flight (LLVM: N = 0 then). It is sound iff WMMAs (XDL) complete in order among
themselves: at most N VALU in flight and N WMMAs younger than X  =>  X has completed and read its operands.

Each kernel (one config) runs ITERS trips per wave of
    load A_old -> R, B -> RB (s_wait_dscnt 0, va_vdst 0: a clean start)
    X:  v_wmma D = R x RB (+0)                         R is X's A operand (or RB its B operand: --opnd b)
    k younger WMMAs on other registers (independent accumulators or one dependent chain), with filler VALU of the
    chosen class mix interleaved (core VALU, TRANS, DPP) -- the other classes LLVM's rule would drain for
    s_wait_alu depctr_va_vdst(W)                       W = the config's wait (k: the rule; k+1, k+2, none: controls)
    ds_load_b128 / ds_load_tr16_b128 of A_new -> R    (the WAR-V write)
    s_wait_dscnt 0, va_vdst 0; compare D with D_ref (computed once with full drains): mismatching lanes -> counter
and writes the per-wave mismatch count. A_new != A_old everywhere, so a WMMA that read any part of R after the
load landed gives D != D_ref. Controls with looser waits must show mismatches somewhere, or the test cannot see
the hazard at all (then it proves nothing and says so).

    python3 micro/warv_stress_gen.py > micro/warv_stress.cpp
"""

import itertools

KS = (0, 1, 2, 3, 4, 6)
FILL = ("none", "mix")
OPS = ("b128_a", "tr16_a", "b128_b")
CHAIN = ("indep", "chain")

# fixed VGPRs of the asm body (declared clobbered): R v[64:71], RB v[72:79], D v[80:87], DREF v[88:95],
# filler WMMA operands FA v[96:103], FB v[104:111], filler accumulators v[112:159] (6 x 8), VALU fillers v[160:175],
# compare scratch v176, per-lane mismatch count v177
R, RB, D, DREF, FA, FB, FACC, VF, T, CNT, DC = 64, 72, 80, 88, 96, 104, 112, 160, 176, 177, 184


def v(r, n=1):
    return f"v{r}" if n == 1 else f"v[{r}:{r + n - 1}]"


def filler(cls, j):
    a, b = VF + (2 * j) % 16, VF + (2 * j + 1) % 16
    if cls == "valu":
        return [f"v_pk_fma_f32 {v(VF + (4 * j) % 16, 2)}, {v(VF + (4 * j + 2) % 16, 2)}, {v(VF + (4 * j + 2) % 16, 2)}, "
                f"{v(VF + (4 * j) % 16, 2)}"]  # fmt: skip
    if cls == "trans":
        return [f"v_exp_f32 {v(a)}, {v(b)}"]
    if cls == "dpp":
        return [f"v_mov_b32_dpp {v(a)}, {v(b)} row_ror:8 row_mask:0xf bank_mask:0xc"]
    return []


def pre_lines(pre):
    """A backlog in front of X: ("chain", P) = P dependent WMMAs into DC (X then accumulates onto DC: it cannot start
    before the chain is done), ("indep", M) = M independent WMMAs on the filler accumulators."""
    kind, m = pre
    out = []
    if kind == "chain":
        out += [f"v_mov_b32 {v(DC + q)}, 0" for q in range(8)]
        out += [f"v_wmma_f32_16x16x32_bf16 {v(DC, 8)}, {v(FA, 8)}, {v(FB, 8)}, {v(DC, 8)}" for _ in range(m)]
    elif kind == "indep":
        out += [f"v_wmma_f32_16x16x32_bf16 {v(FACC + 8 * (j % 6), 8)}, {v(FA, 8)}, {v(FB, 8)}, {v(FACC + 8 * (j % 6), 8)}"
                for j in range(m)]  # fmt: skip
    return out


def x_line(pre):
    c = v(DC, 8) if pre[0] == "chain" else "0"
    return f"v_wmma_f32_16x16x32_bf16 {v(D, 8)}, {v(R, 8)}, {v(RB, 8)}, {c}"


def body(k, fill, op, chain, wait, pre=("none", 0)):
    """The asm lines of one trip (after the clean start)."""
    out = pre_lines(pre)
    out.append(x_line(pre))
    for j in range(k):
        acc = FACC + 8 * (0 if chain == "chain" else j % 6)
        out.append(f"v_wmma_f32_16x16x32_bf16 {v(acc, 8)}, {v(FA, 8)}, {v(FB, 8)}, {v(acc, 8)}")
        mix = {"none": [], "valu": ["valu"], "trans": ["trans"], "mix": ["valu", "trans", "dpp"]}[fill]
        for c in mix:
            out += filler(c, j)
    if k == 0:
        mix = {"none": [], "valu": ["valu"], "trans": ["trans"], "mix": ["valu", "trans", "dpp"]}[fill]
        for c in mix:
            out += filler(c, 0)
    if wait is not None:
        out.append(f"s_wait_alu depctr_va_vdst({wait})")
    tgt = RB if op == "b128_b" else R
    ld = "ds_load_tr16_b128" if op == "tr16_a" else "ds_load_b128"
    out.append(f"{ld} {v(tgt, 4)}, %[anew]")
    out.append(f"{ld} {v(tgt + 4, 4)}, %[anew] offset:16" if op != "tr16_a" else f"{ld} {v(tgt + 4, 4)}, %[anew] offset:512")
    return out


def kernel(name, k, fill, op, chain, wait, pre=("none", 0)):
    trip = body(k, fill, op, chain, wait, pre)
    restore = "rb" if op == "b128_b" else "r"
    lines = []
    a = lines.append
    a(f'extern "C" __global__ __launch_bounds__(WAVES * 32) void {name}(unsigned* __restrict__ out, int iters) {{')
    a("  __shared__ __attribute__((aligned(16))) unsigned char lds[WAVES * LDS_W];")
    a("  const int lane = threadIdx.x % 32, w = threadIdx.x / 32;")
    a("  unsigned char* base = lds + w * LDS_W;")
    a("  fill_lds(base, lane, blockIdx.x * WAVES + w);")
    a("  const unsigned aold = (unsigned)(uintptr_t)(base + lane * 16), anew = aold + 1024, bb = aold + 2048;")
    a("  unsigned bad = 0;")
    a("  asm volatile(")
    asm = []
    # set-up: filler operands, filler VALU inputs, D_ref with full drains
    asm += [f"ds_load_b128 {v(FA, 4)}, %[bb]", f"ds_load_b128 {v(FA + 4, 4)}, %[bb] offset:16",
            f"ds_load_b128 {v(FB, 4)}, %[aold]", f"ds_load_b128 {v(FB + 4, 4)}, %[aold] offset:16",
            f"ds_load_b128 {v(VF, 4)}, %[bb] offset:512", f"ds_load_b128 {v(VF + 4, 4)}, %[bb] offset:528",
            f"ds_load_b128 {v(VF + 8, 4)}, %[aold] offset:512", f"ds_load_b128 {v(VF + 12, 4)}, %[aold] offset:528"]  # fmt: skip
    for j in range(6):
        for q in range(8):
            asm.append(f"v_mov_b32 {v(FACC + 8 * j + q)}, 0")
    asm += [f"ds_load_b128 {v(R, 4)}, %[aold]", f"ds_load_b128 {v(R + 4, 4)}, %[aold] offset:16",
            f"ds_load_b128 {v(RB, 4)}, %[bb]", f"ds_load_b128 {v(RB + 4, 4)}, %[bb] offset:16",
            "s_wait_dscnt 0x0", "s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)"]
    asm += [l.replace(v(D, 8), v(DREF, 8)) for l in pre_lines(pre) + [x_line(pre)]]
    asm += ["s_wait_alu depctr_va_vdst(0)", "v_nop", "v_nop", "v_nop", "v_nop", "v_nop",
            "s_mov_b32 s100, %[iters]", f"v_mov_b32 {v(CNT)}, 0",
            "1:"]  # fmt: skip
    # clean start of a trip: R / RB = old values
    asm += [f"ds_load_b128 {v(R, 4)}, %[aold]", f"ds_load_b128 {v(R + 4, 4)}, %[aold] offset:16"] if restore == "r" else \
           [f"ds_load_b128 {v(RB, 4)}, %[bb]", f"ds_load_b128 {v(RB + 4, 4)}, %[bb] offset:16"]  # fmt: skip
    asm += ["s_wait_dscnt 0x0", "s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)"]
    asm += trip
    asm += ["s_wait_dscnt 0x0", "s_wait_alu depctr_va_vdst(0)", "v_nop", "v_nop", "v_nop", "v_nop", "v_nop"]
    # compare D with DREF in VALU only: T = OR of the XORs; CNT += (T != 0)
    asm.append(f"v_xor_b32 {v(T)}, {v(D)}, {v(DREF)}")
    for q in range(1, 8):
        asm += [f"v_xor_b32 {v(D + q)}, {v(D + q)}, {v(DREF + q)}", f"v_or_b32 {v(T)}, {v(T)}, {v(D + q)}"]
    asm += [f"v_cmp_ne_u32 vcc_lo, 0, {v(T)}", f"v_cndmask_b32 {v(T)}, 0, 1, vcc_lo",
            f"v_add_nc_u32 {v(CNT)}, {v(CNT)}, {v(T)}", "s_wait_alu depctr_va_vdst(0)",
            "s_add_co_i32 s100, s100, -1", "s_cmp_lg_u32 s100, 0", "s_cbranch_scc1 1b",
            f"v_mov_b32 %[bad], {v(CNT)}", "s_wait_alu depctr_va_vdst(0)"]  # fmt: skip
    a("      " + "\n      ".join(f'"{l}\\n"' for l in asm))
    a('      : [bad] "=v"(bad) : [aold] "v"(aold), [anew] "v"(anew), [bb] "v"(bb), [iters] "s"(iters)')
    clob = ", ".join(f'"v{r}"' for r in range(64, 192))
    a(f'      : {clob}, "vcc", "scc", "s100", "memory");')
    a("  out[(blockIdx.x * WAVES + w) * 32 + lane] = bad;")
    a("}")
    return "\n".join(lines)


HEAD = r'''// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

// GENERATED by micro/warv_stress_gen.py -- the WAR-V-by-class wait rule on the card (see the generator's docstring).
//   warv_stress [--iters N] [--reps R] [--waves W]
// prints one line per config: mismatching lane-trips (should be 0 for wait == k) and the number of hazard events.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x)                                                                       \
  do {                                                                                 \
    hipError_t e = (x);                                                                \
    if (e != hipSuccess) {                                                             \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e)); \
      exit(1);                                                                         \
    }                                                                                  \
  } while (0)

#ifndef WAVES
#define WAVES 4
#endif
constexpr int LDS_W = 4096;  // per wave: A_old [0, 1024), A_new [1024, 2048), B [2048, 3072), spare

// distinct, finite bf16 data per (wave, lane): A_new differs from A_old in every element
__device__ inline unsigned short bf(float x) { return (unsigned short)(__float_as_uint(x) >> 16); }
__device__ void fill_lds(unsigned char* base, int lane, int gw) {
  unsigned short* p = (unsigned short*)base;
  for (int i = lane; i < LDS_W / 2; i += 32) {
    const unsigned h = (unsigned)(i * 2654435761u) ^ (unsigned)(gw * 40503u);
    float x = (float)((h >> 8) % 2001) / 1000.f - 1.f;
    if (i >= 512 && i < 1024) x = x + 3.0f;  // A_new = shifted: never equal to A_old
    p[i] = bf(x);
  }
  __syncthreads();
}
'''


def main():
    cfgs = []
    for op, k, fill, chain in itertools.product(OPS, KS, FILL, CHAIN):
        if chain == "chain" and k < 2:
            continue
        for wtag, wait in (("rule", k), ("k+1", k + 1), ("k+2", k + 2), ("none", None)):
            if wait is not None and wait > 14:
                continue
            cfgs.append((op, k, fill, chain, wtag, wait))
    out = [HEAD]
    names = []
    for op, k, pre in itertools.product(OPS, (0, 1, 2, 4), (("chain", 2), ("chain", 4), ("chain", 8), ("indep", 4),
                                                           ("indep", 8))):  # fmt: skip
        for wtag, wait in (("rule", k), ("k+1", k + 1), ("k+2", k + 2), ("none", None)):
            cfgs.append((op, k, "none", f"{pre[0]}{pre[1]}", wtag, wait))
    for i, (op, k, fill, chain, wtag, wait) in enumerate(cfgs):
        name = f"k_ws{i}"
        names.append((name, op, k, fill, chain, wtag))
        pre = (("chain", int(chain[5:])) if chain.startswith("chain") and chain != "chain" else
               ("indep", int(chain[5:])) if chain.startswith("indep") and chain != "indep" else ("none", 0))  # fmt: skip
        out.append(kernel(name, k, fill, op, "chain" if chain == "chain" else "indep", wait, pre))
    out.append("struct Cfg { void (*fn)(unsigned*, int); const char* op; int k; const char* fill; const char* chain; "
               "const char* wait; };")  # fmt: skip
    out.append("static const Cfg CFGS[] = {")
    for name, op, k, fill, chain, wtag in names:
        out.append(f'  {{{name}, "{op}", {k}, "{fill}", "{chain}", "{wtag}"}},')
    out.append("};")
    out.append(r'''
int main(int argc, char** argv) {
  int iters = 2000, reps = 4;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!strcmp(argv[i], "--iters")) iters = atoi(argv[i + 1]);
    if (!strcmp(argv[i], "--reps")) reps = atoi(argv[i + 1]);
  }
  hipDeviceProp_t p;
  CHECK(hipGetDeviceProperties(&p, 0));
  const int grid = p.multiProcessorCount * 2;  // 2 workgroups of WAVES waves per CU
  unsigned* out;
  CHECK(hipMalloc(&out, (size_t)grid * WAVES * 32 * sizeof(unsigned)));
  std::vector<unsigned> h((size_t)grid * WAVES * 32);
  const int n = sizeof(CFGS) / sizeof(CFGS[0]);
  long long rule_bad = 0, rule_events = 0, ctrl_bad = 0;
  printf("op      k fill  chain wait  mismatching-lane-trips  events\n");
  for (int c = 0; c < n; ++c) {
    long long bad = 0;
    for (int r = 0; r < reps; ++r) {
      hipLaunchKernelGGL(CFGS[c].fn, dim3(grid), dim3(WAVES * 32), 0, 0, out, iters);
      CHECK(hipDeviceSynchronize());
      CHECK(hipMemcpy(h.data(), out, h.size() * sizeof(unsigned), hipMemcpyDeviceToHost));
      for (unsigned x : h) bad += x;
    }
    const long long ev = (long long)reps * grid * WAVES * 32 * iters;
    printf("%-7s %d %-5s %-7s %-5s %12lld %14lld\n", CFGS[c].op, CFGS[c].k, CFGS[c].fill, CFGS[c].chain, CFGS[c].wait,
           bad, ev);
    if (!strcmp(CFGS[c].wait, "rule")) {
      rule_bad += bad;
      rule_events += ev;
    } else {
      ctrl_bad += bad;
    }
  }
  printf("RULE: %lld mismatching lane-trips in %lld lane-trips; CONTROLS (looser waits): %lld mismatching\n", rule_bad,
         rule_events, ctrl_bad);
  printf("VERDICT %s\n", rule_bad == 0 ? (ctrl_bad > 0 ? "PASS (rule clean, controls caught)"
                                                       : "INCONCLUSIVE (rule clean, controls never caught)")
                                       : "FAIL (the rule let a WMMA read an overwritten operand)");
  return rule_bad == 0 ? 0 : 1;
}''')
    print("\n\n".join(out))


if __name__ == "__main__":
    main()
