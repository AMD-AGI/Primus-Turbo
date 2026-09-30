#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef unsigned long long u64;
struct Rec { u64 cyc, rt; unsigned h1, h2; };

extern "C" __global__ __launch_bounds__(256) void k_p1w1d(float* out, const bf16x16* in, int iters, Rec* rec, int seg) {
  extern __shared__ i32x4 lds4[];
  for (int i = threadIdx.x; i < 256 * 1024 / 16; i += blockDim.x) lds4[i] = i32x4{i, i + 1, i + 2, i + 3};
  __syncthreads();
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned addr = ((lane * 16 + w * 4096) & 0xFFFF) + (seg ? ((w & 1) << 16) : 0u);
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tds_load_b128 v[96:99], %10 offset:0\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tds_load_b128 v[100:103], %10 offset:2048\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tds_load_b128 v[104:107], %10 offset:4096\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tds_load_b128 v[108:111], %10 offset:6144\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tds_load_b128 v[112:115], %10 offset:8192\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tds_load_b128 v[116:119], %10 offset:10240\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tds_load_b128 v[120:123], %10 offset:12288\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tds_load_b128 v[124:127], %10 offset:14336\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tds_load_b128 v[128:131], %10 offset:16384\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tds_load_b128 v[132:135], %10 offset:18432\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tds_load_b128 v[136:139], %10 offset:20480\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tds_load_b128 v[140:143], %10 offset:22528\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tds_load_b128 v[144:147], %10 offset:24576\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tds_load_b128 v[148:151], %10 offset:26624\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tds_load_b128 v[152:155], %10 offset:28672\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tds_load_b128 v[156:159], %10 offset:30720\n\ts_wait_dscnt 16" : "+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3),"+v"(c4),"+v"(c5),"+v"(c6),"+v"(c7),"+v"(a),"+v"(b),"+v"(addr) :: "memory", "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v107", "v108", "v109", "v110", "v111", "v112", "v113", "v114", "v115", "v116", "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v125", "v126", "v127", "v128", "v129", "v130", "v131", "v132", "v133", "v134", "v135", "v136", "v137", "v138", "v139", "v140", "v141", "v142", "v143", "v144", "v145", "v146", "v147", "v148", "v149", "v150", "v151", "v152", "v153", "v154", "v155", "v156", "v157", "v158", "v159", "v160", "v161", "v162", "v163", "v164", "v165", "v166", "v167", "v168", "v169", "v170", "v171", "v172", "v173", "v174", "v175", "v176", "v177", "v178", "v179", "v180", "v181", "v182", "v183", "v184", "v185", "v186", "v187", "v188", "v189", "v190", "v191", "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247", "v248", "v249", "v250", "v251", "v252", "v253", "v254", "v255");
  }
  asm volatile("s_wait_dscnt 0" ::: "memory");
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if (lane == 0) rec[blockIdx.x * (blockDim.x >> 5) + w] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 cs = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
  out[blockIdx.x * blockDim.x + threadIdx.x] = cs[0] + addr;
}

extern "C" __global__ __launch_bounds__(256) void k_p1w2d(float* out, const bf16x16* in, int iters, Rec* rec, int seg) {
  extern __shared__ i32x4 lds4[];
  for (int i = threadIdx.x; i < 256 * 1024 / 16; i += blockDim.x) lds4[i] = i32x4{i, i + 1, i + 2, i + 3};
  __syncthreads();
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned addr = ((lane * 16 + w * 4096) & 0xFFFF) + (seg ? ((w & 1) << 16) : 0u);
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tds_load_b128 v[96:99], %10 offset:0\n\tds_load_b128 v[100:103], %10 offset:2048\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tds_load_b128 v[104:107], %10 offset:4096\n\tds_load_b128 v[108:111], %10 offset:6144\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tds_load_b128 v[112:115], %10 offset:8192\n\tds_load_b128 v[116:119], %10 offset:10240\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tds_load_b128 v[120:123], %10 offset:12288\n\tds_load_b128 v[124:127], %10 offset:14336\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tds_load_b128 v[128:131], %10 offset:16384\n\tds_load_b128 v[132:135], %10 offset:18432\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tds_load_b128 v[136:139], %10 offset:20480\n\tds_load_b128 v[140:143], %10 offset:22528\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tds_load_b128 v[144:147], %10 offset:24576\n\tds_load_b128 v[148:151], %10 offset:26624\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tds_load_b128 v[152:155], %10 offset:28672\n\tds_load_b128 v[156:159], %10 offset:30720\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tds_load_b128 v[160:163], %10 offset:0\n\tds_load_b128 v[164:167], %10 offset:2048\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tds_load_b128 v[168:171], %10 offset:4096\n\tds_load_b128 v[172:175], %10 offset:6144\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tds_load_b128 v[176:179], %10 offset:8192\n\tds_load_b128 v[180:183], %10 offset:10240\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tds_load_b128 v[184:187], %10 offset:12288\n\tds_load_b128 v[188:191], %10 offset:14336\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tds_load_b128 v[192:195], %10 offset:16384\n\tds_load_b128 v[196:199], %10 offset:18432\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tds_load_b128 v[200:203], %10 offset:20480\n\tds_load_b128 v[204:207], %10 offset:22528\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tds_load_b128 v[208:211], %10 offset:24576\n\tds_load_b128 v[212:215], %10 offset:26624\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tds_load_b128 v[216:219], %10 offset:28672\n\tds_load_b128 v[220:223], %10 offset:30720\n\ts_wait_dscnt 16" : "+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3),"+v"(c4),"+v"(c5),"+v"(c6),"+v"(c7),"+v"(a),"+v"(b),"+v"(addr) :: "memory", "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v107", "v108", "v109", "v110", "v111", "v112", "v113", "v114", "v115", "v116", "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v125", "v126", "v127", "v128", "v129", "v130", "v131", "v132", "v133", "v134", "v135", "v136", "v137", "v138", "v139", "v140", "v141", "v142", "v143", "v144", "v145", "v146", "v147", "v148", "v149", "v150", "v151", "v152", "v153", "v154", "v155", "v156", "v157", "v158", "v159", "v160", "v161", "v162", "v163", "v164", "v165", "v166", "v167", "v168", "v169", "v170", "v171", "v172", "v173", "v174", "v175", "v176", "v177", "v178", "v179", "v180", "v181", "v182", "v183", "v184", "v185", "v186", "v187", "v188", "v189", "v190", "v191", "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247", "v248", "v249", "v250", "v251", "v252", "v253", "v254", "v255");
  }
  asm volatile("s_wait_dscnt 0" ::: "memory");
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if (lane == 0) rec[blockIdx.x * (blockDim.x >> 5) + w] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 cs = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
  out[blockIdx.x * blockDim.x + threadIdx.x] = cs[0] + addr;
}

extern "C" __global__ __launch_bounds__(256) void k_p1w4d(float* out, const bf16x16* in, int iters, Rec* rec, int seg) {
  extern __shared__ i32x4 lds4[];
  for (int i = threadIdx.x; i < 256 * 1024 / 16; i += blockDim.x) lds4[i] = i32x4{i, i + 1, i + 2, i + 3};
  __syncthreads();
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned addr = ((lane * 16 + w * 4096) & 0xFFFF) + (seg ? ((w & 1) << 16) : 0u);
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tds_load_b128 v[96:99], %10 offset:0\n\tds_load_b128 v[100:103], %10 offset:2048\n\tds_load_b128 v[104:107], %10 offset:4096\n\tds_load_b128 v[108:111], %10 offset:6144\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tds_load_b128 v[112:115], %10 offset:8192\n\tds_load_b128 v[116:119], %10 offset:10240\n\tds_load_b128 v[120:123], %10 offset:12288\n\tds_load_b128 v[124:127], %10 offset:14336\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tds_load_b128 v[128:131], %10 offset:16384\n\tds_load_b128 v[132:135], %10 offset:18432\n\tds_load_b128 v[136:139], %10 offset:20480\n\tds_load_b128 v[140:143], %10 offset:22528\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tds_load_b128 v[144:147], %10 offset:24576\n\tds_load_b128 v[148:151], %10 offset:26624\n\tds_load_b128 v[152:155], %10 offset:28672\n\tds_load_b128 v[156:159], %10 offset:30720\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tds_load_b128 v[160:163], %10 offset:0\n\tds_load_b128 v[164:167], %10 offset:2048\n\tds_load_b128 v[168:171], %10 offset:4096\n\tds_load_b128 v[172:175], %10 offset:6144\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tds_load_b128 v[176:179], %10 offset:8192\n\tds_load_b128 v[180:183], %10 offset:10240\n\tds_load_b128 v[184:187], %10 offset:12288\n\tds_load_b128 v[188:191], %10 offset:14336\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tds_load_b128 v[192:195], %10 offset:16384\n\tds_load_b128 v[196:199], %10 offset:18432\n\tds_load_b128 v[200:203], %10 offset:20480\n\tds_load_b128 v[204:207], %10 offset:22528\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tds_load_b128 v[208:211], %10 offset:24576\n\tds_load_b128 v[212:215], %10 offset:26624\n\tds_load_b128 v[216:219], %10 offset:28672\n\tds_load_b128 v[220:223], %10 offset:30720\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tds_load_b128 v[224:227], %10 offset:0\n\tds_load_b128 v[228:231], %10 offset:2048\n\tds_load_b128 v[232:235], %10 offset:4096\n\tds_load_b128 v[236:239], %10 offset:6144\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tds_load_b128 v[240:243], %10 offset:8192\n\tds_load_b128 v[244:247], %10 offset:10240\n\tds_load_b128 v[248:251], %10 offset:12288\n\tds_load_b128 v[252:255], %10 offset:14336\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tds_load_b128 v[96:99], %10 offset:16384\n\tds_load_b128 v[100:103], %10 offset:18432\n\tds_load_b128 v[104:107], %10 offset:20480\n\tds_load_b128 v[108:111], %10 offset:22528\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tds_load_b128 v[112:115], %10 offset:24576\n\tds_load_b128 v[116:119], %10 offset:26624\n\tds_load_b128 v[120:123], %10 offset:28672\n\tds_load_b128 v[124:127], %10 offset:30720\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tds_load_b128 v[128:131], %10 offset:0\n\tds_load_b128 v[132:135], %10 offset:2048\n\tds_load_b128 v[136:139], %10 offset:4096\n\tds_load_b128 v[140:143], %10 offset:6144\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tds_load_b128 v[144:147], %10 offset:8192\n\tds_load_b128 v[148:151], %10 offset:10240\n\tds_load_b128 v[152:155], %10 offset:12288\n\tds_load_b128 v[156:159], %10 offset:14336\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tds_load_b128 v[160:163], %10 offset:16384\n\tds_load_b128 v[164:167], %10 offset:18432\n\tds_load_b128 v[168:171], %10 offset:20480\n\tds_load_b128 v[172:175], %10 offset:22528\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tds_load_b128 v[176:179], %10 offset:24576\n\tds_load_b128 v[180:183], %10 offset:26624\n\tds_load_b128 v[184:187], %10 offset:28672\n\tds_load_b128 v[188:191], %10 offset:30720\n\ts_wait_dscnt 16" : "+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3),"+v"(c4),"+v"(c5),"+v"(c6),"+v"(c7),"+v"(a),"+v"(b),"+v"(addr) :: "memory", "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v107", "v108", "v109", "v110", "v111", "v112", "v113", "v114", "v115", "v116", "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v125", "v126", "v127", "v128", "v129", "v130", "v131", "v132", "v133", "v134", "v135", "v136", "v137", "v138", "v139", "v140", "v141", "v142", "v143", "v144", "v145", "v146", "v147", "v148", "v149", "v150", "v151", "v152", "v153", "v154", "v155", "v156", "v157", "v158", "v159", "v160", "v161", "v162", "v163", "v164", "v165", "v166", "v167", "v168", "v169", "v170", "v171", "v172", "v173", "v174", "v175", "v176", "v177", "v178", "v179", "v180", "v181", "v182", "v183", "v184", "v185", "v186", "v187", "v188", "v189", "v190", "v191", "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247", "v248", "v249", "v250", "v251", "v252", "v253", "v254", "v255");
  }
  asm volatile("s_wait_dscnt 0" ::: "memory");
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if (lane == 0) rec[blockIdx.x * (blockDim.x >> 5) + w] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 cs = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
  out[blockIdx.x * blockDim.x + threadIdx.x] = cs[0] + addr;
}

extern "C" __global__ __launch_bounds__(256) void k_p2w2d(float* out, const bf16x16* in, int iters, Rec* rec, int seg) {
  extern __shared__ i32x4 lds4[];
  for (int i = threadIdx.x; i < 256 * 1024 / 16; i += blockDim.x) lds4[i] = i32x4{i, i + 1, i + 2, i + 3};
  __syncthreads();
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned addr = ((lane * 16 + w * 4096) & 0xFFFF) + (seg ? ((w & 1) << 16) : 0u);
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tds_load_b128 v[96:99], %10 offset:0\n\tds_load_b128 v[100:103], %10 offset:2048\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tds_load_b128 v[104:107], %10 offset:4096\n\tds_load_b128 v[108:111], %10 offset:6144\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tds_load_b128 v[112:115], %10 offset:8192\n\tds_load_b128 v[116:119], %10 offset:10240\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tds_load_b128 v[120:123], %10 offset:12288\n\tds_load_b128 v[124:127], %10 offset:14336\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tds_load_b128 v[128:131], %10 offset:16384\n\tds_load_b128 v[132:135], %10 offset:18432\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tds_load_b128 v[136:139], %10 offset:20480\n\tds_load_b128 v[140:143], %10 offset:22528\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tds_load_b128 v[144:147], %10 offset:24576\n\tds_load_b128 v[148:151], %10 offset:26624\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tds_load_b128 v[152:155], %10 offset:28672\n\tds_load_b128 v[156:159], %10 offset:30720\n\ts_wait_dscnt 16" : "+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3),"+v"(c4),"+v"(c5),"+v"(c6),"+v"(c7),"+v"(a),"+v"(b),"+v"(addr) :: "memory", "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v107", "v108", "v109", "v110", "v111", "v112", "v113", "v114", "v115", "v116", "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v125", "v126", "v127", "v128", "v129", "v130", "v131", "v132", "v133", "v134", "v135", "v136", "v137", "v138", "v139", "v140", "v141", "v142", "v143", "v144", "v145", "v146", "v147", "v148", "v149", "v150", "v151", "v152", "v153", "v154", "v155", "v156", "v157", "v158", "v159", "v160", "v161", "v162", "v163", "v164", "v165", "v166", "v167", "v168", "v169", "v170", "v171", "v172", "v173", "v174", "v175", "v176", "v177", "v178", "v179", "v180", "v181", "v182", "v183", "v184", "v185", "v186", "v187", "v188", "v189", "v190", "v191", "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247", "v248", "v249", "v250", "v251", "v252", "v253", "v254", "v255");
  }
  asm volatile("s_wait_dscnt 0" ::: "memory");
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if (lane == 0) rec[blockIdx.x * (blockDim.x >> 5) + w] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 cs = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
  out[blockIdx.x * blockDim.x + threadIdx.x] = cs[0] + addr;
}

extern "C" __global__ __launch_bounds__(256) void k_p4w4d(float* out, const bf16x16* in, int iters, Rec* rec, int seg) {
  extern __shared__ i32x4 lds4[];
  for (int i = threadIdx.x; i < 256 * 1024 / 16; i += blockDim.x) lds4[i] = i32x4{i, i + 1, i + 2, i + 3};
  __syncthreads();
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned addr = ((lane * 16 + w * 4096) & 0xFFFF) + (seg ? ((w & 1) << 16) : 0u);
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tds_load_b128 v[96:99], %10 offset:0\n\tds_load_b128 v[100:103], %10 offset:2048\n\tds_load_b128 v[104:107], %10 offset:4096\n\tds_load_b128 v[108:111], %10 offset:6144\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tds_load_b128 v[112:115], %10 offset:8192\n\tds_load_b128 v[116:119], %10 offset:10240\n\tds_load_b128 v[120:123], %10 offset:12288\n\tds_load_b128 v[124:127], %10 offset:14336\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tds_load_b128 v[128:131], %10 offset:16384\n\tds_load_b128 v[132:135], %10 offset:18432\n\tds_load_b128 v[136:139], %10 offset:20480\n\tds_load_b128 v[140:143], %10 offset:22528\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tds_load_b128 v[144:147], %10 offset:24576\n\tds_load_b128 v[148:151], %10 offset:26624\n\tds_load_b128 v[152:155], %10 offset:28672\n\tds_load_b128 v[156:159], %10 offset:30720\n\ts_wait_dscnt 16" : "+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3),"+v"(c4),"+v"(c5),"+v"(c6),"+v"(c7),"+v"(a),"+v"(b),"+v"(addr) :: "memory", "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v107", "v108", "v109", "v110", "v111", "v112", "v113", "v114", "v115", "v116", "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v125", "v126", "v127", "v128", "v129", "v130", "v131", "v132", "v133", "v134", "v135", "v136", "v137", "v138", "v139", "v140", "v141", "v142", "v143", "v144", "v145", "v146", "v147", "v148", "v149", "v150", "v151", "v152", "v153", "v154", "v155", "v156", "v157", "v158", "v159", "v160", "v161", "v162", "v163", "v164", "v165", "v166", "v167", "v168", "v169", "v170", "v171", "v172", "v173", "v174", "v175", "v176", "v177", "v178", "v179", "v180", "v181", "v182", "v183", "v184", "v185", "v186", "v187", "v188", "v189", "v190", "v191", "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247", "v248", "v249", "v250", "v251", "v252", "v253", "v254", "v255");
  }
  asm volatile("s_wait_dscnt 0" ::: "memory");
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if (lane == 0) rec[blockIdx.x * (blockDim.x >> 5) + w] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 cs = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
  out[blockIdx.x * blockDim.x + threadIdx.x] = cs[0] + addr;
}

extern "C" __global__ __launch_bounds__(256) void k_p8w8d(float* out, const bf16x16* in, int iters, Rec* rec, int seg) {
  extern __shared__ i32x4 lds4[];
  for (int i = threadIdx.x; i < 256 * 1024 / 16; i += blockDim.x) lds4[i] = i32x4{i, i + 1, i + 2, i + 3};
  __syncthreads();
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned addr = ((lane * 16 + w * 4096) & 0xFFFF) + (seg ? ((w & 1) << 16) : 0u);
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tds_load_b128 v[96:99], %10 offset:0\n\tds_load_b128 v[100:103], %10 offset:2048\n\tds_load_b128 v[104:107], %10 offset:4096\n\tds_load_b128 v[108:111], %10 offset:6144\n\tds_load_b128 v[112:115], %10 offset:8192\n\tds_load_b128 v[116:119], %10 offset:10240\n\tds_load_b128 v[120:123], %10 offset:12288\n\tds_load_b128 v[124:127], %10 offset:14336\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tds_load_b128 v[128:131], %10 offset:16384\n\tds_load_b128 v[132:135], %10 offset:18432\n\tds_load_b128 v[136:139], %10 offset:20480\n\tds_load_b128 v[140:143], %10 offset:22528\n\tds_load_b128 v[144:147], %10 offset:24576\n\tds_load_b128 v[148:151], %10 offset:26624\n\tds_load_b128 v[152:155], %10 offset:28672\n\tds_load_b128 v[156:159], %10 offset:30720\n\ts_wait_dscnt 16" : "+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3),"+v"(c4),"+v"(c5),"+v"(c6),"+v"(c7),"+v"(a),"+v"(b),"+v"(addr) :: "memory", "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v107", "v108", "v109", "v110", "v111", "v112", "v113", "v114", "v115", "v116", "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v125", "v126", "v127", "v128", "v129", "v130", "v131", "v132", "v133", "v134", "v135", "v136", "v137", "v138", "v139", "v140", "v141", "v142", "v143", "v144", "v145", "v146", "v147", "v148", "v149", "v150", "v151", "v152", "v153", "v154", "v155", "v156", "v157", "v158", "v159", "v160", "v161", "v162", "v163", "v164", "v165", "v166", "v167", "v168", "v169", "v170", "v171", "v172", "v173", "v174", "v175", "v176", "v177", "v178", "v179", "v180", "v181", "v182", "v183", "v184", "v185", "v186", "v187", "v188", "v189", "v190", "v191", "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247", "v248", "v249", "v250", "v251", "v252", "v253", "v254", "v255");
  }
  asm volatile("s_wait_dscnt 0" ::: "memory");
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if (lane == 0) rec[blockIdx.x * (blockDim.x >> 5) + w] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 cs = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
  out[blockIdx.x * blockDim.x + threadIdx.x] = cs[0] + addr;
}

extern "C" __global__ __launch_bounds__(256) void k_p2w1d(float* out, const bf16x16* in, int iters, Rec* rec, int seg) {
  extern __shared__ i32x4 lds4[];
  for (int i = threadIdx.x; i < 256 * 1024 / 16; i += blockDim.x) lds4[i] = i32x4{i, i + 1, i + 2, i + 3};
  __syncthreads();
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned addr = ((lane * 16 + w * 4096) & 0xFFFF) + (seg ? ((w & 1) << 16) : 0u);
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tds_load_b128 v[96:99], %10 offset:0\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tds_load_b128 v[100:103], %10 offset:2048\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tds_load_b128 v[104:107], %10 offset:4096\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tds_load_b128 v[108:111], %10 offset:6144\n\tv_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tds_load_b128 v[112:115], %10 offset:8192\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tds_load_b128 v[116:119], %10 offset:10240\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tds_load_b128 v[120:123], %10 offset:12288\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tds_load_b128 v[124:127], %10 offset:14336\n\ts_wait_dscnt 16" : "+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3),"+v"(c4),"+v"(c5),"+v"(c6),"+v"(c7),"+v"(a),"+v"(b),"+v"(addr) :: "memory", "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v107", "v108", "v109", "v110", "v111", "v112", "v113", "v114", "v115", "v116", "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v125", "v126", "v127", "v128", "v129", "v130", "v131", "v132", "v133", "v134", "v135", "v136", "v137", "v138", "v139", "v140", "v141", "v142", "v143", "v144", "v145", "v146", "v147", "v148", "v149", "v150", "v151", "v152", "v153", "v154", "v155", "v156", "v157", "v158", "v159", "v160", "v161", "v162", "v163", "v164", "v165", "v166", "v167", "v168", "v169", "v170", "v171", "v172", "v173", "v174", "v175", "v176", "v177", "v178", "v179", "v180", "v181", "v182", "v183", "v184", "v185", "v186", "v187", "v188", "v189", "v190", "v191", "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247", "v248", "v249", "v250", "v251", "v252", "v253", "v254", "v255");
  }
  asm volatile("s_wait_dscnt 0" ::: "memory");
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if (lane == 0) rec[blockIdx.x * (blockDim.x >> 5) + w] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 cs = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
  out[blockIdx.x * blockDim.x + threadIdx.x] = cs[0] + addr;
}

extern "C" __global__ __launch_bounds__(256) void k_p4w1d(float* out, const bf16x16* in, int iters, Rec* rec, int seg) {
  extern __shared__ i32x4 lds4[];
  for (int i = threadIdx.x; i < 256 * 1024 / 16; i += blockDim.x) lds4[i] = i32x4{i, i + 1, i + 2, i + 3};
  __syncthreads();
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned addr = ((lane * 16 + w * 4096) & 0xFFFF) + (seg ? ((w & 1) << 16) : 0u);
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tds_load_b128 v[96:99], %10 offset:0\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tds_load_b128 v[100:103], %10 offset:2048\n\tv_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tds_load_b128 v[104:107], %10 offset:4096\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tds_load_b128 v[108:111], %10 offset:6144" : "+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3),"+v"(c4),"+v"(c5),"+v"(c6),"+v"(c7),"+v"(a),"+v"(b),"+v"(addr) :: "memory", "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v107", "v108", "v109", "v110", "v111", "v112", "v113", "v114", "v115", "v116", "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v125", "v126", "v127", "v128", "v129", "v130", "v131", "v132", "v133", "v134", "v135", "v136", "v137", "v138", "v139", "v140", "v141", "v142", "v143", "v144", "v145", "v146", "v147", "v148", "v149", "v150", "v151", "v152", "v153", "v154", "v155", "v156", "v157", "v158", "v159", "v160", "v161", "v162", "v163", "v164", "v165", "v166", "v167", "v168", "v169", "v170", "v171", "v172", "v173", "v174", "v175", "v176", "v177", "v178", "v179", "v180", "v181", "v182", "v183", "v184", "v185", "v186", "v187", "v188", "v189", "v190", "v191", "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247", "v248", "v249", "v250", "v251", "v252", "v253", "v254", "v255");
  }
  asm volatile("s_wait_dscnt 0" ::: "memory");
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if (lane == 0) rec[blockIdx.x * (blockDim.x >> 5) + w] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 cs = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
  out[blockIdx.x * blockDim.x + threadIdx.x] = cs[0] + addr;
}

extern "C" __global__ __launch_bounds__(256) void k_p8w1d(float* out, const bf16x16* in, int iters, Rec* rec, int seg) {
  extern __shared__ i32x4 lds4[];
  for (int i = threadIdx.x; i < 256 * 1024 / 16; i += blockDim.x) lds4[i] = i32x4{i, i + 1, i + 2, i + 3};
  __syncthreads();
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned addr = ((lane * 16 + w * 4096) & 0xFFFF) + (seg ? ((w & 1) << 16) : 0u);
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tds_load_b128 v[96:99], %10 offset:0\n\tv_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tds_load_b128 v[100:103], %10 offset:2048" : "+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3),"+v"(c4),"+v"(c5),"+v"(c6),"+v"(c7),"+v"(a),"+v"(b),"+v"(addr) :: "memory", "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v107", "v108", "v109", "v110", "v111", "v112", "v113", "v114", "v115", "v116", "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v125", "v126", "v127", "v128", "v129", "v130", "v131", "v132", "v133", "v134", "v135", "v136", "v137", "v138", "v139", "v140", "v141", "v142", "v143", "v144", "v145", "v146", "v147", "v148", "v149", "v150", "v151", "v152", "v153", "v154", "v155", "v156", "v157", "v158", "v159", "v160", "v161", "v162", "v163", "v164", "v165", "v166", "v167", "v168", "v169", "v170", "v171", "v172", "v173", "v174", "v175", "v176", "v177", "v178", "v179", "v180", "v181", "v182", "v183", "v184", "v185", "v186", "v187", "v188", "v189", "v190", "v191", "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247", "v248", "v249", "v250", "v251", "v252", "v253", "v254", "v255");
  }
  asm volatile("s_wait_dscnt 0" ::: "memory");
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if (lane == 0) rec[blockIdx.x * (blockDim.x >> 5) + w] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 cs = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
  out[blockIdx.x * blockDim.x + threadIdx.x] = cs[0] + addr;
}

extern "C" __global__ __launch_bounds__(256) void k_p16w1d(float* out, const bf16x16* in, int iters, Rec* rec, int seg) {
  extern __shared__ i32x4 lds4[];
  for (int i = threadIdx.x; i < 256 * 1024 / 16; i += blockDim.x) lds4[i] = i32x4{i, i + 1, i + 2, i + 3};
  __syncthreads();
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned addr = ((lane * 16 + w * 4096) & 0xFFFF) + (seg ? ((w & 1) << 16) : 0u);
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tds_load_b128 v[96:99], %10 offset:0" : "+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3),"+v"(c4),"+v"(c5),"+v"(c6),"+v"(c7),"+v"(a),"+v"(b),"+v"(addr) :: "memory", "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v107", "v108", "v109", "v110", "v111", "v112", "v113", "v114", "v115", "v116", "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v125", "v126", "v127", "v128", "v129", "v130", "v131", "v132", "v133", "v134", "v135", "v136", "v137", "v138", "v139", "v140", "v141", "v142", "v143", "v144", "v145", "v146", "v147", "v148", "v149", "v150", "v151", "v152", "v153", "v154", "v155", "v156", "v157", "v158", "v159", "v160", "v161", "v162", "v163", "v164", "v165", "v166", "v167", "v168", "v169", "v170", "v171", "v172", "v173", "v174", "v175", "v176", "v177", "v178", "v179", "v180", "v181", "v182", "v183", "v184", "v185", "v186", "v187", "v188", "v189", "v190", "v191", "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247", "v248", "v249", "v250", "v251", "v252", "v253", "v254", "v255");
  }
  asm volatile("s_wait_dscnt 0" ::: "memory");
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if (lane == 0) rec[blockIdx.x * (blockDim.x >> 5) + w] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 cs = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
  out[blockIdx.x * blockDim.x + threadIdx.x] = cs[0] + addr;
}

extern "C" __global__ __launch_bounds__(256) void k_p1w1dv(float* out, const bf16x16* in, int iters, Rec* rec, int seg) {
  extern __shared__ i32x4 lds4[];
  for (int i = threadIdx.x; i < 256 * 1024 / 16; i += blockDim.x) lds4[i] = i32x4{i, i + 1, i + 2, i + 3};
  __syncthreads();
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned addr = ((lane * 16 + w * 4096) & 0xFFFF) + (seg ? ((w & 1) << 16) : 0u);
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[96:99], %10 offset:0\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[100:103], %10 offset:2048\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[104:107], %10 offset:4096\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[108:111], %10 offset:6144\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[112:115], %10 offset:8192\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[116:119], %10 offset:10240\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[120:123], %10 offset:12288\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[124:127], %10 offset:14336\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[128:131], %10 offset:16384\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[132:135], %10 offset:18432\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[136:139], %10 offset:20480\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[140:143], %10 offset:22528\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[144:147], %10 offset:24576\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[148:151], %10 offset:26624\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[152:155], %10 offset:28672\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[156:159], %10 offset:30720\n\ts_wait_dscnt 16" : "+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3),"+v"(c4),"+v"(c5),"+v"(c6),"+v"(c7),"+v"(a),"+v"(b),"+v"(addr) :: "memory", "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v107", "v108", "v109", "v110", "v111", "v112", "v113", "v114", "v115", "v116", "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v125", "v126", "v127", "v128", "v129", "v130", "v131", "v132", "v133", "v134", "v135", "v136", "v137", "v138", "v139", "v140", "v141", "v142", "v143", "v144", "v145", "v146", "v147", "v148", "v149", "v150", "v151", "v152", "v153", "v154", "v155", "v156", "v157", "v158", "v159", "v160", "v161", "v162", "v163", "v164", "v165", "v166", "v167", "v168", "v169", "v170", "v171", "v172", "v173", "v174", "v175", "v176", "v177", "v178", "v179", "v180", "v181", "v182", "v183", "v184", "v185", "v186", "v187", "v188", "v189", "v190", "v191", "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247", "v248", "v249", "v250", "v251", "v252", "v253", "v254", "v255");
  }
  asm volatile("s_wait_dscnt 0" ::: "memory");
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if (lane == 0) rec[blockIdx.x * (blockDim.x >> 5) + w] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 cs = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
  out[blockIdx.x * blockDim.x + threadIdx.x] = cs[0] + addr;
}

extern "C" __global__ __launch_bounds__(256) void k_p1w1ds(float* out, const bf16x16* in, int iters, Rec* rec, int seg) {
  extern __shared__ i32x4 lds4[];
  for (int i = threadIdx.x; i < 256 * 1024 / 16; i += blockDim.x) lds4[i] = i32x4{i, i + 1, i + 2, i + 3};
  __syncthreads();
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned addr = ((lane * 16 + w * 4096) & 0xFFFF) + (seg ? ((w & 1) << 16) : 0u);
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\ts_nop 0\n\tds_load_b128 v[96:99], %10 offset:0\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\ts_nop 0\n\tds_load_b128 v[100:103], %10 offset:2048\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\ts_nop 0\n\tds_load_b128 v[104:107], %10 offset:4096\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\ts_nop 0\n\tds_load_b128 v[108:111], %10 offset:6144\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\ts_nop 0\n\tds_load_b128 v[112:115], %10 offset:8192\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\ts_nop 0\n\tds_load_b128 v[116:119], %10 offset:10240\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\ts_nop 0\n\tds_load_b128 v[120:123], %10 offset:12288\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\ts_nop 0\n\tds_load_b128 v[124:127], %10 offset:14336\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\ts_nop 0\n\tds_load_b128 v[128:131], %10 offset:16384\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\ts_nop 0\n\tds_load_b128 v[132:135], %10 offset:18432\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\ts_nop 0\n\tds_load_b128 v[136:139], %10 offset:20480\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\ts_nop 0\n\tds_load_b128 v[140:143], %10 offset:22528\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\ts_nop 0\n\tds_load_b128 v[144:147], %10 offset:24576\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\ts_nop 0\n\tds_load_b128 v[148:151], %10 offset:26624\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\ts_nop 0\n\tds_load_b128 v[152:155], %10 offset:28672\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\ts_nop 0\n\tds_load_b128 v[156:159], %10 offset:30720\n\ts_wait_dscnt 16" : "+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3),"+v"(c4),"+v"(c5),"+v"(c6),"+v"(c7),"+v"(a),"+v"(b),"+v"(addr) :: "memory", "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v107", "v108", "v109", "v110", "v111", "v112", "v113", "v114", "v115", "v116", "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v125", "v126", "v127", "v128", "v129", "v130", "v131", "v132", "v133", "v134", "v135", "v136", "v137", "v138", "v139", "v140", "v141", "v142", "v143", "v144", "v145", "v146", "v147", "v148", "v149", "v150", "v151", "v152", "v153", "v154", "v155", "v156", "v157", "v158", "v159", "v160", "v161", "v162", "v163", "v164", "v165", "v166", "v167", "v168", "v169", "v170", "v171", "v172", "v173", "v174", "v175", "v176", "v177", "v178", "v179", "v180", "v181", "v182", "v183", "v184", "v185", "v186", "v187", "v188", "v189", "v190", "v191", "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247", "v248", "v249", "v250", "v251", "v252", "v253", "v254", "v255");
  }
  asm volatile("s_wait_dscnt 0" ::: "memory");
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if (lane == 0) rec[blockIdx.x * (blockDim.x >> 5) + w] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 cs = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
  out[blockIdx.x * blockDim.x + threadIdx.x] = cs[0] + addr;
}

extern "C" __global__ __launch_bounds__(256) void k_p4w4dv(float* out, const bf16x16* in, int iters, Rec* rec, int seg) {
  extern __shared__ i32x4 lds4[];
  for (int i = threadIdx.x; i < 256 * 1024 / 16; i += blockDim.x) lds4[i] = i32x4{i, i + 1, i + 2, i + 3};
  __syncthreads();
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned addr = ((lane * 16 + w * 4096) & 0xFFFF) + (seg ? ((w & 1) << 16) : 0u);
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[96:99], %10 offset:0\n\tds_load_b128 v[100:103], %10 offset:2048\n\tds_load_b128 v[104:107], %10 offset:4096\n\tds_load_b128 v[108:111], %10 offset:6144\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[112:115], %10 offset:8192\n\tds_load_b128 v[116:119], %10 offset:10240\n\tds_load_b128 v[120:123], %10 offset:12288\n\tds_load_b128 v[124:127], %10 offset:14336\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[128:131], %10 offset:16384\n\tds_load_b128 v[132:135], %10 offset:18432\n\tds_load_b128 v[136:139], %10 offset:20480\n\tds_load_b128 v[140:143], %10 offset:22528\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[144:147], %10 offset:24576\n\tds_load_b128 v[148:151], %10 offset:26624\n\tds_load_b128 v[152:155], %10 offset:28672\n\tds_load_b128 v[156:159], %10 offset:30720\n\ts_wait_dscnt 16" : "+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3),"+v"(c4),"+v"(c5),"+v"(c6),"+v"(c7),"+v"(a),"+v"(b),"+v"(addr) :: "memory", "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v107", "v108", "v109", "v110", "v111", "v112", "v113", "v114", "v115", "v116", "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v125", "v126", "v127", "v128", "v129", "v130", "v131", "v132", "v133", "v134", "v135", "v136", "v137", "v138", "v139", "v140", "v141", "v142", "v143", "v144", "v145", "v146", "v147", "v148", "v149", "v150", "v151", "v152", "v153", "v154", "v155", "v156", "v157", "v158", "v159", "v160", "v161", "v162", "v163", "v164", "v165", "v166", "v167", "v168", "v169", "v170", "v171", "v172", "v173", "v174", "v175", "v176", "v177", "v178", "v179", "v180", "v181", "v182", "v183", "v184", "v185", "v186", "v187", "v188", "v189", "v190", "v191", "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247", "v248", "v249", "v250", "v251", "v252", "v253", "v254", "v255");
  }
  asm volatile("s_wait_dscnt 0" ::: "memory");
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if (lane == 0) rec[blockIdx.x * (blockDim.x >> 5) + w] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 cs = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
  out[blockIdx.x * blockDim.x + threadIdx.x] = cs[0] + addr;
}

extern "C" __global__ __launch_bounds__(256) void k_p1w4dv(float* out, const bf16x16* in, int iters, Rec* rec, int seg) {
  extern __shared__ i32x4 lds4[];
  for (int i = threadIdx.x; i < 256 * 1024 / 16; i += blockDim.x) lds4[i] = i32x4{i, i + 1, i + 2, i + 3};
  __syncthreads();
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned addr = ((lane * 16 + w * 4096) & 0xFFFF) + (seg ? ((w & 1) << 16) : 0u);
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[96:99], %10 offset:0\n\tds_load_b128 v[100:103], %10 offset:2048\n\tds_load_b128 v[104:107], %10 offset:4096\n\tds_load_b128 v[108:111], %10 offset:6144\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[112:115], %10 offset:8192\n\tds_load_b128 v[116:119], %10 offset:10240\n\tds_load_b128 v[120:123], %10 offset:12288\n\tds_load_b128 v[124:127], %10 offset:14336\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[128:131], %10 offset:16384\n\tds_load_b128 v[132:135], %10 offset:18432\n\tds_load_b128 v[136:139], %10 offset:20480\n\tds_load_b128 v[140:143], %10 offset:22528\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[144:147], %10 offset:24576\n\tds_load_b128 v[148:151], %10 offset:26624\n\tds_load_b128 v[152:155], %10 offset:28672\n\tds_load_b128 v[156:159], %10 offset:30720\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[160:163], %10 offset:0\n\tds_load_b128 v[164:167], %10 offset:2048\n\tds_load_b128 v[168:171], %10 offset:4096\n\tds_load_b128 v[172:175], %10 offset:6144\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[176:179], %10 offset:8192\n\tds_load_b128 v[180:183], %10 offset:10240\n\tds_load_b128 v[184:187], %10 offset:12288\n\tds_load_b128 v[188:191], %10 offset:14336\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[192:195], %10 offset:16384\n\tds_load_b128 v[196:199], %10 offset:18432\n\tds_load_b128 v[200:203], %10 offset:20480\n\tds_load_b128 v[204:207], %10 offset:22528\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[208:211], %10 offset:24576\n\tds_load_b128 v[212:215], %10 offset:26624\n\tds_load_b128 v[216:219], %10 offset:28672\n\tds_load_b128 v[220:223], %10 offset:30720\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[224:227], %10 offset:0\n\tds_load_b128 v[228:231], %10 offset:2048\n\tds_load_b128 v[232:235], %10 offset:4096\n\tds_load_b128 v[236:239], %10 offset:6144\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[240:243], %10 offset:8192\n\tds_load_b128 v[244:247], %10 offset:10240\n\tds_load_b128 v[248:251], %10 offset:12288\n\tds_load_b128 v[252:255], %10 offset:14336\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[96:99], %10 offset:16384\n\tds_load_b128 v[100:103], %10 offset:18432\n\tds_load_b128 v[104:107], %10 offset:20480\n\tds_load_b128 v[108:111], %10 offset:22528\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[112:115], %10 offset:24576\n\tds_load_b128 v[116:119], %10 offset:26624\n\tds_load_b128 v[120:123], %10 offset:28672\n\tds_load_b128 v[124:127], %10 offset:30720\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[128:131], %10 offset:0\n\tds_load_b128 v[132:135], %10 offset:2048\n\tds_load_b128 v[136:139], %10 offset:4096\n\tds_load_b128 v[140:143], %10 offset:6144\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[144:147], %10 offset:8192\n\tds_load_b128 v[148:151], %10 offset:10240\n\tds_load_b128 v[152:155], %10 offset:12288\n\tds_load_b128 v[156:159], %10 offset:14336\n\ts_wait_dscnt 16\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[160:163], %10 offset:16384\n\tds_load_b128 v[164:167], %10 offset:18432\n\tds_load_b128 v[168:171], %10 offset:20480\n\tds_load_b128 v[172:175], %10 offset:22528\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_add_f32 %10, 0, %10\n\tds_load_b128 v[176:179], %10 offset:24576\n\tds_load_b128 v[180:183], %10 offset:26624\n\tds_load_b128 v[184:187], %10 offset:28672\n\tds_load_b128 v[188:191], %10 offset:30720\n\ts_wait_dscnt 16" : "+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3),"+v"(c4),"+v"(c5),"+v"(c6),"+v"(c7),"+v"(a),"+v"(b),"+v"(addr) :: "memory", "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v107", "v108", "v109", "v110", "v111", "v112", "v113", "v114", "v115", "v116", "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v125", "v126", "v127", "v128", "v129", "v130", "v131", "v132", "v133", "v134", "v135", "v136", "v137", "v138", "v139", "v140", "v141", "v142", "v143", "v144", "v145", "v146", "v147", "v148", "v149", "v150", "v151", "v152", "v153", "v154", "v155", "v156", "v157", "v158", "v159", "v160", "v161", "v162", "v163", "v164", "v165", "v166", "v167", "v168", "v169", "v170", "v171", "v172", "v173", "v174", "v175", "v176", "v177", "v178", "v179", "v180", "v181", "v182", "v183", "v184", "v185", "v186", "v187", "v188", "v189", "v190", "v191", "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247", "v248", "v249", "v250", "v251", "v252", "v253", "v254", "v255");
  }
  asm volatile("s_wait_dscnt 0" ::: "memory");
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if (lane == 0) rec[blockIdx.x * (blockDim.x >> 5) + w] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 cs = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
  out[blockIdx.x * blockDim.x + threadIdx.x] = cs[0] + addr;
}
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIPERR %s line %d\n", hipGetErrorString(e_), __LINE__); return 3; } } while (0)
typedef void (*KFn)(float*, const bf16x16*, int, Rec*, int);
static u64 med(std::vector<u64> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
int main(int argc, char** argv) {
  bool toy = argc > 1 && !strcmp(argv[1], "toy");
  struct C { const char* n; KFn f; int loads, wmmas; } cs[] = { {"p1w1d", k_p1w1d, 16, 16}, {"p1w2d", k_p1w2d, 32, 16}, {"p1w4d", k_p1w4d, 64, 16}, {"p2w2d", k_p2w2d, 16, 16}, {"p4w4d", k_p4w4d, 16, 16}, {"p8w8d", k_p8w8d, 16, 16}, {"p2w1d", k_p2w1d, 8, 16}, {"p4w1d", k_p4w1d, 4, 16}, {"p8w1d", k_p8w1d, 2, 16}, {"p16w1d", k_p16w1d, 1, 16}, {"p1w1dv", k_p1w1dv, 16, 16}, {"p1w1ds", k_p1w1ds, 16, 16}, {"p4w4dv", k_p4w4dv, 16, 16}, {"p1w4dv", k_p1w4dv, 64, 16} };
  const size_t LDS = 256 * 1024;
  std::vector<__bf16> hin(1024 * 16); for (size_t i = 0; i < hin.size(); i++) hin[i] = (__bf16)((i * 37 % 101) / 101.f - 0.5f);
  bf16x16* din; CK(hipMalloc(&din, hin.size() * 2)); CK(hipMemcpy(din, hin.data(), hin.size() * 2, hipMemcpyHostToDevice));
  float* dout; CK(hipMalloc(&dout, 256 * 512 * 4)); Rec* drec; CK(hipMalloc(&drec, 256 * 16 * sizeof(Rec)));
  hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
  for (auto& c : cs) {
    CK(hipFuncSetAttribute((const void*)c.f, hipFuncAttributeMaxDynamicSharedMemorySize, (int)LDS));
    for (int seg = 0; seg <= 0; seg++)
    for (int wps = 1; wps <= 2; wps *= 2) {
      if (toy && (wps > 1 || seg)) continue;
      int grid = toy ? 1 : 256, block = 128 * wps, nw = grid * block / 32, iters = toy ? 16 : 16384 / wps;
      std::vector<u64> cy, rt; std::vector<float> ev; std::vector<Rec> h(nw);
      for (int rep = 0; rep < 5; rep++) {
        CK(hipMemset(drec, 0, nw * sizeof(Rec)));
        CK(hipEventRecord(e0)); hipLaunchKernelGGL(c.f, dim3(grid), dim3(block), LDS, 0, dout, din, iters, drec, seg);
        CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipGetLastError());
        float ms; CK(hipEventElapsedTime(&ms, e0, e1));
        CK(hipMemcpy(h.data(), drec, nw * sizeof(Rec), hipMemcpyDeviceToHost));
        std::vector<u64> a, b; int miss = 0; for (auto& r : h) { if (!r.cyc) miss++; a.push_back(r.cyc); b.push_back(r.rt); }
        if (miss) printf("BAD %s missing=%d\n", c.n, miss);
        if (rep >= 1) { cy.push_back(*std::max_element(a.begin(), a.end())); rt.push_back(*std::max_element(b.begin(), b.end())); ev.push_back(ms); }
      }
      std::sort(ev.begin(), ev.end());
      u64 C = med(cy), R = med(rt);
      // per CU: 4*wps waves, each iters*loads loads of 512 B; C = slowest wave's span (all waves co-resident)
      double bytes_cu = 4.0 * wps * iters * c.loads * 512.0;
      printf("MB5 %s seg=%d wps=%d ev_ms=%.4f cyc_max=%llu f_rt=%.0f B_per_clk_CU=%.1f cyc_per_wmma_simd=%.3f\n", c.n, seg, wps, ev[ev.size() / 2], C,
             (double)C / R * 100, bytes_cu / C, c.wmmas ? (double)C / ((double)iters * c.wmmas * wps) : 0.0);
      fflush(stdout);
    }
  }
  printf("MBDONE\n"); return 0;
}
