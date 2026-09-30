#include <hip/hip_runtime.h>
extern "C" __global__ void k(unsigned* o){
  unsigned h1, h2;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID2)" : "=s"(h2));
  o[threadIdx.x]=h1; o[threadIdx.x+64]=h2; unsigned x; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(x)); o[threadIdx.x+128]=x;}
