#include <hip/hip_runtime.h>
typedef unsigned int uint2_t __attribute__((ext_vector_type(2)));
__global__ void k(unsigned* o){
  uint2_t r = __builtin_amdgcn_permlane32_swap(o[0], o[1], false, true);
  o[0]=r.x; o[1]=r.y;
}
