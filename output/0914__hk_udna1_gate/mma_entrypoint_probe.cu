// Which of HipKittens' three matrix entry points actually compile for gfx1250?
// Flip the -DPROBE_* define to select one.
#include "kittens.cuh"
using namespace kittens;
using AccS = ducks::rt_shape::rt_16x16;
using OpS  = ducks::rt_shape::rt_16x32;
using OpST = ducks::rt_shape::rt_32x16;

__global__ void k() {
    rt_base<float, ducks::rt_layout::col, AccS> d, c;
#if defined(PROBE_ABT)
    rt_base<bf16, ducks::rt_layout::row, OpS> a, b;
    mma_ABt_base(d, a, b, c);          // -> wmma_f32_16x16x32_bf16
#elif defined(PROBE_AB)
    rt_base<bf16, ducks::rt_layout::row, OpS>  a;
    rt_base<bf16, ducks::rt_layout::col, OpST> b;
    mma_AB_base(d, a, b, c);           // -> mfma161632  (CDNA-only)
#elif defined(PROBE_ATB)
    rt_base<bf16, ducks::rt_layout::col, OpST> a, b;
    mma_AtB_base(d, a, b, c);          // -> mfma161632  (CDNA-only)
#endif
    (void)d; (void)c;
}
int main(){return 0;}
