__global__ void k() {
  typedef unsigned int v4u32 __attribute__((ext_vector_type(4)));
  typedef unsigned int v8u32 __attribute__((ext_vector_type(8)));
  v4u32 g0={0,0,0,0}, g2={0,0,0,0}, g3={0,0,0,0};
  v8u32 g1={0,0,0,0,0,0,0,0};
  __builtin_amdgcn_tensor_load_to_lds(g0,g1,g2,g3,g1,0);
}
