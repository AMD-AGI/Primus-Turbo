/***************************************************************************************************
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

#include <torch/extension.h>

#include "extensions.h"

namespace primus_turbo::pytorch {

/********************************************/

TORCH_LIBRARY(primus_turbo_cpp_extension, m) {
    // ********* Gemm *********
    m.def("hipblaslt_gemm(Tensor A, Tensor B, "
          "ScalarType out_dtype, bool transA, bool transB, bool transC,"
          "float beta=0.0, Tensor(a!)? out=None) -> Tensor");
    m.def("hipblaslt_gemm_fp8(Tensor A, Tensor scaleA_inv, Tensor B, Tensor scaleB_inv,"
          "ScalarType out_dtype, bool transA, bool transB, bool transC, str granularity,"
          "float beta=0.0, Tensor(a!)? out=None) -> Tensor");
    m.def("hipblaslt_gemm_fp4(Tensor A, Tensor scaleA_inv, Tensor B, Tensor scaleB_inv,"
          "ScalarType out_dtype, bool transA, bool transB, bool transC, str granularity,"
          "float beta=0.0, Tensor(a!)? out=None) -> Tensor");
    m.def("ck_gemm_fp8(Tensor a, Tensor b, Tensor a_scales, Tensor b_scales, bool transA,"
          "bool transB, ScalarType out_dtype, str granularity) -> Tensor");

    m.def(
        "turbo_gemm_fp8(Tensor A, Tensor scaleA_inv, Tensor B, Tensor scaleB_inv,"
        "ScalarType out_dtype, bool transA, bool transB, bool transC, str granularity) -> Tensor");

    // ********* Quantization *********
    m.def("quantize_fp8_tensorwise(Tensor input, ScalarType dest_dtype, Tensor? scale_opt=None, "
          "int padding_align_size=128, int pad_penultimate_align_size=1) -> Tensor[]");
    m.def("quantize_fp8_rowwise(Tensor input, ScalarType dest_dtype, int axis, Tensor? "
          "scale_opt=None) -> Tensor[]");
    m.def("quantize_fp8_blockwise_segment_m_row_col(Tensor input, ScalarType dest_dtype, "
          "int block_size, Tensor group_lens, Tensor group_offs) -> Tensor[]");
    m.def("quantize_fp8_blockwise_for_weight(Tensor input, ScalarType dest_dtype, int block_size) "
          "-> Tensor[]");

    // ********* Transpose *********
    m.def("transpose_2d(Tensor input, int dim0, int dim1) -> Tensor");

    m.def("dequantize_fp8_tensorwise(Tensor input, Tensor scale_inv, ScalarType dest_dtype) -> "
          "Tensor");
    m.def("dequantize_fp8_rowwise(Tensor input, Tensor scale_inv, int axis, "
          "ScalarType dest_dtype) -> Tensor");

    // ********* MXFP4 Quantization *********
    m.def("quantize_mxfp4_dual(Tensor input, ScalarType dest_dtype, "
          "int padding_align_size, "
          "bool rowwise_use_2d_block, bool rowwise_use_sr, bool rowwise_use_rht, "
          "bool colwise_use_2d_block, bool colwise_use_sr, bool colwise_use_rht, "
          "bool shuffle_rowwise_scale=False, bool shuffle_rowwise=False, "
          "bool shuffle_colwise_scale=False, bool shuffle_colwise=False) -> Tensor[]");
    m.def("quantize_mxfp4(Tensor input, ScalarType dest_dtype, int axis, "
          "int padding_align_size, "
          "bool use_2d_block, bool use_sr, bool use_rht, "
          "bool shuffle_scale=False, bool shuffle_out=False) -> Tensor[]");
    m.def("dequantize_mxfp4(Tensor input, Tensor scale_inv, int axis, int block_size, "
          "ScalarType dest_dtype) -> Tensor");
    m.def("grouped_quantize_mxfp4_dual(Tensor input, Tensor group_lens, Tensor group_offs, "
          "ScalarType dest_dtype, "
          "bool rowwise_use_2d_block, bool rowwise_use_sr, bool rowwise_use_rht, "
          "bool colwise_use_2d_block, bool colwise_use_sr, bool colwise_use_rht) -> Tensor[]");
    m.def("grouped_quantize_mxfp4(Tensor input, Tensor group_lens, Tensor group_offs, "
          "ScalarType dest_dtype, int axis, "
          "bool use_2d_block, bool use_sr, bool use_rht) -> Tensor[]");

    // ********* MXFP8 Quantization *********
    m.def("quantize_mxfp8_dual(Tensor input, ScalarType dest_dtype, "
          "int padding_align_size, "
          "bool rowwise_use_2d_block, bool colwise_use_2d_block, "
          "bool shuffle_rowwise_scale=False, bool shuffle_rowwise=False, "
          "bool shuffle_colwise_scale=False, bool shuffle_colwise=False) -> Tensor[]");
    m.def("grouped_quantize_mxfp8_dual(Tensor input, Tensor group_lens, Tensor group_offs, "
          "ScalarType dest_dtype, "
          "bool rowwise_use_2d_block, bool colwise_use_2d_block, "
          "bool shuffle_rowwise_scale=False, bool shuffle_rowwise=False, "
          "bool shuffle_colwise_scale=False, bool shuffle_colwise=False) -> Tensor[]");
    m.def("grouped_quantize_mxfp8(Tensor input, Tensor group_lens, Tensor group_offs, "
          "ScalarType dest_dtype, int axis, int padding_align_size, "
          "bool use_2d_block, bool shuffle_scale=False, bool shuffle_out=False) -> Tensor[]");
    m.def("quantize_mxfp8(Tensor input, ScalarType dest_dtype, int axis, "
          "int padding_align_size, "
          "bool use_2d_block, bool shuffle_scale=False, bool shuffle_out=False) -> Tensor[]");
    m.def("dequantize_mxfp8(Tensor input, Tensor scale_inv, int axis, int block_size, "
          "ScalarType dest_dtype) -> Tensor");
    m.def("grouped_dequantize_mxfp8(Tensor input, Tensor scale_inv, Tensor group_offs, "
          "Tensor group_offs_padded, int axis, int block_size, ScalarType dest_dtype, "
          "int? total_M=None) -> Tensor");

    // ********* Shuffle *********
    m.def("shuffle_scale(Tensor scale, int[] layout) -> Tensor");
    m.def("shuffle_weight(Tensor weight, int[] layout) -> Tensor");

    // ********* Permute (MoE token (un)permute) *********
    m.def("permute_preprocessing(Tensor expert_map, int num_local_experts, int num_topk, "
          "int pad_multiple, int num_permuted_tokens, int probs_topk_stride=0) "
          "-> (Tensor, Tensor, Tensor, Tensor)");
    m.def("permute(Tensor tokens, Tensor output_tokens, Tensor? scaling_factor, "
          "Tensor? output_scaling_factor, Tensor? probs, Tensor? output_probs, "
          "Tensor row_id_map, Tensor num_dispatched_token_tensor, "
          "int pad_multiple, int num_local_experts, int hidden_size, int scales_per_token, "
          "bool use_fp8, bool with_probs, int num_permuted_token, "
          "int probs_stride=0) -> ()");
    m.def("unpermute(Tensor permuted_tokens, Tensor output_tokens, "
          "Tensor? permuted_probs, Tensor? output_probs, Tensor row_id_map, "
          "Tensor num_dispatched_tokens_tensor, int num_local_experts, int hidden_size, "
          "bool with_probs, int probs_stride=0) -> ()");

    // ********* Grouped Gemm *********
    m.def("ck_grouped_gemm(Tensor a, Tensor b, Tensor group_lens, Tensor group_offs, bool transA, "
          "bool transB, int? num_cu=None, bool work_steal=False, Tensor? ws_counter=None, "
          "int ws_local_per_xcd=0) -> Tensor");
    m.def("ck_grouped_gemm_variable_k(Tensor a, Tensor b, Tensor group_lens, Tensor group_offs, "
          "bool transA, bool transB, int? num_cu=None, bool work_steal=False, "
          "Tensor? ws_counter=None, int ws_local_per_xcd=0) -> Tensor");
    m.def("ck_grouped_gemm_fp8(Tensor a, Tensor b, Tensor a_scales, Tensor b_scales, "
          "Tensor group_lens, Tensor group_offs, bool transA, bool transB, "
          "ScalarType out_dtype, str granularity, int? num_cu) -> Tensor");
    m.def("ck_grouped_gemm_fp8_variable_k(Tensor a, Tensor b, Tensor a_scales, Tensor b_scales, "
          "Tensor group_lens, Tensor group_offs, bool transA, bool transB, "
          "ScalarType out_dtype, str granularity, int? num_cu) -> Tensor");
    m.def("hipblaslt_grouped_gemm(Tensor a, Tensor b, Tensor group_lens, Tensor group_offs, "
          "bool transA, bool transB, bool pre_sync,"
          "float beta=0.0, Tensor(a!)? out=None) -> Tensor");
    m.def("hipblaslt_grouped_gemm_fp8(Tensor a, Tensor b, Tensor a_scales, Tensor b_scales, "
          "Tensor group_lens, Tensor group_offs, bool transA, bool transB, "
          "ScalarType out_dtype, str granularity, bool pre_sync,"
          "float beta=0.0, Tensor(a!)? out=None) -> Tensor");
    m.def("grouped_gemm_compute_offs(Tensor group_lens) -> Tensor");

    // ********* HipKittens attention (gfx950) *********
    // Registered only where the *_gfx950.cu kernels were built, i.e. where gfx950 is among the
    // offload archs; hipkittens_attn_supported reports their absence rather than letting the op
    // lookup fail. Even then they are only functional ON gfx950 -- the kernel bodies are guarded
    // on __gfx950__ and the Python layer refuses any other device before launch. The kernels
    // write through their outputs, so these are mutating and return nothing.
#ifdef BUILD_HIPKITTENS_BACKEND
    m.def("hk_attn_fwd_d64(Tensor q, Tensor k, Tensor v, Tensor(a!) o, Tensor(b!) lse, int Sq, "
          "int Skv, int B, int Hq, int Hkv, int window_left, float softmax_scale) -> ()");
    m.def("hk_attn_fwd_d128(Tensor q, Tensor k, Tensor v, Tensor(a!) o, Tensor(b!) lse, int Sq, "
          "int Skv, int B, int Hq, int Hkv, int window_left, float softmax_scale) -> ()");
    m.def("hk_attn_bwd_d64(Tensor q, Tensor k, Tensor v, Tensor o, Tensor dO, Tensor(a!) dq, "
          "Tensor(b!) dk, Tensor(c!) dv, Tensor lse, Tensor(d!) delta, Tensor(e!) lneg, "
          "Tensor(f!) wsk, Tensor(g!) wsv, int Sq, int Skv, int B, int Hq, int Hkv, "
          "int window_left, float softmax_scale, int n_split_req) -> ()");
    m.def("hk_attn_bwd_d128(Tensor q, Tensor k, Tensor v, Tensor o, Tensor dO, Tensor(a!) dq, "
          "Tensor(b!) dk, Tensor(c!) dv, Tensor lse, Tensor(d!) delta, Tensor(e!) lneg, "
          "Tensor(f!) wsk, Tensor(g!) wsv, int Sq, int Skv, int B, int Hq, int Hkv, "
          "int window_left, float softmax_scale, int n_split_req) -> ()");
    m.def("hk_attn_bwd_fused_d64(Tensor q, Tensor k, Tensor v, Tensor o, Tensor dO, "
          "Tensor(a!) dq, Tensor(b!) dk, Tensor(c!) dv, Tensor(d!) ws, Tensor lse, "
          "Tensor(e!) delta, int Sq, int Skv, int B, int Hq, int Hkv, int window_left, "
          "float softmax_scale) -> ()");
    m.def("hk_attn_bwd_fused_d128(Tensor q, Tensor k, Tensor v, Tensor o, Tensor dO, "
          "Tensor(a!) dq, Tensor(b!) dk, Tensor(c!) dv, Tensor(d!) ws, Tensor lse, "
          "Tensor(e!) delta, int Sq, int Skv, int B, int Hq, int Hkv, int window_left, "
          "float softmax_scale) -> ()");
    m.def("hk_attn_dkdv_head_split(int head_dim, int Sq, int Skv, int B, int Hq, int Hkv, "
          "int window_left) -> int");
    m.def("hk_attn_block_sizes(int head_dim) -> int[]");
#endif // BUILD_HIPKITTENS_BACKEND
}

TORCH_LIBRARY_IMPL(primus_turbo_cpp_extension, CUDA, m) {
    // ********* Gemm *********
    m.impl("hipblaslt_gemm", hipblaslt_gemm);
    m.impl("hipblaslt_gemm_fp8", hipblaslt_gemm_fp8);
    m.impl("hipblaslt_gemm_fp4", hipblaslt_gemm_fp4);
    m.impl("ck_gemm_fp8", ck_gemm_fp8);
    m.impl("turbo_gemm_fp8", turbo_gemm_fp8);
    // ********* Quantization *********
    m.impl("quantize_fp8_tensorwise", quantize_fp8_tensorwise);
    m.impl("transpose_2d", transpose_2d);
    m.impl("dequantize_fp8_tensorwise", dequantize_fp8_tensorwise);
    m.impl("quantize_fp8_rowwise", quantize_fp8_rowwise);
    m.impl("dequantize_fp8_rowwise", dequantize_fp8_rowwise);
    m.impl("quantize_fp8_blockwise_segment_m_row_col", quantize_fp8_blockwise_segment_m_row_col);
    m.impl("quantize_fp8_blockwise_for_weight", quantize_fp8_blockwise_for_weight);

    // ********* MXFP4 Quantization *********
    m.impl("quantize_mxfp4_dual", quantize_mxfp4_dual);
    m.impl("quantize_mxfp4", quantize_mxfp4);
    m.impl("dequantize_mxfp4", dequantize_mxfp4);
    m.impl("grouped_quantize_mxfp4_dual", grouped_quantize_mxfp4_dual);
    m.impl("grouped_quantize_mxfp4", grouped_quantize_mxfp4);

    // ********* MXFP8 Quantization *********
    m.impl("quantize_mxfp8_dual", quantize_mxfp8_dual);
    m.impl("grouped_quantize_mxfp8_dual", grouped_quantize_mxfp8_dual);
    m.impl("grouped_quantize_mxfp8", grouped_quantize_mxfp8);
    m.impl("quantize_mxfp8", quantize_mxfp8);
    m.impl("dequantize_mxfp8", dequantize_mxfp8);
    m.impl("grouped_dequantize_mxfp8", grouped_dequantize_mxfp8);

    // ********* Shuffle *********
    m.impl("shuffle_scale", shuffle_scale_impl);
    m.impl("shuffle_weight", shuffle_weight_impl);

    // ********* Permute *********
    m.impl("permute_preprocessing", permute_preprocessing);
    m.impl("permute", permute);
    m.impl("unpermute", unpermute);

    // ********* Grouped Gemm *********
    m.impl("ck_grouped_gemm", ck_grouped_gemm);
    m.impl("ck_grouped_gemm_variable_k", ck_grouped_gemm_variable_k);
    m.impl("ck_grouped_gemm_fp8", ck_grouped_gemm_fp8);
    m.impl("ck_grouped_gemm_fp8_variable_k", ck_grouped_gemm_fp8_variable_k);
    m.impl("grouped_gemm_compute_offs", grouped_gemm_compute_offs);
    m.impl("hipblaslt_grouped_gemm", hipblaslt_grouped_gemm);
    m.impl("hipblaslt_grouped_gemm_fp8", hipblaslt_grouped_gemm_fp8);

    // ********* HipKittens attention (gfx950) *********
#ifdef BUILD_HIPKITTENS_BACKEND
    m.impl("hk_attn_fwd_d64", hk_attn_fwd_d64);
    m.impl("hk_attn_fwd_d128", hk_attn_fwd_d128);
    m.impl("hk_attn_bwd_d64", hk_attn_bwd_d64);
    m.impl("hk_attn_bwd_d128", hk_attn_bwd_d128);
    m.impl("hk_attn_bwd_fused_d64", hk_attn_bwd_fused_d64);
    m.impl("hk_attn_bwd_fused_d128", hk_attn_bwd_fused_d128);
#endif // BUILD_HIPKITTENS_BACKEND
}

TORCH_LIBRARY_IMPL(primus_turbo_cpp_extension, Meta, m) {
    // ********* Gemm *********
    m.impl("hipblaslt_gemm", hipblaslt_gemm_meta);
    m.impl("hipblaslt_gemm_fp8", hipblaslt_gemm_fp8_meta);
    m.impl("ck_gemm_fp8", ck_gemm_fp8_meta);
    m.impl("turbo_gemm_fp8", turbo_gemm_fp8_meta);

    // ********* Quantization *********
    m.impl("quantize_fp8_tensorwise", quantize_fp8_tensorwise_meta);
    m.impl("transpose_2d", transpose_2d_meta);
    m.impl("dequantize_fp8_tensorwise", dequantize_fp8_tensorwise_meta);
    m.impl("quantize_fp8_rowwise", quantize_fp8_rowwise_meta);
    m.impl("dequantize_fp8_rowwise", dequantize_fp8_rowwise_meta);
    m.impl("quantize_fp8_blockwise_segment_m_row_col",
           quantize_fp8_blockwise_segment_m_row_col_meta);
    m.impl("quantize_fp8_blockwise_for_weight", quantize_fp8_blockwise_for_weight_meta);

    // ********* MXFP4 Quantization *********
    m.impl("quantize_mxfp4_dual", quantize_mxfp4_dual_meta);
    m.impl("quantize_mxfp4", quantize_mxfp4_meta);
    m.impl("dequantize_mxfp4", dequantize_mxfp4_meta);
    m.impl("grouped_quantize_mxfp4_dual", grouped_quantize_mxfp4_dual_meta);
    m.impl("grouped_quantize_mxfp4", grouped_quantize_mxfp4_meta);

    // ********* MXFP8 Quantization *********
    m.impl("quantize_mxfp8_dual", quantize_mxfp8_dual_meta);
    m.impl("grouped_quantize_mxfp8_dual", grouped_quantize_mxfp8_dual_meta);
    m.impl("grouped_quantize_mxfp8", grouped_quantize_mxfp8_meta);
    m.impl("quantize_mxfp8", quantize_mxfp8_meta);
    m.impl("dequantize_mxfp8", dequantize_mxfp8_meta);
    m.impl("grouped_dequantize_mxfp8", grouped_dequantize_mxfp8_meta);

    // ********* Shuffle *********
    m.impl("shuffle_scale", shuffle_scale_impl_meta);
    m.impl("shuffle_weight", shuffle_weight_impl_meta);

    // ********* Permute *********
    m.impl("permute_preprocessing", permute_preprocessing_meta);
    m.impl("permute", permute_meta);
    m.impl("unpermute", unpermute_meta);

    // ********* Grouped Gemm *********
    m.impl("ck_grouped_gemm", ck_grouped_gemm_meta);
    m.impl("ck_grouped_gemm_variable_k", ck_grouped_gemm_variable_k_meta);
    m.impl("ck_grouped_gemm_fp8", ck_grouped_gemm_fp8_meta);
    m.impl("ck_grouped_gemm_fp8_variable_k", ck_grouped_gemm_fp8_variable_k_meta);
    m.impl("grouped_gemm_compute_offs", grouped_gemm_compute_offs_meta);
    m.impl("hipblaslt_grouped_gemm", hipblaslt_grouped_gemm_meta);
    m.impl("hipblaslt_grouped_gemm_fp8", hipblaslt_grouped_gemm_fp8_meta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // ********* DeepEP *********
    // ROCm: the inline class_<Buffer> block -> deep_ep/python_api.cpp's registration
    auto deep_ep_module =
        m.def_submodule("deep_ep", "DeepEP: an efficient expert-parallel communication library");
    primus_turbo::deep_ep::register_deep_ep_apis(deep_ep_module);

    // ********* Runtime *********
    auto runtime_module = m.def_submodule("runtime", "Runtime utilities");
    runtime_module.def("create_stream_with_cu_masks", &create_stream_with_cu_masks);
    runtime_module.def("destroy_stream", &destroy_stream);

    // ********* ODC rocSHMEM distributed backends *********
#ifndef DISABLE_ROCSHMEM
    register_odc_rocshmem_host(m);
    register_odc_rocshmem_gda(m);
#endif
}

// Shape-only HipKittens queries: no tensors, so they answer the same on any device and do not
// belong on a device-specific key.
#ifdef BUILD_HIPKITTENS_BACKEND
TORCH_LIBRARY_IMPL(primus_turbo_cpp_extension, CompositeExplicitAutograd, m) {
    m.impl("hk_attn_dkdv_head_split", hk_attn_dkdv_head_split);
    m.impl("hk_attn_block_sizes", hk_attn_block_sizes);
}
#endif // BUILD_HIPKITTENS_BACKEND

/********************************************/

} // namespace primus_turbo::pytorch
