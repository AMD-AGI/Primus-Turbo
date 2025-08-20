#pragma once
#include <hip/hip_runtime.h>
#include "primus_turbo/ck_attention.h"
//#include "ck_tile/ops/fmha/block/variants.hpp"

namespace primus_turbo {

/**
 * Dtype traits
 */
struct FmhaFwdFp16 {};

struct FmhaFwdBf16 {};

struct FmhaFwdFp8 {};

struct FmhaFwdBf8 {};

struct FmhaFwdFp8Fp16 {};

struct FmhaFwdFp8Bf16 {};

template <typename DataType>
struct FmhaFwdTypeConfig;

template <>
struct FmhaFwdTypeConfig<FmhaFwdFp16> {
    using QDataType             = ck_tile::half_t;
    using KDataType             = ck_tile::half_t;
    using VDataType             = ck_tile::half_t;
    using BiasDataType          = ck_tile::half_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::half_t; // data type for A matrix of second gemm
    using OaccDataType          = float;           // data type for second gemm accumulation
    using ODataType             = ck_tile::half_t;
};

template <>
struct FmhaFwdTypeConfig<FmhaFwdBf16> {
    using QDataType             = ck_tile::bf16_t;
    using KDataType             = ck_tile::bf16_t;
    using VDataType             = ck_tile::bf16_t;
    using BiasDataType          = ck_tile::bf16_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::bf16_t; // data type for A matrix of second gemm
    using OaccDataType          = float;           // data type for second gemm accumulation
    using ODataType             = ck_tile::bf16_t;
};

template <>
struct FmhaFwdTypeConfig<FmhaFwdFp8> {
    using QDataType             = ck_tile::fp8_t;
    using KDataType             = ck_tile::fp8_t;
    using VDataType             = ck_tile::fp8_t;
    using BiasDataType          = float;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::fp8_t; // data type for A matrix of second gemm
    using OaccDataType          = float;          // data type for second gemm accumulation
    using ODataType             = ck_tile::fp8_t;
};

template <>
struct FmhaFwdTypeConfig<FmhaFwdBf8> {
    using QDataType             = ck_tile::bf8_t;
    using KDataType             = ck_tile::bf8_t;
    using VDataType             = ck_tile::bf8_t;
    using BiasDataType          = ck_tile::bf8_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::bf8_t; // data type for A matrix of second gemm
    using OaccDataType          = float;          // data type for second gemm accumulation
    using ODataType             = ck_tile::bf8_t;
};


/**
 * Bias Traits
 */

// template <bias_enum>
// struct BiasTypeTraits {
// };

// template <bias_enum::no_bias>
// struct BiasTypeTraits<> {
//     static constexpr ck_tile::BlockAttentionBiasEnum val = ck_tile::BlockAttentionBiasEnum::NO_BIAS;
// };

// template <>
// struct BiasTypeTraits<> {
// };

// template <>
// struct BiasTypeTraits<> {
// };

/**
 * TileSize Traits: Determain the tile size for the given HDimQK, HDimV and DataType.
 */
template <int HDimQK, int HDimV, typename DataType_>
struct FmhaTileSizeTraits;

template <>
struct FmhaTileSizeTraits<32, 32, FmhaFwdBf16> {
    static constexpr ck_tile::index_t kM0            = 128;
    static constexpr ck_tile::index_t kN0            = 64;
    static constexpr ck_tile::index_t kK0            = 16;
    static constexpr ck_tile::index_t kN1            = 32;
    static constexpr ck_tile::index_t kK1            = 32;
    static constexpr ck_tile::index_t kK0BlockLength = 32;

    using Gemm0BlockWarps =  ck_tile::sequence<2, 1, 1>;
    using Gemm0WarpTile   = ck_tile::sequence<32, 32, 16>;
    using Gemm1BlockWarps = ck_tile::sequence<2, 1, 1>;
    using Gemm1WarpTile   = ck_tile::sequence<32, 32, 16>;
    static constexpr int kOccupy = -1;
};

template <>
struct FmhaTileSizeTraits<64, 64, FmhaFwdBf16> {
    static constexpr ck_tile::index_t kM0            = 128;
    static constexpr ck_tile::index_t kN0            = 64;
    static constexpr ck_tile::index_t kK0            = 32;
    static constexpr ck_tile::index_t kN1            = 64;
    static constexpr ck_tile::index_t kK1            = 32;
    static constexpr ck_tile::index_t kK0BlockLength = 64;

    using Gemm0BlockWarps =  ck_tile::sequence<4, 1, 1>;
    using Gemm0WarpTile   = ck_tile::sequence<32, 32, 16>;
    using Gemm1BlockWarps = ck_tile::sequence<4, 1, 1>;
    using Gemm1WarpTile   = ck_tile::sequence<32, 32, 16>;
    static constexpr int kOccupy = -1;
};

template <>
struct FmhaTileSizeTraits<128, 128, FmhaFwdBf16> {
    static constexpr ck_tile::index_t kM0            = 128;
    static constexpr ck_tile::index_t kN0            = 128;
    static constexpr ck_tile::index_t kK0            = 32;
    static constexpr ck_tile::index_t kN1            = 128;
    static constexpr ck_tile::index_t kK1            = 32;
    static constexpr ck_tile::index_t kK0BlockLength = 128;

    using Gemm0BlockWarps =  ck_tile::sequence<4, 1, 1>;
    using Gemm0WarpTile   = ck_tile::sequence<32, 32, 16>;
    using Gemm1BlockWarps = ck_tile::sequence<4, 1, 1>;
    using Gemm1WarpTile   = ck_tile::sequence<32, 32, 16>;
    static constexpr int kOccupy = -1;
};

template <>
struct FmhaTileSizeTraits<192, 128, FmhaFwdBf16> {
    static constexpr ck_tile::index_t kM0            = 128;
    static constexpr ck_tile::index_t kN0            = 128;
    static constexpr ck_tile::index_t kK0            = 32;
    static constexpr ck_tile::index_t kN1            = 128;
    static constexpr ck_tile::index_t kK1            = 32;
    static constexpr ck_tile::index_t kK0BlockLength = 192;

    using Gemm0BlockWarps =  ck_tile::sequence<4, 1, 1>;
    using Gemm0WarpTile   = ck_tile::sequence<32, 32, 16>;
    using Gemm1BlockWarps = ck_tile::sequence<4, 1, 1>;
    using Gemm1WarpTile   = ck_tile::sequence<32, 32, 16>;
    static constexpr int kOccupy = -1;
};

template <>
struct FmhaTileSizeTraits<256, 256, FmhaFwdBf16> {
    static constexpr ck_tile::index_t kM0            = 128;
    static constexpr ck_tile::index_t kN0            = 128;
    static constexpr ck_tile::index_t kK0            = 32;
    static constexpr ck_tile::index_t kN1            = 256;
    static constexpr ck_tile::index_t kK1            = 32;
    static constexpr ck_tile::index_t kK0BlockLength = 256;

    using Gemm0BlockWarps =  ck_tile::sequence<4, 1, 1>;
    using Gemm0WarpTile   = ck_tile::sequence<32, 32, 16>;
    using Gemm1BlockWarps = ck_tile::sequence<4, 1, 1>;
    using Gemm1WarpTile   = ck_tile::sequence<32, 32, 16>;
    static constexpr int kOccupy = -1;
};

/**
 * Pipeline traits: Used to determine the pipeline type based on the BiasEnum.
 */
// bias
template <ck_tile::BlockAttentionBiasEnum BiasEnum_, typename PipelineProblem>
struct FmhaPipelineTraits {
    using PipeLine = ck_tile::BlockFmhaPipelineQRKSVS<PipelineProblem>;
};

// nobias
template <typename PipelineProblem>
struct FmhaPipelineTraits<ck_tile::BlockAttentionBiasEnum::NO_BIAS, PipelineProblem> {
    using PipeLine = ck_tile::BlockFmhaPipelineQRKSVSAsync<PipelineProblem>;
};

/**
 * UserInterface, use these type below to determine which kernel invoked
 */
template <ck_tile::index_t HDim_,
          ck_tile::index_t HDimV_,
          typename DataType_,
          bool kIsGroupMode_,
          bool kIsVLayoutRowMajor_,
          bool kHasLogitsSoftCap_,
          typename FmhaMask_,
          ck_tile::BlockAttentionBiasEnum BiasEnum_,
          bool kStoreLse_,
          bool kHasDropout_,
          bool kDoFp8StaticQuant_,
          bool kPadS_,
          bool kPadSK_,
          bool kPadD_,
          bool kPadDv_,
          bool kSkipMinSeqlenQ_ = false>
struct FmhaFwdKernelTraitsParam {
    static constexpr ck_tile::index_t HDim           = HDim_;
    static constexpr ck_tile::index_t HDimV          = HDimV_;
    using DataType                                   = ck_tile::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode               = kIsGroupMode_;
    static constexpr bool kIsVLayoutRowMajor         = kIsVLayoutRowMajor_;
    //static constexpr auto FmhaPipelineEnum           = FmhaPipelineEnum_;
    static constexpr bool kHasLogitsSoftCap          = kHasLogitsSoftCap_;
    using FmhaMask                                   = ck_tile::remove_cvref_t<FmhaMask_>;
    static constexpr auto BiasEnum                   = BiasEnum_;
    static constexpr bool kStoreLse                  = kStoreLse_;
    static constexpr bool kHasDropout                = kHasDropout_;
    static constexpr bool kDoFp8StaticQuant          = kDoFp8StaticQuant_;
    static constexpr bool kPadS                      = kPadS_;
    static constexpr bool kPadSK                     = kPadSK_;
    static constexpr bool kPadD                      = kPadD_;
    static constexpr bool kPadDv                     = kPadDv_;
    static constexpr bool kSkipMinSeqlenQ            = kSkipMinSeqlenQ_;
};

/**
 * A bridge from user-defined TraitsParam to ck kernel
 */
template <typename T>
struct FmhaFwdKernelTraits {
    using fmha_dtype_0 = typename T::DataType;

    using tile_size_0 = FmhaTileSizeTraits<T::HDim, T::HDimV, typename T::DataType>;
    using fmha_block_tile_0 = ck_tile::sequence<tile_size_0::kM0, tile_size_0::kN0, tile_size_0::kK0, tile_size_0::kN1, tile_size_0::kK1, tile_size_0::kK0BlockLength>;

    using fmha_shape_0 = ck_tile::TileFmhaShape<fmha_block_tile_0,
        typename tile_size_0::Gemm0BlockWarps,
        typename tile_size_0::Gemm0WarpTile,
        typename tile_size_0::Gemm1BlockWarps,
        typename tile_size_0::Gemm1WarpTile,
        T::kIsVLayoutRowMajor
    >;

    using fmha_trait_0 = ck_tile::TileFmhaTraits<
        T::kPadS, // Fspad
        T::kPadSK, // Fskpad
        T::kPadD, // Fdpad
        T::kPadDv, // Fdvpad
        T::kHasLogitsSoftCap,     // Flogits
        T::BiasEnum, // F_bias
        false, // fixed
        T::kStoreLse, // F_lse
        T::kHasDropout, // F_dropout
        T::kDoFp8StaticQuant, // F_squant
        tile_size_0::kOccupy, // F_occupacy
        T::kSkipMinSeqlenQ // F_skip
    >;

    using fmha_variant_0 = ck_tile::ComposedAttention<T::kHasLogitsSoftCap * ck_tile::LOGITS_SOFT_CAP, CK_TILE_FMHA_FWD_FAST_EXP2>;
    
    // TODO: add mask traits
    using fmha_pipeline_problem_0 = ck_tile::BlockFmhaPipelineProblem<
        typename FmhaFwdTypeConfig<fmha_dtype_0>::QDataType,
        typename FmhaFwdTypeConfig<fmha_dtype_0>::KDataType,
        typename FmhaFwdTypeConfig<fmha_dtype_0>::VDataType,
        typename FmhaFwdTypeConfig<fmha_dtype_0>::SaccDataType,
        typename FmhaFwdTypeConfig<fmha_dtype_0>::SMPLComputeDataType,
        typename FmhaFwdTypeConfig<fmha_dtype_0>::BiasDataType,
        typename FmhaFwdTypeConfig<fmha_dtype_0>::RandValOutputDataType,
        typename FmhaFwdTypeConfig<fmha_dtype_0>::LSEDataType,
        typename FmhaFwdTypeConfig<fmha_dtype_0>::PDataType,
        typename FmhaFwdTypeConfig<fmha_dtype_0>::OaccDataType,
        typename FmhaFwdTypeConfig<fmha_dtype_0>::ODataType,
        fmha_shape_0,
        T::kIsGroupMode, 
        fmha_variant_0,
        typename T::FmhaMask,
        false, // TR load
        fmha_trait_0
    >;

    using fmha_pipeline_0 = FmhaPipelineTraits<T::BiasEnum, fmha_pipeline_problem_0>::PipeLine;
    
    using fmha_epilogue_0 = ck_tile::Default2DEpilogue<
        ck_tile::Default2DEpilogueProblem<
            typename FmhaFwdTypeConfig<fmha_dtype_0>::OaccDataType,
            typename FmhaFwdTypeConfig<fmha_dtype_0>::ODataType,
            T::kPadS, 
            T::kPadDv
        >
    >;

    using fmha_kernel_0 = ck_tile::FmhaFwdKernel<fmha_pipeline_0, fmha_epilogue_0>;
};

}