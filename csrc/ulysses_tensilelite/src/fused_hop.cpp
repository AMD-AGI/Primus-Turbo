/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Adapted from attn-a2a-demo (https://github.com/DDEle/attn-a2a-demo) at 7a7343ef47,
// csrc/fused_hop.cpp.
#include "fused_hop.hpp"

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/FusedA2A.hpp>
#include <Tensile/FusedA2AKernArg.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <sstream>
#include <stdexcept>

namespace aa {
using namespace TensileLite;

const size_t kFlagBlockBytes = FUSED_A2A_FLAG_BLOCK_BYTES;

namespace {
// TN (A transposed, as in the forward) and NN (A read as stored, the backward dgrads)
constexpr char const *kOpIdTN = "Contraction_l_Alik_Bljk_Cijk_Dijk";
constexpr char const *kOpIdNN = "Contraction_l_Ailk_Bljk_Cijk_Dijk";

FusedA2A parseMode(std::string const &name) {
    std::istringstream in(name);
    FusedA2A           mode;
    in >> mode;
    return mode;
}

ContractionProblemGemm makeProblem(HopShape const &s) {
    // indexSizes = [M, N, batch, K]; empty strides = packed.
    auto p = ContractionProblemGemm::FromIndexSizes(
        s.trans_a ? kOpIdTN : kOpIdNN, {s.m, s.n, 1, s.k}, rocisa::DataType::BFloat16, {},
        rocisa::DataType::BFloat16, {}, rocisa::DataType::BFloat16, {}, rocisa::DataType::BFloat16,
        {}, s.beta);
    p.setComputeInputTypeA(rocisa::DataType::BFloat16);
    p.setComputeInputTypeB(rocisa::DataType::BFloat16);
    p.setAlphaRestriction(ScalarValue::One);
    p.setCEqualsD(false);
    p.setAlphaType(rocisa::DataType::Float);
    p.setBetaType(rocisa::DataType::Float);
    p.setStridedBatched(true);
    p.setHighPrecisionAccumulate(true);
    p.setKernelLanguage(KernelLanguage::Assembly);
    p.setDeterministicMode(false);
    p.setActivationType(ActivationType::None);
    p.setWorkspaceSize(0);
    p.setFusedA2A(parseMode(s.mode));
    p.setWorld(s.world);
    p.setFusedA2AExtent(s.extent);
    p.setA2ABuf();
    return p;
}
} // namespace

struct FusedHop::Impl {
    using Library = SolutionLibrary<ContractionProblemGemm>;

    HopShape                              shape;
    std::shared_ptr<Hardware>             hardware;
    std::shared_ptr<AMDGPU>               amd;
    std::shared_ptr<hip::SolutionAdapter> adapter;
    ContractionProblemGemm                problem;
    std::shared_ptr<Library>              library;
    std::shared_ptr<ContractionSolution>  solution;

    std::array<std::vector<KernelInvocation>, kChannels> kernels;
    std::array<bool, kChannels>                          bound{};
};

static_assert(std::is_same_v<FusedHop::PeerFields, FusedA2APeerFields>);

FusedHop::FusedHop(HopShape const &shape, std::string const &dat, std::string const &coDir,
                   std::string const &co)
    : m_impl(std::make_unique<Impl>()) {
    auto &d    = *m_impl;
    d.shape    = shape;
    d.hardware = hip::GetCurrentDevice();
    d.amd      = std::dynamic_pointer_cast<AMDGPU>(d.hardware);
    if (!d.amd || d.amd->computeUnitCount == 0)
        throw std::runtime_error("FusedHop: hardware has no CU count");
    d.adapter = std::make_shared<hip::SolutionAdapter>();
    d.problem = makeProblem(shape);
    if (d.problem.world() != (size_t) shape.world)
        throw std::runtime_error("FusedHop: problem world did not take");

    d.library = LoadLibraryFile<ContractionProblemGemm>(dat);
    if (!d.library)
        throw std::runtime_error("FusedHop: could not load " + dat);
    d.adapter->setLazyLoadingContext(d.amd->archName(), coDir);
    HIP_CHECK_EXC(d.adapter->loadCodeObjectFile(co));
    HIP_CHECK_EXC(d.adapter->initializeLazyLoading(d.amd->archName(), coDir));

    d.solution = d.library->findBestSolution(d.problem, *d.hardware);
    if (!d.solution)
        throw std::runtime_error("FusedHop: no solution for " + d.problem.description());
    if (d.solution->requiredWorkspaceSize(d.problem, *d.hardware) != 0)
        throw std::runtime_error("FusedHop: solution needs a workspace");
}

FusedHop::~FusedHop() = default;

void FusedHop::bind(int channel, std::vector<PeerFields> const &peers, void const *a, void const *b,
                    void *d, void *counter, void const *c) {
    auto &s = *m_impl;
    if (channel < 0 || channel >= kChannels)
        throw std::runtime_error("FusedHop::bind: bad channel");
    if (peers.size() != (size_t) s.shape.world)
        throw std::runtime_error("FusedHop::bind: need one peer group per rank");
    for (size_t j = 0; j < peers.size(); j++)
        for (void *f : peers[j])
            if (f == nullptr)
                throw std::runtime_error("FusedHop::bind: null field in peer group " +
                                         std::to_string(j));
    ContractionInputs in;
    in.a = a;
    in.b = b;
    if (s.shape.beta != 0.0 && c == nullptr)
        throw std::runtime_error("FusedHop::bind: beta != 0 needs a C pointer");
    in.c               = c ? c : d;
    in.d               = d;
    in.alpha           = 1.0f;
    in.beta            = (float) s.shape.beta;
    in.Synchronizer    = counter;
    in.fusedA2APeers   = peers;
    in.fusedA2ACounter = counter;
    in.fusedA2AMyRank  = (uint32_t) s.shape.rank;
    in.fusedA2ADrain   = FUSED_A2A_DRAIN_RECV | FUSED_A2A_DRAIN_SEND;
    s.kernels[channel] = s.solution->solve(s.problem, in, *s.hardware);
    s.bound[channel]   = true;
}

void FusedHop::launch(int channel, uintptr_t stream) {
    auto &s = *m_impl;
    if (channel < 0 || channel >= kChannels || !s.bound[channel])
        throw std::runtime_error("FusedHop::launch: channel not bound");
    HIP_CHECK_EXC(s.adapter->launchKernels(
        s.kernels[channel], reinterpret_cast<hipStream_t>(stream), nullptr, nullptr));
}

int FusedHop::world() const {
    return m_impl->shape.world;
}

std::string FusedHop::kernelName() const {
    return m_impl->solution->KernelName();
}

std::string FusedHop::modeName() const {
    return ToString(parseMode(m_impl->shape.mode));
}

std::array<size_t, 2> FusedHop::macroTile() const {
    auto const &mt = m_impl->solution->sizeMapping.macroTile;
    return {(size_t) mt.x, (size_t) mt.y};
}

size_t FusedHop::counterBytes() const {
    // Twinned with client/src/DeviceContext.cpp kCounter*Offset and a2aGemmCounterBytes.
    constexpr size_t kCounter2Offset =
        fusedA2AAlignLine((size_t) FUSED_A2A_MAX_RANKS * 2 * sizeof(uint64_t));
    constexpr size_t kCounter3Offset =
        fusedA2AAlignLine(kCounter2Offset + FUSED_A2A_MAX_RANKS * sizeof(uint32_t));
    constexpr size_t kCounter1Offset = fusedA2AAlignLine(kCounter3Offset + sizeof(uint32_t));

    auto const  &d           = *m_impl;
    const size_t mt0         = d.solution->sizeMapping.macroTile.x;
    const size_t mt1         = d.solution->sizeMapping.macroTile.y;
    const size_t W           = d.shape.world;
    const size_t tokenBlocks = (d.shape.n + mt1 - 1) / mt1;
    if (parseMode(d.shape.mode) == FusedA2A::GemmA2A)
        return kCounter1Offset + W * tokenBlocks * sizeof(uint32_t);

    const size_t featureBlocks = (d.shape.m + mt0 - 1) / mt0;
    const size_t batches       = (tokenBlocks - 1) * featureBlocks / d.amd->computeUnitCount + 1;
    const size_t flagBytes     = fusedA2AAlignLine(
        tokenBlocks * (W - 1) * d.solution->sizeMapping.a2aSubRounds * sizeof(uint32_t));
    const size_t tallyBytes = fusedA2AAlignLine((tokenBlocks + 1) * sizeof(uint32_t));
    return kCounter2Offset + flagBytes + tallyBytes + batches * sizeof(uint32_t);
}
} // namespace aa
