#include <pybind11/pybind11.h>
#include <torch/python.h>

#include <primus_turbo/deep_ep/common/compiled.cuh>

// ROCm: "jit/api.hpp" and "elastic/buffer.hpp" are dropped, elastic/ is a placeholder
#include "legacy/buffer.hpp"

// ROCm: PYBIND11_MODULE -> a registration function, Turbo owns the module entry point
namespace primus_turbo::deep_ep {

void register_deep_ep_apis(pybind11::module_& m) {
    m.doc() = "DeepEP: an efficient expert-parallel communication library";

    // Whether support FP8 and TMA features
    // ROCm: callers only read this as "FP8 available", and CDNA has FP8 without TMA
    m.def("is_sm90_compiled", []() { return kEnableFP8Features; });

    // The integer type of top-k indices
    m.attr("topk_idx_t") = pybind11::cast(c10::CppTypeToScalarType<topk_idx_t>::value);

    // Register legacy buffer APIs
    legacy::register_apis(m);
}

}  // namespace primus_turbo::deep_ep
