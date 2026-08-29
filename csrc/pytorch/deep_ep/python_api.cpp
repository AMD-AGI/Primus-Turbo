#include <pybind11/pybind11.h>
#include <torch/python.h>

// [agent modifed]: include root deep_ep/ -> primus_turbo/deep_ep/
#include <primus_turbo/deep_ep/common/compiled.cuh>

// [agent modifed]: "jit/api.hpp" and "elastic/buffer.hpp" are dropped -- the JIT runtime is
// elastic-only and elastic/ is a placeholder here (see its README).
#include "legacy/buffer.hpp"

// [agent modifed]: `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` -> a plain registration function.
// Turbo already owns the one module entry point (csrc/pytorch/bindings_pytorch.cpp); DeepEP
// hangs off it as the `deep_ep` submodule, which is what upstream's module became here.
namespace primus_turbo::deep_ep {

void register_deep_ep_apis(pybind11::module_& m) {
    m.doc() = "DeepEP: an efficient expert-parallel communication library";

    // Whether support FP8 and TMA features
    m.def("is_sm90_compiled", []() { return kEnableSM90Features; });

    // The integer type of top-k indices
    m.attr("topk_idx_t") = pybind11::cast(c10::CppTypeToScalarType<topk_idx_t>::value);

    // Register legacy buffer APIs
    register_apis(m);
}

}  // namespace primus_turbo::deep_ep
