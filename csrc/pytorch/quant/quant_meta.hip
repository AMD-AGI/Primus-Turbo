#include <hip/hip_runtime.h>

#include "../extensions.h"

namespace primus_turbo::pytorch {

at::Tensor fp8_quantize_meta(const at::Tensor input, const at::Tensor scale, const at::ScalarType dest_dtype) {
  return torch::empty_like(input, torch::dtype(dest_dtype).device(at::kMeta));
}


at::Tensor fp8_dequantize_meta(const at::Tensor input, const at::Tensor scale_inv, const at::ScalarType dest_dtype) {
  return torch::empty_like(input, torch::dtype(dest_dtype).device(at::kMeta));
}

}
