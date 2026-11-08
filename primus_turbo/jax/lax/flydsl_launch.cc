// Generic XLA FFI trampoline that launches a FlyDSL-compiled (MLIR ExecutionEngine) host function.
//
// A FlyDSL @flyc.jit launcher compiles to `void f(void **packed)`, where packed[i] points at the storage of
// ABI slot i (pointers, scalars, dynamic-layout buffers and the stream). Python captures one filled copy of
// those slots ("template"); this handler copies it, patches in the device pointers of the XLA operands/results
// and the XLA stream, and calls the function. Nothing here touches Python, so it is safe inside XLA's executor.
//
//   build: hipcc -shared -fPIC -O2 -std=c++17 -I<jax.ffi.include_dir()> flydsl_launch.cc -o libflydsl_launch.so
#include <hip/hip_runtime.h>
#include <xla/ffi/api/ffi.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ffi = xla::ffi;

namespace {

constexpr int kMaxSlots = 32;

ffi::Error FlyDSLLaunchImpl(hipStream_t stream, ffi::RemainingArgs args, ffi::RemainingRets rets,
                            ffi::Span<const int64_t> fns,       // JIT'd function address per device ordinal
                            ffi::Span<const uint8_t> tmpl,      // concatenated slot storage
                            ffi::Span<const int32_t> slot_off,  // byte offset of slot i in tmpl (n_slots entries)
                            ffi::Span<const int32_t> buf_slot,  // slot index patched with buffer j's pointer (-1 = none)
                            ffi::Span<const int64_t> ptr_off,   // byte offset added to buffer j's pointer (empty = none)
                            int32_t stream_slot,                // slot that receives the stream (-1 = none)
                            int32_t zero_ret) {                 // bit j set: memset result j to 0 before launching
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return ffi::Error::Internal("hipGetDevice failed");
    if (dev < 0 || dev >= (int) fns.size() || fns[dev] == 0)
        return ffi::Error::Internal("flydsl_launch: no compiled function for device " + std::to_string(dev));
    const int n_slots = (int) slot_off.size();
    if (n_slots > kMaxSlots) return ffi::Error::Internal("flydsl_launch: too many slots");

    uint8_t storage[1024];
    if (tmpl.size() > sizeof(storage)) return ffi::Error::Internal("flydsl_launch: template too large");
    std::memcpy(storage, tmpl.begin(), tmpl.size());

    int j = 0;
    for (size_t i = 0; i < args.size(); ++i, ++j) {
        auto buf = args.get<ffi::AnyBuffer>(i);
        if (buf.has_error()) return buf.error();
        if (j < (int) buf_slot.size() && buf_slot[j] >= 0) {
            void *p = (char *) buf->untyped_data() + (j < (int) ptr_off.size() ? ptr_off[j] : 0);
            std::memcpy(storage + slot_off[buf_slot[j]], &p, sizeof(void *));
        }
    }
    for (size_t i = 0; i < rets.size(); ++i, ++j) {
        auto buf = rets.get<ffi::AnyBuffer>(i);
        if (buf.has_error()) return buf.error();
        void *p = (*buf)->untyped_data();
        if (j < (int) buf_slot.size() && buf_slot[j] >= 0) {
            void *pp = (char *) p + (j < (int) ptr_off.size() ? ptr_off[j] : 0);
            std::memcpy(storage + slot_off[buf_slot[j]], &pp, sizeof(void *));
        }
        if (zero_ret & (1 << i)) {
            size_t bytes = ffi::ByteWidth((*buf)->element_type()) * (*buf)->element_count();
            if (hipMemsetAsync(p, 0, bytes, stream) != hipSuccess) return ffi::Error::Internal("hipMemsetAsync failed");
        }
    }
    if (stream_slot >= 0) std::memcpy(storage + slot_off[stream_slot], &stream, sizeof(void *));

    void *packed[kMaxSlots];
    for (int i = 0; i < n_slots; ++i) packed[i] = storage + slot_off[i];
    reinterpret_cast<void (*)(void *)>(fns[dev])(packed);
    return ffi::Error::Success();
}

} // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(FlyDSLLaunchHandler, FlyDSLLaunchImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>()
                                  .RemainingArgs()
                                  .RemainingRets()
                                  .Attr<ffi::Span<const int64_t>>("fns")
                                  .Attr<ffi::Span<const uint8_t>>("tmpl")
                                  .Attr<ffi::Span<const int32_t>>("slot_off")
                                  .Attr<ffi::Span<const int32_t>>("buf_slot")
                                  .Attr<ffi::Span<const int64_t>>("ptr_off")
                                  .Attr<int32_t>("stream_slot")
                                  .Attr<int32_t>("zero_ret"));
