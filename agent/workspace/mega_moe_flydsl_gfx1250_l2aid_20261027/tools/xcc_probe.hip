// Which XCC does workgroup `bid` run on, in a launch shaped like the Mega GEMMs (256 threads, ~280 KB LDS, so
// one workgroup per WGP, grid >> 256)?
//
//     xcc_probe [grid=4160] [launches=6] [busy_us=20] [slow_every=0] [slow_factor=4]
//
// Lane 0 of each workgroup reads MSG_RTN_GET_SE_AID_ID (XCC in bits [18:16]) and the 100 MHz realtime counter at
// start and end, then busy-waits busy_us (x slow_factor when slow_every > 0 and bid % slow_every == 0). Per launch:
// s = (xcc - bid) mod 8 histogram (one value = strict round-robin from XCC s), the AID-run pattern, and how
// many workgroups break the round-robin.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s: %s\n", #x, hipGetErrorString(e)); exit(1); } } while (0)

__device__ inline unsigned se_aid_id() {
    unsigned v;
    asm volatile("s_sendmsg_rtn_b32 %0, sendmsg(MSG_RTN_GET_SE_AID_ID)\n\ts_wait_kmcnt 0x0" : "=s"(v));
    return v;
}

__global__ void probe(unsigned* raw, unsigned long long* t0, unsigned long long* t1, int busy_ticks, int slow_every,
                      int slow_factor) {
    extern __shared__ char lds[];
    const int bid = blockIdx.x;
    if (threadIdx.x == 0) {
        unsigned long long s = wall_clock64();
        raw[bid] = se_aid_id();
        t0[bid] = s;
        long long ticks = busy_ticks;
        if (slow_every > 0 && bid % slow_every == 0) ticks *= slow_factor;
        while ((long long)(wall_clock64() - s) < ticks) lds[0] = 1;
        t1[bid] = wall_clock64();
    }
    __syncthreads();
}

int main(int argc, char** argv) {
    const int grid = argc > 1 ? atoi(argv[1]) : 4160;
    const int launches = argc > 2 ? atoi(argv[2]) : 6;
    const int busy_us = argc > 3 ? atoi(argv[3]) : 20;
    const int slow_every = argc > 4 ? atoi(argv[4]) : 0;
    const int slow_factor = argc > 5 ? atoi(argv[5]) : 4;
    const size_t lds = 280 * 1024;
    CHECK(hipFuncSetAttribute((const void*)probe, hipFuncAttributeMaxDynamicSharedMemorySize, (int)lds));
    int rate_khz = 0;
    CHECK(hipDeviceGetAttribute(&rate_khz, hipDeviceAttributeWallClockRate, 0));
    const int busy_ticks = (int)((long long)busy_us * rate_khz / 1000);
    unsigned* raw; unsigned long long *t0, *t1;
    CHECK(hipMalloc(&raw, grid * 4)); CHECK(hipMalloc(&t0, grid * 8)); CHECK(hipMalloc(&t1, grid * 8));
    std::vector<unsigned> h(grid); std::vector<unsigned long long> a(grid), b(grid);
    printf("grid %d, busy %d us (wall clock %d kHz), slow_every %d x%d, lds %zu\n", grid, busy_us, rate_khz,
           slow_every, slow_factor, lds);
    for (int l = 0; l < launches; ++l) {
        hipLaunchKernelGGL(probe, dim3(grid), dim3(256), lds, 0, raw, t0, t1, busy_ticks, slow_every, slow_factor);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(h.data(), raw, grid * 4, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(a.data(), t0, grid * 8, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(b.data(), t1, grid * 8, hipMemcpyDeviceToHost));
        int shist[8] = {0}, per_xcc[8] = {0};
        for (int i = 0; i < grid; ++i) {
            int x = (h[i] >> 16) & 7;
            shist[((x - i) % 8 + 8) % 8]++;
            per_xcc[x]++;
        }
        int s = 0;
        for (int k = 1; k < 8; ++k) if (shist[k] > shist[s]) s = k;
        int breaks = 0, first_break = -1;
        for (int i = 0; i < grid; ++i)
            if ((((h[i] >> 16) & 7) - i - s) % 8 != 0) { if (first_break < 0) first_break = i; ++breaks; }
        unsigned long long tmin = a[0], tmax = 0;
        for (int i = 0; i < grid; ++i) { tmin = a[i] < tmin ? a[i] : tmin; tmax = b[i] > tmax ? b[i] : tmax; }
        printf("launch %d: s hist", l);
        for (int k = 0; k < 8; ++k) printf(" %d", shist[k]);
        printf(" | s=%d, %d off round-robin (first at bid %d) | per xcc", s, breaks, first_break);
        for (int k = 0; k < 8; ++k) printf(" %d", per_xcc[k]);
        printf(" | span %.1f us\n", (tmax - tmin) * 1000.0 / rate_khz);
        if (l == 0) {
            printf("  bid: xcc (start us) for bids 0-15, 256-271, %d-%d:\n  ", grid - 16, grid - 1);
            const int lo[3] = {0, 256, grid - 16};
            for (int r = 0; r < 3; ++r) {
                for (int i = lo[r]; i < lo[r] + 16 && i < grid; ++i)
                    printf("%d:%d(%.0f) ", i, (h[i] >> 16) & 7, (a[i] - tmin) * 1000.0 / rate_khz);
                printf("\n  ");
            }
            printf("raw[0..3] = %08x %08x %08x %08x\n", h[0], h[1], h[2], h[3]);
        }
    }
    return 0;
}
