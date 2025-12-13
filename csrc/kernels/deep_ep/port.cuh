#pragma once

#define WARP_SIZE 64
#define WARP_MASK 0xffffffffffffffffu
#define MAX_NTHREADS 1024
#define MAX_GROUPS (MAX_NTHREADS / WARP_SIZE) // 16 warps in the block
#define MAX_GROUPS_MASK 0xf
