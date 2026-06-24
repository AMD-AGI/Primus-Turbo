.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _grouped_fp8_persistent_gemm_kernel
.p2align 8
.type _grouped_fp8_persistent_gemm_kernel,@function
_grouped_fp8_persistent_gemm_kernel:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .L0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
.L0:
	s_mov_b64 s[24:25], s[2:3]
	s_load_dword s2, s[0:1], 0x38
	s_mov_b64 s[36:37], s[14:15]
	s_mov_b64 s[20:21], s[6:7]
	s_cmpk_gt_i32 s16, 0x100
	v_readfirstlane_b32 s18, v0
	s_cbranch_scc1 .L1
	s_ashr_i32 s3, s16, 31
	s_lshr_b32 s6, s3, 29
	s_add_i32 s6, s16, s6
	s_ashr_i32 s14, s6, 31
	s_ashr_i32 s7, s6, 3
	s_lshr_b32 s3, s3, 24
	s_lshr_b32 s14, s14, 27
	s_and_b32 s6, s6, 0x7fffff8
	s_add_i32 s3, s16, s3
	s_add_i32 s14, s7, s14
	s_sub_i32 s6, s16, s6
	s_andn2_b32 s14, s14, 31
	s_and_b32 s3, s3, 0xffffff00
	s_lshl_b32 s6, s6, 5
	s_sub_i32 s7, s7, s14
	s_add_i32 s3, s3, s6
	s_add_i32 s16, s3, s7
.L1:
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[6:7], s[2:3], 2
	s_add_u32 s6, s36, s6
	s_addc_u32 s7, s37, s7
	s_load_dword s17, s[6:7], 0x0
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_i32 s16, s17
	s_cbranch_scc1 .L8
	v_and_b32_e32 v129, 63, v0
	v_lshlrev_b32_e32 v1, 2, v129
	v_bfrev_b32_e32 v131, 1
	v_cmp_ge_i32_e32 vcc, s2, v129
	s_and_b32 s37, s37, 0xffff
	s_mov_b32 s39, 0x27000
	s_mov_b32 s38, 0x7ffffffe
	v_cndmask_b32_e32 v1, v131, v1, vcc
	buffer_load_dword v1, v1, s[36:39], 0 offen
	s_load_dwordx4 s[28:31], s[0:1], 0x3c
	s_load_dwordx2 s[14:15], s[0:1], 0x4c
	s_load_dword s22, s[8:9], 0x0
	s_load_dword s23, s[10:11], 0x0
	s_bfe_u32 s0, s18, 0x30006
	s_movk_i32 s2, 0xff
	s_waitcnt lgkmcnt(0)
	s_add_i32 s8, s28, 0xff
	s_add_i32 s9, s29, 0x7f
	s_ashr_i32 s10, s8, 31
	s_ashr_i32 s11, s9, 31
	v_and_b32_e32 v3, 15, v0
	s_lshr_b32 s1, s18, 4
	v_lshlrev_b32_e32 v4, 4, v0
	s_movk_i32 s19, 0x70
	s_movk_i32 s3, 0x100
	v_and_b32_e32 v5, 48, v0
	v_lshlrev_b32_e32 v6, 3, v0
	v_lshl_or_b32 v7, s0, 6, v129
	s_lshl_b32 s33, s0, 10
	s_lshr_b32 s10, s10, 24
	s_lshr_b32 s11, s11, 25
	v_and_or_b32 v141, s1, 16, v3
	v_and_b32_e32 v142, 0x70, v4
	v_cmp_gt_u32_e64 s[0:1], s3, v0
	v_cmp_lt_u32_e64 s[2:3], s2, v0
	v_lshlrev_b32_e32 v0, 7, v3
	v_bitop3_b32 v3, v6, v5, s19 bitop3:0x6c
	v_lshrrev_b32_e32 v4, 2, v7
	v_lshlrev_b32_e32 v6, 4, v7
	s_add_i32 s34, s33, 0
	s_add_i32 s8, s8, s10
	s_add_i32 s10, s9, s11
	s_and_b32 s25, s25, 0xffff
	s_and_b32 s5, s5, 0xffff
	s_mov_b32 s26, s38
	s_mov_b32 s27, s39
	v_and_b32_e32 v147, 60, v4
	v_bitop3_b32 v4, v6, v7, s19 bitop3:0x78
	s_add_i32 s35, s34, 0x2000
	s_add_i32 s36, s34, 0x4000
	s_add_i32 s37, s34, 0x6000
	s_add_i32 s38, s34, 0x10000
	s_add_i32 s39, s34, 0x12000
	s_add_i32 s40, s34, 0x14000
	s_add_i32 s41, s34, 0x16000
	s_ashr_i32 s29, s8, 8
	s_ashr_i32 s19, s10, 7
	s_cmpk_gt_i32 s9, 0xff
	s_cselect_b64 s[8:9], -1, 0
	s_lshl_b32 s10, s18, 5
	s_and_b32 s10, s10, 0x1800
	v_lshlrev_b32_e32 v8, 3, v7
	s_mul_i32 s29, s29, 6
	s_cmp_gt_i32 s28, -1
	v_lshrrev_b32_e32 v143, 3, v7
	v_and_b32_e32 v7, 0x870, v8
	v_or_b32_e32 v3, s10, v3
	s_cselect_b64 s[10:11], -1, 0
	s_abs_i32 s42, s29
	v_bitop3_b32 v148, v7, v5, v0 bitop3:0x36
	v_or_b32_e32 v151, v3, v0
	v_bitop3_b32 v152, v3, 64, v0 bitop3:0x36
	v_cvt_f32_u32_e32 v0, s42
	v_bfrev_b32_e32 v2, -2
	s_abs_i32 s28, s28
	v_cmp_ne_u32_e64 s[6:7], 0, v129
	v_rcp_iflag_f32_e32 v0, v0
	s_max_i32 s45, s19, 2
	s_and_b64 s[18:19], s[6:7], vcc
	s_sub_i32 s6, 0, s42
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	v_sub_u32_e32 v4, v4, v6
	v_ashrrev_i32_e32 v4, 4, v4
	v_add_u32_e32 v128, v4, v129
	v_readfirstlane_b32 s7, v0
	s_mul_i32 s6, s6, s7
	s_mul_hi_u32 s6, s7, s6
	s_add_i32 s44, s7, s6
	s_sub_i32 s6, 0, s28
	s_lshl_b32 s45, s45, 7
	s_waitcnt vmcnt(0)
	v_cndmask_b32_e32 v153, v2, v1, vcc
	v_cvt_f32_u32_e32 v1, s28
	v_mov_b32_e32 v140, 0
	v_or_b32_e32 v144, 64, v143
	v_or_b32_e32 v145, 0x80, v143
	v_rcp_iflag_f32_e32 v1, v1
	v_or_b32_e32 v146, 0xc0, v143
	v_xor_b32_e32 v149, 64, v148
	v_lshlrev_b32_e32 v150, 2, v128
	v_mul_f32_e32 v0, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v0, v0
	s_and_b32 s21, s21, 0xffff
	s_ashr_i32 s43, s29, 31
	v_or_b32_e32 v130, 0x80, v142
	v_mul_lo_u32 v1, s6, v0
	v_mul_hi_u32 v1, v0, v1
	v_add_u32_e32 v154, v0, v1
	v_mov_b32_e32 v0, s23
	v_mul_f32_e32 v155, s22, v0
	s_addk_i32 s45, 0xff80
	s_branch .L3
.L2:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v132, s48, v141
	s_ashr_i32 s6, s48, 31
	v_add_u32_e32 v132, s6, v132
	v_xor_b32_e32 v133, s6, v132
	v_mul_hi_u32 v134, v133, v156
	v_mul_lo_u32 v134, v134, s46
	v_sub_u32_e32 v133, v133, v134
	v_subrev_u32_e32 v134, s46, v133
	v_cmp_le_u32_e32 vcc, s46, v133
	v_add_u32_e32 v177, s50, v151
	v_add_u32_e32 v178, s50, v152
	v_cndmask_b32_e32 v133, v133, v134, vcc
	v_subrev_u32_e32 v134, s46, v133
	v_cmp_le_u32_e32 vcc, s46, v133
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_nop 0
	v_cndmask_b32_e32 v133, v133, v134, vcc
	v_xor_b32_e32 v133, s6, v133
	v_subrev_u32_e32 v135, s6, v133
	v_add_u32_e32 v133, 32, v132
	v_xor_b32_e32 v133, s6, v133
	v_mul_hi_u32 v134, v133, v156
	v_mul_lo_u32 v134, v134, s46
	v_sub_u32_e32 v133, v133, v134
	v_subrev_u32_e32 v134, s46, v133
	v_cmp_le_u32_e32 vcc, s46, v133
	s_barrier
	ds_read_b128 v[202:205], v178 offset:16384
	v_cndmask_b32_e32 v133, v133, v134, vcc
	v_subrev_u32_e32 v134, s46, v133
	v_cmp_le_u32_e32 vcc, s46, v133
	v_add_u32_e32 v180, s49, v149
	v_add_u32_e32 v179, s49, v148
	v_cndmask_b32_e32 v133, v133, v134, vcc
	v_xor_b32_e32 v133, s6, v133
	v_subrev_u32_e32 v136, s6, v133
	v_add_u32_e32 v133, 64, v132
	v_xor_b32_e32 v133, s6, v133
	v_mul_hi_u32 v134, v133, v156
	v_mul_lo_u32 v134, v134, s46
	v_sub_u32_e32 v133, v133, v134
	v_subrev_u32_e32 v134, s46, v133
	v_cmp_le_u32_e32 vcc, s46, v133
	ds_read_b128 v[186:189], v180
	ds_read_b128 v[182:185], v179
	ds_read_b128 v[190:193], v179 offset:4096
	ds_read_b128 v[194:197], v180 offset:4096
	ds_read_b128 v[168:171], v177 offset:8192
	ds_read_b128 v[172:175], v178 offset:8192
	v_cndmask_b32_e32 v133, v133, v134, vcc
	v_subrev_u32_e32 v134, s46, v133
	v_cmp_le_u32_e32 vcc, s46, v133
	ds_read_b128 v[198:201], v177 offset:16384
	ds_read_b128 v[206:209], v177 offset:24576
	ds_read_b128 v[210:213], v178 offset:24576
	v_cndmask_b32_e32 v133, v133, v134, vcc
	v_xor_b32_e32 v133, s6, v133
	v_subrev_u32_e32 v137, s6, v133
	v_add_u32_e32 v133, 0x60, v132
	v_xor_b32_e32 v133, s6, v133
	v_mul_hi_u32 v134, v133, v156
	v_mul_lo_u32 v134, v134, s46
	v_sub_u32_e32 v133, v133, v134
	v_subrev_u32_e32 v134, s46, v133
	v_cmp_le_u32_e32 vcc, s46, v133
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x128_f8f6f4 v[120:123], v[168:175], v[182:189], v[120:123]
	ds_read_b128 v[164:167], v178
	v_cndmask_b32_e32 v133, v133, v134, vcc
	v_subrev_u32_e32 v134, s46, v133
	v_cmp_le_u32_e32 vcc, s46, v133
	s_nop 1
	v_cndmask_b32_e32 v133, v133, v134, vcc
	v_xor_b32_e32 v133, s6, v133
	v_subrev_u32_e32 v138, s6, v133
	v_add_u32_e32 v133, 0x80, v132
	v_xor_b32_e32 v133, s6, v133
	v_mul_hi_u32 v134, v133, v156
	v_mul_lo_u32 v134, v134, s46
	v_sub_u32_e32 v133, v133, v134
	v_subrev_u32_e32 v134, s46, v133
	v_cmp_le_u32_e32 vcc, s46, v133
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x128_f8f6f4 v[116:119], v[198:205], v[182:189], v[116:119]
	v_mul_f32_e32 v120, v155, v120
	v_cndmask_b32_e32 v133, v133, v134, vcc
	v_subrev_u32_e32 v134, s46, v133
	v_cmp_le_u32_e32 vcc, s46, v133
	v_mul_f32_e32 v121, v155, v121
	v_mul_f32_e32 v122, v155, v122
	v_cndmask_b32_e32 v133, v133, v134, vcc
	v_xor_b32_e32 v133, s6, v133
	v_subrev_u32_e32 v139, s6, v133
	v_add_u32_e32 v133, 0xa0, v132
	v_xor_b32_e32 v133, s6, v133
	v_mul_hi_u32 v134, v133, v156
	v_mul_lo_u32 v134, v134, s46
	v_sub_u32_e32 v133, v133, v134
	v_subrev_u32_e32 v134, s46, v133
	v_cmp_le_u32_e32 vcc, s46, v133
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x128_f8f6f4 v[112:115], v[206:213], v[182:189], v[112:115]
	v_mul_f32_e32 v123, v155, v123
	v_cndmask_b32_e32 v133, v133, v134, vcc
	v_subrev_u32_e32 v134, s46, v133
	v_cmp_le_u32_e32 vcc, s46, v133
	v_mul_f32_e32 v116, v155, v116
	v_mul_f32_e32 v117, v155, v117
	v_cndmask_b32_e32 v133, v133, v134, vcc
	v_xor_b32_e32 v133, s6, v133
	v_subrev_u32_e32 v157, s6, v133
	v_add_u32_e32 v133, 0xc0, v132
	v_xor_b32_e32 v133, s6, v133
	v_mul_hi_u32 v134, v133, v156
	v_mul_lo_u32 v134, v134, s46
	v_sub_u32_e32 v133, v133, v134
	v_subrev_u32_e32 v134, s46, v133
	v_cmp_le_u32_e32 vcc, s46, v133
	v_add_u32_e32 v132, 0xe0, v132
	v_xor_b32_e32 v132, s6, v132
	v_cndmask_b32_e32 v133, v133, v134, vcc
	v_subrev_u32_e32 v134, s46, v133
	v_cmp_le_u32_e32 vcc, s46, v133
	v_mfma_f32_16x16x128_f8f6f4 v[104:107], v[168:175], v[190:197], v[104:107]
	v_mul_f32_e32 v118, v155, v118
	v_cndmask_b32_e32 v133, v133, v134, vcc
	v_xor_b32_e32 v133, s6, v133
	v_subrev_u32_e32 v158, s6, v133
	v_mul_hi_u32 v133, v132, v156
	v_mul_lo_u32 v133, v133, s46
	v_sub_u32_e32 v132, v132, v133
	v_subrev_u32_e32 v133, s46, v132
	v_cmp_le_u32_e32 vcc, s46, v132
	v_mfma_f32_16x16x128_f8f6f4 v[100:103], v[198:205], v[190:197], v[100:103]
	v_mul_f32_e32 v119, v155, v119
	v_cndmask_b32_e32 v132, v132, v133, vcc
	v_subrev_u32_e32 v133, s46, v132
	v_cmp_le_u32_e32 vcc, s46, v132
	v_mul_f32_e32 v112, v155, v112
	v_mul_f32_e32 v113, v155, v113
	v_cndmask_b32_e32 v132, v132, v133, vcc
	v_xor_b32_e32 v132, s6, v132
	v_subrev_u32_e32 v156, s6, v132
	v_or_b32_e32 v132, s47, v147
	s_ashr_i32 s6, s47, 31
	v_add_u32_e32 v159, s6, v132
	v_xor_b32_e32 v132, s6, v159
	v_mul_hi_u32 v133, v132, v154
	v_mul_lo_u32 v133, v133, s28
	v_sub_u32_e32 v132, v132, v133
	v_subrev_u32_e32 v133, s28, v132
	v_cmp_le_u32_e32 vcc, s28, v132
	v_mfma_f32_16x16x128_f8f6f4 v[96:99], v[206:213], v[190:197], v[96:99]
	s_cmp_gt_i32 s23, -1
	v_cndmask_b32_e32 v132, v132, v133, vcc
	v_subrev_u32_e32 v133, s28, v132
	v_cmp_le_u32_e32 vcc, s28, v132
	s_mov_b32 s23, s27
	v_mul_f32_e32 v114, v155, v114
	v_cndmask_b32_e32 v132, v132, v133, vcc
	v_add_u32_e32 v133, 64, v159
	v_xor_b32_e32 v133, s6, v133
	v_mul_hi_u32 v134, v133, v154
	v_mul_lo_u32 v134, v134, s28
	v_sub_u32_e32 v133, v133, v134
	v_subrev_u32_e32 v134, s28, v133
	v_cmp_le_u32_e32 vcc, s28, v133
	v_xor_b32_e32 v132, s6, v132
	v_subrev_u32_e32 v132, s6, v132
	v_cndmask_b32_e32 v133, v133, v134, vcc
	v_subrev_u32_e32 v134, s28, v133
	v_cmp_le_u32_e32 vcc, s28, v133
	v_mul_f32_e32 v115, v155, v115
	v_mul_f32_e32 v104, v155, v104
	v_cndmask_b32_e32 v133, v133, v134, vcc
	v_add_u32_e32 v134, 0x80, v159
	v_xor_b32_e32 v134, s6, v134
	v_mul_hi_u32 v160, v134, v154
	v_mul_lo_u32 v160, v160, s28
	v_sub_u32_e32 v134, v134, v160
	v_subrev_u32_e32 v160, s28, v134
	v_cmp_le_u32_e32 vcc, s28, v134
	v_add_u32_e32 v159, 0xc0, v159
	v_xor_b32_e32 v159, s6, v159
	v_cndmask_b32_e32 v134, v134, v160, vcc
	v_subrev_u32_e32 v160, s28, v134
	v_cmp_le_u32_e32 vcc, s28, v134
	v_mul_hi_u32 v176, v159, v154
	v_mul_lo_u32 v176, v176, s28
	v_cndmask_b32_e32 v134, v134, v160, vcc
	ds_read_b128 v[160:163], v177
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x128_f8f6f4 v[124:127], v[160:167], v[182:189], v[124:127]
	v_sub_u32_e32 v159, v159, v176
	v_subrev_u32_e32 v176, s28, v159
	v_cmp_le_u32_e32 vcc, s28, v159
	v_xor_b32_e32 v133, s6, v133
	v_xor_b32_e32 v134, s6, v134
	v_cndmask_b32_e32 v159, v159, v176, vcc
	v_subrev_u32_e32 v176, s28, v159
	v_mfma_f32_16x16x128_f8f6f4 v[108:111], v[160:167], v[190:197], v[108:111]
	ds_read_b128 v[186:189], v180 offset:8192
	ds_read_b128 v[182:185], v179 offset:8192
	ds_read_b128 v[190:193], v179 offset:12288
	ds_read_b128 v[194:197], v180 offset:12288
	v_cmp_le_u32_e32 vcc, s28, v159
	v_subrev_u32_e32 v133, s6, v133
	v_subrev_u32_e32 v134, s6, v134
	v_cndmask_b32_e32 v159, v159, v176, vcc
	v_xor_b32_e32 v159, s6, v159
	v_subrev_u32_e32 v159, s6, v159
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x128_f8f6f4 v[92:95], v[160:167], v[182:189], v[92:95]
	s_cselect_b64 s[6:7], -1, 0
	v_mul_f32_e32 v124, v155, v124
	v_mul_f32_e32 v125, v155, v125
	v_mul_f32_e32 v126, v155, v126
	v_mul_f32_e32 v127, v155, v127
	s_and_b64 vcc, s[10:11], s[6:7]
	v_mul_f32_e32 v108, v155, v108
	v_mfma_f32_16x16x128_f8f6f4 v[88:91], v[168:175], v[182:189], v[88:91]
	v_mul_f32_e32 v109, v155, v109
	v_mul_f32_e32 v110, v155, v110
	v_mul_f32_e32 v111, v155, v111
	v_mul_f32_e32 v105, v155, v105
	v_mul_f32_e32 v106, v155, v106
	v_mul_f32_e32 v107, v155, v107
	v_mul_f32_e32 v100, v155, v100
	v_mfma_f32_16x16x128_f8f6f4 v[84:87], v[198:205], v[182:189], v[84:87]
	v_mul_f32_e32 v101, v155, v101
	v_mul_f32_e32 v102, v155, v102
	v_mul_f32_e32 v103, v155, v103
	v_mul_f32_e32 v96, v155, v96
	v_mul_f32_e32 v97, v155, v97
	v_mul_f32_e32 v98, v155, v98
	v_mul_f32_e32 v99, v155, v99
	v_mfma_f32_16x16x128_f8f6f4 v[80:83], v[206:213], v[182:189], v[80:83]
	v_mul_f32_e32 v92, v155, v92
	v_mul_f32_e32 v93, v155, v93
	v_mul_f32_e32 v94, v155, v94
	v_mul_f32_e32 v95, v155, v95
	v_mul_f32_e32 v88, v155, v88
	v_mul_f32_e32 v89, v155, v89
	v_mul_f32_e32 v90, v155, v90
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x128_f8f6f4 v[76:79], v[160:167], v[190:197], v[76:79]
	v_mul_f32_e32 v91, v155, v91
	v_mul_f32_e32 v84, v155, v84
	v_mul_f32_e32 v85, v155, v85
	v_mul_f32_e32 v86, v155, v86
	v_mul_f32_e32 v87, v155, v87
	v_mul_f32_e32 v80, v155, v80
	v_mul_f32_e32 v81, v155, v81
	v_mfma_f32_16x16x128_f8f6f4 v[72:75], v[168:175], v[190:197], v[72:75]
	v_mul_f32_e32 v82, v155, v82
	v_mul_f32_e32 v83, v155, v83
	s_nop 1
	v_mul_f32_e32 v76, v155, v76
	v_mul_f32_e32 v77, v155, v77
	v_mul_f32_e32 v78, v155, v78
	v_mul_f32_e32 v79, v155, v79
	s_addk_i32 s16, 0x100
	v_mfma_f32_16x16x128_f8f6f4 v[68:71], v[198:205], v[190:197], v[68:71]
	s_nop 1
	v_mul_f32_e32 v72, v155, v72
	v_mul_f32_e32 v73, v155, v73
	v_mul_f32_e32 v74, v155, v74
	v_mul_f32_e32 v75, v155, v75
	s_cmp_lt_i32 s16, s17
	s_nop 4
	v_mul_f32_e32 v68, v155, v68
	v_mfma_f32_16x16x128_f8f6f4 v[64:67], v[206:213], v[190:197], v[64:67]
	ds_read_b128 v[186:189], v180 offset:16384
	ds_read_b128 v[182:185], v179 offset:16384
	ds_read_b128 v[190:193], v179 offset:20480
	ds_read_b128 v[194:197], v180 offset:20480
	v_mul_f32_e32 v69, v155, v69
	v_mul_f32_e32 v70, v155, v70
	v_mul_f32_e32 v71, v155, v71
	s_nop 4
	v_mul_f32_e32 v64, v155, v64
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x128_f8f6f4 v[60:63], v[160:167], v[182:189], v[60:63]
	v_mul_f32_e32 v65, v155, v65
	v_mul_f32_e32 v66, v155, v66
	v_mul_f32_e32 v67, v155, v67
	v_mfma_f32_16x16x128_f8f6f4 v[56:59], v[168:175], v[182:189], v[56:59]
	s_nop 7
	v_mul_f32_e32 v60, v155, v60
	v_mul_f32_e32 v61, v155, v61
	v_mul_f32_e32 v62, v155, v62
	v_mul_f32_e32 v63, v155, v63
	v_mfma_f32_16x16x128_f8f6f4 v[52:55], v[198:205], v[182:189], v[52:55]
	v_mul_f32_e32 v56, v155, v56
	v_mul_f32_e32 v57, v155, v57
	v_mul_f32_e32 v58, v155, v58
	v_mul_f32_e32 v59, v155, v59
	v_mfma_f32_16x16x128_f8f6f4 v[48:51], v[206:213], v[182:189], v[48:51]
	s_nop 6
	v_mul_f32_e32 v52, v155, v52
	v_mul_f32_e32 v53, v155, v53
	v_mul_f32_e32 v54, v155, v54
	v_mul_f32_e32 v55, v155, v55
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x128_f8f6f4 v[44:47], v[160:167], v[190:197], v[44:47]
	v_mul_f32_e32 v48, v155, v48
	v_mul_f32_e32 v49, v155, v49
	v_mul_f32_e32 v50, v155, v50
	v_mul_f32_e32 v51, v155, v51
	v_mfma_f32_16x16x128_f8f6f4 v[40:43], v[168:175], v[190:197], v[40:43]
	s_nop 6
	v_mul_f32_e32 v44, v155, v44
	v_mul_f32_e32 v45, v155, v45
	v_mul_f32_e32 v46, v155, v46
	v_mul_f32_e32 v47, v155, v47
	v_mfma_f32_16x16x128_f8f6f4 v[36:39], v[198:205], v[190:197], v[36:39]
	v_mul_f32_e32 v43, v155, v43
	v_mfma_f32_16x16x128_f8f6f4 v[32:35], v[206:213], v[190:197], v[32:35]
	ds_read_b128 v[186:189], v180 offset:24576
	ds_read_b128 v[182:185], v179 offset:24576
	ds_read_b128 v[190:193], v179 offset:28672
	ds_read_b128 v[194:197], v180 offset:28672
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x128_f8f6f4 v[28:31], v[160:167], v[182:189], v[28:31]
	v_mfma_f32_16x16x128_f8f6f4 v[24:27], v[168:175], v[182:189], v[24:27]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[206:213], v[190:197], v[0:3]
	s_nop 9
	v_mul_f32_e32 v177, v155, v26
	v_cvt_pk_bf16_f32 v26, v72, v73
	v_mul_f32_e32 v176, v155, v25
	v_cvt_pk_bf16_f32 v25, v78, v79
	v_mul_f32_e32 v178, v155, v27
	v_cvt_pk_bf16_f32 v27, v74, v75
	v_mfma_f32_16x16x128_f8f6f4 v[4:7], v[198:205], v[190:197], v[4:7]
	v_mfma_f32_16x16x128_f8f6f4 v[8:11], v[168:175], v[190:197], v[8:11]
	v_mul_f32_e32 v173, v155, v30
	v_cvt_pk_bf16_f32 v30, v64, v65
	v_add_u32_e32 v64, s22, v135
	v_mul_lo_u32 v64, v64, s15
	v_add_lshl_u32 v72, v64, v132, 1
	v_mul_f32_e32 v171, v155, v28
	v_mul_f32_e32 v172, v155, v29
	v_mfma_f32_16x16x128_f8f6f4 v[20:23], v[198:205], v[182:189], v[20:23]
	v_mul_f32_e32 v174, v155, v31
	v_mul_f32_e32 v199, v155, v0
	v_mul_f32_e32 v200, v155, v1
	v_cvt_pk_bf16_f32 v0, v124, v125
	v_cvt_pk_bf16_f32 v1, v126, v127
	v_cvt_pk_bf16_f32 v28, v68, v69
	v_cvt_pk_bf16_f32 v29, v70, v71
	v_cvt_pk_bf16_f32 v31, v66, v67
	v_add_u32_e32 v65, s22, v136
	v_add_u32_e32 v66, s22, v137
	v_add_u32_e32 v67, s22, v138
	v_add_u32_e32 v68, s22, v139
	v_add_u32_e32 v69, s22, v157
	v_add_u32_e32 v70, s22, v158
	v_add_u32_e32 v71, s22, v156
	v_cndmask_b32_e32 v72, v131, v72, vcc
	s_mov_b32 s22, s26
	buffer_store_dwordx2 v[0:1], v72, s[20:23], 0 offen
	v_add_lshl_u32 v0, v64, v133, 1
	v_mul_f32_e32 v201, v155, v2
	v_mul_f32_e32 v202, v155, v3
	v_cvt_pk_bf16_f32 v2, v120, v121
	v_cvt_pk_bf16_f32 v3, v122, v123
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[2:3], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v64, v134, 1
	v_mfma_f32_16x16x128_f8f6f4 v[12:15], v[160:167], v[190:197], v[12:15]
	v_mul_f32_e32 v195, v155, v4
	v_mul_f32_e32 v196, v155, v5
	v_cvt_pk_bf16_f32 v4, v116, v117
	v_cvt_pk_bf16_f32 v5, v118, v119
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[4:5], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v64, v159, 1
	v_mul_f32_e32 v197, v155, v6
	v_mul_f32_e32 v198, v155, v7
	v_cvt_pk_bf16_f32 v6, v112, v113
	v_cvt_pk_bf16_f32 v7, v114, v115
	v_mul_lo_u32 v65, v65, s15
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[6:7], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v65, v132, 1
	v_mfma_f32_16x16x128_f8f6f4 v[16:19], v[206:213], v[182:189], v[16:19]
	v_mul_f32_e32 v191, v155, v8
	v_mul_f32_e32 v192, v155, v9
	v_cvt_pk_bf16_f32 v8, v108, v109
	v_cvt_pk_bf16_f32 v9, v110, v111
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[8:9], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v65, v133, 1
	v_mul_f32_e32 v193, v155, v10
	v_mul_f32_e32 v194, v155, v11
	v_cvt_pk_bf16_f32 v10, v104, v105
	v_cvt_pk_bf16_f32 v11, v106, v107
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[10:11], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v65, v134, 1
	v_mul_f32_e32 v187, v155, v12
	v_mul_f32_e32 v188, v155, v13
	v_cvt_pk_bf16_f32 v12, v100, v101
	v_cvt_pk_bf16_f32 v13, v102, v103
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[12:13], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v65, v159, 1
	v_mul_f32_e32 v189, v155, v14
	v_mul_f32_e32 v190, v155, v15
	v_cvt_pk_bf16_f32 v14, v96, v97
	v_cvt_pk_bf16_f32 v15, v98, v99
	v_mul_lo_u32 v66, v66, s15
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[14:15], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v66, v132, 1
	v_mul_f32_e32 v183, v155, v16
	v_mul_f32_e32 v184, v155, v17
	v_cvt_pk_bf16_f32 v16, v92, v93
	v_cvt_pk_bf16_f32 v17, v94, v95
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[16:17], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v66, v133, 1
	v_mul_f32_e32 v185, v155, v18
	v_mul_f32_e32 v186, v155, v19
	v_cvt_pk_bf16_f32 v18, v88, v89
	v_cvt_pk_bf16_f32 v19, v90, v91
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[18:19], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v66, v134, 1
	v_mul_f32_e32 v179, v155, v20
	v_mul_f32_e32 v180, v155, v21
	v_cvt_pk_bf16_f32 v20, v84, v85
	v_cvt_pk_bf16_f32 v21, v86, v87
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[20:21], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v66, v159, 1
	v_mul_f32_e32 v181, v155, v22
	v_mul_f32_e32 v182, v155, v23
	v_cvt_pk_bf16_f32 v22, v80, v81
	v_cvt_pk_bf16_f32 v23, v82, v83
	v_mul_lo_u32 v67, v67, s15
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[22:23], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v67, v132, 1
	v_mul_f32_e32 v175, v155, v24
	v_cvt_pk_bf16_f32 v24, v76, v77
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[24:25], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v67, v133, 1
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[26:27], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v67, v134, 1
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[28:29], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v67, v159, 1
	v_mul_lo_u32 v68, v68, s15
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[30:31], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v68, v132, 1
	v_mul_f32_e32 v167, v155, v32
	v_mul_f32_e32 v168, v155, v33
	v_cvt_pk_bf16_f32 v32, v60, v61
	v_cvt_pk_bf16_f32 v33, v62, v63
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[32:33], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v68, v133, 1
	v_mul_f32_e32 v169, v155, v34
	v_mul_f32_e32 v170, v155, v35
	v_cvt_pk_bf16_f32 v34, v56, v57
	v_cvt_pk_bf16_f32 v35, v58, v59
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[34:35], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v68, v134, 1
	v_mul_f32_e32 v163, v155, v36
	v_mul_f32_e32 v164, v155, v37
	v_cvt_pk_bf16_f32 v36, v52, v53
	v_cvt_pk_bf16_f32 v37, v54, v55
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[36:37], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v68, v159, 1
	v_mul_f32_e32 v165, v155, v38
	v_mul_f32_e32 v166, v155, v39
	v_cvt_pk_bf16_f32 v38, v48, v49
	v_cvt_pk_bf16_f32 v39, v50, v51
	v_mul_lo_u32 v69, v69, s15
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[38:39], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v69, v132, 1
	v_mul_f32_e32 v160, v155, v40
	v_mul_f32_e32 v161, v155, v41
	v_cvt_pk_bf16_f32 v40, v44, v45
	v_cvt_pk_bf16_f32 v41, v46, v47
	v_cndmask_b32_e32 v0, v131, v0, vcc
	v_mul_f32_e32 v162, v155, v42
	buffer_store_dwordx2 v[40:41], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v69, v133, 1
	v_cvt_pk_bf16_f32 v42, v160, v161
	v_cvt_pk_bf16_f32 v43, v162, v43
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[42:43], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v69, v134, 1
	v_cvt_pk_bf16_f32 v44, v163, v164
	v_cvt_pk_bf16_f32 v45, v165, v166
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[44:45], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v69, v159, 1
	v_cvt_pk_bf16_f32 v46, v167, v168
	v_cvt_pk_bf16_f32 v47, v169, v170
	v_mul_lo_u32 v70, v70, s15
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[46:47], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v70, v132, 1
	v_cvt_pk_bf16_f32 v48, v171, v172
	v_cvt_pk_bf16_f32 v49, v173, v174
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[48:49], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v70, v133, 1
	v_cvt_pk_bf16_f32 v50, v175, v176
	v_cvt_pk_bf16_f32 v51, v177, v178
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[50:51], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v70, v134, 1
	v_cvt_pk_bf16_f32 v52, v179, v180
	v_cvt_pk_bf16_f32 v53, v181, v182
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[52:53], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v70, v159, 1
	v_cvt_pk_bf16_f32 v54, v183, v184
	v_cvt_pk_bf16_f32 v55, v185, v186
	v_mul_lo_u32 v71, v71, s15
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[54:55], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v71, v132, 1
	v_cvt_pk_bf16_f32 v56, v187, v188
	v_cvt_pk_bf16_f32 v57, v189, v190
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[56:57], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v71, v133, 1
	v_cvt_pk_bf16_f32 v58, v191, v192
	v_cvt_pk_bf16_f32 v59, v193, v194
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[58:59], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v71, v134, 1
	v_cvt_pk_bf16_f32 v60, v195, v196
	v_cvt_pk_bf16_f32 v61, v197, v198
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[60:61], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v71, v159, 1
	v_cvt_pk_bf16_f32 v62, v199, v200
	v_cvt_pk_bf16_f32 v63, v201, v202
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[62:63], v0, s[20:23], 0 offen
	s_cbranch_scc0 .L8
.L3:
	v_cmp_ge_i32_e32 vcc, s16, v153
	s_and_b64 s[6:7], vcc, s[18:19]
	v_cndmask_b32_e64 v0, 0, 1, s[6:7]
	s_mov_b32 m0, s34
	s_nop 0
	v_add_u32_dpp v0, v0, v0 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_u32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_u32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_u32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_e32 v1, v0
	s_nop 1
	v_mov_b32_dpp v1, v1 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:1
	v_add_u32_e32 v0, v0, v1
	s_nop 1
	v_add_u32_dpp v0, v0, v0 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_readlane_b32 s6, v0, 63
	s_nop 1
	v_cmp_eq_u32_e32 vcc, s6, v129
	s_nop 1
	v_cndmask_b32_e32 v0, 0, v153, vcc
	s_nop 1
	v_add_u32_dpp v0, v0, v0 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_u32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_u32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_u32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_e32 v1, v0
	s_nop 1
	v_mov_b32_dpp v1, v1 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:1
	v_add_u32_e32 v0, v0, v1
	s_nop 1
	v_add_u32_dpp v0, v0, v0 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_readlane_b32 s7, v0, 63
	s_sub_i32 s46, s16, s7
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[22:23], s[6:7], 3
	s_add_u32 s22, s12, s22
	s_addc_u32 s23, s13, s23
	global_load_dwordx4 v[0:3], v140, s[22:23]
	s_abs_i32 s22, s46
	s_mul_hi_u32 s47, s22, s44
	s_mul_i32 s23, s47, s42
	s_sub_i32 s49, s22, s23
	s_ashr_i32 s7, s46, 31
	s_xor_b32 s7, s7, s43
	s_add_i32 s48, s47, 1
	s_sub_i32 s50, s49, s42
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v0
	v_readfirstlane_b32 s23, v2
	s_sub_i32 s23, s23, s22
	s_add_i32 s51, s23, 0xff
	s_ashr_i32 s52, s51, 31
	s_lshr_b32 s52, s52, 24
	s_add_i32 s51, s51, s52
	s_ashr_i32 s51, s51, 8
	s_cmp_ge_u32 s49, s42
	s_cselect_b32 s47, s48, s47
	s_cselect_b32 s48, s50, s49
	s_add_i32 s49, s47, 1
	s_cmp_ge_u32 s48, s42
	s_cselect_b32 s47, s49, s47
	s_xor_b32 s47, s47, s7
	s_sub_i32 s7, s47, s7
	s_mul_i32 s48, s7, 6
	s_sub_i32 s47, s51, s48
	s_min_i32 s47, s47, 6
	s_abs_i32 s49, s47
	v_cvt_f32_u32_e32 v0, s49
	s_sub_i32 s51, 0, s49
	s_mul_i32 s7, s7, s29
	s_sub_i32 s7, s46, s7
	v_rcp_iflag_f32_e32 v0, v0
	s_abs_i32 s46, s7
	s_xor_b32 s50, s7, s47
	s_ashr_i32 s50, s50, 31
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_nop 0
	v_readfirstlane_b32 s52, v0
	s_mul_i32 s51, s51, s52
	s_mul_hi_u32 s51, s52, s51
	s_add_i32 s52, s52, s51
	s_mul_hi_u32 s51, s46, s52
	s_mul_i32 s52, s51, s49
	s_sub_i32 s46, s46, s52
	s_add_i32 s53, s51, 1
	s_sub_i32 s52, s46, s49
	s_cmp_ge_u32 s46, s49
	s_cselect_b32 s51, s53, s51
	s_cselect_b32 s46, s52, s46
	s_add_i32 s52, s51, 1
	s_cmp_ge_u32 s46, s49
	s_cselect_b32 s49, s52, s51
	s_abs_i32 s46, s23
	v_cvt_f32_u32_e32 v0, s46
	s_xor_b32 s49, s49, s50
	s_sub_i32 s49, s49, s50
	s_mul_i32 s50, s49, s47
	s_lshl_b32 s47, s49, 8
	s_bfe_i32 s52, s49, 0x10017
	v_or_b32_e32 v1, s47, v143
	v_rcp_iflag_f32_e32 v0, v0
	v_add_u32_e32 v1, s52, v1
	v_xor_b32_e32 v1, s52, v1
	v_mul_hi_u32 v9, v1, v154
	v_mul_lo_u32 v9, v9, s28
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_sub_u32_e32 v1, v1, v9
	v_cvt_u32_f32_e32 v0, v0
	s_sub_i32 s7, s7, s50
	v_subrev_u32_e32 v9, s28, v1
	v_cmp_le_u32_e32 vcc, s28, v1
	s_add_i32 s7, s7, s48
	s_sub_i32 s51, 0, s46
	v_cndmask_b32_e32 v1, v1, v9, vcc
	s_lshl_b32 s48, s7, 8
	v_subrev_u32_e32 v9, s28, v1
	v_cmp_le_u32_e32 vcc, s28, v1
	s_bfe_i32 s49, s7, 0x10017
	v_or_b32_e32 v2, s48, v143
	v_cndmask_b32_e32 v1, v1, v9, vcc
	v_mul_lo_u32 v9, s51, v0
	v_or_b32_e32 v3, s48, v144
	v_or_b32_e32 v7, s48, v145
	v_add_u32_e32 v2, s49, v2
	v_xor_b32_e32 v10, s52, v1
	v_mul_hi_u32 v1, v0, v9
	v_or_b32_e32 v8, s48, v146
	v_add_u32_e32 v3, s49, v3
	v_add_u32_e32 v7, s49, v7
	v_xor_b32_e32 v2, s49, v2
	v_add_u32_e32 v156, v0, v1
	v_add_u32_e32 v8, s49, v8
	v_xor_b32_e32 v3, s49, v3
	v_xor_b32_e32 v7, s49, v7
	v_mul_hi_u32 v0, v2, v156
	v_xor_b32_e32 v8, s49, v8
	v_mul_hi_u32 v1, v3, v156
	v_mul_hi_u32 v9, v7, v156
	v_mul_lo_u32 v0, v0, s46
	v_mul_hi_u32 v11, v8, v156
	v_mul_lo_u32 v1, v1, s46
	v_mul_lo_u32 v9, v9, s46
	v_sub_u32_e32 v0, v2, v0
	v_mul_lo_u32 v11, v11, s46
	v_sub_u32_e32 v1, v3, v1
	v_sub_u32_e32 v2, v7, v9
	v_subrev_u32_e32 v7, s46, v0
	v_cmp_le_u32_e32 vcc, s46, v0
	v_sub_u32_e32 v3, v8, v11
	v_subrev_u32_e32 v8, s46, v1
	v_cndmask_b32_e32 v0, v0, v7, vcc
	v_cmp_le_u32_e32 vcc, s46, v1
	v_subrev_u32_e32 v9, s46, v2
	v_or_b32_e32 v4, s47, v144
	v_cndmask_b32_e32 v1, v1, v8, vcc
	v_cmp_le_u32_e32 vcc, s46, v2
	v_subrev_u32_e32 v11, s46, v3
	v_subrev_u32_e32 v7, s46, v0
	v_cndmask_b32_e32 v2, v2, v9, vcc
	v_cmp_le_u32_e32 vcc, s46, v3
	v_add_u32_e32 v4, s52, v4
	v_subrev_u32_e32 v8, s46, v1
	v_cndmask_b32_e32 v3, v3, v11, vcc
	v_cmp_le_u32_e32 vcc, s46, v0
	v_xor_b32_e32 v4, s52, v4
	v_subrev_u32_e32 v9, s46, v2
	v_cndmask_b32_e32 v0, v0, v7, vcc
	v_cmp_le_u32_e32 vcc, s46, v1
	v_subrev_u32_e32 v11, s46, v3
	v_or_b32_e32 v5, s47, v145
	v_cndmask_b32_e32 v1, v1, v8, vcc
	v_mul_hi_u32 v8, v4, v154
	v_cmp_le_u32_e32 vcc, s46, v2
	v_mul_lo_u32 v8, v8, s28
	v_sub_u32_e32 v4, v4, v8
	v_cndmask_b32_e32 v2, v2, v9, vcc
	v_cmp_le_u32_e32 vcc, s46, v3
	v_subrev_u32_e32 v8, s28, v4
	v_or_b32_e32 v6, s47, v146
	v_cndmask_b32_e32 v3, v3, v11, vcc
	v_cmp_le_u32_e32 vcc, s28, v4
	v_xor_b32_e32 v0, s49, v0
	s_mul_i32 s50, s6, s31
	v_cndmask_b32_e32 v4, v4, v8, vcc
	v_subrev_u32_e32 v8, s28, v4
	v_cmp_le_u32_e32 vcc, s28, v4
	s_mul_i32 s6, s30, s22
	v_xor_b32_e32 v1, s49, v1
	v_cndmask_b32_e32 v4, v4, v8, vcc
	v_xor_b32_e32 v4, s52, v4
	v_subrev_u32_e32 v17, s52, v4
	v_add_u32_e32 v4, s52, v5
	v_xor_b32_e32 v4, s52, v4
	v_mul_hi_u32 v5, v4, v154
	v_mul_lo_u32 v5, v5, s28
	v_sub_u32_e32 v4, v4, v5
	v_subrev_u32_e32 v5, s28, v4
	v_cmp_le_u32_e32 vcc, s28, v4
	v_xor_b32_e32 v2, s49, v2
	v_xor_b32_e32 v3, s49, v3
	v_cndmask_b32_e32 v4, v4, v5, vcc
	v_subrev_u32_e32 v5, s28, v4
	v_cmp_le_u32_e32 vcc, s28, v4
	v_subrev_u32_e32 v7, s49, v0
	v_subrev_u32_e32 v11, s49, v1
	v_cndmask_b32_e32 v4, v4, v5, vcc
	v_xor_b32_e32 v4, s52, v4
	v_subrev_u32_e32 v18, s52, v4
	v_add_u32_e32 v4, s52, v6
	v_xor_b32_e32 v4, s52, v4
	v_mul_hi_u32 v5, v4, v154
	v_mul_lo_u32 v5, v5, s28
	v_sub_u32_e32 v4, v4, v5
	v_subrev_u32_e32 v5, s28, v4
	v_cmp_le_u32_e32 vcc, s28, v4
	v_subrev_u32_e32 v12, s49, v2
	v_subrev_u32_e32 v14, s49, v3
	v_cndmask_b32_e32 v4, v4, v5, vcc
	v_subrev_u32_e32 v5, s28, v4
	v_cmp_le_u32_e32 vcc, s28, v4
	v_subrev_u32_e32 v16, s52, v10
	v_mul_lo_u32 v6, v18, s14
	v_cndmask_b32_e32 v4, v4, v5, vcc
	v_xor_b32_e32 v4, s52, v4
	v_subrev_u32_e32 v19, s52, v4
	v_add_u32_e32 v4, s6, v142
	v_mad_u64_u32 v[8:9], s[6:7], v7, s30, v[4:5]
	v_mad_u64_u32 v[10:11], s[6:7], v11, s30, v[4:5]
	v_mad_u64_u32 v[12:13], s[6:7], v12, s30, v[4:5]
	v_mad_u64_u32 v[14:15], s[6:7], v14, s30, v[4:5]
	v_mul_lo_u32 v4, v16, s14
	v_mul_lo_u32 v5, v17, s14
	v_mul_lo_u32 v7, v19, s14
	v_add_u32_e32 v9, s50, v142
	v_add_u32_e32 v11, v9, v4
	v_add_u32_e32 v13, v9, v5
	v_add_u32_e32 v15, v9, v6
	v_add_u32_e32 v16, v9, v7
	ds_bpermute_b32 v17, v150, v8
	v_lshrrev_b64 v[8:9], v128, exec
	ds_bpermute_b32 v9, v150, v10
	v_and_b32_e32 v8, 1, v8
	v_cmp_eq_u32_e32 vcc, 1, v8
	ds_bpermute_b32 v10, v150, v11
	s_mov_b32 s6, s26
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e32 v8, v131, v17, vcc
	buffer_load_dwordx4 v8, s[24:27], 0 offen lds
	ds_bpermute_b32 v8, v150, v12
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e32 v9, v131, v9, vcc
	s_mov_b32 m0, s35
	s_mov_b32 s7, s27
	buffer_load_dwordx4 v9, s[24:27], 0 offen lds
	ds_bpermute_b32 v9, v150, v14
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e32 v8, v131, v8, vcc
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dwordx4 v8, s[24:27], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e32 v8, v131, v9, vcc
	s_mov_b32 m0, s37
	ds_bpermute_b32 v9, v150, v13
	buffer_load_dwordx4 v8, s[24:27], 0 offen lds
	v_cndmask_b32_e32 v8, v131, v10, vcc
	s_mov_b32 m0, s38
	ds_bpermute_b32 v10, v150, v16
	buffer_load_dwordx4 v8, s[4:7], 0 offen lds
	ds_bpermute_b32 v8, v150, v15
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e32 v9, v131, v9, vcc
	s_mov_b32 m0, s39
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e32 v8, v131, v8, vcc
	buffer_load_dwordx4 v9, s[4:7], 0 offen lds
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v8, s[4:7], 0 offen lds
	v_cndmask_b32_e32 v8, v131, v10, vcc
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v8, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .L4
	s_barrier
.L4:
	s_or_b64 exec, exec, s[6:7]
	s_andn2_b64 vcc, exec, s[8:9]
	s_cbranch_vccnz .L6
	v_add_u32_e32 v3, s22, v3
	v_add_u32_e32 v2, s22, v2
	v_add_u32_e32 v1, s22, v1
	v_add_u32_e32 v0, s22, v0
	v_add_u32_e32 v8, s50, v130
	v_subrev_u32_e32 v3, s49, v3
	v_subrev_u32_e32 v2, s49, v2
	v_subrev_u32_e32 v1, s49, v1
	v_subrev_u32_e32 v0, s49, v0
	v_mov_b32_e32 v124, 0
	v_add_u32_e32 v157, v8, v7
	v_add_u32_e32 v158, v8, v6
	v_add_u32_e32 v159, v8, v5
	v_add_u32_e32 v160, v8, v4
	v_mad_u64_u32 v[132:133], s[6:7], s30, v3, v[130:131]
	v_mad_u64_u32 v[134:135], s[6:7], s30, v2, v[130:131]
	v_mad_u64_u32 v[136:137], s[6:7], s30, v1, v[130:131]
	v_mad_u64_u32 v[138:139], s[6:7], s30, v0, v[130:131]
	s_mov_b32 s49, 0
	s_add_i32 s50, 0, 0x10000
	s_mov_b32 s51, 0
	s_mov_b32 s52, 0
	v_mov_b32_e32 v125, v124
	v_mov_b32_e32 v126, v124
	v_mov_b32_e32 v127, v124
	v_mov_b32_e32 v120, v124
	v_mov_b32_e32 v121, v124
	v_mov_b32_e32 v122, v124
	v_mov_b32_e32 v123, v124
	v_mov_b32_e32 v116, v124
	v_mov_b32_e32 v117, v124
	v_mov_b32_e32 v118, v124
	v_mov_b32_e32 v119, v124
	v_mov_b32_e32 v112, v124
	v_mov_b32_e32 v113, v124
	v_mov_b32_e32 v114, v124
	v_mov_b32_e32 v115, v124
	v_mov_b32_e32 v108, v124
	v_mov_b32_e32 v109, v124
	v_mov_b32_e32 v110, v124
	v_mov_b32_e32 v111, v124
	v_mov_b32_e32 v104, v124
	v_mov_b32_e32 v105, v124
	v_mov_b32_e32 v106, v124
	v_mov_b32_e32 v107, v124
	v_mov_b32_e32 v100, v124
	v_mov_b32_e32 v101, v124
	v_mov_b32_e32 v102, v124
	v_mov_b32_e32 v103, v124
	v_mov_b32_e32 v96, v124
	v_mov_b32_e32 v97, v124
	v_mov_b32_e32 v98, v124
	v_mov_b32_e32 v99, v124
	v_mov_b32_e32 v92, v124
	v_mov_b32_e32 v93, v124
	v_mov_b32_e32 v94, v124
	v_mov_b32_e32 v95, v124
	v_mov_b32_e32 v88, v124
	v_mov_b32_e32 v89, v124
	v_mov_b32_e32 v90, v124
	v_mov_b32_e32 v91, v124
	v_mov_b32_e32 v84, v124
	v_mov_b32_e32 v85, v124
	v_mov_b32_e32 v86, v124
	v_mov_b32_e32 v87, v124
	v_mov_b32_e32 v80, v124
	v_mov_b32_e32 v81, v124
	v_mov_b32_e32 v82, v124
	v_mov_b32_e32 v83, v124
	v_mov_b32_e32 v76, v124
	v_mov_b32_e32 v77, v124
	v_mov_b32_e32 v78, v124
	v_mov_b32_e32 v79, v124
	v_mov_b32_e32 v72, v124
	v_mov_b32_e32 v73, v124
	v_mov_b32_e32 v74, v124
	v_mov_b32_e32 v75, v124
	v_mov_b32_e32 v68, v124
	v_mov_b32_e32 v69, v124
	v_mov_b32_e32 v70, v124
	v_mov_b32_e32 v71, v124
	v_mov_b32_e32 v64, v124
	v_mov_b32_e32 v65, v124
	v_mov_b32_e32 v66, v124
	v_mov_b32_e32 v67, v124
	v_mov_b32_e32 v60, v124
	v_mov_b32_e32 v61, v124
	v_mov_b32_e32 v62, v124
	v_mov_b32_e32 v63, v124
	v_mov_b32_e32 v56, v124
	v_mov_b32_e32 v57, v124
	v_mov_b32_e32 v58, v124
	v_mov_b32_e32 v59, v124
	v_mov_b32_e32 v52, v124
	v_mov_b32_e32 v53, v124
	v_mov_b32_e32 v54, v124
	v_mov_b32_e32 v55, v124
	v_mov_b32_e32 v48, v124
	v_mov_b32_e32 v49, v124
	v_mov_b32_e32 v50, v124
	v_mov_b32_e32 v51, v124
	v_mov_b32_e32 v44, v124
	v_mov_b32_e32 v45, v124
	v_mov_b32_e32 v46, v124
	v_mov_b32_e32 v47, v124
	v_mov_b32_e32 v40, v124
	v_mov_b32_e32 v41, v124
	v_mov_b32_e32 v42, v124
	v_mov_b32_e32 v43, v124
	v_mov_b32_e32 v36, v124
	v_mov_b32_e32 v37, v124
	v_mov_b32_e32 v38, v124
	v_mov_b32_e32 v39, v124
	v_mov_b32_e32 v32, v124
	v_mov_b32_e32 v33, v124
	v_mov_b32_e32 v34, v124
	v_mov_b32_e32 v35, v124
	v_mov_b32_e32 v28, v124
	v_mov_b32_e32 v29, v124
	v_mov_b32_e32 v30, v124
	v_mov_b32_e32 v31, v124
	v_mov_b32_e32 v24, v124
	v_mov_b32_e32 v25, v124
	v_mov_b32_e32 v26, v124
	v_mov_b32_e32 v27, v124
	v_mov_b32_e32 v20, v124
	v_mov_b32_e32 v21, v124
	v_mov_b32_e32 v22, v124
	v_mov_b32_e32 v23, v124
	v_mov_b32_e32 v16, v124
	v_mov_b32_e32 v17, v124
	v_mov_b32_e32 v18, v124
	v_mov_b32_e32 v19, v124
	v_mov_b32_e32 v12, v124
	v_mov_b32_e32 v13, v124
	v_mov_b32_e32 v14, v124
	v_mov_b32_e32 v15, v124
	v_mov_b32_e32 v8, v124
	v_mov_b32_e32 v9, v124
	v_mov_b32_e32 v10, v124
	v_mov_b32_e32 v11, v124
	v_mov_b32_e32 v4, v124
	v_mov_b32_e32 v5, v124
	v_mov_b32_e32 v6, v124
	v_mov_b32_e32 v7, v124
	v_mov_b32_e32 v0, v124
	v_mov_b32_e32 v1, v124
	v_mov_b32_e32 v2, v124
	v_mov_b32_e32 v3, v124
.L5:
	s_mov_b32 s53, s49
	v_add_u32_e32 v133, s51, v138
	s_add_i32 s49, s52, 1
	v_add_u32_e32 v135, s51, v136
	s_cmp_lt_i32 s49, 2
	ds_bpermute_b32 v133, v150, v133
	v_add_u32_e32 v137, s51, v134
	ds_bpermute_b32 v135, v150, v135
	s_cselect_b32 s52, s49, 0
	v_add_u32_e32 v139, s51, v132
	v_lshrrev_b64 v[162:163], v128, exec
	ds_bpermute_b32 v137, v150, v137
	s_lshl_b32 s49, s52, 15
	v_add_u32_e32 v161, s51, v160
	v_and_b32_e32 v162, 1, v162
	ds_bpermute_b32 v139, v150, v139
	s_add_i32 s49, s49, 0
	v_add_u32_e32 v163, s51, v159
	ds_bpermute_b32 v161, v150, v161
	v_cmp_eq_u32_e32 vcc, 1, v162
	s_add_i32 s55, s49, s33
	v_add_u32_e32 v164, s51, v158
	ds_bpermute_b32 v163, v150, v163
	s_waitcnt lgkmcnt(5)
	v_cndmask_b32_e32 v133, v131, v133, vcc
	s_mov_b32 m0, s55
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v165, s51, v157
	ds_bpermute_b32 v164, v150, v164
	v_cndmask_b32_e32 v135, v131, v135, vcc
	buffer_load_dwordx4 v133, s[24:27], 0 offen lds
	s_add_i32 m0, s55, 0x2000
	s_mov_b32 s54, s50
	ds_bpermute_b32 v165, v150, v165
	s_add_i32 s50, s49, 0x10000
	v_cndmask_b32_e32 v137, v131, v137, vcc
	buffer_load_dwordx4 v135, s[24:27], 0 offen lds
	s_add_i32 m0, s55, 0x4000
	v_cndmask_b32_e32 v139, v131, v139, vcc
	s_add_i32 s56, s50, s33
	buffer_load_dwordx4 v137, s[24:27], 0 offen lds
	s_add_i32 m0, s55, 0x6000
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	v_cndmask_b32_e32 v161, v131, v161, vcc
	buffer_load_dwordx4 v139, s[24:27], 0 offen lds
	s_mov_b32 m0, s56
	v_cndmask_b32_e32 v162, v131, v163, vcc
	buffer_load_dwordx4 v161, s[4:7], 0 offen lds
	s_add_i32 m0, s56, 0x2000
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e32 v163, v131, v164, vcc
	buffer_load_dwordx4 v162, s[4:7], 0 offen lds
	s_add_i32 m0, s56, 0x4000
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e32 v164, v131, v165, vcc
	buffer_load_dwordx4 v163, s[4:7], 0 offen lds
	s_add_i32 m0, s56, 0x6000
	s_nop 0
	buffer_load_dwordx4 v164, s[4:7], 0 offen lds
	v_add_u32_e32 v133, s54, v151
	v_add_u32_e32 v135, s54, v152
	v_add_u32_e32 v139, s53, v149
	v_add_u32_e32 v137, s53, v148
	s_barrier
	s_setprio 2
	ds_read_b128 v[162:165], v133
	ds_read_b128 v[166:169], v135
	ds_read_b128 v[190:193], v135 offset:16384
	ds_read_b128 v[198:201], v139
	ds_read_b128 v[194:197], v137
	ds_read_b128 v[202:205], v137 offset:4096
	ds_read_b128 v[206:209], v139 offset:4096
	ds_read_b128 v[170:173], v133 offset:8192
	ds_read_b128 v[174:177], v135 offset:8192
	ds_read_b128 v[186:189], v133 offset:16384
	ds_read_b128 v[178:181], v133 offset:24576
	ds_read_b128 v[182:185], v135 offset:24576
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x128_f8f6f4 v[120:123], v[170:177], v[194:201], v[120:123]
	v_mfma_f32_16x16x128_f8f6f4 v[124:127], v[162:169], v[194:201], v[124:127]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x128_f8f6f4 v[116:119], v[186:193], v[194:201], v[116:119]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x128_f8f6f4 v[112:115], v[178:185], v[194:201], v[112:115]
	v_mfma_f32_16x16x128_f8f6f4 v[108:111], v[162:169], v[202:209], v[108:111]
	v_mfma_f32_16x16x128_f8f6f4 v[104:107], v[170:177], v[202:209], v[104:107]
	v_mfma_f32_16x16x128_f8f6f4 v[100:103], v[186:193], v[202:209], v[100:103]
	v_mfma_f32_16x16x128_f8f6f4 v[96:99], v[178:185], v[202:209], v[96:99]
	ds_read_b128 v[198:201], v139 offset:8192
	ds_read_b128 v[194:197], v137 offset:8192
	ds_read_b128 v[202:205], v137 offset:12288
	ds_read_b128 v[206:209], v139 offset:12288
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x128_f8f6f4 v[92:95], v[162:169], v[194:201], v[92:95]
	v_mfma_f32_16x16x128_f8f6f4 v[88:91], v[170:177], v[194:201], v[88:91]
	v_mfma_f32_16x16x128_f8f6f4 v[84:87], v[186:193], v[194:201], v[84:87]
	v_mfma_f32_16x16x128_f8f6f4 v[80:83], v[178:185], v[194:201], v[80:83]
	ds_read_b128 v[194:197], v137 offset:16384
	ds_read_b128 v[210:213], v137 offset:20480
	ds_read_b128 v[218:221], v137 offset:24576
	ds_read_b128 v[226:229], v137 offset:28672
	ds_read_b128 v[198:201], v139 offset:16384
	ds_read_b128 v[214:217], v139 offset:20480
	ds_read_b128 v[222:225], v139 offset:24576
	ds_read_b128 v[230:233], v139 offset:28672
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x128_f8f6f4 v[76:79], v[162:169], v[202:209], v[76:79]
	v_mfma_f32_16x16x128_f8f6f4 v[72:75], v[170:177], v[202:209], v[72:75]
	v_mfma_f32_16x16x128_f8f6f4 v[68:71], v[186:193], v[202:209], v[68:71]
	v_mfma_f32_16x16x128_f8f6f4 v[64:67], v[178:185], v[202:209], v[64:67]
	s_setprio 1
	s_barrier
	s_setprio 2
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x128_f8f6f4 v[60:63], v[162:169], v[194:201], v[60:63]
	s_addk_i32 s51, 0x80
	s_cmp_lg_u32 s45, s51
	v_mfma_f32_16x16x128_f8f6f4 v[56:59], v[170:177], v[194:201], v[56:59]
	v_mfma_f32_16x16x128_f8f6f4 v[52:55], v[186:193], v[194:201], v[52:55]
	v_mfma_f32_16x16x128_f8f6f4 v[48:51], v[178:185], v[194:201], v[48:51]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x128_f8f6f4 v[44:47], v[162:169], v[210:217], v[44:47]
	v_mfma_f32_16x16x128_f8f6f4 v[40:43], v[170:177], v[210:217], v[40:43]
	v_mfma_f32_16x16x128_f8f6f4 v[36:39], v[186:193], v[210:217], v[36:39]
	v_mfma_f32_16x16x128_f8f6f4 v[32:35], v[178:185], v[210:217], v[32:35]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x128_f8f6f4 v[28:31], v[162:169], v[218:225], v[28:31]
	v_mfma_f32_16x16x128_f8f6f4 v[24:27], v[170:177], v[218:225], v[24:27]
	v_mfma_f32_16x16x128_f8f6f4 v[20:23], v[186:193], v[218:225], v[20:23]
	v_mfma_f32_16x16x128_f8f6f4 v[16:19], v[178:185], v[218:225], v[16:19]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x128_f8f6f4 v[12:15], v[162:169], v[226:233], v[12:15]
	v_mfma_f32_16x16x128_f8f6f4 v[8:11], v[170:177], v[226:233], v[8:11]
	v_mfma_f32_16x16x128_f8f6f4 v[4:7], v[186:193], v[226:233], v[4:7]
	v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[178:185], v[226:233], v[0:3]
	s_setprio 0
	s_cbranch_scc1 .L5
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .L2
	s_branch .L7
.L6:
	v_mov_b32_e32 v3, 0
	s_mov_b32 s49, 0
	s_add_i32 s50, 0, 0x10000
	v_mov_b32_e32 v2, v3
	v_mov_b32_e32 v1, v3
	v_mov_b32_e32 v0, v3
	v_mov_b32_e32 v7, v3
	v_mov_b32_e32 v6, v3
	v_mov_b32_e32 v5, v3
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v11, v3
	v_mov_b32_e32 v10, v3
	v_mov_b32_e32 v9, v3
	v_mov_b32_e32 v8, v3
	v_mov_b32_e32 v15, v3
	v_mov_b32_e32 v14, v3
	v_mov_b32_e32 v13, v3
	v_mov_b32_e32 v12, v3
	v_mov_b32_e32 v19, v3
	v_mov_b32_e32 v18, v3
	v_mov_b32_e32 v17, v3
	v_mov_b32_e32 v16, v3
	v_mov_b32_e32 v23, v3
	v_mov_b32_e32 v22, v3
	v_mov_b32_e32 v21, v3
	v_mov_b32_e32 v20, v3
	v_mov_b32_e32 v27, v3
	v_mov_b32_e32 v26, v3
	v_mov_b32_e32 v25, v3
	v_mov_b32_e32 v24, v3
	v_mov_b32_e32 v31, v3
	v_mov_b32_e32 v30, v3
	v_mov_b32_e32 v29, v3
	v_mov_b32_e32 v28, v3
	v_mov_b32_e32 v35, v3
	v_mov_b32_e32 v34, v3
	v_mov_b32_e32 v33, v3
	v_mov_b32_e32 v32, v3
	v_mov_b32_e32 v39, v3
	v_mov_b32_e32 v38, v3
	v_mov_b32_e32 v37, v3
	v_mov_b32_e32 v36, v3
	v_mov_b32_e32 v43, v3
	v_mov_b32_e32 v42, v3
	v_mov_b32_e32 v41, v3
	v_mov_b32_e32 v40, v3
	v_mov_b32_e32 v47, v3
	v_mov_b32_e32 v46, v3
	v_mov_b32_e32 v45, v3
	v_mov_b32_e32 v44, v3
	v_mov_b32_e32 v51, v3
	v_mov_b32_e32 v50, v3
	v_mov_b32_e32 v49, v3
	v_mov_b32_e32 v48, v3
	v_mov_b32_e32 v55, v3
	v_mov_b32_e32 v54, v3
	v_mov_b32_e32 v53, v3
	v_mov_b32_e32 v52, v3
	v_mov_b32_e32 v59, v3
	v_mov_b32_e32 v58, v3
	v_mov_b32_e32 v57, v3
	v_mov_b32_e32 v56, v3
	v_mov_b32_e32 v63, v3
	v_mov_b32_e32 v62, v3
	v_mov_b32_e32 v61, v3
	v_mov_b32_e32 v60, v3
	v_mov_b32_e32 v67, v3
	v_mov_b32_e32 v66, v3
	v_mov_b32_e32 v65, v3
	v_mov_b32_e32 v64, v3
	v_mov_b32_e32 v71, v3
	v_mov_b32_e32 v70, v3
	v_mov_b32_e32 v69, v3
	v_mov_b32_e32 v68, v3
	v_mov_b32_e32 v75, v3
	v_mov_b32_e32 v74, v3
	v_mov_b32_e32 v73, v3
	v_mov_b32_e32 v72, v3
	v_mov_b32_e32 v79, v3
	v_mov_b32_e32 v78, v3
	v_mov_b32_e32 v77, v3
	v_mov_b32_e32 v76, v3
	v_mov_b32_e32 v83, v3
	v_mov_b32_e32 v82, v3
	v_mov_b32_e32 v81, v3
	v_mov_b32_e32 v80, v3
	v_mov_b32_e32 v87, v3
	v_mov_b32_e32 v86, v3
	v_mov_b32_e32 v85, v3
	v_mov_b32_e32 v84, v3
	v_mov_b32_e32 v91, v3
	v_mov_b32_e32 v90, v3
	v_mov_b32_e32 v89, v3
	v_mov_b32_e32 v88, v3
	v_mov_b32_e32 v95, v3
	v_mov_b32_e32 v94, v3
	v_mov_b32_e32 v93, v3
	v_mov_b32_e32 v92, v3
	v_mov_b32_e32 v99, v3
	v_mov_b32_e32 v98, v3
	v_mov_b32_e32 v97, v3
	v_mov_b32_e32 v96, v3
	v_mov_b32_e32 v103, v3
	v_mov_b32_e32 v102, v3
	v_mov_b32_e32 v101, v3
	v_mov_b32_e32 v100, v3
	v_mov_b32_e32 v107, v3
	v_mov_b32_e32 v106, v3
	v_mov_b32_e32 v105, v3
	v_mov_b32_e32 v104, v3
	v_mov_b32_e32 v111, v3
	v_mov_b32_e32 v110, v3
	v_mov_b32_e32 v109, v3
	v_mov_b32_e32 v108, v3
	v_mov_b32_e32 v115, v3
	v_mov_b32_e32 v114, v3
	v_mov_b32_e32 v113, v3
	v_mov_b32_e32 v112, v3
	v_mov_b32_e32 v119, v3
	v_mov_b32_e32 v118, v3
	v_mov_b32_e32 v117, v3
	v_mov_b32_e32 v116, v3
	v_mov_b32_e32 v123, v3
	v_mov_b32_e32 v122, v3
	v_mov_b32_e32 v121, v3
	v_mov_b32_e32 v120, v3
	v_mov_b32_e32 v127, v3
	v_mov_b32_e32 v126, v3
	v_mov_b32_e32 v125, v3
	v_mov_b32_e32 v124, v3
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .L2
.L7:
	s_barrier
	s_branch .L2
.L8:
	s_endpgm
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
.Lfunc_end:
.size _grouped_fp8_persistent_gemm_kernel, .Lfunc_end-_grouped_fp8_persistent_gemm_kernel

.rodata
.p2align 6
.amdhsa_kernel _grouped_fp8_persistent_gemm_kernel
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 96
  .amdhsa_next_free_vgpr 224
  .amdhsa_next_free_sgpr 59
  .amdhsa_accum_offset 224
  .amdhsa_float_round_mode_32 3
  .amdhsa_float_round_mode_16_64 3
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_ieee_mode 1
  .amdhsa_dx10_clamp 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.kernels:
  - .name: _grouped_fp8_persistent_gemm_kernel
    .symbol: _grouped_fp8_persistent_gemm_kernel.kd
    .kernarg_segment_size: 96
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 59
    .vgpr_count: 221
    .agpr_count: 0
    .max_flat_workgroup_size: 512
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .uses_dynamic_stack: false
    .uniform_work_group_size: 1
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 8
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 16
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 24
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 32
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 40
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 48
        .size: 4
        .value_kind: by_value
      - .offset: 52
        .size: 4
        .value_kind: by_value
      - .offset: 56
        .size: 4
        .value_kind: by_value
      - .offset: 60
        .size: 4
        .value_kind: by_value
      - .offset: 64
        .size: 4
        .value_kind: by_value
      - .offset: 68
        .size: 4
        .value_kind: by_value
      - .offset: 72
        .size: 4
        .value_kind: by_value
      - .offset: 80
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 88
        .size: 8
        .value_kind: global_buffer
        .address_space: global
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...
.end_amdgpu_metadata
