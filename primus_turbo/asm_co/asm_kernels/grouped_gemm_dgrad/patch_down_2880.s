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
	s_cbranch_scc1 .L9
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
	s_bfe_u32 s0, s18, 0x30006
	s_movk_i32 s2, 0xff
	v_and_b32_e32 v3, 15, v0
	s_lshr_b32 s1, s18, 4
	v_lshlrev_b32_e32 v4, 4, v0
	s_movk_i32 s19, 0x70
	s_movk_i32 s3, 0x100
	v_and_b32_e32 v5, 48, v0
	v_lshlrev_b32_e32 v6, 3, v0
	v_lshl_or_b32 v7, s0, 6, v129
	v_and_or_b32 v140, s1, 16, v3
	v_and_b32_e32 v141, 0x70, v4
	s_lshl_b32 s33, s0, 10
	v_cmp_gt_u32_e64 s[0:1], s3, v0
	v_cmp_lt_u32_e64 s[2:3], s2, v0
	v_lshlrev_b32_e32 v0, 7, v3
	v_bitop3_b32 v3, v6, v5, s19 bitop3:0x6c
	v_lshrrev_b32_e32 v4, 2, v7
	v_lshlrev_b32_e32 v6, 4, v7
	s_load_dword s22, s[8:9], 0x0
	s_load_dword s23, s[10:11], 0x0
	v_and_b32_e32 v146, 60, v4
	v_bitop3_b32 v4, v6, v7, s19 bitop3:0x78
	s_waitcnt lgkmcnt(0)
	s_add_i32 s8, s28, 0xff
	s_add_i32 s19, s29, 0x7f
	s_ashr_i32 s9, s8, 31
	s_ashr_i32 s10, s19, 31
	s_lshr_b32 s9, s9, 24
	s_lshr_b32 s10, s10, 25
	s_add_i32 s8, s8, s9
	s_add_i32 s9, s19, s10
	s_mov_b32 s26, s38
	s_add_i32 s38, s33, 0
	s_and_b32 s47, s9, 0xffffff80
	s_and_b32 s25, s25, 0xffff
	s_and_b32 s5, s5, 0xffff
	s_mov_b32 s27, s39
	s_add_i32 s39, s38, 0x2000
	s_add_i32 s40, s38, 0x4000
	s_add_i32 s41, s38, 0x6000
	s_add_i32 s42, s38, 0x10000
	s_add_i32 s43, s38, 0x12000
	s_add_i32 s44, s38, 0x14000
	s_add_i32 s45, s38, 0x16000
	s_ashr_i32 s46, s8, 8
	s_ashr_i32 s36, s9, 7
	s_addk_i32 s47, 0xff80
	s_cmpk_gt_i32 s19, 0x17f
	v_lshlrev_b32_e32 v8, 3, v7
	s_cselect_b64 s[10:11], -1, 0
	s_lshl_b32 s18, s18, 5
	v_lshrrev_b32_e32 v142, 3, v7
	v_and_b32_e32 v7, 0x870, v8
	v_sub_u32_e32 v6, v4, v6
	s_and_b32 s18, s18, 0x1800
	v_bitop3_b32 v147, v7, v5, v0 bitop3:0x36
	v_ashrrev_i32_e32 v5, 4, v6
	s_cmpk_gt_i32 s19, 0xff
	v_add_u32_e32 v128, v5, v129
	s_mul_i32 s46, s46, 6
	v_or_b32_e32 v5, s47, v141
	v_or_b32_e32 v3, s18, v3
	s_cselect_b64 s[18:19], -1, 0
	s_cmp_gt_i32 s28, -1
	v_bfrev_b32_e32 v2, -2
	v_cmp_gt_i32_e64 s[8:9], s29, v5
	s_cselect_b64 s[34:35], -1, 0
	s_abs_i32 s29, s46
	v_or_b32_e32 v150, v3, v0
	v_bitop3_b32 v152, v3, 64, v0 bitop3:0x36
	s_abs_i32 s28, s28
	v_cmp_ne_u32_e64 s[6:7], 0, v129
	s_max_i32 s50, s36, 3
	s_and_b64 s[36:37], s[6:7], vcc
	s_sub_i32 s6, 0, s29
	s_lshl_b32 s50, s50, 7
	v_or_b32_e32 v143, 64, v142
	v_or_b32_e32 v144, 0x80, v142
	v_or_b32_e32 v145, 0xc0, v142
	s_waitcnt vmcnt(0)
	v_cndmask_b32_e32 v151, v2, v1, vcc
	v_cvt_f32_u32_e32 v1, s29
	v_xor_b32_e32 v148, 64, v147
	v_lshlrev_b32_e32 v149, 2, v128
	s_and_b32 s21, s21, 0xffff
	v_rcp_iflag_f32_e32 v0, v1
	v_cvt_f32_u32_e32 v1, s28
	s_ashr_i32 s48, s46, 31
	v_or_b32_e32 v130, 0x80, v141
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	v_rcp_iflag_f32_e32 v1, v1
	s_addk_i32 s50, 0xff00
	v_add_u32_e32 v155, 0, v4
	v_readfirstlane_b32 s7, v0
	v_mul_f32_e32 v0, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v0, v0
	s_mul_i32 s6, s6, s7
	s_mul_hi_u32 s6, s7, s6
	s_add_i32 s49, s7, s6
	s_sub_i32 s6, 0, s28
	v_mul_lo_u32 v1, s6, v0
	v_mul_hi_u32 v1, v0, v1
	v_add_u32_e32 v153, v0, v1
	v_mov_b32_e32 v0, s23
	v_mul_f32_e32 v154, s22, v0
	s_branch .L3
.L2:
	v_add_u32_e32 v132, s47, v161
	v_add_u32_e32 v133, s47, v162
	v_add_u32_e32 v134, s47, v163
	v_cndmask_b32_e64 v132, v131, v132, s[8:9]
	v_add_u32_e32 v135, s47, v164
	v_cndmask_b32_e64 v133, v131, v133, s[8:9]
	buffer_load_dwordx4 v[162:165], v132, s[24:27], 0 offen
	buffer_load_dwordx4 v[166:169], v133, s[24:27], 0 offen
	v_cndmask_b32_e64 v132, v131, v134, s[8:9]
	v_cndmask_b32_e64 v133, v131, v135, s[8:9]
	buffer_load_dwordx4 v[170:173], v132, s[24:27], 0 offen
	buffer_load_dwordx4 v[174:177], v133, s[24:27], 0 offen
	v_add_u32_e32 v132, s47, v157
	v_add_u32_e32 v133, s47, v158
	v_add_u32_e32 v134, s47, v159
	v_add_u32_e32 v135, s47, v160
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	v_cndmask_b32_e64 v132, v131, v132, s[8:9]
	v_cndmask_b32_e64 v133, v131, v133, s[8:9]
	v_cndmask_b32_e64 v134, v131, v134, s[8:9]
	v_cndmask_b32_e64 v135, v131, v135, s[8:9]
	buffer_load_dwordx4 v[158:161], v132, s[4:7], 0 offen
	buffer_load_dwordx4 v[178:181], v133, s[4:7], 0 offen
	buffer_load_dwordx4 v[190:193], v134, s[4:7], 0 offen
	buffer_load_dwordx4 v[196:199], v135, s[4:7], 0 offen
	v_add_u32_e32 v132, 0, v147
	v_add_u32_e32 v136, 0, v148
	v_add_u32_e32 v194, 0, v152
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v157, 0, v150
	s_ashr_i32 s6, s53, 31
	s_waitcnt vmcnt(7)
	ds_write_b128 v155, v[162:165]
	s_waitcnt vmcnt(6)
	ds_write_b128 v155, v[166:169] offset:8192
	s_waitcnt vmcnt(5)
	ds_write_b128 v155, v[170:173] offset:16384
	s_waitcnt vmcnt(4)
	ds_write_b128 v155, v[174:177] offset:24576
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[222:225], v132
	ds_read_b128 v[230:233], v132 offset:4096
	ds_read_b128 v[238:241], v132 offset:8192
	ds_read_b128 v[214:217], v132 offset:12288
	ds_read_b128 v[206:209], v132 offset:16384
	ds_read_b128 v[182:185], v132 offset:20480
	ds_read_b128 v[166:169], v132 offset:24576
	ds_read_b128 v[132:135], v132 offset:28672
	ds_read_b128 v[226:229], v136
	ds_read_b128 v[234:237], v136 offset:4096
	ds_read_b128 v[242:245], v136 offset:8192
	ds_read_b128 v[218:221], v136 offset:12288
	ds_read_b128 v[210:213], v136 offset:16384
	ds_read_b128 v[186:189], v136 offset:20480
	ds_read_b128 v[170:173], v136 offset:24576
	ds_read_b128 v[136:139], v136 offset:28672
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(3)
	ds_write_b128 v155, v[158:161]
	s_waitcnt vmcnt(2)
	ds_write_b128 v155, v[178:181] offset:8192
	s_waitcnt vmcnt(1)
	ds_write_b128 v155, v[190:193] offset:16384
	s_waitcnt vmcnt(0)
	ds_write_b128 v155, v[196:199] offset:24576
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[202:205], v194 offset:16384
	ds_read_b128 v[178:181], v194
	ds_read_b128 v[174:177], v157
	ds_read_b128 v[158:161], v157 offset:8192
	ds_read_b128 v[162:165], v194 offset:8192
	ds_read_b128 v[198:201], v157 offset:16384
	ds_read_b128 v[190:193], v157 offset:24576
	ds_read_b128 v[194:197], v194 offset:24576
	v_or_b32_e32 v157, s53, v140
	v_add_u32_e32 v157, s6, v157
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x128_f8f6f4 v[124:127], v[174:181], v[222:229], v[124:127]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x128_f8f6f4 v[120:123], v[158:165], v[222:229], v[120:123]
	s_nop 9
	v_mul_f32_e32 v124, v154, v124
	v_mul_f32_e32 v125, v154, v125
	v_mul_f32_e32 v126, v154, v126
	v_mul_f32_e32 v127, v154, v127
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x128_f8f6f4 v[116:119], v[198:205], v[222:229], v[116:119]
	v_mul_f32_e32 v120, v154, v120
	v_mul_f32_e32 v121, v154, v121
	v_mul_f32_e32 v122, v154, v122
	v_mul_f32_e32 v123, v154, v123
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x128_f8f6f4 v[112:115], v[190:197], v[222:229], v[112:115]
	v_xor_b32_e32 v222, s6, v157
	v_mul_hi_u32 v223, v222, v156
	v_mul_lo_u32 v223, v223, s51
	v_sub_u32_e32 v222, v222, v223
	v_subrev_u32_e32 v223, s51, v222
	v_cmp_le_u32_e32 vcc, s51, v222
	v_mul_f32_e32 v116, v154, v116
	v_mfma_f32_16x16x128_f8f6f4 v[76:79], v[174:181], v[214:221], v[76:79]
	v_cndmask_b32_e32 v222, v222, v223, vcc
	v_subrev_u32_e32 v223, s51, v222
	v_cmp_le_u32_e32 vcc, s51, v222
	v_mul_f32_e32 v117, v154, v117
	v_mul_f32_e32 v118, v154, v118
	v_cndmask_b32_e32 v222, v222, v223, vcc
	v_add_u32_e32 v223, 32, v157
	v_xor_b32_e32 v223, s6, v223
	v_mul_hi_u32 v224, v223, v156
	v_mul_lo_u32 v224, v224, s51
	v_sub_u32_e32 v223, v223, v224
	v_subrev_u32_e32 v224, s51, v223
	v_cmp_le_u32_e32 vcc, s51, v223
	v_mfma_f32_16x16x128_f8f6f4 v[72:75], v[158:165], v[214:221], v[72:75]
	v_xor_b32_e32 v222, s6, v222
	v_cndmask_b32_e32 v223, v223, v224, vcc
	v_subrev_u32_e32 v224, s51, v223
	v_cmp_le_u32_e32 vcc, s51, v223
	v_subrev_u32_e32 v222, s6, v222
	v_mul_f32_e32 v119, v154, v119
	v_cndmask_b32_e32 v223, v223, v224, vcc
	v_add_u32_e32 v224, 64, v157
	v_xor_b32_e32 v224, s6, v224
	v_mul_hi_u32 v225, v224, v156
	v_mul_lo_u32 v225, v225, s51
	v_sub_u32_e32 v224, v224, v225
	v_subrev_u32_e32 v225, s51, v224
	v_cmp_le_u32_e32 vcc, s51, v224
	v_mfma_f32_16x16x128_f8f6f4 v[68:71], v[198:205], v[214:221], v[68:71]
	v_xor_b32_e32 v223, s6, v223
	v_cndmask_b32_e32 v224, v224, v225, vcc
	v_subrev_u32_e32 v225, s51, v224
	v_cmp_le_u32_e32 vcc, s51, v224
	v_subrev_u32_e32 v223, s6, v223
	v_mul_f32_e32 v72, v154, v72
	v_cndmask_b32_e32 v224, v224, v225, vcc
	v_add_u32_e32 v225, 0x60, v157
	v_xor_b32_e32 v225, s6, v225
	v_mul_hi_u32 v226, v225, v156
	v_mul_lo_u32 v226, v226, s51
	v_mfma_f32_16x16x128_f8f6f4 v[64:67], v[190:197], v[214:221], v[64:67]
	v_sub_u32_e32 v214, v225, v226
	v_subrev_u32_e32 v215, s51, v214
	v_cmp_le_u32_e32 vcc, s51, v214
	v_xor_b32_e32 v224, s6, v224
	v_subrev_u32_e32 v224, s6, v224
	v_cndmask_b32_e32 v214, v214, v215, vcc
	v_subrev_u32_e32 v215, s51, v214
	v_cmp_le_u32_e32 vcc, s51, v214
	v_mfma_f32_16x16x128_f8f6f4 v[60:63], v[174:181], v[206:213], v[60:63]
	s_nop 2
	v_mul_f32_e32 v64, v154, v64
	v_cndmask_b32_e32 v214, v214, v215, vcc
	v_add_u32_e32 v215, 0x80, v157
	v_xor_b32_e32 v215, s6, v215
	v_mul_hi_u32 v216, v215, v156
	v_mul_lo_u32 v216, v216, s51
	v_sub_u32_e32 v215, v215, v216
	v_subrev_u32_e32 v216, s51, v215
	v_cmp_le_u32_e32 vcc, s51, v215
	v_mfma_f32_16x16x128_f8f6f4 v[56:59], v[158:165], v[206:213], v[56:59]
	v_xor_b32_e32 v214, s6, v214
	v_cndmask_b32_e32 v215, v215, v216, vcc
	v_subrev_u32_e32 v216, s51, v215
	v_cmp_le_u32_e32 vcc, s51, v215
	v_subrev_u32_e32 v214, s6, v214
	v_mul_f32_e32 v65, v154, v65
	v_cndmask_b32_e32 v215, v215, v216, vcc
	v_add_u32_e32 v216, 0xa0, v157
	v_xor_b32_e32 v216, s6, v216
	v_mul_hi_u32 v217, v216, v156
	v_mul_lo_u32 v217, v217, s51
	v_sub_u32_e32 v216, v216, v217
	v_subrev_u32_e32 v217, s51, v216
	v_cmp_le_u32_e32 vcc, s51, v216
	v_mfma_f32_16x16x128_f8f6f4 v[52:55], v[198:205], v[206:213], v[52:55]
	v_xor_b32_e32 v215, s6, v215
	v_subrev_u32_e32 v215, s6, v215
	v_mul_f32_e32 v73, v154, v73
	v_mul_f32_e32 v68, v154, v68
	v_mul_f32_e32 v69, v154, v69
	v_mul_f32_e32 v70, v154, v70
	v_mul_f32_e32 v71, v154, v71
	v_mfma_f32_16x16x128_f8f6f4 v[48:51], v[190:197], v[206:213], v[48:51]
	v_cndmask_b32_e32 v206, v216, v217, vcc
	v_subrev_u32_e32 v207, s51, v206
	v_cmp_le_u32_e32 vcc, s51, v206
	v_mul_f32_e32 v66, v154, v66
	v_mul_f32_e32 v67, v154, v67
	v_cndmask_b32_e32 v206, v206, v207, vcc
	v_add_u32_e32 v207, 0xc0, v157
	v_xor_b32_e32 v207, s6, v207
	v_mul_hi_u32 v208, v207, v156
	v_mul_lo_u32 v208, v208, s51
	v_add_u32_e32 v157, 0xe0, v157
	v_sub_u32_e32 v207, v207, v208
	v_xor_b32_e32 v157, s6, v157
	v_subrev_u32_e32 v208, s51, v207
	v_cmp_le_u32_e32 vcc, s51, v207
	v_mul_hi_u32 v156, v157, v156
	v_mul_lo_u32 v156, v156, s51
	v_cndmask_b32_e32 v207, v207, v208, vcc
	v_subrev_u32_e32 v208, s51, v207
	v_cmp_le_u32_e32 vcc, s51, v207
	v_sub_u32_e32 v156, v157, v156
	v_subrev_u32_e32 v157, s51, v156
	v_cndmask_b32_e32 v207, v207, v208, vcc
	v_cmp_le_u32_e32 vcc, s51, v156
	v_xor_b32_e32 v206, s6, v206
	v_xor_b32_e32 v207, s6, v207
	v_cndmask_b32_e32 v156, v156, v157, vcc
	v_subrev_u32_e32 v157, s51, v156
	v_cmp_le_u32_e32 vcc, s51, v156
	v_subrev_u32_e32 v206, s6, v206
	v_subrev_u32_e32 v207, s6, v207
	v_cndmask_b32_e32 v156, v156, v157, vcc
	v_xor_b32_e32 v156, s6, v156
	v_subrev_u32_e32 v156, s6, v156
	v_or_b32_e32 v157, s52, v146
	s_ashr_i32 s6, s52, 31
	v_add_u32_e32 v157, s6, v157
	v_mfma_f32_16x16x128_f8f6f4 v[44:47], v[174:181], v[182:189], v[44:47]
	s_cmp_gt_i32 s23, -1
	s_mov_b32 s23, s27
	v_mul_f32_e32 v112, v154, v112
	v_mul_f32_e32 v113, v154, v113
	v_mul_f32_e32 v114, v154, v114
	v_mul_f32_e32 v115, v154, v115
	v_mul_f32_e32 v76, v154, v76
	v_mfma_f32_16x16x128_f8f6f4 v[40:43], v[158:165], v[182:189], v[40:43]
	v_mul_f32_e32 v77, v154, v77
	v_mul_f32_e32 v78, v154, v78
	v_mul_f32_e32 v79, v154, v79
	v_mul_f32_e32 v74, v154, v74
	v_mul_f32_e32 v75, v154, v75
	v_mul_f32_e32 v60, v154, v60
	v_mul_f32_e32 v61, v154, v61
	v_mfma_f32_16x16x128_f8f6f4 v[36:39], v[198:205], v[182:189], v[36:39]
	v_mul_f32_e32 v62, v154, v62
	v_mul_f32_e32 v63, v154, v63
	v_mul_f32_e32 v56, v154, v56
	v_mul_f32_e32 v57, v154, v57
	v_mul_f32_e32 v58, v154, v58
	v_mul_f32_e32 v59, v154, v59
	v_mul_f32_e32 v52, v154, v52
	v_mfma_f32_16x16x128_f8f6f4 v[32:35], v[190:197], v[182:189], v[32:35]
	v_xor_b32_e32 v182, s6, v157
	v_mul_hi_u32 v183, v182, v153
	v_mul_lo_u32 v183, v183, s28
	v_sub_u32_e32 v182, v182, v183
	v_subrev_u32_e32 v183, s28, v182
	v_cmp_le_u32_e32 vcc, s28, v182
	v_mul_f32_e32 v53, v154, v53
	v_mfma_f32_16x16x128_f8f6f4 v[28:31], v[174:181], v[166:173], v[28:31]
	v_cndmask_b32_e32 v182, v182, v183, vcc
	v_subrev_u32_e32 v183, s28, v182
	v_cmp_le_u32_e32 vcc, s28, v182
	v_mul_f32_e32 v54, v154, v54
	v_mul_f32_e32 v55, v154, v55
	v_cndmask_b32_e32 v182, v182, v183, vcc
	v_add_u32_e32 v183, 64, v157
	v_xor_b32_e32 v183, s6, v183
	v_mul_hi_u32 v184, v183, v153
	v_mfma_f32_16x16x128_f8f6f4 v[24:27], v[158:165], v[166:173], v[24:27]
	v_mul_lo_u32 v184, v184, s28
	v_sub_u32_e32 v183, v183, v184
	v_subrev_u32_e32 v184, s28, v183
	v_cmp_le_u32_e32 vcc, s28, v183
	v_xor_b32_e32 v182, s6, v182
	v_subrev_u32_e32 v182, s6, v182
	v_cndmask_b32_e32 v183, v183, v184, vcc
	v_mfma_f32_16x16x128_f8f6f4 v[20:23], v[198:205], v[166:173], v[20:23]
	v_subrev_u32_e32 v184, s28, v183
	v_cmp_le_u32_e32 vcc, s28, v183
	v_mul_f32_e32 v48, v154, v48
	v_mul_f32_e32 v49, v154, v49
	v_cndmask_b32_e32 v183, v183, v184, vcc
	v_mul_f32_e32 v50, v154, v50
	v_mul_f32_e32 v51, v154, v51
	v_mfma_f32_16x16x128_f8f6f4 v[16:19], v[190:197], v[166:173], v[16:19]
	v_add_u32_e32 v167, 0x80, v157
	v_xor_b32_e32 v167, s6, v167
	v_mul_hi_u32 v168, v167, v153
	v_mul_lo_u32 v168, v168, s28
	v_add_u32_e32 v157, 0xc0, v157
	v_sub_u32_e32 v167, v167, v168
	v_xor_b32_e32 v157, s6, v157
	v_mfma_f32_16x16x128_f8f6f4 v[104:107], v[158:165], v[230:237], v[104:107]
	v_subrev_u32_e32 v168, s28, v167
	v_cmp_le_u32_e32 vcc, s28, v167
	v_xor_b32_e32 v166, s6, v183
	v_subrev_u32_e32 v166, s6, v166
	v_cndmask_b32_e32 v167, v167, v168, vcc
	v_subrev_u32_e32 v168, s28, v167
	v_cmp_le_u32_e32 vcc, s28, v167
	v_mfma_f32_16x16x128_f8f6f4 v[88:91], v[158:165], v[238:245], v[88:91]
	v_mul_f32_e32 v169, v154, v26
	v_cndmask_b32_e32 v167, v167, v168, vcc
	v_xor_b32_e32 v167, s6, v167
	v_cvt_pk_bf16_f32 v26, v72, v73
	v_mul_f32_e32 v104, v154, v104
	v_mul_f32_e32 v105, v154, v105
	v_mul_f32_e32 v106, v154, v106
	v_mfma_f32_16x16x128_f8f6f4 v[8:11], v[158:165], v[132:139], v[8:11]
	v_mul_hi_u32 v159, v157, v153
	v_mul_lo_u32 v159, v159, s28
	v_sub_u32_e32 v157, v157, v159
	v_subrev_u32_e32 v159, s28, v157
	v_cmp_le_u32_e32 vcc, s28, v157
	v_mul_f32_e32 v164, v154, v30
	v_cvt_pk_bf16_f32 v30, v64, v65
	v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[190:197], v[132:139], v[0:3]
	v_cndmask_b32_e32 v157, v157, v159, vcc
	v_subrev_u32_e32 v159, s28, v157
	v_cmp_le_u32_e32 vcc, s28, v157
	v_add_u32_e32 v64, s22, v222
	v_subrev_u32_e32 v158, s6, v167
	v_cndmask_b32_e32 v157, v157, v159, vcc
	v_xor_b32_e32 v157, s6, v157
	v_mfma_f32_16x16x128_f8f6f4 v[4:7], v[198:205], v[132:139], v[4:7]
	v_subrev_u32_e32 v157, s6, v157
	s_cselect_b64 s[6:7], -1, 0
	v_mul_lo_u32 v64, v64, s15
	v_add_lshl_u32 v72, v64, v182, 1
	s_and_b64 vcc, s[34:35], s[6:7]
	v_mul_f32_e32 v162, v154, v28
	v_mul_f32_e32 v163, v154, v29
	v_mfma_f32_16x16x128_f8f6f4 v[108:111], v[174:181], v[230:237], v[108:111]
	v_mul_f32_e32 v165, v154, v31
	v_cvt_pk_bf16_f32 v28, v68, v69
	v_cvt_pk_bf16_f32 v29, v70, v71
	v_cvt_pk_bf16_f32 v31, v66, v67
	v_add_u32_e32 v65, s22, v223
	v_add_u32_e32 v66, s22, v224
	v_add_u32_e32 v67, s22, v214
	v_mfma_f32_16x16x128_f8f6f4 v[96:99], v[190:197], v[230:237], v[96:99]
	v_add_u32_e32 v68, s22, v215
	v_add_u32_e32 v69, s22, v206
	v_add_u32_e32 v70, s22, v207
	v_add_u32_e32 v71, s22, v156
	v_cndmask_b32_e32 v72, v131, v72, vcc
	s_mov_b32 s22, s26
	v_mul_f32_e32 v188, v154, v4
	v_mfma_f32_16x16x128_f8f6f4 v[80:83], v[190:197], v[238:245], v[80:83]
	v_mul_f32_e32 v192, v154, v0
	v_mul_f32_e32 v193, v154, v1
	v_cvt_pk_bf16_f32 v0, v124, v125
	v_cvt_pk_bf16_f32 v1, v126, v127
	buffer_store_dwordx2 v[0:1], v72, s[20:23], 0 offen
	v_add_lshl_u32 v0, v64, v166, 1
	v_mul_f32_e32 v194, v154, v2
	v_mul_f32_e32 v195, v154, v3
	v_cvt_pk_bf16_f32 v2, v120, v121
	v_cvt_pk_bf16_f32 v3, v122, v123
	v_cndmask_b32_e32 v0, v131, v0, vcc
	v_mfma_f32_16x16x128_f8f6f4 v[100:103], v[198:205], v[230:237], v[100:103]
	buffer_store_dwordx2 v[2:3], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v64, v158, 1
	v_mul_f32_e32 v189, v154, v5
	v_cvt_pk_bf16_f32 v4, v116, v117
	v_cvt_pk_bf16_f32 v5, v118, v119
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[4:5], v0, s[20:23], 0 offen
	v_mfma_f32_16x16x128_f8f6f4 v[12:15], v[174:181], v[132:139], v[12:15]
	v_add_lshl_u32 v0, v64, v157, 1
	v_mul_f32_e32 v190, v154, v6
	v_mul_f32_e32 v191, v154, v7
	v_cvt_pk_bf16_f32 v6, v112, v113
	v_cvt_pk_bf16_f32 v7, v114, v115
	v_mul_lo_u32 v65, v65, s15
	v_cndmask_b32_e32 v0, v131, v0, vcc
	v_mfma_f32_16x16x128_f8f6f4 v[92:95], v[174:181], v[238:245], v[92:95]
	v_mul_f32_e32 v108, v154, v108
	v_mul_f32_e32 v109, v154, v109
	v_mul_f32_e32 v110, v154, v110
	v_mul_f32_e32 v111, v154, v111
	buffer_store_dwordx2 v[6:7], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v65, v182, 1
	v_mul_f32_e32 v184, v154, v8
	v_mul_f32_e32 v185, v154, v9
	v_cvt_pk_bf16_f32 v8, v108, v109
	v_cvt_pk_bf16_f32 v9, v110, v111
	v_cndmask_b32_e32 v0, v131, v0, vcc
	v_mul_f32_e32 v107, v154, v107
	buffer_store_dwordx2 v[8:9], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v65, v166, 1
	v_mul_f32_e32 v186, v154, v10
	v_mul_f32_e32 v187, v154, v11
	v_cvt_pk_bf16_f32 v10, v104, v105
	v_cvt_pk_bf16_f32 v11, v106, v107
	v_cndmask_b32_e32 v0, v131, v0, vcc
	v_mfma_f32_16x16x128_f8f6f4 v[84:87], v[198:205], v[238:245], v[84:87]
	v_mul_f32_e32 v100, v154, v100
	v_mul_f32_e32 v101, v154, v101
	v_mul_f32_e32 v102, v154, v102
	v_mul_f32_e32 v103, v154, v103
	buffer_store_dwordx2 v[10:11], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v65, v158, 1
	v_mul_f32_e32 v179, v154, v12
	v_mul_f32_e32 v180, v154, v13
	v_cvt_pk_bf16_f32 v12, v100, v101
	v_cvt_pk_bf16_f32 v13, v102, v103
	v_cndmask_b32_e32 v0, v131, v0, vcc
	v_mul_f32_e32 v96, v154, v96
	v_mul_f32_e32 v97, v154, v97
	v_mul_f32_e32 v98, v154, v98
	v_mul_f32_e32 v99, v154, v99
	buffer_store_dwordx2 v[12:13], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v65, v157, 1
	v_mul_f32_e32 v181, v154, v14
	v_mul_f32_e32 v183, v154, v15
	v_cvt_pk_bf16_f32 v14, v96, v97
	v_cvt_pk_bf16_f32 v15, v98, v99
	v_mul_lo_u32 v66, v66, s15
	v_cndmask_b32_e32 v0, v131, v0, vcc
	v_mul_f32_e32 v92, v154, v92
	v_mul_f32_e32 v93, v154, v93
	v_mul_f32_e32 v94, v154, v94
	v_mul_f32_e32 v95, v154, v95
	buffer_store_dwordx2 v[14:15], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v66, v182, 1
	v_mul_f32_e32 v175, v154, v16
	v_mul_f32_e32 v176, v154, v17
	v_cvt_pk_bf16_f32 v16, v92, v93
	v_cvt_pk_bf16_f32 v17, v94, v95
	v_cndmask_b32_e32 v0, v131, v0, vcc
	v_mul_f32_e32 v88, v154, v88
	v_mul_f32_e32 v89, v154, v89
	v_mul_f32_e32 v90, v154, v90
	v_mul_f32_e32 v91, v154, v91
	buffer_store_dwordx2 v[16:17], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v66, v166, 1
	v_mul_f32_e32 v177, v154, v18
	v_mul_f32_e32 v178, v154, v19
	v_cvt_pk_bf16_f32 v18, v88, v89
	v_cvt_pk_bf16_f32 v19, v90, v91
	v_cndmask_b32_e32 v0, v131, v0, vcc
	v_mul_f32_e32 v84, v154, v84
	v_mul_f32_e32 v85, v154, v85
	v_mul_f32_e32 v86, v154, v86
	v_mul_f32_e32 v87, v154, v87
	buffer_store_dwordx2 v[18:19], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v66, v158, 1
	v_mul_f32_e32 v171, v154, v20
	v_mul_f32_e32 v172, v154, v21
	v_cvt_pk_bf16_f32 v20, v84, v85
	v_cvt_pk_bf16_f32 v21, v86, v87
	v_cndmask_b32_e32 v0, v131, v0, vcc
	v_mul_f32_e32 v80, v154, v80
	v_mul_f32_e32 v81, v154, v81
	v_mul_f32_e32 v82, v154, v82
	v_mul_f32_e32 v83, v154, v83
	buffer_store_dwordx2 v[20:21], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v66, v157, 1
	v_mul_f32_e32 v173, v154, v22
	v_mul_f32_e32 v174, v154, v23
	v_cvt_pk_bf16_f32 v22, v80, v81
	v_cvt_pk_bf16_f32 v23, v82, v83
	v_mul_lo_u32 v67, v67, s15
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[22:23], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v67, v182, 1
	v_mul_f32_e32 v167, v154, v24
	v_mul_f32_e32 v168, v154, v25
	v_cvt_pk_bf16_f32 v24, v76, v77
	v_cvt_pk_bf16_f32 v25, v78, v79
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[24:25], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v67, v166, 1
	v_mul_f32_e32 v170, v154, v27
	v_cvt_pk_bf16_f32 v27, v74, v75
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[26:27], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v67, v158, 1
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[28:29], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v67, v157, 1
	v_mul_lo_u32 v68, v68, s15
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[30:31], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v68, v182, 1
	v_mul_f32_e32 v139, v154, v32
	v_mul_f32_e32 v159, v154, v33
	v_cvt_pk_bf16_f32 v32, v60, v61
	v_cvt_pk_bf16_f32 v33, v62, v63
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[32:33], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v68, v166, 1
	v_mul_f32_e32 v160, v154, v34
	v_mul_f32_e32 v161, v154, v35
	v_cvt_pk_bf16_f32 v34, v56, v57
	v_cvt_pk_bf16_f32 v35, v58, v59
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[34:35], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v68, v158, 1
	v_mul_f32_e32 v135, v154, v36
	v_mul_f32_e32 v136, v154, v37
	v_cvt_pk_bf16_f32 v36, v52, v53
	v_cvt_pk_bf16_f32 v37, v54, v55
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[36:37], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v68, v157, 1
	v_mul_f32_e32 v137, v154, v38
	v_mul_f32_e32 v138, v154, v39
	v_cvt_pk_bf16_f32 v38, v48, v49
	v_cvt_pk_bf16_f32 v39, v50, v51
	v_mul_lo_u32 v69, v69, s15
	v_cndmask_b32_e32 v0, v131, v0, vcc
	v_mul_f32_e32 v44, v154, v44
	v_mul_f32_e32 v45, v154, v45
	v_mul_f32_e32 v46, v154, v46
	v_mul_f32_e32 v47, v154, v47
	buffer_store_dwordx2 v[38:39], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v69, v182, 1
	v_mul_f32_e32 v132, v154, v40
	v_mul_f32_e32 v133, v154, v41
	v_cvt_pk_bf16_f32 v40, v44, v45
	v_cvt_pk_bf16_f32 v41, v46, v47
	v_cndmask_b32_e32 v0, v131, v0, vcc
	v_mul_f32_e32 v134, v154, v42
	v_mul_f32_e32 v43, v154, v43
	buffer_store_dwordx2 v[40:41], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v69, v166, 1
	v_cvt_pk_bf16_f32 v42, v132, v133
	v_cvt_pk_bf16_f32 v43, v134, v43
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[42:43], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v69, v158, 1
	v_cvt_pk_bf16_f32 v44, v135, v136
	v_cvt_pk_bf16_f32 v45, v137, v138
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[44:45], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v69, v157, 1
	v_cvt_pk_bf16_f32 v46, v139, v159
	v_cvt_pk_bf16_f32 v47, v160, v161
	v_mul_lo_u32 v70, v70, s15
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[46:47], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v70, v182, 1
	v_cvt_pk_bf16_f32 v48, v162, v163
	v_cvt_pk_bf16_f32 v49, v164, v165
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[48:49], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v70, v166, 1
	v_cvt_pk_bf16_f32 v50, v167, v168
	v_cvt_pk_bf16_f32 v51, v169, v170
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[50:51], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v70, v158, 1
	v_cvt_pk_bf16_f32 v52, v171, v172
	v_cvt_pk_bf16_f32 v53, v173, v174
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[52:53], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v70, v157, 1
	v_cvt_pk_bf16_f32 v54, v175, v176
	v_cvt_pk_bf16_f32 v55, v177, v178
	v_mul_lo_u32 v71, v71, s15
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[54:55], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v71, v182, 1
	v_cvt_pk_bf16_f32 v56, v179, v180
	v_cvt_pk_bf16_f32 v57, v181, v183
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[56:57], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v71, v166, 1
	v_cvt_pk_bf16_f32 v58, v184, v185
	v_cvt_pk_bf16_f32 v59, v186, v187
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[58:59], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v71, v158, 1
	v_cvt_pk_bf16_f32 v60, v188, v189
	v_cvt_pk_bf16_f32 v61, v190, v191
	v_cndmask_b32_e32 v0, v131, v0, vcc
	buffer_store_dwordx2 v[60:61], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v71, v157, 1
	s_addk_i32 s16, 0x100
	v_cvt_pk_bf16_f32 v62, v192, v193
	v_cvt_pk_bf16_f32 v63, v194, v195
	v_cndmask_b32_e32 v0, v131, v0, vcc
	s_cmp_lt_i32 s16, s17
	buffer_store_dwordx2 v[62:63], v0, s[20:23], 0 offen
	s_cbranch_scc0 .L9
.L3:
	v_cmp_ge_i32_e32 vcc, s16, v151
	s_and_b64 s[6:7], vcc, s[36:37]
	v_cndmask_b32_e64 v0, 0, 1, s[6:7]
	s_mov_b32 m0, s38
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
	v_cndmask_b32_e32 v0, 0, v151, vcc
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
	s_sub_i32 s51, s16, s7
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[22:23], s[6:7], 3
	s_add_u32 s22, s12, s22
	s_addc_u32 s23, s13, s23
	v_mov_b32_e32 v0, 0
	global_load_dwordx4 v[0:3], v0, s[22:23]
	s_abs_i32 s22, s51
	s_mul_hi_u32 s52, s22, s49
	s_mul_i32 s23, s52, s29
	s_sub_i32 s54, s22, s23
	s_ashr_i32 s7, s51, 31
	s_xor_b32 s7, s7, s48
	s_add_i32 s53, s52, 1
	s_sub_i32 s55, s54, s29
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v0
	v_readfirstlane_b32 s23, v2
	s_sub_i32 s23, s23, s22
	s_add_i32 s56, s23, 0xff
	s_ashr_i32 s57, s56, 31
	s_lshr_b32 s57, s57, 24
	s_add_i32 s56, s56, s57
	s_ashr_i32 s56, s56, 8
	s_cmp_ge_u32 s54, s29
	s_cselect_b32 s52, s53, s52
	s_cselect_b32 s53, s55, s54
	s_add_i32 s54, s52, 1
	s_cmp_ge_u32 s53, s29
	s_cselect_b32 s52, s54, s52
	s_xor_b32 s52, s52, s7
	s_sub_i32 s7, s52, s7
	s_mul_i32 s53, s7, 6
	s_sub_i32 s52, s56, s53
	s_min_i32 s52, s52, 6
	s_abs_i32 s54, s52
	v_cvt_f32_u32_e32 v0, s54
	s_sub_i32 s56, 0, s54
	s_mul_i32 s7, s7, s46
	s_sub_i32 s7, s51, s7
	v_rcp_iflag_f32_e32 v0, v0
	s_abs_i32 s51, s7
	s_xor_b32 s55, s7, s52
	s_ashr_i32 s55, s55, 31
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_nop 0
	v_readfirstlane_b32 s57, v0
	s_mul_i32 s56, s56, s57
	s_mul_hi_u32 s56, s57, s56
	s_add_i32 s57, s57, s56
	s_mul_hi_u32 s56, s51, s57
	s_mul_i32 s57, s56, s54
	s_sub_i32 s51, s51, s57
	s_add_i32 s58, s56, 1
	s_sub_i32 s57, s51, s54
	s_cmp_ge_u32 s51, s54
	s_cselect_b32 s56, s58, s56
	s_cselect_b32 s51, s57, s51
	s_add_i32 s57, s56, 1
	s_cmp_ge_u32 s51, s54
	s_cselect_b32 s54, s57, s56
	s_abs_i32 s51, s23
	v_cvt_f32_u32_e32 v0, s51
	s_xor_b32 s54, s54, s55
	s_sub_i32 s54, s54, s55
	s_mul_i32 s55, s54, s52
	s_lshl_b32 s52, s54, 8
	s_bfe_i32 s57, s54, 0x10017
	v_or_b32_e32 v1, s52, v142
	v_rcp_iflag_f32_e32 v0, v0
	v_add_u32_e32 v1, s57, v1
	v_xor_b32_e32 v1, s57, v1
	v_mul_hi_u32 v9, v1, v153
	v_mul_lo_u32 v9, v9, s28
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_sub_u32_e32 v1, v1, v9
	v_cvt_u32_f32_e32 v0, v0
	s_sub_i32 s7, s7, s55
	v_subrev_u32_e32 v9, s28, v1
	v_cmp_le_u32_e32 vcc, s28, v1
	s_add_i32 s7, s7, s53
	s_sub_i32 s56, 0, s51
	v_cndmask_b32_e32 v1, v1, v9, vcc
	s_lshl_b32 s53, s7, 8
	v_subrev_u32_e32 v9, s28, v1
	v_cmp_le_u32_e32 vcc, s28, v1
	s_bfe_i32 s54, s7, 0x10017
	v_or_b32_e32 v2, s53, v142
	v_cndmask_b32_e32 v1, v1, v9, vcc
	v_mul_lo_u32 v9, s56, v0
	v_or_b32_e32 v3, s53, v143
	v_or_b32_e32 v7, s53, v144
	v_add_u32_e32 v2, s54, v2
	v_xor_b32_e32 v10, s57, v1
	v_mul_hi_u32 v1, v0, v9
	v_or_b32_e32 v8, s53, v145
	v_add_u32_e32 v3, s54, v3
	v_add_u32_e32 v7, s54, v7
	v_xor_b32_e32 v2, s54, v2
	v_add_u32_e32 v156, v0, v1
	v_add_u32_e32 v8, s54, v8
	v_xor_b32_e32 v3, s54, v3
	v_xor_b32_e32 v7, s54, v7
	v_mul_hi_u32 v0, v2, v156
	v_xor_b32_e32 v8, s54, v8
	v_mul_hi_u32 v1, v3, v156
	v_mul_hi_u32 v9, v7, v156
	v_mul_lo_u32 v0, v0, s51
	v_mul_hi_u32 v11, v8, v156
	v_mul_lo_u32 v1, v1, s51
	v_mul_lo_u32 v9, v9, s51
	v_sub_u32_e32 v0, v2, v0
	v_mul_lo_u32 v11, v11, s51
	v_sub_u32_e32 v1, v3, v1
	v_sub_u32_e32 v2, v7, v9
	v_subrev_u32_e32 v7, s51, v0
	v_cmp_le_u32_e32 vcc, s51, v0
	v_sub_u32_e32 v3, v8, v11
	v_subrev_u32_e32 v8, s51, v1
	v_cndmask_b32_e32 v0, v0, v7, vcc
	v_cmp_le_u32_e32 vcc, s51, v1
	v_or_b32_e32 v4, s52, v143
	v_subrev_u32_e32 v9, s51, v2
	v_cndmask_b32_e32 v1, v1, v8, vcc
	v_cmp_le_u32_e32 vcc, s51, v2
	v_subrev_u32_e32 v11, s51, v3
	v_add_u32_e32 v4, s57, v4
	v_cndmask_b32_e32 v2, v2, v9, vcc
	v_cmp_le_u32_e32 vcc, s51, v3
	v_subrev_u32_e32 v7, s51, v0
	v_xor_b32_e32 v4, s57, v4
	v_cndmask_b32_e32 v3, v3, v11, vcc
	v_cmp_le_u32_e32 vcc, s51, v0
	v_subrev_u32_e32 v8, s51, v1
	v_mul_hi_u32 v12, v4, v153
	v_cndmask_b32_e32 v0, v0, v7, vcc
	v_cmp_le_u32_e32 vcc, s51, v1
	v_subrev_u32_e32 v9, s51, v2
	v_mul_lo_u32 v12, v12, s28
	v_cndmask_b32_e32 v1, v1, v8, vcc
	v_cmp_le_u32_e32 vcc, s51, v2
	v_subrev_u32_e32 v11, s51, v3
	v_sub_u32_e32 v4, v4, v12
	v_cndmask_b32_e32 v2, v2, v9, vcc
	v_cmp_le_u32_e32 vcc, s51, v3
	v_subrev_u32_e32 v12, s28, v4
	v_or_b32_e32 v5, s52, v144
	v_cndmask_b32_e32 v3, v3, v11, vcc
	v_cmp_le_u32_e32 vcc, s28, v4
	v_or_b32_e32 v6, s52, v145
	v_xor_b32_e32 v0, s54, v0
	v_cndmask_b32_e32 v4, v4, v12, vcc
	v_subrev_u32_e32 v12, s28, v4
	v_cmp_le_u32_e32 vcc, s28, v4
	v_xor_b32_e32 v1, s54, v1
	v_subrev_u32_e32 v7, s54, v0
	v_cndmask_b32_e32 v4, v4, v12, vcc
	v_xor_b32_e32 v4, s57, v4
	v_subrev_u32_e32 v12, s57, v4
	v_add_u32_e32 v4, s57, v5
	v_xor_b32_e32 v4, s57, v4
	v_mul_hi_u32 v5, v4, v153
	v_mul_lo_u32 v5, v5, s28
	v_sub_u32_e32 v4, v4, v5
	v_subrev_u32_e32 v5, s28, v4
	v_cmp_le_u32_e32 vcc, s28, v4
	v_xor_b32_e32 v2, s54, v2
	v_subrev_u32_e32 v8, s54, v1
	v_cndmask_b32_e32 v4, v4, v5, vcc
	v_subrev_u32_e32 v5, s28, v4
	v_cmp_le_u32_e32 vcc, s28, v4
	s_mul_i32 s55, s6, s31
	s_mul_i32 s6, s30, s22
	v_cndmask_b32_e32 v4, v4, v5, vcc
	v_xor_b32_e32 v4, s57, v4
	v_subrev_u32_e32 v13, s57, v4
	v_add_u32_e32 v4, s57, v6
	v_xor_b32_e32 v4, s57, v4
	v_mul_hi_u32 v5, v4, v153
	v_mul_lo_u32 v5, v5, s28
	v_sub_u32_e32 v4, v4, v5
	v_subrev_u32_e32 v5, s28, v4
	v_cmp_le_u32_e32 vcc, s28, v4
	v_subrev_u32_e32 v9, s54, v2
	v_subrev_u32_e32 v10, s57, v10
	v_cndmask_b32_e32 v4, v4, v5, vcc
	v_subrev_u32_e32 v5, s28, v4
	v_cmp_le_u32_e32 vcc, s28, v4
	v_mul_lo_u32 v6, v9, s30
	v_xor_b32_e32 v3, s54, v3
	v_cndmask_b32_e32 v4, v4, v5, vcc
	v_xor_b32_e32 v4, s57, v4
	v_subrev_u32_e32 v14, s57, v4
	v_mul_lo_u32 v4, v7, s30
	v_mul_lo_u32 v5, v8, s30
	v_add3_u32 v161, v4, v141, s6
	v_add3_u32 v162, v5, v141, s6
	v_mul_lo_u32 v4, v10, s14
	ds_bpermute_b32 v10, v149, v161
	v_lshrrev_b64 v[8:9], v128, exec
	ds_bpermute_b32 v9, v149, v162
	v_and_b32_e32 v8, 1, v8
	v_subrev_u32_e32 v11, s54, v3
	v_cmp_eq_u32_e32 vcc, 1, v8
	v_mul_lo_u32 v7, v11, s30
	v_add3_u32 v163, v6, v141, s6
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e32 v8, v131, v10, vcc
	v_add3_u32 v164, v7, v141, s6
	buffer_load_dwordx4 v8, s[24:27], 0 offen lds
	ds_bpermute_b32 v8, v149, v163
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e32 v9, v131, v9, vcc
	s_mov_b32 m0, s39
	v_add3_u32 v157, v4, v141, s55
	buffer_load_dwordx4 v9, s[24:27], 0 offen lds
	ds_bpermute_b32 v9, v149, v164
	ds_bpermute_b32 v10, v149, v157
	v_mul_lo_u32 v5, v12, s14
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e32 v8, v131, v8, vcc
	s_mov_b32 m0, s40
	v_mul_lo_u32 v6, v13, s14
	v_add3_u32 v158, v5, v141, s55
	buffer_load_dwordx4 v8, s[24:27], 0 offen lds
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e32 v8, v131, v9, vcc
	s_mov_b32 m0, s41
	v_mul_lo_u32 v7, v14, s14
	v_add3_u32 v159, v6, v141, s55
	buffer_load_dwordx4 v8, s[24:27], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e32 v8, v131, v10, vcc
	s_mov_b32 s6, s26
	ds_bpermute_b32 v9, v149, v158
	s_mov_b32 s7, s27
	s_mov_b32 m0, s42
	v_add3_u32 v160, v7, v141, s55
	buffer_load_dwordx4 v8, s[4:7], 0 offen lds
	ds_bpermute_b32 v8, v149, v159
	ds_bpermute_b32 v10, v149, v160
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e32 v9, v131, v9, vcc
	s_mov_b32 m0, s43
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e32 v8, v131, v8, vcc
	buffer_load_dwordx4 v9, s[4:7], 0 offen lds
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dwordx4 v8, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e32 v8, v131, v10, vcc
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dwordx4 v8, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .L4
	s_barrier
.L4:
	s_or_b64 exec, exec, s[6:7]
	s_andn2_b64 vcc, exec, s[10:11]
	s_cbranch_vccnz .L8
	v_add_u32_e32 v3, s22, v3
	v_add_u32_e32 v2, s22, v2
	v_add_u32_e32 v1, s22, v1
	v_add_u32_e32 v0, s22, v0
	v_add_u32_e32 v8, s55, v130
	v_subrev_u32_e32 v3, s54, v3
	v_subrev_u32_e32 v2, s54, v2
	v_subrev_u32_e32 v1, s54, v1
	v_subrev_u32_e32 v0, s54, v0
	v_mov_b32_e32 v124, 0
	v_add_u32_e32 v165, v8, v7
	v_add_u32_e32 v166, v8, v6
	v_add_u32_e32 v167, v8, v5
	v_add_u32_e32 v168, v8, v4
	v_mad_u64_u32 v[132:133], s[6:7], s30, v3, v[130:131]
	v_mad_u64_u32 v[134:135], s[6:7], s30, v2, v[130:131]
	v_mad_u64_u32 v[136:137], s[6:7], s30, v1, v[130:131]
	v_mad_u64_u32 v[138:139], s[6:7], s30, v0, v[130:131]
	s_mov_b32 s54, 0
	s_add_i32 s55, 0, 0x10000
	s_mov_b32 s56, 0
	s_mov_b32 s57, 0
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
	s_mov_b32 s58, s54
	v_add_u32_e32 v133, s56, v138
	s_add_i32 s54, s57, 1
	v_add_u32_e32 v135, s56, v136
	s_cmp_lt_i32 s54, 2
	ds_bpermute_b32 v133, v149, v133
	v_add_u32_e32 v137, s56, v134
	ds_bpermute_b32 v135, v149, v135
	s_cselect_b32 s57, s54, 0
	v_add_u32_e32 v139, s56, v132
	v_lshrrev_b64 v[170:171], v128, exec
	ds_bpermute_b32 v137, v149, v137
	s_lshl_b32 s54, s57, 15
	v_add_u32_e32 v169, s56, v168
	v_and_b32_e32 v170, 1, v170
	ds_bpermute_b32 v139, v149, v139
	s_add_i32 s54, s54, 0
	v_add_u32_e32 v171, s56, v167
	ds_bpermute_b32 v169, v149, v169
	v_cmp_eq_u32_e32 vcc, 1, v170
	s_add_i32 s60, s54, s33
	v_add_u32_e32 v172, s56, v166
	ds_bpermute_b32 v171, v149, v171
	s_waitcnt lgkmcnt(5)
	v_cndmask_b32_e32 v133, v131, v133, vcc
	s_mov_b32 m0, s60
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v173, s56, v165
	ds_bpermute_b32 v172, v149, v172
	v_cndmask_b32_e32 v135, v131, v135, vcc
	buffer_load_dwordx4 v133, s[24:27], 0 offen lds
	s_add_i32 m0, s60, 0x2000
	s_mov_b32 s59, s55
	ds_bpermute_b32 v173, v149, v173
	s_add_i32 s55, s54, 0x10000
	v_cndmask_b32_e32 v137, v131, v137, vcc
	buffer_load_dwordx4 v135, s[24:27], 0 offen lds
	s_add_i32 m0, s60, 0x4000
	v_cndmask_b32_e32 v139, v131, v139, vcc
	s_add_i32 s61, s55, s33
	buffer_load_dwordx4 v137, s[24:27], 0 offen lds
	s_add_i32 m0, s60, 0x6000
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	v_cndmask_b32_e32 v169, v131, v169, vcc
	buffer_load_dwordx4 v139, s[24:27], 0 offen lds
	s_mov_b32 m0, s61
	v_cndmask_b32_e32 v170, v131, v171, vcc
	buffer_load_dwordx4 v169, s[4:7], 0 offen lds
	s_add_i32 m0, s61, 0x2000
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e32 v171, v131, v172, vcc
	buffer_load_dwordx4 v170, s[4:7], 0 offen lds
	s_add_i32 m0, s61, 0x4000
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e32 v172, v131, v173, vcc
	buffer_load_dwordx4 v171, s[4:7], 0 offen lds
	s_add_i32 m0, s61, 0x6000
	s_nop 0
	buffer_load_dwordx4 v172, s[4:7], 0 offen lds
	v_add_u32_e32 v133, s59, v150
	v_add_u32_e32 v135, s59, v152
	v_add_u32_e32 v139, s58, v148
	v_add_u32_e32 v137, s58, v147
	s_barrier
	s_setprio 2
	ds_read_b128 v[170:173], v133
	ds_read_b128 v[174:177], v135
	ds_read_b128 v[198:201], v135 offset:16384
	ds_read_b128 v[206:209], v139
	ds_read_b128 v[202:205], v137
	ds_read_b128 v[210:213], v137 offset:4096
	ds_read_b128 v[214:217], v139 offset:4096
	ds_read_b128 v[178:181], v133 offset:8192
	ds_read_b128 v[182:185], v135 offset:8192
	ds_read_b128 v[194:197], v133 offset:16384
	ds_read_b128 v[186:189], v133 offset:24576
	ds_read_b128 v[190:193], v135 offset:24576
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x128_f8f6f4 v[120:123], v[178:185], v[202:209], v[120:123]
	v_mfma_f32_16x16x128_f8f6f4 v[124:127], v[170:177], v[202:209], v[124:127]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x128_f8f6f4 v[116:119], v[194:201], v[202:209], v[116:119]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x128_f8f6f4 v[112:115], v[186:193], v[202:209], v[112:115]
	v_mfma_f32_16x16x128_f8f6f4 v[108:111], v[170:177], v[210:217], v[108:111]
	v_mfma_f32_16x16x128_f8f6f4 v[104:107], v[178:185], v[210:217], v[104:107]
	v_mfma_f32_16x16x128_f8f6f4 v[100:103], v[194:201], v[210:217], v[100:103]
	v_mfma_f32_16x16x128_f8f6f4 v[96:99], v[186:193], v[210:217], v[96:99]
	ds_read_b128 v[206:209], v139 offset:8192
	ds_read_b128 v[202:205], v137 offset:8192
	ds_read_b128 v[210:213], v137 offset:12288
	ds_read_b128 v[214:217], v139 offset:12288
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x128_f8f6f4 v[92:95], v[170:177], v[202:209], v[92:95]
	v_mfma_f32_16x16x128_f8f6f4 v[88:91], v[178:185], v[202:209], v[88:91]
	v_mfma_f32_16x16x128_f8f6f4 v[84:87], v[194:201], v[202:209], v[84:87]
	v_mfma_f32_16x16x128_f8f6f4 v[80:83], v[186:193], v[202:209], v[80:83]
	ds_read_b128 v[202:205], v137 offset:16384
	ds_read_b128 v[218:221], v137 offset:20480
	ds_read_b128 v[226:229], v137 offset:24576
	ds_read_b128 v[234:237], v137 offset:28672
	ds_read_b128 v[206:209], v139 offset:16384
	ds_read_b128 v[222:225], v139 offset:20480
	ds_read_b128 v[230:233], v139 offset:24576
	ds_read_b128 v[238:241], v139 offset:28672
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x128_f8f6f4 v[76:79], v[170:177], v[210:217], v[76:79]
	v_mfma_f32_16x16x128_f8f6f4 v[72:75], v[178:185], v[210:217], v[72:75]
	v_mfma_f32_16x16x128_f8f6f4 v[68:71], v[194:201], v[210:217], v[68:71]
	v_mfma_f32_16x16x128_f8f6f4 v[64:67], v[186:193], v[210:217], v[64:67]
	s_setprio 1
	s_barrier
	s_setprio 2
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x128_f8f6f4 v[60:63], v[170:177], v[202:209], v[60:63]
	s_addk_i32 s56, 0x80
	s_cmp_lg_u32 s50, s56
	v_mfma_f32_16x16x128_f8f6f4 v[56:59], v[178:185], v[202:209], v[56:59]
	v_mfma_f32_16x16x128_f8f6f4 v[52:55], v[194:201], v[202:209], v[52:55]
	v_mfma_f32_16x16x128_f8f6f4 v[48:51], v[186:193], v[202:209], v[48:51]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x128_f8f6f4 v[44:47], v[170:177], v[218:225], v[44:47]
	v_mfma_f32_16x16x128_f8f6f4 v[40:43], v[178:185], v[218:225], v[40:43]
	v_mfma_f32_16x16x128_f8f6f4 v[36:39], v[194:201], v[218:225], v[36:39]
	v_mfma_f32_16x16x128_f8f6f4 v[32:35], v[186:193], v[218:225], v[32:35]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x128_f8f6f4 v[28:31], v[170:177], v[226:233], v[28:31]
	v_mfma_f32_16x16x128_f8f6f4 v[24:27], v[178:185], v[226:233], v[24:27]
	v_mfma_f32_16x16x128_f8f6f4 v[20:23], v[194:201], v[226:233], v[20:23]
	v_mfma_f32_16x16x128_f8f6f4 v[16:19], v[186:193], v[226:233], v[16:19]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x128_f8f6f4 v[12:15], v[170:177], v[234:241], v[12:15]
	v_mfma_f32_16x16x128_f8f6f4 v[8:11], v[178:185], v[234:241], v[8:11]
	v_mfma_f32_16x16x128_f8f6f4 v[4:7], v[194:201], v[234:241], v[4:7]
	v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[186:193], v[234:241], v[0:3]
	s_setprio 0
	s_cbranch_scc1 .L5
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .L7
.L6:
	s_barrier
.L7:
	s_or_b64 exec, exec, s[6:7]
	s_andn2_b64 vcc, exec, s[18:19]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .L2
	v_add_u32_e32 v165, s55, v152
	v_add_u32_e32 v174, s55, v150
	ds_read_b128 v[136:139], v165
	ds_read_b128 v[132:135], v174
	ds_read_b128 v[182:185], v174 offset:16384
	v_add_u32_e32 v190, s54, v148
	v_add_u32_e32 v191, s54, v147
	ds_read_b128 v[196:199], v190
	ds_read_b128 v[192:195], v191
	ds_read_b128 v[170:173], v165 offset:8192
	ds_read_b128 v[166:169], v174 offset:8192
	ds_read_b128 v[204:207], v190 offset:4096
	ds_read_b128 v[200:203], v191 offset:4096
	ds_read_b128 v[186:189], v165 offset:16384
	ds_read_b128 v[178:181], v165 offset:24576
	ds_read_b128 v[174:177], v174 offset:24576
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x128_f8f6f4 v[124:127], v[132:139], v[192:199], v[124:127]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x128_f8f6f4 v[120:123], v[166:173], v[192:199], v[120:123]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x128_f8f6f4 v[116:119], v[182:189], v[192:199], v[116:119]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x128_f8f6f4 v[112:115], v[174:181], v[192:199], v[112:115]
	v_mfma_f32_16x16x128_f8f6f4 v[108:111], v[132:139], v[200:207], v[108:111]
	v_mfma_f32_16x16x128_f8f6f4 v[104:107], v[166:173], v[200:207], v[104:107]
	v_mfma_f32_16x16x128_f8f6f4 v[100:103], v[182:189], v[200:207], v[100:103]
	v_mfma_f32_16x16x128_f8f6f4 v[96:99], v[174:181], v[200:207], v[96:99]
	ds_read_b128 v[192:195], v191 offset:8192
	ds_read_b128 v[196:199], v190 offset:8192
	ds_read_b128 v[204:207], v190 offset:12288
	ds_read_b128 v[200:203], v191 offset:12288
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x128_f8f6f4 v[92:95], v[132:139], v[192:199], v[92:95]
	v_mfma_f32_16x16x128_f8f6f4 v[88:91], v[166:173], v[192:199], v[88:91]
	v_mfma_f32_16x16x128_f8f6f4 v[84:87], v[182:189], v[192:199], v[84:87]
	v_mfma_f32_16x16x128_f8f6f4 v[80:83], v[174:181], v[192:199], v[80:83]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x128_f8f6f4 v[76:79], v[132:139], v[200:207], v[76:79]
	v_mfma_f32_16x16x128_f8f6f4 v[72:75], v[166:173], v[200:207], v[72:75]
	v_mfma_f32_16x16x128_f8f6f4 v[68:71], v[182:189], v[200:207], v[68:71]
	v_mfma_f32_16x16x128_f8f6f4 v[64:67], v[174:181], v[200:207], v[64:67]
	ds_read_b128 v[192:195], v191 offset:16384
	ds_read_b128 v[196:199], v190 offset:16384
	ds_read_b128 v[204:207], v190 offset:20480
	ds_read_b128 v[200:203], v191 offset:20480
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x128_f8f6f4 v[60:63], v[132:139], v[192:199], v[60:63]
	v_mfma_f32_16x16x128_f8f6f4 v[56:59], v[166:173], v[192:199], v[56:59]
	v_mfma_f32_16x16x128_f8f6f4 v[52:55], v[182:189], v[192:199], v[52:55]
	v_mfma_f32_16x16x128_f8f6f4 v[48:51], v[174:181], v[192:199], v[48:51]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x128_f8f6f4 v[44:47], v[132:139], v[200:207], v[44:47]
	v_mfma_f32_16x16x128_f8f6f4 v[40:43], v[166:173], v[200:207], v[40:43]
	v_mfma_f32_16x16x128_f8f6f4 v[36:39], v[182:189], v[200:207], v[36:39]
	v_mfma_f32_16x16x128_f8f6f4 v[32:35], v[174:181], v[200:207], v[32:35]
	ds_read_b128 v[192:195], v191 offset:24576
	ds_read_b128 v[196:199], v190 offset:24576
	ds_read_b128 v[204:207], v190 offset:28672
	ds_read_b128 v[200:203], v191 offset:28672
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x128_f8f6f4 v[28:31], v[132:139], v[192:199], v[28:31]
	v_mfma_f32_16x16x128_f8f6f4 v[24:27], v[166:173], v[192:199], v[24:27]
	v_mfma_f32_16x16x128_f8f6f4 v[20:23], v[182:189], v[192:199], v[20:23]
	v_mfma_f32_16x16x128_f8f6f4 v[16:19], v[174:181], v[192:199], v[16:19]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x128_f8f6f4 v[12:15], v[132:139], v[200:207], v[12:15]
	v_mfma_f32_16x16x128_f8f6f4 v[8:11], v[166:173], v[200:207], v[8:11]
	v_mfma_f32_16x16x128_f8f6f4 v[4:7], v[182:189], v[200:207], v[4:7]
	v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[174:181], v[200:207], v[0:3]
	s_branch .L2
.L8:
	v_mov_b32_e32 v3, 0
	s_mov_b32 s54, 0
	s_add_i32 s55, 0, 0x10000
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
	s_cbranch_execnz .L6
	s_branch .L7
.L9:
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
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
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
