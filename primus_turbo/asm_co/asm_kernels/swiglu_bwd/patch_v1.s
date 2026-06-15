	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	swiglu_with_mask_bwd_kernel     ; -- Begin function swiglu_with_mask_bwd_kernel
	.p2align	8
	.type	swiglu_with_mask_bwd_kernel,@function
swiglu_with_mask_bwd_kernel:            ; @swiglu_with_mask_bwd_kernel
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.13:
	.file	1 "/workspace/code/Primus-Turbo/primus_turbo/triton/activation" "swiglu_kernel.py"
	.loc	1 61 0 prologue_end             ; swiglu_kernel.py:61:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.14:
.LBB0_0:
	s_mov_b64 s[36:37], s[10:11]
	s_mov_b64 s[10:11], s[6:7]
	s_load_dword s6, s[0:1], 0x38
.Ltmp1:
	.loc	1 117 12 is_stmt 1              ; swiglu_kernel.py:117:12
	v_readfirstlane_b32 s7, v0
	.loc	1 88 44                         ; swiglu_kernel.py:88:44
	s_lshr_b32 s0, s15, 31
	.loc	1 133 12                        ; swiglu_kernel.py:133:12
	s_lshr_b32 s18, s7, 6
	.loc	1 88 44                         ; swiglu_kernel.py:88:44
	s_add_i32 s20, s15, s0
	.loc	1 93 31                         ; swiglu_kernel.py:93:31
	v_and_b32_e32 v1, 63, v0
	s_and_b32 s0, s7, 0xc0
	.loc	1 132 80                        ; swiglu_kernel.py:132:80
	s_waitcnt lgkmcnt(0)
	s_lshr_b32 s17, s6, 31
	s_bfe_u32 s7, s7, 0x20006
	.loc	1 93 31                         ; swiglu_kernel.py:93:31
	v_or_b32_e32 v2, s0, v1
	.loc	1 132 80                        ; swiglu_kernel.py:132:80
	s_add_i32 s21, s6, s17
	s_lshl2_add_u32 s33, s7, 0
	.loc	1 90 22                         ; swiglu_kernel.py:90:22
	s_mul_i32 s22, s16, s6
	s_lshl_b32 s7, s7, 10
	s_lshl_b32 s48, s6, 14
	s_mul_i32 s6, s16, s14
	.loc	1 93 31                         ; swiglu_kernel.py:93:31
	v_lshlrev_b32_e32 v2, 3, v2
	v_cmp_gt_u32_e64 s[24:25], 4, v0
	v_lshl_add_u32 v80, v0, 2, 0
	v_and_b32_e32 v0, 3, v0
	.loc	1 90 22                         ; swiglu_kernel.py:90:22
	s_lshl1_add_u32 s49, s6, s7
	s_mul_i32 s6, s16, s15
	s_mov_b64 s[40:41], s[2:3]
	.loc	1 88 44                         ; swiglu_kernel.py:88:44
	s_ashr_i32 s2, s20, 1
	.loc	1 93 31                         ; swiglu_kernel.py:93:31
	v_or_b32_e32 v3, 0x800, v2
	s_mov_b32 s43, 0x27000
	s_mov_b32 s42, 0x7ffffffe
	v_cmp_eq_u32_e32 vcc, 0, v0
	v_or_b32_e32 v0, s18, v1
	.loc	1 90 22                         ; swiglu_kernel.py:90:22
	s_lshl1_add_u32 s46, s22, s7
	s_and_b32 s21, s21, -2
	s_lshl1_add_u32 s51, s6, s7
	s_and_b32 s6, s20, -2
	.loc	1 94 29                         ; swiglu_kernel.py:94:29
	v_cmp_gt_i32_e64 s[0:1], s2, v2
	v_cmp_gt_i32_e64 s[2:3], s2, v3
	s_and_b32 s5, s5, 0xffff
	s_and_b32 s41, s41, 0xffff
	s_mov_b32 s17, 0
	v_cmp_eq_u32_e64 s[34:35], 0, v1
	s_and_b64 s[44:45], s[24:25], vcc
	v_cmp_eq_u32_e64 s[18:19], 0, v0
	s_and_b32 s37, s37, 0xffff
	.loc	1 90 22                         ; swiglu_kernel.py:90:22
	v_lshlrev_b32_e32 v81, 4, v1
	s_add_i32 s47, s46, s21
	s_lshl_b32 s50, s14, 14
	s_lshl_b32 s52, s15, 14
	s_add_i32 s53, s51, s6
	v_mov_b32_e32 v82, 0
	v_bfrev_b32_e32 v83, 1
	s_mov_b32 s54, 0x3fb8aa3b
	s_mov_b32 s55, 0xc2fc0000
	v_mov_b32_e32 v84, 0x42800000
	v_not_b32_e32 v85, 63
	s_mov_b32 s6, s42
	s_mov_b32 s7, s43
	s_branch .LBB0_2
.LBB0_1:                                ; %._crit_edge
                                        ;   in Loop: Header=BB0_2 Depth=1
	.loc	1 0 22 is_stmt 0                ; swiglu_kernel.py:0:22
	s_or_b64 exec, exec, s[26:27]
	.loc	1 121 24 is_stmt 1              ; swiglu_kernel.py:121:24
	s_lshl_b64 s[14:15], s[14:15], 2
	s_add_u32 s14, s10, s14
	s_addc_u32 s15, s11, s15
	global_load_dword v86, v82, s[14:15]
	.loc	1 125 49                        ; swiglu_kernel.py:125:49
	v_pk_add_f32 v[88:89], v[20:21], 1.0 op_sel_hi:[1,0] neg_lo:[1,0] neg_hi:[1,0]
	v_pk_add_f32 v[90:91], v[34:35], 1.0 op_sel_hi:[1,0] neg_lo:[1,0] neg_hi:[1,0]
	v_pk_add_f32 v[92:93], v[2:3], 1.0 op_sel_hi:[1,0] neg_lo:[1,0] neg_hi:[1,0]
	.loc	1 125 37 is_stmt 0              ; swiglu_kernel.py:125:37
	v_pk_fma_f32 v[28:29], v[88:89], v[28:29], 1.0 op_sel_hi:[1,1,0]
	.loc	1 125 49                        ; swiglu_kernel.py:125:49
	v_pk_add_f32 v[88:89], v[54:55], 1.0 op_sel_hi:[1,0] neg_lo:[1,0] neg_hi:[1,0]
	.loc	1 125 37                        ; swiglu_kernel.py:125:37
	v_pk_fma_f32 v[30:31], v[90:91], v[30:31], 1.0 op_sel_hi:[1,1,0]
	.loc	1 125 49                        ; swiglu_kernel.py:125:49
	v_pk_add_f32 v[90:91], v[56:57], 1.0 op_sel_hi:[1,0] neg_lo:[1,0] neg_hi:[1,0]
	.loc	1 125 37                        ; swiglu_kernel.py:125:37
	v_pk_fma_f32 v[38:39], v[92:93], v[38:39], 1.0 op_sel_hi:[1,1,0]
	.loc	1 125 49                        ; swiglu_kernel.py:125:49
	v_pk_add_f32 v[92:93], v[58:59], 1.0 op_sel_hi:[1,0] neg_lo:[1,0] neg_hi:[1,0]
	.loc	1 125 37                        ; swiglu_kernel.py:125:37
	v_pk_fma_f32 v[40:41], v[88:89], v[40:41], 1.0 op_sel_hi:[1,1,0]
	.loc	1 125 49                        ; swiglu_kernel.py:125:49
	v_pk_add_f32 v[88:89], v[60:61], 1.0 op_sel_hi:[1,0] neg_lo:[1,0] neg_hi:[1,0]
	.loc	1 125 37                        ; swiglu_kernel.py:125:37
	v_pk_fma_f32 v[44:45], v[90:91], v[44:45], 1.0 op_sel_hi:[1,1,0]
	.loc	1 125 49                        ; swiglu_kernel.py:125:49
	v_pk_add_f32 v[90:91], v[64:65], 1.0 op_sel_hi:[1,0] neg_lo:[1,0] neg_hi:[1,0]
	.loc	1 129 66 is_stmt 1              ; swiglu_kernel.py:129:66
	v_add_u32_e32 v87, s46, v81
	.loc	1 125 37                        ; swiglu_kernel.py:125:37
	v_pk_fma_f32 v[46:47], v[92:93], v[46:47], 1.0 op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[0:1], v[88:89], v[0:1], 1.0 op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[50:51], v[90:91], v[50:51], 1.0 op_sel_hi:[1,1,0]
	.loc	1 129 66                        ; swiglu_kernel.py:129:66
	v_cndmask_b32_e64 v88, v83, v87, s[22:23]
	v_add_u32_e32 v87, 0x1000, v87
	.loc	1 125 31                        ; swiglu_kernel.py:125:31
	v_pk_mul_f32 v[20:21], v[20:21], v[28:29]
	v_pk_mul_f32 v[28:29], v[34:35], v[30:31]
	v_pk_mul_f32 v[2:3], v[2:3], v[38:39]
	v_pk_mul_f32 v[30:31], v[54:55], v[40:41]
	v_pk_mul_f32 v[34:35], v[56:57], v[44:45]
	v_pk_mul_f32 v[38:39], v[58:59], v[46:47]
	v_pk_mul_f32 v[0:1], v[60:61], v[0:1]
	v_pk_mul_f32 v[40:41], v[64:65], v[50:51]
	.loc	1 126 48                        ; swiglu_kernel.py:126:48
	v_pk_mul_f32 v[20:21], v[20:21], v[24:25]
	v_pk_mul_f32 v[24:25], v[28:29], v[26:27]
	v_pk_mul_f32 v[2:3], v[2:3], v[12:13]
	v_pk_mul_f32 v[12:13], v[30:31], v[18:19]
	v_pk_mul_f32 v[14:15], v[34:35], v[14:15]
	v_pk_mul_f32 v[8:9], v[38:39], v[8:9]
	v_pk_mul_f32 v[0:1], v[0:1], v[10:11]
	v_pk_mul_f32 v[10:11], v[40:41], v[48:49]
	.loc	1 129 66                        ; swiglu_kernel.py:129:66
	s_mov_b32 s38, s42
	s_mov_b32 s39, s43
	.loc	1 133 12                        ; swiglu_kernel.py:133:12
	v_add_u32_e32 v92, s47, v81
	v_cndmask_b32_e64 v89, v83, v92, s[22:23]
	.loc	1 129 66                        ; swiglu_kernel.py:129:66
	v_cndmask_b32_e64 v44, v83, v87, s[20:21]
	.loc	1 90 22                         ; swiglu_kernel.py:90:22
	s_addk_i32 s17, 0x2000
	s_add_i32 s47, s47, s48
	s_add_i32 s46, s46, s48
	s_add_i32 s49, s49, s50
	s_add_i32 s51, s51, s52
	s_add_i32 s53, s53, s52
	s_cmp_lg_u32 s17, 0x20000
	.loc	1 123 41                        ; swiglu_kernel.py:123:41
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[16:17], v[86:87], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[86:87], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[86:87], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[86:87], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[86:87], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[86:87], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[86:87], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[86:87], v[52:53] op_sel_hi:[0,1]
	.loc	1 124 42                        ; swiglu_kernel.py:124:42
	v_pk_mul_f32 v[32:33], v[62:63], v[16:17]
	.loc	1 126 41                        ; swiglu_kernel.py:126:41
	v_pk_mul_f32 v[16:17], v[20:21], v[16:17]
	.loc	1 124 42                        ; swiglu_kernel.py:124:42
	v_pk_mul_f32 v[20:21], v[66:67], v[18:19]
	.loc	1 126 41                        ; swiglu_kernel.py:126:41
	v_pk_mul_f32 v[18:19], v[24:25], v[18:19]
	v_pk_mul_f32 v[2:3], v[2:3], v[26:27]
	v_pk_mul_f32 v[12:13], v[12:13], v[22:23]
	.loc	1 124 42                        ; swiglu_kernel.py:124:42
	v_pk_mul_f32 v[24:25], v[68:69], v[26:27]
	v_pk_mul_f32 v[26:27], v[70:71], v[22:23]
	.loc	1 126 41                        ; swiglu_kernel.py:126:41
	v_pk_mul_f32 v[14:15], v[14:15], v[28:29]
	v_pk_mul_f32 v[8:9], v[8:9], v[4:5]
	.loc	1 124 42                        ; swiglu_kernel.py:124:42
	v_pk_mul_f32 v[34:35], v[76:77], v[6:7]
	.loc	1 126 41                        ; swiglu_kernel.py:126:41
	v_pk_mul_f32 v[6:7], v[0:1], v[6:7]
	v_pk_mul_f32 v[10:11], v[10:11], v[30:31]
	.loc	1 129 77                        ; swiglu_kernel.py:129:77
	v_cvt_pk_bf16_f32 v0, v16, v17
	v_cvt_pk_bf16_f32 v1, v18, v19
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_cvt_pk_bf16_f32 v3, v12, v13
	.loc	1 124 42                        ; swiglu_kernel.py:124:42
	v_pk_mul_f32 v[22:23], v[72:73], v[28:29]
	v_pk_mul_f32 v[28:29], v[74:75], v[4:5]
	v_pk_mul_f32 v[36:37], v[78:79], v[30:31]
	.loc	1 129 77                        ; swiglu_kernel.py:129:77
	v_cvt_pk_bf16_f32 v4, v14, v15
	v_cvt_pk_bf16_f32 v5, v8, v9
	v_cvt_pk_bf16_f32 v6, v6, v7
	v_cvt_pk_bf16_f32 v7, v10, v11
	.loc	1 133 25                        ; swiglu_kernel.py:133:25
	v_cvt_pk_bf16_f32 v8, v32, v33
	v_cvt_pk_bf16_f32 v9, v20, v21
	v_cvt_pk_bf16_f32 v10, v24, v25
	v_cvt_pk_bf16_f32 v11, v26, v27
	.loc	1 129 66                        ; swiglu_kernel.py:129:66
	buffer_store_dwordx4 v[0:3], v88, s[36:39], 0 offen
	buffer_store_dwordx4 v[4:7], v44, s[36:39], 0 offen
	.loc	1 133 12                        ; swiglu_kernel.py:133:12
	buffer_store_dwordx4 v[8:11], v89, s[36:39], 0 offen
	v_add_u32_e32 v0, 0x1000, v92
	.loc	1 133 25 is_stmt 0              ; swiglu_kernel.py:133:25
	v_cvt_pk_bf16_f32 v12, v22, v23
	v_cvt_pk_bf16_f32 v13, v28, v29
	v_cvt_pk_bf16_f32 v14, v34, v35
	v_cvt_pk_bf16_f32 v15, v36, v37
	.loc	1 133 12                        ; swiglu_kernel.py:133:12
	v_cndmask_b32_e64 v0, v83, v0, s[20:21]
	buffer_store_dwordx4 v[12:15], v0, s[36:39], 0 offen
	.loc	1 90 22 is_stmt 1               ; swiglu_kernel.py:90:22
	s_cbranch_scc0 .LBB0_12
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	.loc	1 92 29                         ; swiglu_kernel.py:92:29
	s_add_i32 s14, s16, s17
	s_cmp_lt_i32 s14, 0x20000
	s_cselect_b64 s[38:39], -1, 0
	s_cmp_gt_i32 s14, 0x1ffff
	.loc	1 96 29                         ; swiglu_kernel.py:96:29
	v_mov_b64_e32 v[0:1], 0
	s_cbranch_scc1 .LBB0_4
; %bb.3:                                ;   in Loop: Header=BB0_2 Depth=1
	.loc	1 0 29 is_stmt 0                ; swiglu_kernel.py:0:29
	s_ashr_i32 s15, s14, 31
	s_lshl_b64 s[20:21], s[14:15], 3
	s_add_u32 s20, s8, s20
	s_addc_u32 s21, s9, s21
	.loc	1 96 29                         ; swiglu_kernel.py:96:29
	global_load_dwordx2 v[0:1], v82, s[20:21]
.LBB0_4:                                ;   in Loop: Header=BB0_2 Depth=1
	.loc	1 97 27 is_stmt 1               ; swiglu_kernel.py:97:27
	s_and_b64 s[20:21], s[0:1], s[38:39]
	.loc	1 97 54 is_stmt 0               ; swiglu_kernel.py:97:54
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, 0, v[0:1]
	.loc	1 102 21 is_stmt 1              ; swiglu_kernel.py:102:21
	v_add_u32_e32 v0, s51, v81
	.loc	1 97 40                         ; swiglu_kernel.py:97:40
	s_and_b64 s[22:23], s[20:21], vcc
	.loc	1 102 21                        ; swiglu_kernel.py:102:21
	v_cndmask_b32_e64 v1, v83, v0, s[22:23]
	buffer_load_dwordx4 v[16:19], v1, s[4:7], 0 offen
	.loc	1 103 23                        ; swiglu_kernel.py:103:23
	v_add_u32_e32 v1, s53, v81
	v_cndmask_b32_e64 v2, v83, v1, s[22:23]
	buffer_load_dwordx4 v[20:23], v2, s[4:7], 0 offen
	.loc	1 108 27                        ; swiglu_kernel.py:108:27
	v_add_u32_e32 v2, s49, v81
	v_cndmask_b32_e64 v3, v83, v2, s[22:23]
	buffer_load_dwordx4 v[12:15], v3, s[40:43], 0 offen
	.loc	1 97 27                         ; swiglu_kernel.py:97:27
	s_and_b64 s[20:21], s[2:3], s[38:39]
	.loc	1 102 21                        ; swiglu_kernel.py:102:21
	v_add_u32_e32 v0, 0x1000, v0
	.loc	1 97 40                         ; swiglu_kernel.py:97:40
	s_and_b64 s[20:21], s[20:21], vcc
	.loc	1 103 23                        ; swiglu_kernel.py:103:23
	v_add_u32_e32 v1, 0x1000, v1
	.loc	1 108 27                        ; swiglu_kernel.py:108:27
	v_add_u32_e32 v2, 0x1000, v2
	.loc	1 102 21                        ; swiglu_kernel.py:102:21
	v_cndmask_b32_e64 v0, v83, v0, s[20:21]
	.loc	1 103 23                        ; swiglu_kernel.py:103:23
	v_cndmask_b32_e64 v24, v83, v1, s[20:21]
	.loc	1 108 27                        ; swiglu_kernel.py:108:27
	v_cndmask_b32_e64 v25, v83, v2, s[20:21]
	.loc	1 102 21                        ; swiglu_kernel.py:102:21
	buffer_load_dwordx4 v[8:11], v0, s[4:7], 0 offen
	.loc	1 103 23                        ; swiglu_kernel.py:103:23
	buffer_load_dwordx4 v[4:7], v24, s[4:7], 0 offen
	.loc	1 108 27                        ; swiglu_kernel.py:108:27
	s_nop 0
	buffer_load_dwordx4 v[0:3], v25, s[40:43], 0 offen
.Ltmp2:
	.file	2 "/opt/venv/lib/python3.12/site-packages/triton/language" "standard.py"
	.loc	2 293 36                        ; standard.py:293:36 @[ swiglu_kernel.py:113:32 ]
	s_waitcnt lgkmcnt(0)
	s_barrier
.Ltmp3:
	.loc	1 102 53                        ; swiglu_kernel.py:102:53
	s_waitcnt vmcnt(5)
	v_and_b32_e32 v29, 0xffff0000, v16
	v_lshlrev_b32_e32 v28, 16, v16
.Ltmp4:
	.loc	2 50 29                         ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_mul_f32_e64 v32, -v29, s54
.Ltmp5:
	.loc	1 103 57                        ; swiglu_kernel.py:103:57
	s_waitcnt vmcnt(4)
	v_and_b32_e32 v27, 0xffff0000, v21
	v_lshlrev_b32_e32 v26, 16, v21
.Ltmp6:
	.loc	2 50 29                         ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_mul_f32_e64 v21, -v28, s54
	v_cmp_gt_f32_e32 vcc, s55, v21
	v_cmp_gt_f32_e64 s[26:27], s55, v32
.Ltmp7:
	.loc	1 102 53                        ; swiglu_kernel.py:102:53
	v_and_b32_e32 v31, 0xffff0000, v17
	v_lshlrev_b32_e32 v30, 16, v17
	.loc	1 103 57                        ; swiglu_kernel.py:103:57
	v_and_b32_e32 v25, 0xffff0000, v20
	v_lshlrev_b32_e32 v24, 16, v20
	.loc	1 109 12                        ; swiglu_kernel.py:109:12
	s_waitcnt vmcnt(3)
	v_and_b32_e32 v17, 0xffff0000, v12
	v_lshlrev_b32_e32 v16, 16, v12
.Ltmp8:
	.loc	2 50 30                         ; standard.py:50:30 @[ swiglu_kernel.py:105:29 ]
	v_sub_f32_e32 v12, 0, v28
	v_sub_f32_e32 v20, 0, v29
	.loc	2 50 29 is_stmt 0               ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_cndmask_b32_e32 v21, 0, v84, vcc
	v_cndmask_b32_e64 v32, 0, v84, s[26:27]
	v_mul_f32_e64 v35, -v30, s54
	v_fmac_f32_e32 v21, 0x3fb8aa3b, v12
	v_fmac_f32_e32 v32, 0x3fb8aa3b, v20
	v_mul_f32_e64 v36, -v31, s54
	v_cmp_gt_f32_e64 s[28:29], s55, v35
	v_exp_f32_e32 v12, v21
	v_exp_f32_e32 v21, v32
	.loc	2 50 30                         ; standard.py:50:30 @[ swiglu_kernel.py:105:29 ]
	v_sub_f32_e32 v33, 0, v30
	.loc	2 50 29                         ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_cndmask_b32_e64 v35, 0, v84, s[28:29]
	v_cmp_gt_f32_e64 s[30:31], s55, v36
	.loc	2 50 30                         ; standard.py:50:30 @[ swiglu_kernel.py:105:29 ]
	v_sub_f32_e32 v34, 0, v31
	.loc	2 50 29                         ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_fmac_f32_e32 v35, 0x3fb8aa3b, v33
	v_cndmask_b32_e64 v36, 0, v84, s[30:31]
	v_cndmask_b32_e32 v37, 0, v85, vcc
	v_cndmask_b32_e64 v38, 0, v85, s[26:27]
	v_fmac_f32_e32 v36, 0x3fb8aa3b, v34
	v_exp_f32_e32 v32, v35
	v_exp_f32_e32 v33, v36
	v_ldexp_f32 v20, v12, v37
	v_ldexp_f32 v21, v21, v38
	.loc	2 50 20                         ; standard.py:50:20 @[ swiglu_kernel.py:105:29 ]
	v_pk_add_f32 v[20:21], v[20:21], 1.0 op_sel_hi:[1,0]
	.loc	2 50 29                         ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_cndmask_b32_e64 v39, 0, v85, s[28:29]
	.loc	2 50 16                         ; standard.py:50:16 @[ swiglu_kernel.py:105:29 ]
	v_div_scale_f32 v12, s[26:27], v21, v21, 1.0
	.loc	2 50 29                         ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_cndmask_b32_e64 v40, 0, v85, s[30:31]
	v_ldexp_f32 v32, v32, v39
	.loc	2 50 16                         ; standard.py:50:16 @[ swiglu_kernel.py:105:29 ]
	v_div_scale_f32 v35, s[26:27], v20, v20, 1.0
	v_rcp_f32_e32 v39, v12
	.loc	2 50 29                         ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_ldexp_f32 v33, v33, v40
	.loc	2 50 16                         ; standard.py:50:16 @[ swiglu_kernel.py:105:29 ]
	v_rcp_f32_e32 v40, v35
	.loc	2 50 20                         ; standard.py:50:20 @[ swiglu_kernel.py:105:29 ]
	v_pk_add_f32 v[32:33], v[32:33], 1.0 op_sel_hi:[1,0]
	.loc	2 50 16                         ; standard.py:50:16 @[ swiglu_kernel.py:105:29 ]
	v_fma_f32 v43, -v12, v39, 1.0
	v_div_scale_f32 v37, s[28:29], v33, v33, 1.0
	v_div_scale_f32 v34, vcc, 1.0, v21, 1.0
	v_rcp_f32_e32 v41, v37
	v_fma_f32 v44, -v35, v40, 1.0
	v_fmac_f32_e32 v39, v43, v39
	v_div_scale_f32 v36, s[26:27], 1.0, v20, 1.0
	v_fmac_f32_e32 v40, v44, v40
	v_mul_f32_e32 v43, v34, v39
	v_mul_f32_e32 v44, v36, v40
	v_fma_f32 v46, -v12, v43, v34
	v_fma_f32 v47, -v35, v44, v36
	v_fmac_f32_e32 v43, v46, v39
	v_fma_f32 v45, -v37, v41, 1.0
	v_fmac_f32_e32 v44, v47, v40
	v_fma_f32 v12, -v12, v43, v34
	v_div_scale_f32 v38, s[28:29], 1.0, v33, 1.0
	v_fmac_f32_e32 v41, v45, v41
	v_fma_f32 v34, -v35, v44, v36
	v_div_fmas_f32 v12, v12, v39, v43
	s_mov_b64 vcc, s[26:27]
	v_div_scale_f32 v42, s[30:31], v32, v32, 1.0
	v_mul_f32_e32 v45, v38, v41
	v_div_fixup_f32 v21, v12, v21, 1.0
	v_div_fmas_f32 v12, v34, v40, v44
	v_fma_f32 v48, -v37, v45, v38
	v_div_fixup_f32 v20, v12, v20, 1.0
	v_rcp_f32_e32 v12, v42
	v_fmac_f32_e32 v45, v48, v41
	v_fma_f32 v35, -v37, v45, v38
	s_mov_b64 vcc, s[28:29]
	v_div_fmas_f32 v34, v35, v41, v45
	v_div_fixup_f32 v35, v34, v33, 1.0
	v_fma_f32 v33, -v42, v12, 1.0
	v_fmac_f32_e32 v12, v33, v12
	v_div_scale_f32 v33, vcc, 1.0, v32, 1.0
	v_mul_f32_e32 v34, v33, v12
	v_fma_f32 v36, -v42, v34, v33
	v_fmac_f32_e32 v34, v36, v12
.Ltmp9:
	.loc	1 102 53 is_stmt 1              ; swiglu_kernel.py:102:53
	v_lshlrev_b32_e32 v38, 16, v18
.Ltmp10:
	.loc	2 50 16                         ; standard.py:50:16 @[ swiglu_kernel.py:105:29 ]
	v_fma_f32 v33, -v42, v34, v33
	.loc	2 50 29 is_stmt 0               ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_mul_f32_e64 v53, -v38, s54
	.loc	2 50 16                         ; standard.py:50:16 @[ swiglu_kernel.py:105:29 ]
	v_div_fmas_f32 v12, v33, v12, v34
	.loc	2 50 29                         ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_cmp_gt_f32_e32 vcc, s55, v53
	.loc	2 50 16                         ; standard.py:50:16 @[ swiglu_kernel.py:105:29 ]
	v_div_fixup_f32 v34, v12, v32, 1.0
.Ltmp11:
	.loc	1 109 12 is_stmt 1              ; swiglu_kernel.py:109:12
	v_and_b32_e32 v33, 0xffff0000, v13
	v_lshlrev_b32_e32 v32, 16, v13
	.loc	1 102 53                        ; swiglu_kernel.py:102:53
	v_and_b32_e32 v39, 0xffff0000, v18
	.loc	1 103 57                        ; swiglu_kernel.py:103:57
	v_and_b32_e32 v13, 0xffff0000, v22
	v_lshlrev_b32_e32 v12, 16, v22
	.loc	1 109 12                        ; swiglu_kernel.py:109:12
	v_and_b32_e32 v37, 0xffff0000, v14
	v_lshlrev_b32_e32 v36, 16, v14
	.loc	1 102 53                        ; swiglu_kernel.py:102:53
	v_and_b32_e32 v41, 0xffff0000, v19
	v_lshlrev_b32_e32 v40, 16, v19
	.loc	1 103 57                        ; swiglu_kernel.py:103:57
	v_and_b32_e32 v19, 0xffff0000, v23
	v_lshlrev_b32_e32 v18, 16, v23
	.loc	1 109 12                        ; swiglu_kernel.py:109:12
	v_and_b32_e32 v23, 0xffff0000, v15
	v_lshlrev_b32_e32 v22, 16, v15
	.loc	1 102 53                        ; swiglu_kernel.py:102:53
	s_waitcnt vmcnt(2)
	v_and_b32_e32 v45, 0xffff0000, v8
	v_lshlrev_b32_e32 v44, 16, v8
	.loc	1 103 57                        ; swiglu_kernel.py:103:57
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v15, 0xffff0000, v4
	v_lshlrev_b32_e32 v14, 16, v4
	.loc	1 109 12                        ; swiglu_kernel.py:109:12
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v43, 0xffff0000, v0
	v_lshlrev_b32_e32 v42, 16, v0
	.loc	1 102 53                        ; swiglu_kernel.py:102:53
	v_and_b32_e32 v47, 0xffff0000, v9
	v_lshlrev_b32_e32 v46, 16, v9
	.loc	1 103 57                        ; swiglu_kernel.py:103:57
	v_and_b32_e32 v9, 0xffff0000, v5
	v_lshlrev_b32_e32 v8, 16, v5
	.loc	1 109 12                        ; swiglu_kernel.py:109:12
	v_and_b32_e32 v5, 0xffff0000, v1
	v_lshlrev_b32_e32 v4, 16, v1
	.loc	1 102 53                        ; swiglu_kernel.py:102:53
	v_and_b32_e32 v1, 0xffff0000, v10
	v_lshlrev_b32_e32 v0, 16, v10
	v_and_b32_e32 v51, 0xffff0000, v11
	v_lshlrev_b32_e32 v50, 16, v11
	.loc	1 103 57                        ; swiglu_kernel.py:103:57
	v_and_b32_e32 v11, 0xffff0000, v6
	v_lshlrev_b32_e32 v10, 16, v6
.Ltmp12:
	.loc	2 50 30                         ; standard.py:50:30 @[ swiglu_kernel.py:105:29 ]
	v_sub_f32_e32 v6, 0, v38
	.loc	2 50 29 is_stmt 0               ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_cndmask_b32_e32 v53, 0, v84, vcc
	v_fmac_f32_e32 v53, 0x3fb8aa3b, v6
	v_exp_f32_e32 v6, v53
	v_mul_f32_e64 v53, -v39, s54
	v_cmp_gt_f32_e64 s[26:27], s55, v53
	.loc	2 50 30                         ; standard.py:50:30 @[ swiglu_kernel.py:105:29 ]
	v_sub_f32_e32 v52, 0, v39
.Ltmp13:
	.loc	1 103 57 is_stmt 1              ; swiglu_kernel.py:103:57
	v_and_b32_e32 v49, 0xffff0000, v7
.Ltmp14:
	.loc	2 50 29                         ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_cndmask_b32_e64 v53, 0, v84, s[26:27]
	v_fmac_f32_e32 v53, 0x3fb8aa3b, v52
	v_exp_f32_e32 v53, v53
	v_cndmask_b32_e32 v52, 0, v85, vcc
	v_ldexp_f32 v52, v6, v52
	v_cndmask_b32_e64 v6, 0, v85, s[26:27]
	v_ldexp_f32 v53, v53, v6
	.loc	2 50 20 is_stmt 0               ; standard.py:50:20 @[ swiglu_kernel.py:105:29 ]
	v_pk_add_f32 v[54:55], v[52:53], 1.0 op_sel_hi:[1,0]
.Ltmp15:
	.loc	1 103 57 is_stmt 1              ; swiglu_kernel.py:103:57
	v_lshlrev_b32_e32 v48, 16, v7
.Ltmp16:
	.loc	2 50 16                         ; standard.py:50:16 @[ swiglu_kernel.py:105:29 ]
	v_div_scale_f32 v56, s[26:27], v55, v55, 1.0
	v_rcp_f32_e32 v57, v56
.Ltmp17:
	.loc	1 109 12                        ; swiglu_kernel.py:109:12
	v_and_b32_e32 v7, 0xffff0000, v2
	v_lshlrev_b32_e32 v6, 16, v2
	v_and_b32_e32 v53, 0xffff0000, v3
.Ltmp18:
	.loc	2 50 16                         ; standard.py:50:16 @[ swiglu_kernel.py:105:29 ]
	v_fma_f32 v2, -v56, v57, 1.0
	v_fmac_f32_e32 v57, v2, v57
	v_div_scale_f32 v2, vcc, 1.0, v55, 1.0
.Ltmp19:
	.loc	1 109 12                        ; swiglu_kernel.py:109:12
	v_lshlrev_b32_e32 v52, 16, v3
.Ltmp20:
	.loc	2 50 16                         ; standard.py:50:16 @[ swiglu_kernel.py:105:29 ]
	v_mul_f32_e32 v3, v2, v57
	v_fma_f32 v58, -v56, v3, v2
	v_fmac_f32_e32 v3, v58, v57
	v_div_scale_f32 v58, s[26:27], v54, v54, 1.0
	v_rcp_f32_e32 v59, v58
	v_fma_f32 v2, -v56, v3, v2
	v_div_fmas_f32 v60, v2, v57, v3
	.loc	2 50 30 is_stmt 0               ; standard.py:50:30 @[ swiglu_kernel.py:105:29 ]
	v_sub_f32_e32 v56, 0, v41
	.loc	2 50 16                         ; standard.py:50:16 @[ swiglu_kernel.py:105:29 ]
	v_fma_f32 v2, -v58, v59, 1.0
	v_fmac_f32_e32 v59, v2, v59
	v_div_scale_f32 v2, vcc, 1.0, v54, 1.0
	v_mul_f32_e32 v61, v2, v59
	v_fma_f32 v3, -v58, v61, v2
	v_fmac_f32_e32 v61, v3, v59
	.loc	2 50 29                         ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_mul_f32_e64 v3, -v40, s54
	v_cmp_gt_f32_e64 s[26:27], s55, v3
	.loc	2 50 16                         ; standard.py:50:16 @[ swiglu_kernel.py:105:29 ]
	v_fma_f32 v58, -v58, v61, v2
	.loc	2 50 30                         ; standard.py:50:30 @[ swiglu_kernel.py:105:29 ]
	v_sub_f32_e32 v2, 0, v40
	.loc	2 50 29                         ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_cndmask_b32_e64 v3, 0, v84, s[26:27]
	v_fmac_f32_e32 v3, 0x3fb8aa3b, v2
	v_mul_f32_e64 v2, -v41, s54
	v_cmp_gt_f32_e64 s[28:29], s55, v2
	v_exp_f32_e32 v3, v3
	s_nop 0
	v_cndmask_b32_e64 v2, 0, v84, s[28:29]
	v_fmac_f32_e32 v2, 0x3fb8aa3b, v56
	v_exp_f32_e32 v56, v2
	v_cndmask_b32_e64 v2, 0, v85, s[26:27]
	v_ldexp_f32 v2, v3, v2
	v_cndmask_b32_e64 v3, 0, v85, s[28:29]
	v_ldexp_f32 v3, v56, v3
	.loc	2 50 20                         ; standard.py:50:20 @[ swiglu_kernel.py:105:29 ]
	v_pk_add_f32 v[56:57], v[2:3], 1.0 op_sel_hi:[1,0]
	.loc	2 50 16                         ; standard.py:50:16 @[ swiglu_kernel.py:105:29 ]
	v_div_fmas_f32 v2, v58, v59, v61
	v_div_scale_f32 v62, s[26:27], v57, v57, 1.0
	v_rcp_f32_e32 v63, v62
	v_div_fixup_f32 v2, v2, v54, 1.0
	v_div_fixup_f32 v3, v60, v55, 1.0
	v_fma_f32 v54, -v62, v63, 1.0
	v_fmac_f32_e32 v63, v54, v63
	v_div_scale_f32 v54, vcc, 1.0, v57, 1.0
	v_mul_f32_e32 v55, v54, v63
	v_fma_f32 v58, -v62, v55, v54
	v_fmac_f32_e32 v55, v58, v63
	v_div_scale_f32 v58, s[26:27], v56, v56, 1.0
	v_rcp_f32_e32 v60, v58
	v_fma_f32 v54, -v62, v55, v54
	v_div_fmas_f32 v61, v54, v63, v55
	v_fma_f32 v54, -v58, v60, 1.0
	v_fmac_f32_e32 v60, v54, v60
	v_div_scale_f32 v54, vcc, 1.0, v56, 1.0
	v_mul_f32_e32 v62, v54, v60
	v_fma_f32 v55, -v58, v62, v54
	v_fmac_f32_e32 v62, v55, v60
	v_fma_f32 v63, -v58, v62, v54
	.loc	2 50 29                         ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_mul_f32_e64 v54, -v44, s54
	v_cmp_gt_f32_e64 s[26:27], s55, v54
	.loc	2 50 30                         ; standard.py:50:30 @[ swiglu_kernel.py:105:29 ]
	v_sub_f32_e32 v55, 0, v44
	v_sub_f32_e32 v58, 0, v45
	.loc	2 50 29                         ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_cndmask_b32_e64 v54, 0, v84, s[26:27]
	v_fmac_f32_e32 v54, 0x3fb8aa3b, v55
	v_mul_f32_e64 v55, -v45, s54
	v_cmp_gt_f32_e64 s[28:29], s55, v55
	v_exp_f32_e32 v54, v54
	s_nop 0
	v_cndmask_b32_e64 v55, 0, v84, s[28:29]
	v_fmac_f32_e32 v55, 0x3fb8aa3b, v58
	v_exp_f32_e32 v55, v55
	v_cndmask_b32_e64 v58, 0, v85, s[26:27]
	v_ldexp_f32 v54, v54, v58
	v_cndmask_b32_e64 v58, 0, v85, s[28:29]
	v_ldexp_f32 v55, v55, v58
	.loc	2 50 20                         ; standard.py:50:20 @[ swiglu_kernel.py:105:29 ]
	v_pk_add_f32 v[58:59], v[54:55], 1.0 op_sel_hi:[1,0]
	.loc	2 50 16                         ; standard.py:50:16 @[ swiglu_kernel.py:105:29 ]
	v_div_fmas_f32 v54, v63, v60, v62
	v_div_scale_f32 v64, s[26:27], v59, v59, 1.0
	v_rcp_f32_e32 v65, v64
	v_div_fixup_f32 v54, v54, v56, 1.0
	v_div_fixup_f32 v55, v61, v57, 1.0
	v_fma_f32 v56, -v64, v65, 1.0
	v_fmac_f32_e32 v65, v56, v65
	v_div_scale_f32 v56, vcc, 1.0, v59, 1.0
	v_mul_f32_e32 v57, v56, v65
	v_fma_f32 v60, -v64, v57, v56
	v_fmac_f32_e32 v57, v60, v65
	v_div_scale_f32 v60, s[26:27], v58, v58, 1.0
	v_rcp_f32_e32 v62, v60
	v_fma_f32 v56, -v64, v57, v56
	v_div_fmas_f32 v63, v56, v65, v57
	v_fma_f32 v56, -v60, v62, 1.0
	v_fmac_f32_e32 v62, v56, v62
	v_div_scale_f32 v56, vcc, 1.0, v58, 1.0
	v_mul_f32_e32 v64, v56, v62
	v_fma_f32 v57, -v60, v64, v56
	v_fmac_f32_e32 v64, v57, v62
	v_fma_f32 v65, -v60, v64, v56
	.loc	2 50 29                         ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_mul_f32_e64 v56, -v46, s54
	v_cmp_gt_f32_e64 s[26:27], s55, v56
	.loc	2 50 30                         ; standard.py:50:30 @[ swiglu_kernel.py:105:29 ]
	v_sub_f32_e32 v57, 0, v46
	v_sub_f32_e32 v60, 0, v47
	.loc	2 50 29                         ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_cndmask_b32_e64 v56, 0, v84, s[26:27]
	v_fmac_f32_e32 v56, 0x3fb8aa3b, v57
	v_mul_f32_e64 v57, -v47, s54
	v_cmp_gt_f32_e64 s[28:29], s55, v57
	v_exp_f32_e32 v56, v56
	s_nop 0
	v_cndmask_b32_e64 v57, 0, v84, s[28:29]
	v_fmac_f32_e32 v57, 0x3fb8aa3b, v60
	v_exp_f32_e32 v57, v57
	v_cndmask_b32_e64 v60, 0, v85, s[26:27]
	v_ldexp_f32 v56, v56, v60
	v_cndmask_b32_e64 v60, 0, v85, s[28:29]
	v_ldexp_f32 v57, v57, v60
	.loc	2 50 20                         ; standard.py:50:20 @[ swiglu_kernel.py:105:29 ]
	v_pk_add_f32 v[60:61], v[56:57], 1.0 op_sel_hi:[1,0]
	.loc	2 50 16                         ; standard.py:50:16 @[ swiglu_kernel.py:105:29 ]
	v_div_fmas_f32 v56, v65, v62, v64
	v_div_scale_f32 v66, s[26:27], v61, v61, 1.0
	v_rcp_f32_e32 v67, v66
	v_div_fixup_f32 v56, v56, v58, 1.0
	v_div_fixup_f32 v57, v63, v59, 1.0
	v_fma_f32 v58, -v66, v67, 1.0
	v_fmac_f32_e32 v67, v58, v67
	v_div_scale_f32 v58, vcc, 1.0, v61, 1.0
	v_mul_f32_e32 v59, v58, v67
	v_fma_f32 v62, -v66, v59, v58
	v_fmac_f32_e32 v59, v62, v67
	v_div_scale_f32 v62, s[26:27], v60, v60, 1.0
	v_rcp_f32_e32 v64, v62
	v_fma_f32 v58, -v66, v59, v58
	v_div_fmas_f32 v65, v58, v67, v59
	v_fma_f32 v58, -v62, v64, 1.0
	v_fmac_f32_e32 v64, v58, v64
	v_div_scale_f32 v58, vcc, 1.0, v60, 1.0
	v_mul_f32_e32 v66, v58, v64
	v_fma_f32 v59, -v62, v66, v58
	v_fmac_f32_e32 v66, v59, v64
	v_fma_f32 v67, -v62, v66, v58
	.loc	2 50 29                         ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_mul_f32_e64 v58, -v0, s54
	v_cmp_gt_f32_e64 s[26:27], s55, v58
	.loc	2 50 30                         ; standard.py:50:30 @[ swiglu_kernel.py:105:29 ]
	v_sub_f32_e32 v59, 0, v0
	v_sub_f32_e32 v62, 0, v1
	.loc	2 50 29                         ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_cndmask_b32_e64 v58, 0, v84, s[26:27]
	v_fmac_f32_e32 v58, 0x3fb8aa3b, v59
	v_mul_f32_e64 v59, -v1, s54
	v_cmp_gt_f32_e64 s[28:29], s55, v59
	v_exp_f32_e32 v58, v58
	s_nop 0
	v_cndmask_b32_e64 v59, 0, v84, s[28:29]
	v_fmac_f32_e32 v59, 0x3fb8aa3b, v62
	v_exp_f32_e32 v59, v59
	v_cndmask_b32_e64 v62, 0, v85, s[26:27]
	v_ldexp_f32 v58, v58, v62
	v_cndmask_b32_e64 v62, 0, v85, s[28:29]
	v_ldexp_f32 v59, v59, v62
	.loc	2 50 20                         ; standard.py:50:20 @[ swiglu_kernel.py:105:29 ]
	v_pk_add_f32 v[62:63], v[58:59], 1.0 op_sel_hi:[1,0]
	.loc	2 50 16                         ; standard.py:50:16 @[ swiglu_kernel.py:105:29 ]
	v_div_fmas_f32 v58, v67, v64, v66
	v_div_scale_f32 v68, s[26:27], v63, v63, 1.0
	v_rcp_f32_e32 v69, v68
	v_div_fixup_f32 v58, v58, v60, 1.0
	v_div_fixup_f32 v59, v65, v61, 1.0
	v_fma_f32 v60, -v68, v69, 1.0
	v_fmac_f32_e32 v69, v60, v69
	v_div_scale_f32 v60, vcc, 1.0, v63, 1.0
	v_mul_f32_e32 v61, v60, v69
	v_fma_f32 v64, -v68, v61, v60
	v_fmac_f32_e32 v61, v64, v69
	v_div_scale_f32 v64, s[26:27], v62, v62, 1.0
	v_rcp_f32_e32 v66, v64
	v_fma_f32 v60, -v68, v61, v60
	v_div_fmas_f32 v67, v60, v69, v61
	v_fma_f32 v60, -v64, v66, 1.0
	v_fmac_f32_e32 v66, v60, v66
	v_div_scale_f32 v60, vcc, 1.0, v62, 1.0
	v_mul_f32_e32 v68, v60, v66
	v_fma_f32 v61, -v64, v68, v60
	v_fmac_f32_e32 v68, v61, v66
	v_fma_f32 v69, -v64, v68, v60
	.loc	2 50 29                         ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_mul_f32_e64 v60, -v50, s54
	v_cmp_gt_f32_e64 s[26:27], s55, v60
	.loc	2 50 30                         ; standard.py:50:30 @[ swiglu_kernel.py:105:29 ]
	v_sub_f32_e32 v61, 0, v50
	v_sub_f32_e32 v64, 0, v51
	.loc	2 50 29                         ; standard.py:50:29 @[ swiglu_kernel.py:105:29 ]
	v_cndmask_b32_e64 v60, 0, v84, s[26:27]
	v_fmac_f32_e32 v60, 0x3fb8aa3b, v61
	v_mul_f32_e64 v61, -v51, s54
	v_cmp_gt_f32_e64 s[28:29], s55, v61
	v_exp_f32_e32 v60, v60
	s_nop 0
	v_cndmask_b32_e64 v61, 0, v84, s[28:29]
	v_fmac_f32_e32 v61, 0x3fb8aa3b, v64
	v_exp_f32_e32 v61, v61
	v_cndmask_b32_e64 v64, 0, v85, s[26:27]
	v_ldexp_f32 v60, v60, v64
	v_cndmask_b32_e64 v64, 0, v85, s[28:29]
	v_ldexp_f32 v61, v61, v64
	.loc	2 50 20                         ; standard.py:50:20 @[ swiglu_kernel.py:105:29 ]
	v_pk_add_f32 v[64:65], v[60:61], 1.0 op_sel_hi:[1,0]
	.loc	2 50 16                         ; standard.py:50:16 @[ swiglu_kernel.py:105:29 ]
	v_div_fmas_f32 v60, v69, v66, v68
	v_div_scale_f32 v70, s[26:27], v65, v65, 1.0
	v_rcp_f32_e32 v71, v70
	v_div_fixup_f32 v60, v60, v62, 1.0
	v_div_fixup_f32 v61, v67, v63, 1.0
	v_fma_f32 v62, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v62, v71
	v_div_scale_f32 v62, vcc, 1.0, v65, 1.0
	v_mul_f32_e32 v63, v62, v71
	v_fma_f32 v66, -v70, v63, v62
	v_fmac_f32_e32 v63, v66, v71
	v_div_scale_f32 v66, s[26:27], v64, v64, 1.0
	v_rcp_f32_e32 v67, v66
	v_fma_f32 v62, -v70, v63, v62
	v_div_fmas_f32 v62, v62, v71, v63
	v_div_fixup_f32 v65, v62, v65, 1.0
	v_fma_f32 v63, -v66, v67, 1.0
	v_fmac_f32_e32 v67, v63, v67
	v_div_scale_f32 v63, vcc, 1.0, v64, 1.0
	v_mul_f32_e32 v68, v63, v67
	v_fma_f32 v69, -v66, v68, v63
	v_fmac_f32_e32 v68, v69, v67
	v_fma_f32 v63, -v66, v68, v63
	v_div_fmas_f32 v63, v63, v67, v68
	v_div_fixup_f32 v64, v63, v64, 1.0
.Ltmp21:
	.loc	1 106 25 is_stmt 1              ; swiglu_kernel.py:106:25
	v_pk_mul_f32 v[62:63], v[20:21], v[28:29]
	.loc	1 112 32                        ; swiglu_kernel.py:112:32
	v_pk_mul_f32 v[66:67], v[62:63], v[16:17]
	.loc	1 112 39 is_stmt 0              ; swiglu_kernel.py:112:39
	v_pk_mul_f32 v[66:67], v[66:67], v[24:25]
.Ltmp22:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ swiglu_kernel.py:113:32 ] ]
	v_add_f32_e32 v70, v66, v67
.Ltmp23:
	.loc	1 106 25                        ; swiglu_kernel.py:106:25
	v_pk_mul_f32 v[66:67], v[34:35], v[30:31]
	.loc	1 112 32                        ; swiglu_kernel.py:112:32
	v_pk_mul_f32 v[68:69], v[66:67], v[32:33]
	.loc	1 112 39 is_stmt 0              ; swiglu_kernel.py:112:39
	v_pk_mul_f32 v[68:69], v[68:69], v[26:27]
.Ltmp24:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ swiglu_kernel.py:113:32 ] ]
	v_add_f32_e32 v68, v68, v70
	v_add_f32_e32 v72, v69, v68
.Ltmp25:
	.loc	1 106 25                        ; swiglu_kernel.py:106:25
	v_pk_mul_f32 v[68:69], v[2:3], v[38:39]
	.loc	1 112 32                        ; swiglu_kernel.py:112:32
	v_pk_mul_f32 v[70:71], v[68:69], v[36:37]
	.loc	1 112 39 is_stmt 0              ; swiglu_kernel.py:112:39
	v_pk_mul_f32 v[70:71], v[70:71], v[12:13]
.Ltmp26:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ swiglu_kernel.py:113:32 ] ]
	v_add_f32_e32 v70, v70, v72
	v_add_f32_e32 v74, v71, v70
.Ltmp27:
	.loc	1 106 25                        ; swiglu_kernel.py:106:25
	v_pk_mul_f32 v[70:71], v[54:55], v[40:41]
	.loc	1 112 32                        ; swiglu_kernel.py:112:32
	v_pk_mul_f32 v[72:73], v[70:71], v[22:23]
	.loc	1 112 39 is_stmt 0              ; swiglu_kernel.py:112:39
	v_pk_mul_f32 v[72:73], v[72:73], v[18:19]
.Ltmp28:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ swiglu_kernel.py:113:32 ] ]
	v_add_f32_e32 v72, v72, v74
	v_add_f32_e32 v76, v73, v72
.Ltmp29:
	.loc	1 106 25                        ; swiglu_kernel.py:106:25
	v_pk_mul_f32 v[72:73], v[56:57], v[44:45]
	.loc	1 112 32                        ; swiglu_kernel.py:112:32
	v_pk_mul_f32 v[74:75], v[72:73], v[42:43]
	.loc	1 112 39 is_stmt 0              ; swiglu_kernel.py:112:39
	v_pk_mul_f32 v[74:75], v[74:75], v[14:15]
.Ltmp30:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ swiglu_kernel.py:113:32 ] ]
	v_add_f32_e32 v74, v74, v76
	v_add_f32_e32 v78, v75, v74
.Ltmp31:
	.loc	1 106 25                        ; swiglu_kernel.py:106:25
	v_pk_mul_f32 v[74:75], v[58:59], v[46:47]
	.loc	1 112 32                        ; swiglu_kernel.py:112:32
	v_pk_mul_f32 v[76:77], v[74:75], v[4:5]
	.loc	1 112 39 is_stmt 0              ; swiglu_kernel.py:112:39
	v_pk_mul_f32 v[76:77], v[76:77], v[8:9]
.Ltmp32:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ swiglu_kernel.py:113:32 ] ]
	v_add_f32_e32 v76, v76, v78
	v_add_f32_e32 v86, v77, v76
.Ltmp33:
	.loc	1 106 25                        ; swiglu_kernel.py:106:25
	v_pk_mul_f32 v[76:77], v[60:61], v[0:1]
	.loc	1 112 32                        ; swiglu_kernel.py:112:32
	v_pk_mul_f32 v[78:79], v[76:77], v[6:7]
	.loc	1 112 39 is_stmt 0              ; swiglu_kernel.py:112:39
	v_pk_mul_f32 v[78:79], v[78:79], v[10:11]
.Ltmp34:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ swiglu_kernel.py:113:32 ] ]
	v_add_f32_e32 v78, v78, v86
	v_add_f32_e32 v88, v79, v78
.Ltmp35:
	.loc	1 106 25                        ; swiglu_kernel.py:106:25
	v_pk_mul_f32 v[78:79], v[64:65], v[50:51]
	.loc	1 112 32                        ; swiglu_kernel.py:112:32
	v_pk_mul_f32 v[86:87], v[78:79], v[52:53]
	.loc	1 112 39 is_stmt 0              ; swiglu_kernel.py:112:39
	v_pk_mul_f32 v[86:87], v[86:87], v[48:49]
.Ltmp36:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ swiglu_kernel.py:113:32 ] ]
	v_add_f32_e32 v86, v86, v88
	v_add_f32_e32 v86, v87, v86
	s_nop 1
	v_add_f32_dpp v86, v86, v86 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v86, v86, v86 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v86, v86, v86 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v86, v86, v86 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp37:
	.loc	2 293 36                        ; standard.py:293:36 @[ swiglu_kernel.py:113:32 ]
	v_mov_b32_e32 v87, v86
	s_nop 1
	v_mov_b32_dpp v87, v87 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:1
.Ltmp38:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ swiglu_kernel.py:113:32 ] ]
	v_add_f32_e32 v86, v87, v86
	s_nop 1
	v_add_f32_dpp v86, v86, v86 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp39:
	.loc	2 293 36                        ; standard.py:293:36 @[ swiglu_kernel.py:113:32 ]
	s_nop 1
	v_readlane_b32 s15, v86, 63
	s_and_saveexec_b64 s[26:27], s[34:35]
; %bb.5:                                ;   in Loop: Header=BB0_2 Depth=1
	v_mov_b32_e32 v86, s33
	v_mov_b32_e32 v87, s15
	ds_write_b32 v86, v87
.Ltmp40:
; %bb.6:                                ;   in Loop: Header=BB0_2 Depth=1
	.loc	2 0 36 is_stmt 0                ; standard.py:0:36
	s_or_b64 exec, exec, s[26:27]
	.loc	1 113 32 is_stmt 1              ; swiglu_kernel.py:113:32
	v_mov_b32_e32 v86, 0
.Ltmp41:
	.loc	2 293 36                        ; standard.py:293:36 @[ swiglu_kernel.py:113:32 ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[26:27], s[24:25]
; %bb.7:                                ;   in Loop: Header=BB0_2 Depth=1
	ds_read_b32 v86, v80
; %bb.8:                                ;   in Loop: Header=BB0_2 Depth=1
	.loc	2 0 36 is_stmt 0                ; standard.py:0:36
	s_or_b64 exec, exec, s[26:27]
	.loc	2 293 36                        ; standard.py:293:36 @[ swiglu_kernel.py:113:32 ]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v87, v86
	s_nop 1
	v_mov_b32_dpp v87, v87 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp42:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ swiglu_kernel.py:113:32 ] ]
	v_add_f32_e32 v86, v86, v87
.Ltmp43:
	.loc	2 293 36                        ; standard.py:293:36 @[ swiglu_kernel.py:113:32 ]
	v_mov_b32_e32 v87, v86
	s_nop 1
	v_mov_b32_dpp v87, v87 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
	s_and_saveexec_b64 s[26:27], s[44:45]
; %bb.9:                                ;   in Loop: Header=BB0_2 Depth=1
	.loc	2 0 36 is_stmt 0                ; standard.py:0:36
	v_add_f32_e32 v86, v86, v87
	.loc	2 293 36                        ; standard.py:293:36 @[ swiglu_kernel.py:113:32 ]
	ds_write_b32 v80, v86
.Ltmp44:
; %bb.10:                               ;   in Loop: Header=BB0_2 Depth=1
	.loc	2 0 36                          ; standard.py:0:36
	s_or_b64 exec, exec, s[26:27]
	.loc	1 117 12 is_stmt 1              ; swiglu_kernel.py:117:12
	s_and_b64 s[28:29], s[18:19], s[38:39]
	s_ashr_i32 s15, s14, 31
.Ltmp45:
	.loc	2 293 36                        ; standard.py:293:36 @[ swiglu_kernel.py:113:32 ]
	s_waitcnt lgkmcnt(0)
	s_barrier
.Ltmp46:
	.loc	1 117 12                        ; swiglu_kernel.py:117:12
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB0_1
; %bb.11:                               ;   in Loop: Header=BB0_2 Depth=1
	ds_read_b32 v86, v82
	s_lshl_b64 s[28:29], s[14:15], 2
	s_add_u32 s28, s12, s28
	s_addc_u32 s29, s13, s29
	s_waitcnt lgkmcnt(0)
	global_store_dword v82, v86, s[28:29]
	s_branch .LBB0_1
.LBB0_12:
	.loc	1 90 4                          ; swiglu_kernel.py:90:4
	s_endpgm
.Ltmp47:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel swiglu_with_mask_bwd_kernel
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 80
		.amdhsa_user_sgpr_count 16
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 14
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 94
		.amdhsa_next_free_sgpr 56
		.amdhsa_accum_offset 96
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	swiglu_with_mask_bwd_kernel, .Lfunc_end0-swiglu_with_mask_bwd_kernel
	.cfi_endproc
                                        ; -- End function
	.set swiglu_with_mask_bwd_kernel.num_vgpr, 94
	.set swiglu_with_mask_bwd_kernel.num_agpr, 0
	.set swiglu_with_mask_bwd_kernel.numbered_sgpr, 56
	.set swiglu_with_mask_bwd_kernel.num_named_barrier, 0
	.set swiglu_with_mask_bwd_kernel.private_seg_size, 0
	.set swiglu_with_mask_bwd_kernel.uses_vcc, 1
	.set swiglu_with_mask_bwd_kernel.uses_flat_scratch, 0
	.set swiglu_with_mask_bwd_kernel.has_dyn_sized_stack, 0
	.set swiglu_with_mask_bwd_kernel.has_recursion, 0
	.set swiglu_with_mask_bwd_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4568
; TotalNumSgprs: 62
; NumVgprs: 94
; NumAgprs: 0
; TotalNumVgprs: 94
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 11
; NumSGPRsForWavesPerEU: 62
; NumVGPRsForWavesPerEU: 94
; AccumOffset: 96
; Occupancy: 5
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 23
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	14                              ; DW_FORM_strp
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	14                              ; DW_FORM_strp
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	32                              ; DW_AT_inline
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	6                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x5e DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.long	.Linfo_string2                  ; DW_AT_comp_dir
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.byte	2                               ; Abbrev [2] 0x2a:0x6 DW_TAG_subprogram
	.long	.Linfo_string3                  ; DW_AT_name
	.byte	1                               ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x30:0x38 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	113                             ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x4d:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	293                             ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x5b:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	105                             ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp39-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"swiglu_kernel.py"              ; string offset=7
.Linfo_string2:
	.asciz	"/workspace/code/Primus-Turbo/primus_turbo/triton/activation" ; string offset=24
.Linfo_string3:
	.asciz	"swiglu_with_mask_bwd_kernel"   ; string offset=84
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 80
    .max_flat_workgroup_size: 256
    .name:           swiglu_with_mask_bwd_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     62
    .sgpr_spill_count: 0
    .symbol:         swiglu_with_mask_bwd_kernel.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     94
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
