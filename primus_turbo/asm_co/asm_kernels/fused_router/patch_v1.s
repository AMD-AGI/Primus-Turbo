	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	fused_scaling_group_sum_routing_kernel ; -- Begin function fused_scaling_group_sum_routing_kernel
	.p2align	8
	.type	fused_scaling_group_sum_routing_kernel,@function
fused_scaling_group_sum_routing_kernel: ; @fused_scaling_group_sum_routing_kernel
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.8:
	.file	1 "/workspace/code/Primus-Turbo/primus_turbo/triton/moe" "fused_router_kernel.py"
	.loc	1 14 0 prologue_end             ; fused_router_kernel.py:14:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx2 s[12:13], s[0:1], 0x28
	s_load_dword s14, s[0:1], 0x30
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.9:
.LBB0_0:
	s_mov_b64 s[24:25], s[2:3]
.Ltmp1:
	.loc	1 130 44 is_stmt 1              ; fused_router_kernel.py:130:44
	v_readfirstlane_b32 s2, v0
	s_mov_b64 s[36:37], s[6:7]
	s_lshr_b32 s6, s2, 6
	.loc	1 56 52                         ; fused_router_kernel.py:56:52
	s_cmp_lt_i32 s15, 0x8000
	.loc	1 35 31                         ; fused_router_kernel.py:35:31
	v_bfe_u32 v25, v0, 1, 5
	.loc	1 56 52                         ; fused_router_kernel.py:56:52
	s_cselect_b64 vcc, -1, 0
	.loc	1 58 58                         ; fused_router_kernel.py:58:58
	s_lshl_b32 s50, s15, 5
	.loc	1 35 31                         ; fused_router_kernel.py:35:31
	v_and_b32_e32 v26, 31, v0
	.loc	1 58 62                         ; fused_router_kernel.py:58:62
	v_or_b32_e32 v2, s50, v25
	.loc	1 67 38                         ; fused_router_kernel.py:67:38
	v_bfrev_b32_e32 v1, 1
	v_lshlrev_b32_e32 v2, 2, v2
	.loc	1 58 62                         ; fused_router_kernel.py:58:62
	v_or_b32_e32 v3, s50, v26
	.loc	1 67 38                         ; fused_router_kernel.py:67:38
	s_and_b32 s25, s25, 0xffff
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	v_cndmask_b32_e32 v2, v1, v2, vcc
	v_lshlrev_b32_e32 v34, 2, v3
	v_cndmask_b32_e32 v1, v1, v34, vcc
	buffer_load_dword v9, v2, s[24:27], 0 offen
	buffer_load_dword v28, v1, s[24:27], 0 offen
	.loc	1 102 55                        ; fused_router_kernel.py:102:55
	v_lshlrev_b32_e32 v23, 3, v25
	ds_bpermute_b32 v22, v23, v25
	.loc	1 33 31                         ; fused_router_kernel.py:33:31
	s_load_dword s33, s[0:1], 0x48
	.loc	1 35 31                         ; fused_router_kernel.py:35:31
	v_lshrrev_b32_e32 v1, 1, v0
.Ltmp2:
	.file	2 "/workspace/code/Primus-Turbo/primus_turbo/triton/utils" "argsort.py"
	.loc	2 68 58                         ; argsort.py:68:58 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ]
	v_bfe_u32 v8, v0, 1, 1
.Ltmp3:
	.loc	2 42 57                         ; argsort.py:42:57 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_bitop3_b32 v10, v1, 1, v1 bitop3:0xc
	.loc	2 44 54                         ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v24, v22, v10
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v27, v22, v8
.Ltmp4:
	.loc	2 68 58                         ; argsort.py:68:58 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ]
	v_lshrrev_b32_e32 v2, 2, v0
	v_lshrrev_b32_e32 v3, 3, v0
	v_lshrrev_b32_e32 v4, 4, v0
.Ltmp5:
	.file	3 "/opt/venv/lib/python3.12/site-packages/triton/language" "standard.py"
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v32, v24
.Ltmp6:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v35, v27
.Ltmp7:
	.loc	1 67 38 is_stmt 1               ; fused_router_kernel.py:67:38
	v_mov_b32_e32 v33, 0xff800000
.Ltmp8:
	.loc	2 68 58                         ; argsort.py:68:58 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ]
	v_lshrrev_b32_e32 v5, 5, v0
	v_bfe_u32 v12, v0, 2, 1
	v_bfe_u32 v14, v0, 3, 1
	v_bfe_u32 v16, v0, 4, 1
	v_and_b32_e32 v30, 32, v0
	v_and_b32_e32 v11, 4, v0
	v_and_b32_e32 v20, 8, v0
	v_and_b32_e32 v21, 16, v0
.Ltmp9:
	.loc	2 42 57                         ; argsort.py:42:57 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_bitop3_b32 v13, v2, 1, v2 bitop3:0xc
	v_bitop3_b32 v15, v3, 1, v3 bitop3:0xc
	v_bitop3_b32 v18, v4, 1, v4 bitop3:0xc
	s_and_b32 s5, s5, 0xffff
.Ltmp10:
	.loc	1 56 52                         ; fused_router_kernel.py:56:52
	s_sub_i32 s51, 0x8000, s33
.Ltmp11:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v32, v32 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp12:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v35, v35 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
	s_mov_b32 s17, s15
	s_mov_b32 s16, s14
	s_mov_b64 s[20:21], s[10:11]
.Ltmp13:
	.loc	2 68 58 is_stmt 1               ; argsort.py:68:58 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ]
	v_bfe_u32 v17, v0, 5, 1
	v_lshlrev_b32_e32 v29, 3, v0
.Ltmp14:
	.loc	2 42 57                         ; argsort.py:42:57 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_bitop3_b32 v19, v5, 1, v5 bitop3:0xc
	.loc	2 42 53 is_stmt 0               ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cvt_f32_ubyte0_e32 v3, v12
	v_cvt_f32_ubyte0_e32 v5, v14
	v_cvt_f32_ubyte0_e32 v7, v16
	v_cvt_f32_ubyte0_e32 v1, v8
	v_cmp_ne_u32_e64 s[18:19], 0, v11
	v_cmp_ne_u32_e64 s[30:31], 0, v20
	v_cmp_ne_u32_e64 s[34:35], 0, v21
	v_cmp_ne_u32_e64 s[44:45], 0, v30
	v_lshrrev_b32_e32 v31, 3, v30
	v_cvt_f32_ubyte0_e32 v2, v13
	v_cvt_f32_ubyte0_e32 v4, v15
	v_cvt_f32_ubyte0_e32 v6, v18
	v_cvt_f32_ubyte0_e32 v0, v10
.Ltmp15:
	.loc	1 56 52 is_stmt 1               ; fused_router_kernel.py:56:52
	s_cmp_lt_i32 s15, s51
.Ltmp16:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v24, v24, v32
.Ltmp17:
	.loc	3 263 15 is_stmt 0              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v32, v35, v27
.Ltmp18:
	.loc	1 67 38 is_stmt 1               ; fused_router_kernel.py:67:38
	s_waitcnt vmcnt(1)
	v_cndmask_b32_e32 v27, v33, v9, vcc
	s_waitcnt vmcnt(0)
	v_cndmask_b32_e32 v28, v33, v28, vcc
	v_lshlrev_b32_e32 v9, 3, v26
	.loc	1 56 52                         ; fused_router_kernel.py:56:52
	s_cbranch_scc1 .LBB0_2
; %bb.1:                                ; %.._crit_edge_crit_edge
	.loc	1 0 52 is_stmt 0                ; fused_router_kernel.py:0:52
	s_movk_i32 s0, 0xf8
.Ltmp19:
	.loc	2 50 53 is_stmt 1               ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ne_u32_e64 s[48:49], 0, v11
	v_cmp_ne_u32_e64 s[46:47], 0, v20
	v_cmp_ne_u32_e64 s[38:39], 0, v21
	v_cmp_ne_u32_e64 s[14:15], 0, v30
.Ltmp20:
	.loc	1 120 46                        ; fused_router_kernel.py:120:46
	v_and_or_b32 v11, v29, s0, v31
	.loc	1 124 53                        ; fused_router_kernel.py:124:53
	s_and_b32 s29, s37, 0xffff
	s_mov_b32 s28, s36
	.loc	1 129 38                        ; fused_router_kernel.py:129:38
	s_and_b32 s9, s21, 0xffff
	s_mov_b32 s8, s20
	.loc	1 130 44                        ; fused_router_kernel.py:130:44
	s_and_b32 s41, s13, 0xffff
	s_mov_b32 s40, s12
	s_mov_b64 s[2:3], 0
	s_mov_b64 s[22:23], s[26:27]
	s_mov_b64 s[10:11], s[26:27]
	s_mov_b64 s[42:43], s[26:27]
	s_branch .LBB0_3
.LBB0_2:
	.loc	1 0 44 is_stmt 0                ; fused_router_kernel.py:0:44
	s_mov_b64 s[2:3], -1
                                        ; implicit-def: $sgpr14_sgpr15
                                        ; implicit-def: $sgpr38_sgpr39
                                        ; implicit-def: $sgpr46_sgpr47
                                        ; implicit-def: $sgpr48_sgpr49
                                        ; implicit-def: $vgpr11
                                        ; implicit-def: $sgpr28_sgpr29
                                        ; implicit-def: $sgpr8_sgpr9
                                        ; implicit-def: $sgpr40_sgpr41
.LBB0_3:                                ; %Flow7
	v_and_or_b32 v30, s6, 3, v30
	v_cmp_gt_u32_e64 s[0:1], 4, v26
	v_cvt_f32_ubyte0_e32 v20, v19
	v_cvt_f32_ubyte0_e32 v21, v17
	v_xor_b32_e32 v24, v32, v24
	s_andn2_b64 vcc, exec, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], 0, v30
	s_cbranch_vccnz .LBB0_7
; %bb.4:                                ; %.lr.ph
	s_movk_i32 s6, 0xf8
	s_and_b32 s37, s37, 0xffff
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	v_and_or_b32 v11, v29, s6, v31
	s_mov_b64 s[28:29], s[36:37]
	s_and_b64 s[8:9], s[0:1], s[2:3]
	s_and_b32 s21, s21, 0xffff
	s_and_b32 s13, s13, 0xffff
	.loc	1 56 52 is_stmt 1               ; fused_router_kernel.py:56:52
	v_add_u32_e32 v29, s50, v9
	s_lshl_b32 s40, s33, 5
	s_mov_b32 s41, 0xc2fc0000
	v_mov_b32_e32 v30, 0x42800000
	v_not_b32_e32 v31, 63
	v_bfrev_b32_e32 v32, 1
	v_mov_b32_e32 v33, 1
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	s_mov_b32 s38, s22
	s_mov_b32 s39, s23
	s_mov_b32 s42, s17
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	s_add_i32 s42, s42, s33
	s_mov_b32 s43, s50
	.loc	1 58 58                         ; fused_router_kernel.py:58:58
	s_lshl_b32 s50, s42, 5
	.loc	1 58 62 is_stmt 0               ; fused_router_kernel.py:58:62
	v_or_b32_e32 v37, s50, v25
.Ltmp21:
	.loc	3 191 40 is_stmt 1              ; standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ]
	ds_swizzle_b32 v35, v28 offset:swizzle(SWAP,16)
.Ltmp22:
	.loc	1 71 66                         ; fused_router_kernel.py:71:66
	v_or_b32_e32 v34, s43, v26
	.loc	1 58 62                         ; fused_router_kernel.py:58:62
	v_or_b32_e32 v38, s50, v26
	.loc	1 67 38                         ; fused_router_kernel.py:67:38
	v_lshlrev_b32_e32 v41, 2, v37
	.loc	1 72 40                         ; fused_router_kernel.py:72:40
	v_lshlrev_b32_e32 v40, 2, v34
	.loc	1 67 38                         ; fused_router_kernel.py:67:38
	v_lshlrev_b32_e32 v34, 2, v38
	buffer_load_dword v37, v41, s[24:27], 0 offen
	buffer_load_dword v38, v34, s[24:27], 0 offen
.Ltmp23:
	.loc	3 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ] ]
	v_max_f32_e32 v36, v28, v28
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v35, v35, v35
	v_max_f32_e32 v35, v36, v35
.Ltmp24:
	.loc	3 191 40                        ; standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_e32 v36, v35
.Ltmp25:
	.loc	1 124 53                        ; fused_router_kernel.py:124:53
	v_cndmask_b32_e64 v39, v32, v29, s[8:9]
	.loc	1 56 52                         ; fused_router_kernel.py:56:52
	v_add_u32_e32 v29, s40, v29
.Ltmp26:
	.loc	3 191 40                        ; standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_dpp v36, v36 row_shr:8 row_mask:0xf bank_mask:0xc
.Ltmp27:
	.loc	1 72 40                         ; fused_router_kernel.py:72:40
	v_cndmask_b32_e64 v40, v32, v40, s[2:3]
	.loc	1 130 44                        ; fused_router_kernel.py:130:44
	s_mov_b32 s14, s22
.Ltmp28:
	.loc	3 191 40                        ; standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_dpp v36, v35 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp29:
	.loc	3 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ] ]
	v_max_f32_e32 v36, v36, v36
	v_max_f32_e32 v35, v35, v36
.Ltmp30:
	.loc	3 191 40                        ; standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_e32 v36, v35
.Ltmp31:
	.loc	1 130 44                        ; fused_router_kernel.py:130:44
	s_mov_b32 s15, s23
.Ltmp32:
	.loc	3 191 40                        ; standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ]
	s_nop 0
	v_mov_b32_dpp v36, v36 row_shr:4 row_mask:0xf bank_mask:0xa
	s_nop 1
	v_mov_b32_dpp v36, v35 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp33:
	.loc	3 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ] ]
	v_max_f32_e32 v36, v36, v36
	v_max_f32_e32 v35, v35, v36
.Ltmp34:
	.loc	3 191 40                        ; standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_e32 v36, v35
	s_nop 1
	v_mov_b32_dpp v36, v36 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp35:
	.loc	3 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ] ]
	v_max_f32_e32 v36, v36, v36
	v_max_f32_e32 v35, v35, v36
.Ltmp36:
	.loc	3 191 40                        ; standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_e32 v36, v35
	s_nop 1
	v_mov_b32_dpp v36, v36 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
.Ltmp37:
	.loc	3 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ] ]
	v_max_f32_e32 v36, v36, v36
	v_max_f32_e32 v35, v35, v36
.Ltmp38:
	.loc	3 61 12                         ; standard.py:61:12 @[ fused_router_kernel.py:68:35 ]
	v_sub_f32_e32 v27, v27, v35
	v_sub_f32_e32 v28, v28, v35
	.loc	3 62 19                         ; standard.py:62:19 @[ fused_router_kernel.py:68:35 ]
	v_mul_f32_e32 v35, 0x3fb8aa3b, v27
	v_mul_f32_e32 v36, 0x3fb8aa3b, v28
	v_cmp_gt_f32_e32 vcc, s41, v35
	v_cmp_gt_f32_e64 s[10:11], s41, v36
	s_nop 0
	v_cndmask_b32_e32 v35, 0, v30, vcc
	v_cndmask_b32_e64 v36, 0, v30, s[10:11]
	v_fmac_f32_e32 v35, 0x3fb8aa3b, v27
	v_fmac_f32_e32 v36, 0x3fb8aa3b, v28
	v_exp_f32_e32 v35, v35
	v_exp_f32_e32 v36, v36
	v_cndmask_b32_e32 v27, 0, v31, vcc
	v_cndmask_b32_e64 v28, 0, v31, s[10:11]
	v_ldexp_f32 v27, v35, v27
	v_ldexp_f32 v35, v36, v28
.Ltmp39:
	.loc	3 293 36                        ; standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ]
	ds_swizzle_b32 v28, v35 offset:swizzle(SWAP,16)
.Ltmp40:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ] ]
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v28, v35, v28
.Ltmp41:
	.loc	3 293 36                        ; standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_e32 v36, v28
	s_nop 1
	v_mov_b32_dpp v36, v36 row_shr:8 row_mask:0xf bank_mask:0xc
	s_nop 1
	v_mov_b32_dpp v36, v28 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp42:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ] ]
	v_add_f32_e32 v28, v28, v36
.Ltmp43:
	.loc	3 293 36                        ; standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_e32 v36, v28
	s_nop 1
	v_mov_b32_dpp v36, v36 row_shr:4 row_mask:0xf bank_mask:0xa
	s_nop 1
	v_mov_b32_dpp v36, v28 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp44:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ] ]
	v_add_f32_e32 v28, v28, v36
.Ltmp45:
	.loc	3 293 36                        ; standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_e32 v36, v28
	s_nop 1
	v_mov_b32_dpp v36, v36 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp46:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ] ]
	v_add_f32_e32 v28, v28, v36
.Ltmp47:
	.loc	3 293 36                        ; standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_e32 v36, v28
	s_nop 1
	v_mov_b32_dpp v36, v36 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
.Ltmp48:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ] ]
	v_add_f32_e32 v36, v28, v36
.Ltmp49:
	.loc	3 64 31                         ; standard.py:64:31 @[ fused_router_kernel.py:68:35 ]
	v_rcp_f32_e32 v44, v36
	s_nop 3
	v_mul_f32_e32 v27, v27, v44
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
.Ltmp50:
	.loc	1 99 63                         ; fused_router_kernel.py:99:63
	ds_bpermute_b32 v28, v23, v27
.Ltmp51:
	.loc	3 64 31                         ; standard.py:64:31 @[ fused_router_kernel.py:68:35 ]
	v_mul_f32_e32 v27, v35, v44
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
.Ltmp52:
	.loc	1 72 40                         ; fused_router_kernel.py:72:40
	buffer_store_dword v27, v40, s[4:7], 0 offen
.Ltmp53:
	.loc	2 43 52                         ; argsort.py:43:52 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[40:41], v[28:29], v[0:1] op_sel_hi:[0,1]
.Ltmp54:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	s_nop 1
	v_mov_b32_dpp v40, v40 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp55:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v41, v41 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp56:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[40:41], v[28:29], v[0:1], v[40:41] op_sel_hi:[0,1,1]
.Ltmp57:
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v40, v41
	.loc	2 50 67 is_stmt 0               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v27, v41, v40
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_xor_b64 s[10:11], vcc, s[18:19]
	.loc	2 50 77                         ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v27, v27, 0, s[10:11]
	.loc	2 51 15 is_stmt 1               ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v36, v27, v28
	.loc	2 52 77                         ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v27, v24, 0, s[10:11]
	.loc	2 42 53                         ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_waitcnt vmcnt(2)
	v_pk_mul_f32 v[40:41], v[36:37], v[2:3] op_sel_hi:[0,1]
	.loc	2 53 20                         ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v35, v27, v22
.Ltmp58:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v42, v40
.Ltmp59:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v43, v41
.Ltmp60:
	.loc	2 44 54 is_stmt 1               ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v27, v35, v13
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v28, v35, v12
.Ltmp61:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v42, v42 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp62:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v43, v43 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp63:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v44, v27
.Ltmp64:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v45, v28
.Ltmp65:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v42, v40 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp66:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v43, v41 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp67:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v44, v44 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp68:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v45, v45 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp69:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[40:41], v[36:37], v[2:3], v[42:43] op_sel_hi:[0,1,1]
.Ltmp70:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v44, v27 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp71:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v45, v28 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp72:
	.loc	2 50 53 is_stmt 1               ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v40, v41
.Ltmp73:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v27, v27, v44
.Ltmp74:
	.loc	3 263 15 is_stmt 0              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v28, v45, v28
.Ltmp75:
	.loc	2 50 67 is_stmt 1               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v40, v40, v41
	.loc	2 50 53 is_stmt 0               ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_xor_b64 s[10:11], vcc, s[30:31]
	.loc	2 50 77                         ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v40, v40, 0, s[10:11]
	.loc	2 52 67 is_stmt 1               ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v41, v28, v27
	.loc	2 51 15                         ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v36, v40, v36
	.loc	2 52 77                         ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v40, v41, 0, s[10:11]
	.loc	2 53 20                         ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v35, v40, v35
	.loc	2 42 53                         ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[40:41], v[36:37], v[0:1] op_sel_hi:[0,1]
	.loc	2 44 54                         ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v42, v35, v10
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v43, v35, v8
.Ltmp76:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v40, v40 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp77:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v41, v41 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp78:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[40:41], v[36:37], v[0:1], v[40:41] op_sel_hi:[0,1,1]
.Ltmp79:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v44, v42
.Ltmp80:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v45, v43
.Ltmp81:
	.loc	2 50 53 is_stmt 1               ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v40, v41
.Ltmp82:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v44, v44 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp83:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v45, v45 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp84:
	.loc	2 50 67 is_stmt 1               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v40, v40, v41
	.loc	2 50 53 is_stmt 0               ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_xor_b64 s[10:11], vcc, s[30:31]
.Ltmp85:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v41, v42, v44
.Ltmp86:
	.loc	3 263 15 is_stmt 0              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v42, v43, v45
.Ltmp87:
	.loc	2 50 77 is_stmt 1               ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v40, v40, 0, s[10:11]
	.loc	2 51 15                         ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v36, v40, v36
	.loc	2 52 67                         ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v40, v41, v42
	.loc	2 52 77 is_stmt 0               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v42, v40, 0, s[10:11]
	.loc	2 42 53 is_stmt 1               ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[40:41], v[36:37], v[4:5] op_sel_hi:[0,1]
	.loc	2 53 20                         ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v35, v42, v35
.Ltmp88:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v42, v40
.Ltmp89:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v43, v41
.Ltmp90:
	.loc	2 44 54 is_stmt 1               ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v44, v35, v15
.Ltmp91:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v42, v42 row_shr:8 row_mask:0xf bank_mask:0xc
.Ltmp92:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v43, v43 row_shr:8 row_mask:0xf bank_mask:0xc
.Ltmp93:
	.loc	2 45 53 is_stmt 1               ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v45, v35, v14
.Ltmp94:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v42, v40 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp95:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v43, v41 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp96:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v46, v44
.Ltmp97:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v47, v45
.Ltmp98:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[40:41], v[36:37], v[4:5], v[42:43] op_sel_hi:[0,1,1]
.Ltmp99:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v46, v46 row_shr:8 row_mask:0xf bank_mask:0xc
.Ltmp100:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v47, v47 row_shr:8 row_mask:0xf bank_mask:0xc
.Ltmp101:
	.loc	2 50 53 is_stmt 1               ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v40, v41
.Ltmp102:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v46, v44 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp103:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v47, v45 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp104:
	.loc	2 50 67 is_stmt 1               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v40, v40, v41
	.loc	2 50 53 is_stmt 0               ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_xor_b64 s[10:11], vcc, s[34:35]
.Ltmp105:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v41, v44, v46
.Ltmp106:
	.loc	3 263 15 is_stmt 0              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v42, v45, v47
.Ltmp107:
	.loc	2 50 77 is_stmt 1               ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v40, v40, 0, s[10:11]
	.loc	2 51 15                         ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v36, v40, v36
	.loc	2 52 67                         ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v40, v41, v42
	.loc	2 52 77 is_stmt 0               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v42, v40, 0, s[10:11]
	.loc	2 42 53 is_stmt 1               ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[40:41], v[36:37], v[2:3] op_sel_hi:[0,1]
	.loc	2 53 20                         ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v35, v42, v35
.Ltmp108:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v42, v40
.Ltmp109:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v43, v41
.Ltmp110:
	.loc	2 44 54 is_stmt 1               ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v44, v35, v13
.Ltmp111:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v42, v42 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp112:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v43, v43 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp113:
	.loc	2 45 53 is_stmt 1               ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v45, v35, v12
.Ltmp114:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v42, v40 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp115:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v43, v41 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp116:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v46, v44
.Ltmp117:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v47, v45
.Ltmp118:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[40:41], v[36:37], v[2:3], v[42:43] op_sel_hi:[0,1,1]
.Ltmp119:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v46, v46 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp120:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v47, v47 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp121:
	.loc	2 50 53 is_stmt 1               ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v40, v41
.Ltmp122:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v46, v44 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp123:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v47, v45 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp124:
	.loc	2 50 67 is_stmt 1               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v40, v40, v41
	.loc	2 50 53 is_stmt 0               ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_xor_b64 s[10:11], vcc, s[34:35]
.Ltmp125:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v41, v44, v46
.Ltmp126:
	.loc	3 263 15 is_stmt 0              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v42, v45, v47
.Ltmp127:
	.loc	2 50 77 is_stmt 1               ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v40, v40, 0, s[10:11]
	.loc	2 51 15                         ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v36, v40, v36
	.loc	2 52 67                         ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v40, v41, v42
	.loc	2 52 77 is_stmt 0               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v42, v40, 0, s[10:11]
	.loc	2 42 53 is_stmt 1               ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[40:41], v[36:37], v[0:1] op_sel_hi:[0,1]
	.loc	2 53 20                         ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v35, v42, v35
	.loc	2 44 54                         ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v42, v35, v10
.Ltmp128:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v40, v40 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp129:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v41, v41 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp130:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[40:41], v[36:37], v[0:1], v[40:41] op_sel_hi:[0,1,1]
.Ltmp131:
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v43, v35, v8
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v40, v41
.Ltmp132:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v44, v42
.Ltmp133:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v45, v43
.Ltmp134:
	.loc	2 50 67 is_stmt 1               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v40, v40, v41
	.loc	2 50 53 is_stmt 0               ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_xor_b64 s[10:11], vcc, s[34:35]
.Ltmp135:
	.loc	3 293 36 is_stmt 1              ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v44, v44 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp136:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v45, v45 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp137:
	.loc	2 50 77 is_stmt 1               ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v40, v40, 0, s[10:11]
.Ltmp138:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v41, v42, v44
.Ltmp139:
	.loc	3 263 15 is_stmt 0              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v42, v43, v45
.Ltmp140:
	.loc	2 51 15 is_stmt 1               ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v36, v40, v36
	.loc	2 52 67                         ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v42, v41, v42
	.loc	2 42 53                         ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[40:41], v[36:37], v[6:7] op_sel_hi:[0,1]
.Ltmp141:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	ds_swizzle_b32 v40, v40 offset:swizzle(SWAP,16)
.Ltmp142:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	ds_swizzle_b32 v41, v41 offset:swizzle(SWAP,16)
.Ltmp143:
	.loc	2 52 77 is_stmt 1               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v42, v42, 0, s[10:11]
	.loc	2 53 20                         ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v35, v42, v35
	.loc	2 44 54                         ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v42, v35, v18
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v43, v35, v16
.Ltmp144:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	ds_swizzle_b32 v44, v42 offset:swizzle(SWAP,16)
.Ltmp145:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	ds_swizzle_b32 v45, v43 offset:swizzle(SWAP,16)
.Ltmp146:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	s_waitcnt lgkmcnt(2)
	v_pk_fma_f32 v[40:41], v[36:37], v[6:7], v[40:41] op_sel_hi:[0,1,1]
.Ltmp147:
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v40, v41
	.loc	2 50 67 is_stmt 0               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v40, v40, v41
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_xor_b64 s[10:11], vcc, s[44:45]
	.loc	2 50 77                         ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v40, v40, 0, s[10:11]
.Ltmp148:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	s_waitcnt lgkmcnt(1)
	v_add_u32_e32 v41, v42, v44
.Ltmp149:
	.loc	3 263 15 is_stmt 0              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	s_waitcnt lgkmcnt(0)
	v_add_u32_e32 v42, v43, v45
.Ltmp150:
	.loc	2 51 15 is_stmt 1               ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v36, v40, v36
	.loc	2 52 67                         ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v42, v41, v42
	.loc	2 42 53                         ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[40:41], v[36:37], v[4:5] op_sel_hi:[0,1]
	.loc	2 52 77                         ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v44, v42, 0, s[10:11]
.Ltmp151:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v42, v40
.Ltmp152:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v43, v41
.Ltmp153:
	.loc	2 53 20 is_stmt 1               ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v35, v44, v35
.Ltmp154:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v42, v42 row_shr:8 row_mask:0xf bank_mask:0xc
.Ltmp155:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v43, v43 row_shr:8 row_mask:0xf bank_mask:0xc
.Ltmp156:
	.loc	2 44 54 is_stmt 1               ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v44, v35, v15
.Ltmp157:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v42, v40 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp158:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v43, v41 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp159:
	.loc	2 45 53 is_stmt 1               ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v45, v35, v14
.Ltmp160:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[40:41], v[36:37], v[4:5], v[42:43] op_sel_hi:[0,1,1]
.Ltmp161:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v46, v44
.Ltmp162:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v47, v45
.Ltmp163:
	.loc	2 50 53 is_stmt 1               ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v40, v41
.Ltmp164:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v46, v46 row_shr:8 row_mask:0xf bank_mask:0xc
.Ltmp165:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v47, v47 row_shr:8 row_mask:0xf bank_mask:0xc
.Ltmp166:
	.loc	2 50 67 is_stmt 1               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v40, v40, v41
	.loc	2 50 53 is_stmt 0               ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_xor_b64 s[10:11], vcc, s[44:45]
.Ltmp167:
	.loc	3 293 36 is_stmt 1              ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v46, v44 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp168:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v47, v45 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp169:
	.loc	2 50 77 is_stmt 1               ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v40, v40, 0, s[10:11]
.Ltmp170:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v41, v44, v46
.Ltmp171:
	.loc	3 263 15 is_stmt 0              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v42, v45, v47
.Ltmp172:
	.loc	2 51 15 is_stmt 1               ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v36, v40, v36
	.loc	2 52 67                         ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v42, v41, v42
	.loc	2 42 53                         ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[40:41], v[36:37], v[2:3] op_sel_hi:[0,1]
	.loc	2 52 77                         ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v44, v42, 0, s[10:11]
.Ltmp173:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v42, v40
.Ltmp174:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v43, v41
.Ltmp175:
	.loc	2 53 20 is_stmt 1               ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v35, v44, v35
.Ltmp176:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v42, v42 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp177:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v43, v43 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp178:
	.loc	2 44 54 is_stmt 1               ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v44, v35, v13
.Ltmp179:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v42, v40 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp180:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v43, v41 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp181:
	.loc	2 45 53 is_stmt 1               ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v45, v35, v12
.Ltmp182:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[40:41], v[36:37], v[2:3], v[42:43] op_sel_hi:[0,1,1]
.Ltmp183:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v46, v44
.Ltmp184:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v47, v45
.Ltmp185:
	.loc	2 50 53 is_stmt 1               ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v40, v41
.Ltmp186:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v46, v46 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp187:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v47, v47 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp188:
	.loc	2 50 67 is_stmt 1               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v40, v40, v41
	.loc	2 50 53 is_stmt 0               ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_xor_b64 s[10:11], vcc, s[44:45]
.Ltmp189:
	.loc	3 293 36 is_stmt 1              ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v46, v44 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp190:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v47, v45 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp191:
	.loc	2 50 77 is_stmt 1               ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v40, v40, 0, s[10:11]
.Ltmp192:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v41, v44, v46
.Ltmp193:
	.loc	3 263 15 is_stmt 0              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v42, v45, v47
.Ltmp194:
	.loc	2 51 15 is_stmt 1               ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v36, v40, v36
	.loc	2 52 67                         ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v42, v41, v42
	.loc	2 42 53                         ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[40:41], v[36:37], v[0:1] op_sel_hi:[0,1]
	.loc	2 52 77                         ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v42, v42, 0, s[10:11]
	.loc	2 53 20                         ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v35, v42, v35
.Ltmp195:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v40, v40 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp196:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v41, v41 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp197:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[40:41], v[36:37], v[0:1], v[40:41] op_sel_hi:[0,1,1]
.Ltmp198:
	.loc	2 44 54                         ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v42, v35, v10
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v43, v35, v8
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v40, v41
	.loc	2 50 67 is_stmt 0               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v40, v40, v41
.Ltmp199:
	.loc	3 293 36 is_stmt 1              ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v41, v42
.Ltmp200:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v44, v43
.Ltmp201:
	.loc	2 50 53 is_stmt 1               ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_xor_b64 s[10:11], vcc, s[44:45]
	.loc	2 50 77 is_stmt 0               ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v40, v40, 0, s[10:11]
.Ltmp202:
	.loc	3 293 36 is_stmt 1              ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v41, v41 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp203:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v44, v44 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp204:
	.loc	2 51 15 is_stmt 1               ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v36, v40, v36
.Ltmp205:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v42, v42, v41
.Ltmp206:
	.loc	3 263 15 is_stmt 0              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v43, v43, v44
.Ltmp207:
	.loc	2 42 53 is_stmt 1               ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_f32_e32 v41, v20, v36
	.loc	2 43 52                         ; argsort.py:43:52 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_f32_e32 v40, v21, v36
	.loc	2 52 67                         ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v44, v42, v43
.Ltmp208:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v43, v41
.Ltmp209:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v42, v40
.Ltmp210:
	.loc	2 52 77 is_stmt 1               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v44, v44, 0, s[10:11]
.Ltmp211:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_permlane32_swap_b32_e32 v41, v43
.Ltmp212:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_permlane32_swap_b32_e32 v40, v42
.Ltmp213:
	.loc	2 53 20 is_stmt 1               ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v35, v44, v35
.Ltmp214:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_add_f32 v[40:41], v[40:41], v[42:43]
.Ltmp215:
	.loc	2 44 54                         ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v42, v35, v19
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v43, v35, v17
	.loc	2 50 67                         ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v44, v40, v41
	.loc	2 50 33 is_stmt 0               ; argsort.py:50:33 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v41, v40
.Ltmp216:
	.loc	3 293 36 is_stmt 1              ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v40, v42
.Ltmp217:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v41, v43
.Ltmp218:
	.loc	2 50 77 is_stmt 1               ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e32 v44, 0, v44, vcc
.Ltmp219:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_permlane32_swap_b32_e32 v42, v40
.Ltmp220:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_permlane32_swap_b32_e32 v43, v41
.Ltmp221:
	.loc	2 51 15 is_stmt 1               ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v36, v36, v44
.Ltmp222:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v42, v42, v40
.Ltmp223:
	.loc	3 263 15 is_stmt 0              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v43, v43, v41
.Ltmp224:
	.loc	2 42 53 is_stmt 1               ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[40:41], v[36:37], v[6:7] op_sel_hi:[0,1]
	.loc	2 52 67                         ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v42, v43, v42
.Ltmp225:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	ds_swizzle_b32 v40, v40 offset:swizzle(SWAP,16)
.Ltmp226:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	ds_swizzle_b32 v41, v41 offset:swizzle(SWAP,16)
.Ltmp227:
	.loc	2 52 77 is_stmt 1               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e32 v42, 0, v42, vcc
	.loc	2 53 20                         ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v35, v35, v42
	.loc	2 44 54                         ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v42, v35, v18
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v43, v35, v16
.Ltmp228:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	ds_swizzle_b32 v44, v42 offset:swizzle(SWAP,16)
.Ltmp229:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	ds_swizzle_b32 v45, v43 offset:swizzle(SWAP,16)
.Ltmp230:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	s_waitcnt lgkmcnt(2)
	v_pk_fma_f32 v[40:41], v[36:37], v[6:7], v[40:41] op_sel_hi:[0,1,1]
.Ltmp231:
	.loc	2 50 67                         ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v46, v40, v41
	.loc	2 50 33 is_stmt 0               ; argsort.py:50:33 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v40, v41
.Ltmp232:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	s_waitcnt lgkmcnt(1)
	v_add_u32_e32 v42, v42, v44
.Ltmp233:
	.loc	3 263 15 is_stmt 0              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	s_waitcnt lgkmcnt(0)
	v_add_u32_e32 v43, v43, v45
.Ltmp234:
	.loc	2 50 77 is_stmt 1               ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e32 v40, 0, v46, vcc
	.loc	2 51 15                         ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v36, v40, v36
	.loc	2 42 53                         ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[40:41], v[36:37], v[4:5] op_sel_hi:[0,1]
	.loc	2 52 67                         ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v44, v42, v43
.Ltmp235:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v42, v40
.Ltmp236:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v43, v41
.Ltmp237:
	.loc	2 52 77 is_stmt 1               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e32 v44, 0, v44, vcc
.Ltmp238:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v42, v42 row_shr:8 row_mask:0xf bank_mask:0xc
.Ltmp239:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v43, v43 row_shr:8 row_mask:0xf bank_mask:0xc
.Ltmp240:
	.loc	2 53 20 is_stmt 1               ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v35, v44, v35
.Ltmp241:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v42, v40 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp242:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v43, v41 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp243:
	.loc	2 44 54 is_stmt 1               ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v44, v35, v15
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v45, v35, v14
.Ltmp244:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[40:41], v[36:37], v[4:5], v[42:43] op_sel_hi:[0,1,1]
.Ltmp245:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v42, v44
.Ltmp246:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v43, v45
.Ltmp247:
	.loc	2 50 67 is_stmt 1               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v46, v40, v41
	.loc	2 50 33 is_stmt 0               ; argsort.py:50:33 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v40, v41
.Ltmp248:
	.loc	3 293 36 is_stmt 1              ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v42, v42 row_shr:8 row_mask:0xf bank_mask:0xc
.Ltmp249:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v43, v43 row_shr:8 row_mask:0xf bank_mask:0xc
.Ltmp250:
	.loc	2 50 77 is_stmt 1               ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e32 v40, 0, v46, vcc
.Ltmp251:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v42, v44 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp252:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v43, v45 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp253:
	.loc	2 51 15 is_stmt 1               ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v36, v40, v36
.Ltmp254:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v42, v44, v42
.Ltmp255:
	.loc	3 263 15 is_stmt 0              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v43, v45, v43
.Ltmp256:
	.loc	2 42 53 is_stmt 1               ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[40:41], v[36:37], v[2:3] op_sel_hi:[0,1]
	.loc	2 52 67                         ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v44, v42, v43
.Ltmp257:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v42, v40
.Ltmp258:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v43, v41
.Ltmp259:
	.loc	2 52 77 is_stmt 1               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e32 v44, 0, v44, vcc
.Ltmp260:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v42, v42 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp261:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v43, v43 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp262:
	.loc	2 53 20 is_stmt 1               ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v35, v44, v35
.Ltmp263:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v42, v40 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp264:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v43, v41 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp265:
	.loc	2 44 54 is_stmt 1               ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v44, v35, v13
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v45, v35, v12
.Ltmp266:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[40:41], v[36:37], v[2:3], v[42:43] op_sel_hi:[0,1,1]
.Ltmp267:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v42, v44
.Ltmp268:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v43, v45
.Ltmp269:
	.loc	2 50 67 is_stmt 1               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v46, v40, v41
	.loc	2 50 33 is_stmt 0               ; argsort.py:50:33 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v40, v41
.Ltmp270:
	.loc	3 293 36 is_stmt 1              ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v42, v42 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp271:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v43, v43 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp272:
	.loc	2 50 77 is_stmt 1               ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e32 v40, 0, v46, vcc
.Ltmp273:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v42, v44 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp274:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v43, v45 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp275:
	.loc	2 51 15 is_stmt 1               ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v36, v40, v36
.Ltmp276:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v42, v44, v42
.Ltmp277:
	.loc	3 263 15 is_stmt 0              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v43, v45, v43
.Ltmp278:
	.loc	2 42 53 is_stmt 1               ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[40:41], v[36:37], v[0:1] op_sel_hi:[0,1]
	.loc	2 52 67                         ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v42, v42, v43
	.loc	2 52 77 is_stmt 0               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e32 v42, 0, v42, vcc
.Ltmp279:
	.loc	3 293 36 is_stmt 1              ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v40, v40 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp280:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v41, v41 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp281:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[40:41], v[36:37], v[0:1], v[40:41] op_sel_hi:[0,1,1]
.Ltmp282:
	.loc	2 53 20                         ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v35, v42, v35
	.loc	2 50 67                         ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v42, v40, v41
	.loc	2 50 33 is_stmt 0               ; argsort.py:50:33 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v40, v41
	.loc	2 44 54 is_stmt 1               ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v40, v35, v10
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v41, v35, v8
.Ltmp283:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v43, v40
.Ltmp284:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v44, v41
.Ltmp285:
	.loc	2 50 77 is_stmt 1               ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e32 v42, 0, v42, vcc
.Ltmp286:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v43, v43 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp287:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v44, v44 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp288:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v40, v40, v43
.Ltmp289:
	.loc	3 263 15 is_stmt 0              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v41, v41, v44
.Ltmp290:
	.loc	2 52 67 is_stmt 1               ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v40, v40, v41
	.loc	2 52 77 is_stmt 0               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e32 v40, 0, v40, vcc
	.loc	2 53 20 is_stmt 1               ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v35, v40, v35
.Ltmp291:
	.loc	1 127 63                        ; fused_router_kernel.py:127:63
	v_add_u32_e32 v40, s43, v35
.Ltmp292:
	.loc	2 51 15                         ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v36, v42, v36
.Ltmp293:
	.loc	1 127 63                        ; fused_router_kernel.py:127:63
	ds_bpermute_b32 v42, v11, v40
	.loc	1 120 46                        ; fused_router_kernel.py:120:46
	v_mul_f32_e32 v36, s16, v36
	.loc	1 124 53                        ; fused_router_kernel.py:124:53
	ds_bpermute_b32 v40, v11, v35
	.loc	1 120 46                        ; fused_router_kernel.py:120:46
	ds_bpermute_b32 v36, v11, v36
	v_mov_b32_e32 v27, v37
	.loc	1 129 38                        ; fused_router_kernel.py:129:38
	s_waitcnt lgkmcnt(2)
	v_lshlrev_b32_e32 v35, 2, v42
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v28, v38
	.loc	1 56 52                         ; fused_router_kernel.py:56:52
	s_cmp_lt_i32 s42, s51
	.loc	1 124 53                        ; fused_router_kernel.py:124:53
	s_waitcnt lgkmcnt(1)
	v_ashrrev_i32_e32 v41, 31, v40
	.loc	1 129 38                        ; fused_router_kernel.py:129:38
	v_cndmask_b32_e64 v35, v32, v35, s[8:9]
	.loc	1 124 53                        ; fused_router_kernel.py:124:53
	buffer_store_dwordx2 v[40:41], v39, s[36:39], 0 offen
	.loc	1 129 38                        ; fused_router_kernel.py:129:38
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v36, v35, s[20:23], 0 offen
	.loc	1 130 44                        ; fused_router_kernel.py:130:44
	buffer_store_dword v33, v35, s[12:15], 0 offen
	.loc	1 56 52                         ; fused_router_kernel.py:56:52
	s_cbranch_scc1 .LBB0_5
; %bb.6:                                ; %Flow
	.loc	1 0 52 is_stmt 0                ; fused_router_kernel.py:0:52
	s_mov_b64 s[14:15], s[44:45]
	s_mov_b64 s[38:39], s[34:35]
	s_mov_b64 s[46:47], s[30:31]
	s_mov_b64 s[48:49], s[18:19]
	v_mov_b32_e32 v27, v37
	v_mov_b32_e32 v28, v38
	s_mov_b64 s[8:9], s[20:21]
	s_mov_b64 s[10:11], s[22:23]
	s_mov_b64 s[40:41], s[12:13]
	s_mov_b64 s[42:43], s[22:23]
.LBB0_7:                                ; %._crit_edge
	.loc	1 56 52 is_stmt 1               ; fused_router_kernel.py:56:52
	s_cmp_gt_i32 s33, -1
	s_cselect_b32 s6, -1, 1
	s_abs_i32 s7, s33
	v_cvt_f32_u32_e32 v25, s7
.Ltmp294:
	.loc	3 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ] ]
	v_max_f32_e32 v26, v28, v28
.Ltmp295:
	.loc	1 56 52                         ; fused_router_kernel.py:56:52
	s_sub_i32 s12, s33, s17
	s_add_i32 s6, s12, s6
	v_rcp_iflag_f32_e32 v25, v25
	s_sub_i32 s12, 0, s7
	s_add_i32 s6, s6, 0x8000
	s_xor_b32 s13, s6, s33
	v_mul_f32_e32 v25, 0x4f7ffffe, v25
	v_cvt_u32_f32_e32 v25, v25
	s_abs_i32 s6, s6
	s_ashr_i32 s13, s13, 31
	.loc	1 124 53                        ; fused_router_kernel.py:124:53
	s_mov_b32 s30, s22
	.loc	1 56 52                         ; fused_router_kernel.py:56:52
	v_readfirstlane_b32 s18, v25
.Ltmp296:
	.loc	3 191 40                        ; standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ]
	ds_swizzle_b32 v25, v28 offset:swizzle(SWAP,16)
.Ltmp297:
	.loc	1 56 52                         ; fused_router_kernel.py:56:52
	s_mul_i32 s12, s12, s18
	s_mul_hi_u32 s12, s18, s12
	s_add_i32 s18, s18, s12
	s_mul_hi_u32 s12, s6, s18
.Ltmp298:
	.loc	3 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ] ]
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v25, v25, v25
	v_max_f32_e32 v25, v26, v25
.Ltmp299:
	.loc	3 191 40                        ; standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_e32 v26, v25
.Ltmp300:
	.loc	1 56 52                         ; fused_router_kernel.py:56:52
	s_mul_i32 s18, s12, s7
	s_sub_i32 s6, s6, s18
.Ltmp301:
	.loc	3 191 40                        ; standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_dpp v26, v26 row_shr:8 row_mask:0xf bank_mask:0xc
.Ltmp302:
	.loc	1 56 52                         ; fused_router_kernel.py:56:52
	s_add_i32 s19, s12, 1
	s_sub_i32 s18, s6, s7
.Ltmp303:
	.loc	3 191 40                        ; standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_dpp v26, v25 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp304:
	.loc	3 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ] ]
	v_max_f32_e32 v26, v26, v26
	v_max_f32_e32 v25, v25, v26
.Ltmp305:
	.loc	3 191 40                        ; standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_e32 v26, v25
.Ltmp306:
	.loc	1 56 52                         ; fused_router_kernel.py:56:52
	s_cmp_ge_u32 s6, s7
	s_cselect_b32 s12, s19, s12
.Ltmp307:
	.loc	3 191 40                        ; standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_dpp v26, v26 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp308:
	.loc	1 56 52                         ; fused_router_kernel.py:56:52
	s_cselect_b32 s6, s18, s6
	s_add_i32 s18, s12, 1
.Ltmp309:
	.loc	3 191 40                        ; standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_dpp v26, v25 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp310:
	.loc	3 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ] ]
	v_max_f32_e32 v26, v26, v26
	v_max_f32_e32 v25, v25, v26
.Ltmp311:
	.loc	3 191 40                        ; standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_e32 v26, v25
.Ltmp312:
	.loc	1 56 52                         ; fused_router_kernel.py:56:52
	s_cmp_ge_u32 s6, s7
	s_cselect_b32 s6, s18, s12
.Ltmp313:
	.loc	3 191 40                        ; standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_dpp v26, v26 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp314:
	.loc	3 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ] ]
	v_max_f32_e32 v26, v26, v26
	v_max_f32_e32 v25, v25, v26
.Ltmp315:
	.loc	3 191 40                        ; standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_e32 v26, v25
.Ltmp316:
	.loc	1 56 52                         ; fused_router_kernel.py:56:52
	s_xor_b32 s6, s6, s13
	s_sub_i32 s12, s6, s13
.Ltmp317:
	.loc	3 191 40                        ; standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_dpp v26, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
.Ltmp318:
	.loc	3 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ standard.py:61:19 @[ fused_router_kernel.py:68:35 ] ] ]
	v_max_f32_e32 v26, v26, v26
	v_max_f32_e32 v25, v25, v26
.Ltmp319:
	.loc	3 61 12                         ; standard.py:61:12 @[ fused_router_kernel.py:68:35 ]
	v_sub_f32_e32 v26, v27, v25
	.loc	3 62 19                         ; standard.py:62:19 @[ fused_router_kernel.py:68:35 ]
	v_mul_f32_e32 v27, 0x3fb8aa3b, v26
	s_mov_b32 s6, 0xc2fc0000
	.loc	3 61 12                         ; standard.py:61:12 @[ fused_router_kernel.py:68:35 ]
	v_sub_f32_e32 v25, v28, v25
	.loc	3 62 19                         ; standard.py:62:19 @[ fused_router_kernel.py:68:35 ]
	v_mov_b32_e32 v28, 0x42800000
	v_cmp_gt_f32_e32 vcc, s6, v27
.Ltmp320:
	.loc	1 56 52                         ; fused_router_kernel.py:56:52
	s_add_i32 s13, s12, -1
	.loc	1 124 53                        ; fused_router_kernel.py:124:53
	s_mov_b32 s31, s23
.Ltmp321:
	.loc	3 62 19                         ; standard.py:62:19 @[ fused_router_kernel.py:68:35 ]
	v_cndmask_b32_e32 v27, 0, v28, vcc
	v_fmac_f32_e32 v27, 0x3fb8aa3b, v26
	v_mul_f32_e32 v26, 0x3fb8aa3b, v25
	v_cmp_gt_f32_e64 s[6:7], s6, v26
	s_nop 1
	v_cndmask_b32_e64 v26, 0, v28, s[6:7]
	v_fmac_f32_e32 v26, 0x3fb8aa3b, v25
	v_exp_f32_e32 v25, v26
	v_exp_f32_e32 v26, v27
	v_not_b32_e32 v27, 63
	v_cndmask_b32_e64 v28, 0, v27, s[6:7]
	v_ldexp_f32 v25, v25, v28
.Ltmp322:
	.loc	3 293 36                        ; standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ]
	ds_swizzle_b32 v28, v25 offset:swizzle(SWAP,16)
.Ltmp323:
	.loc	3 62 19                         ; standard.py:62:19 @[ fused_router_kernel.py:68:35 ]
	v_cndmask_b32_e32 v27, 0, v27, vcc
	v_ldexp_f32 v26, v26, v27
.Ltmp324:
	.loc	1 56 52                         ; fused_router_kernel.py:56:52
	s_max_i32 s6, s13, 0
	s_mul_i32 s13, s6, s33
.Ltmp325:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ] ]
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v27, v25, v28
.Ltmp326:
	.loc	3 293 36                        ; standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_e32 v28, v27
.Ltmp327:
	.loc	1 56 52                         ; fused_router_kernel.py:56:52
	s_add_i32 s17, s13, s17
	s_cmp_gt_i32 s12, 0
.Ltmp328:
	.loc	3 293 36                        ; standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_dpp v28, v28 row_shr:8 row_mask:0xf bank_mask:0xc
.Ltmp329:
	.loc	1 56 52                         ; fused_router_kernel.py:56:52
	s_cselect_b64 s[12:13], -1, 0
	s_and_b64 s[0:1], s[0:1], s[12:13]
.Ltmp330:
	.loc	3 293 36                        ; standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_dpp v28, v27 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp331:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ] ]
	v_add_f32_e32 v27, v27, v28
.Ltmp332:
	.loc	3 293 36                        ; standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mov_b32_dpp v28, v28 row_shr:4 row_mask:0xf bank_mask:0xa
	s_nop 1
	v_mov_b32_dpp v28, v27 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp333:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ] ]
	v_add_f32_e32 v27, v27, v28
.Ltmp334:
	.loc	3 293 36                        ; standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mov_b32_dpp v28, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp335:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ] ]
	v_add_f32_e32 v27, v27, v28
.Ltmp336:
	.loc	3 293 36                        ; standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ]
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mov_b32_dpp v28, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
.Ltmp337:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ standard.py:63:19 @[ fused_router_kernel.py:68:35 ] ] ]
	v_add_f32_e32 v27, v27, v28
.Ltmp338:
	.loc	3 64 31                         ; standard.py:64:31 @[ fused_router_kernel.py:68:35 ]
	v_rcp_f32_e32 v29, v27
	s_nop 3
	v_mul_f32_e32 v26, v26, v29
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
.Ltmp339:
	.loc	1 99 63                         ; fused_router_kernel.py:99:63
	ds_bpermute_b32 v26, v23, v26
.Ltmp340:
	.loc	3 64 31                         ; standard.py:64:31 @[ fused_router_kernel.py:68:35 ]
	v_mul_f32_e32 v30, v25, v29
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
.Ltmp341:
	.loc	2 43 52                         ; argsort.py:43:52 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[28:29], v[26:27], v[0:1] op_sel_hi:[0,1]
.Ltmp342:
	.loc	3 64 31                         ; standard.py:64:31 @[ fused_router_kernel.py:68:35 ]
	s_nop 0
.Ltmp343:
	.loc	1 72 40                         ; fused_router_kernel.py:72:40
	v_bfrev_b32_e32 v31, 1
.Ltmp344:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v28, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp345:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v29, v29 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp346:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[28:29], v[26:27], v[0:1], v[28:29] op_sel_hi:[0,1,1]
.Ltmp347:
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v28, v29
	.loc	2 50 67 is_stmt 0               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v23, v29, v28
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_xor_b64 s[6:7], vcc, s[48:49]
	.loc	2 50 77                         ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v23, v23, 0, s[6:7]
	.loc	2 51 15 is_stmt 1               ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v26, v23, v26
	.loc	2 52 77                         ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v23, v24, 0, s[6:7]
	.loc	2 53 20                         ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v27, v23, v22
	.loc	2 44 54                         ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v22, v27, v13
.Ltmp348:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v23, v22
	s_nop 1
	v_mov_b32_dpp v23, v23 row_shr:4 row_mask:0xf bank_mask:0xa
	s_nop 1
	v_mov_b32_dpp v23, v22 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp349:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v28, v22, v23
.Ltmp350:
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v22, v27, v12
.Ltmp351:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v23, v22
	s_nop 1
	v_mov_b32_dpp v23, v23 row_shr:4 row_mask:0xf bank_mask:0xa
	s_nop 1
	v_mov_b32_dpp v23, v22 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp352:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v29, v23, v22
.Ltmp353:
	.loc	2 42 53                         ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[22:23], v[26:27], v[2:3] op_sel_hi:[0,1]
.Ltmp354:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v24, v22
.Ltmp355:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v25, v23
.Ltmp356:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	s_nop 0
	v_mov_b32_dpp v24, v24 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp357:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v25, v25 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp358:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	s_nop 0
	v_mov_b32_dpp v24, v22 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp359:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v25, v23 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp360:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[22:23], v[26:27], v[2:3], v[24:25] op_sel_hi:[0,1,1]
.Ltmp361:
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v22, v23
	.loc	2 50 67 is_stmt 0               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v22, v22, v23
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_xor_b64 s[6:7], vcc, s[46:47]
	.loc	2 52 67 is_stmt 1               ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v23, v29, v28
	.loc	2 52 77 is_stmt 0               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v23, v23, 0, s[6:7]
	.loc	2 53 20 is_stmt 1               ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v23, v23, v27
	.loc	2 44 54                         ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v24, v23, v10
.Ltmp362:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v25, v24
.Ltmp363:
	.loc	2 50 77                         ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v22, v22, 0, s[6:7]
	.loc	2 51 15                         ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v22, v22, v26
.Ltmp364:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v25, v25 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp365:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v26, v24, v25
.Ltmp366:
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v24, v23, v8
.Ltmp367:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v25, v24
	s_nop 1
	v_mov_b32_dpp v25, v25 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp368:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v27, v24, v25
.Ltmp369:
	.loc	2 42 53                         ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[24:25], v[22:23], v[0:1] op_sel_hi:[0,1]
.Ltmp370:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	s_nop 1
	v_mov_b32_dpp v24, v24 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp371:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v25, v25 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp372:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[24:25], v[22:23], v[0:1], v[24:25] op_sel_hi:[0,1,1]
.Ltmp373:
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v24, v25
	.loc	2 50 67 is_stmt 0               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v24, v24, v25
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_xor_b64 s[6:7], vcc, s[46:47]
	.loc	2 50 77                         ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v24, v24, 0, s[6:7]
	.loc	2 51 15 is_stmt 1               ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v22, v24, v22
	.loc	2 52 67                         ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v24, v26, v27
	.loc	2 52 77 is_stmt 0               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v24, v24, 0, s[6:7]
	.loc	2 53 20 is_stmt 1               ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v23, v24, v23
	.loc	2 44 54                         ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v24, v23, v15
.Ltmp374:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v25, v24
	s_nop 1
	v_mov_b32_dpp v25, v25 row_shr:8 row_mask:0xf bank_mask:0xc
	s_nop 1
	v_mov_b32_dpp v25, v24 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp375:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v28, v24, v25
.Ltmp376:
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v24, v23, v14
.Ltmp377:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v25, v24
	s_nop 1
	v_mov_b32_dpp v25, v25 row_shr:8 row_mask:0xf bank_mask:0xc
	s_nop 1
	v_mov_b32_dpp v25, v24 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp378:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v29, v24, v25
.Ltmp379:
	.loc	2 42 53                         ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[24:25], v[22:23], v[4:5] op_sel_hi:[0,1]
.Ltmp380:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v26, v24
.Ltmp381:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v27, v25
.Ltmp382:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	s_nop 0
	v_mov_b32_dpp v26, v26 row_shr:8 row_mask:0xf bank_mask:0xc
.Ltmp383:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v27, v27 row_shr:8 row_mask:0xf bank_mask:0xc
.Ltmp384:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	s_nop 0
	v_mov_b32_dpp v26, v24 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp385:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v27, v25 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp386:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[24:25], v[22:23], v[4:5], v[26:27] op_sel_hi:[0,1,1]
.Ltmp387:
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v24, v25
	.loc	2 50 67 is_stmt 0               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v24, v24, v25
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_xor_b64 s[6:7], vcc, s[38:39]
	.loc	2 50 77                         ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v24, v24, 0, s[6:7]
	.loc	2 51 15 is_stmt 1               ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v22, v24, v22
	.loc	2 52 67                         ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v24, v28, v29
	.loc	2 52 77 is_stmt 0               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v24, v24, 0, s[6:7]
	.loc	2 53 20 is_stmt 1               ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v23, v24, v23
	.loc	2 44 54                         ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v24, v23, v13
.Ltmp388:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v25, v24
	s_nop 1
	v_mov_b32_dpp v25, v25 row_shr:4 row_mask:0xf bank_mask:0xa
	s_nop 1
	v_mov_b32_dpp v25, v24 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp389:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v28, v24, v25
.Ltmp390:
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v24, v23, v12
.Ltmp391:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v25, v24
	s_nop 1
	v_mov_b32_dpp v25, v25 row_shr:4 row_mask:0xf bank_mask:0xa
	s_nop 1
	v_mov_b32_dpp v25, v24 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp392:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v29, v24, v25
.Ltmp393:
	.loc	2 42 53                         ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[24:25], v[22:23], v[2:3] op_sel_hi:[0,1]
.Ltmp394:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v26, v24
.Ltmp395:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v27, v25
.Ltmp396:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	s_nop 0
	v_mov_b32_dpp v26, v26 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp397:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v27, v27 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp398:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	s_nop 0
	v_mov_b32_dpp v26, v24 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp399:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v27, v25 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp400:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[24:25], v[22:23], v[2:3], v[26:27] op_sel_hi:[0,1,1]
.Ltmp401:
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v24, v25
	.loc	2 50 67 is_stmt 0               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v24, v24, v25
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_xor_b64 s[6:7], vcc, s[38:39]
	.loc	2 50 77                         ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v24, v24, 0, s[6:7]
	.loc	2 51 15 is_stmt 1               ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v22, v24, v22
	.loc	2 52 67                         ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v24, v28, v29
	.loc	2 52 77 is_stmt 0               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v24, v24, 0, s[6:7]
	.loc	2 53 20 is_stmt 1               ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v23, v24, v23
	.loc	2 44 54                         ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v24, v23, v10
.Ltmp402:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v25, v24
	s_nop 1
	v_mov_b32_dpp v25, v25 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp403:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v26, v24, v25
.Ltmp404:
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v24, v23, v8
.Ltmp405:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v25, v24
	s_nop 1
	v_mov_b32_dpp v25, v25 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp406:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v27, v24, v25
.Ltmp407:
	.loc	2 42 53                         ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[24:25], v[22:23], v[0:1] op_sel_hi:[0,1]
.Ltmp408:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	s_nop 1
	v_mov_b32_dpp v24, v24 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp409:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v25, v25 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp410:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[24:25], v[22:23], v[0:1], v[24:25] op_sel_hi:[0,1,1]
.Ltmp411:
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v24, v25
	.loc	2 50 67 is_stmt 0               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v24, v24, v25
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_xor_b64 s[6:7], vcc, s[38:39]
	.loc	2 50 77                         ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v24, v24, 0, s[6:7]
	.loc	2 51 15 is_stmt 1               ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v22, v24, v22
	.loc	2 52 67                         ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v24, v26, v27
	.loc	2 52 77 is_stmt 0               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v24, v24, 0, s[6:7]
	.loc	2 53 20 is_stmt 1               ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v23, v24, v23
	.loc	2 42 53                         ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[24:25], v[22:23], v[6:7] op_sel_hi:[0,1]
.Ltmp412:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	ds_swizzle_b32 v24, v24 offset:swizzle(SWAP,16)
.Ltmp413:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	ds_swizzle_b32 v25, v25 offset:swizzle(SWAP,16)
.Ltmp414:
	.loc	2 44 54 is_stmt 1               ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v26, v23, v18
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v28, v23, v16
.Ltmp415:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	ds_swizzle_b32 v27, v26 offset:swizzle(SWAP,16)
.Ltmp416:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	ds_swizzle_b32 v29, v28 offset:swizzle(SWAP,16)
.Ltmp417:
	.loc	1 72 40 is_stmt 1               ; fused_router_kernel.py:72:40
	s_and_b64 vcc, s[2:3], s[12:13]
.Ltmp418:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	s_waitcnt lgkmcnt(2)
	v_pk_fma_f32 v[24:25], v[22:23], v[6:7], v[24:25] op_sel_hi:[0,1,1]
.Ltmp419:
	.loc	1 72 40                         ; fused_router_kernel.py:72:40
	v_cndmask_b32_e32 v32, v31, v34, vcc
.Ltmp420:
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v24, v25
	.loc	2 50 67 is_stmt 0               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v24, v24, v25
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_xor_b64 s[6:7], vcc, s[14:15]
.Ltmp421:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	s_waitcnt lgkmcnt(1)
	v_add_u32_e32 v26, v26, v27
.Ltmp422:
	.loc	3 263 15 is_stmt 0              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	s_waitcnt lgkmcnt(0)
	v_add_u32_e32 v27, v28, v29
.Ltmp423:
	.loc	2 50 77 is_stmt 1               ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v24, v24, 0, s[6:7]
	.loc	2 51 15                         ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v22, v24, v22
	.loc	2 52 67                         ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v24, v26, v27
	.loc	2 52 77 is_stmt 0               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v24, v24, 0, s[6:7]
	.loc	2 53 20 is_stmt 1               ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v23, v24, v23
	.loc	2 44 54                         ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v24, v23, v15
.Ltmp424:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v25, v24
	s_nop 1
	v_mov_b32_dpp v25, v25 row_shr:8 row_mask:0xf bank_mask:0xc
	s_nop 1
	v_mov_b32_dpp v25, v24 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp425:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v28, v24, v25
.Ltmp426:
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v24, v23, v14
.Ltmp427:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v25, v24
	s_nop 1
	v_mov_b32_dpp v25, v25 row_shr:8 row_mask:0xf bank_mask:0xc
	s_nop 1
	v_mov_b32_dpp v25, v24 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp428:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v29, v24, v25
.Ltmp429:
	.loc	2 42 53                         ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[24:25], v[22:23], v[4:5] op_sel_hi:[0,1]
.Ltmp430:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v26, v24
.Ltmp431:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v27, v25
.Ltmp432:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	s_nop 0
	v_mov_b32_dpp v26, v26 row_shr:8 row_mask:0xf bank_mask:0xc
.Ltmp433:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v27, v27 row_shr:8 row_mask:0xf bank_mask:0xc
.Ltmp434:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	s_nop 0
	v_mov_b32_dpp v26, v24 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp435:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v27, v25 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp436:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[24:25], v[22:23], v[4:5], v[26:27] op_sel_hi:[0,1,1]
.Ltmp437:
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v24, v25
	.loc	2 50 67 is_stmt 0               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v24, v24, v25
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_xor_b64 s[6:7], vcc, s[14:15]
	.loc	2 50 77                         ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v24, v24, 0, s[6:7]
	.loc	2 51 15 is_stmt 1               ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v22, v24, v22
	.loc	2 52 67                         ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v24, v28, v29
	.loc	2 52 77 is_stmt 0               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v24, v24, 0, s[6:7]
	.loc	2 53 20 is_stmt 1               ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v23, v24, v23
	.loc	2 44 54                         ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v24, v23, v13
.Ltmp438:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v25, v24
	s_nop 1
	v_mov_b32_dpp v25, v25 row_shr:4 row_mask:0xf bank_mask:0xa
	s_nop 1
	v_mov_b32_dpp v25, v24 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp439:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v28, v24, v25
.Ltmp440:
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v24, v23, v12
.Ltmp441:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v25, v24
	s_nop 1
	v_mov_b32_dpp v25, v25 row_shr:4 row_mask:0xf bank_mask:0xa
	s_nop 1
	v_mov_b32_dpp v25, v24 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp442:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v29, v24, v25
.Ltmp443:
	.loc	2 42 53                         ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[24:25], v[22:23], v[2:3] op_sel_hi:[0,1]
.Ltmp444:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v26, v24
.Ltmp445:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v27, v25
.Ltmp446:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	s_nop 0
	v_mov_b32_dpp v26, v26 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp447:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v27, v27 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp448:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	s_nop 0
	v_mov_b32_dpp v26, v24 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp449:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v27, v25 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp450:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[24:25], v[22:23], v[2:3], v[26:27] op_sel_hi:[0,1,1]
.Ltmp451:
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v24, v25
	.loc	2 50 67 is_stmt 0               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v24, v24, v25
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_xor_b64 s[6:7], vcc, s[14:15]
	.loc	2 50 77                         ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v24, v24, 0, s[6:7]
	.loc	2 51 15 is_stmt 1               ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v22, v24, v22
	.loc	2 52 67                         ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v24, v28, v29
	.loc	2 52 77 is_stmt 0               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v24, v24, 0, s[6:7]
	.loc	2 53 20 is_stmt 1               ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v23, v24, v23
	.loc	2 44 54                         ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v24, v23, v10
.Ltmp452:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v25, v24
	s_nop 1
	v_mov_b32_dpp v25, v25 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp453:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v26, v24, v25
.Ltmp454:
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v24, v23, v8
.Ltmp455:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v25, v24
	s_nop 1
	v_mov_b32_dpp v25, v25 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp456:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v27, v24, v25
.Ltmp457:
	.loc	2 42 53                         ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[24:25], v[22:23], v[0:1] op_sel_hi:[0,1]
.Ltmp458:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	s_nop 1
	v_mov_b32_dpp v24, v24 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp459:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v25, v25 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp460:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[24:25], v[22:23], v[0:1], v[24:25] op_sel_hi:[0,1,1]
.Ltmp461:
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v24, v25
	.loc	2 50 67 is_stmt 0               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v24, v24, v25
	.loc	2 50 53                         ; argsort.py:50:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_xor_b64 s[6:7], vcc, s[14:15]
	.loc	2 50 77                         ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v24, v24, 0, s[6:7]
	.loc	2 51 15 is_stmt 1               ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v28, v24, v22
	.loc	2 52 67                         ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v22, v26, v27
	.loc	2 52 77 is_stmt 0               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e64 v22, v22, 0, s[6:7]
	.loc	2 53 20 is_stmt 1               ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v26, v22, v23
	.loc	2 44 54                         ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v19, v26, v19
	.loc	2 42 53                         ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_f32_e32 v23, v20, v28
.Ltmp462:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v20, v19
.Ltmp463:
	.loc	2 43 52                         ; argsort.py:43:52 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_f32_e32 v22, v21, v28
.Ltmp464:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	s_nop 0
	v_permlane32_swap_b32_e32 v19, v20
.Ltmp465:
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v17, v26, v17
.Ltmp466:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v25, v23
.Ltmp467:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v24, v22
.Ltmp468:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v19, v19, v20
.Ltmp469:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v20, v17
.Ltmp470:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_permlane32_swap_b32_e32 v23, v25
.Ltmp471:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_permlane32_swap_b32_e32 v22, v24
.Ltmp472:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_permlane32_swap_b32_e32 v17, v20
.Ltmp473:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v17, v17, v20
.Ltmp474:
	.loc	3 263 15 is_stmt 0              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_add_f32 v[20:21], v[22:23], v[24:25]
.Ltmp475:
	.loc	2 52 67 is_stmt 1               ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v17, v17, v19
	.loc	2 50 33                         ; argsort.py:50:33 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v21, v20
	.loc	2 50 67 is_stmt 0               ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v22, v20, v21
.Ltmp476:
	.loc	1 72 40 is_stmt 1               ; fused_router_kernel.py:72:40
	s_mov_b32 s6, s26
.Ltmp477:
	.loc	2 52 77                         ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e32 v17, 0, v17, vcc
	.loc	2 53 20                         ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v19, v26, v17
	.loc	2 44 54                         ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v18, v19, v18
.Ltmp478:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	ds_swizzle_b32 v21, v18 offset:swizzle(SWAP,16)
.Ltmp479:
	.loc	2 50 77                         ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e32 v20, 0, v22, vcc
	.loc	2 51 15                         ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v20, v28, v20
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v22, v19, v16
.Ltmp480:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	ds_swizzle_b32 v23, v22 offset:swizzle(SWAP,16)
.Ltmp481:
	.loc	2 42 53                         ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_waitcnt lgkmcnt(1)
	v_pk_mul_f32 v[16:17], v[20:21], v[6:7] op_sel_hi:[0,1]
.Ltmp482:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	ds_swizzle_b32 v16, v16 offset:swizzle(SWAP,16)
.Ltmp483:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	ds_swizzle_b32 v17, v17 offset:swizzle(SWAP,16)
.Ltmp484:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v18, v18, v21
.Ltmp485:
	.loc	3 263 15 is_stmt 0              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	s_waitcnt lgkmcnt(2)
	v_add_u32_e32 v21, v22, v23
.Ltmp486:
	.loc	1 72 40 is_stmt 1               ; fused_router_kernel.py:72:40
	s_mov_b32 s7, s27
	buffer_store_dword v30, v32, s[4:7], 0 offen
.Ltmp487:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	s_waitcnt lgkmcnt(0)
	v_pk_fma_f32 v[6:7], v[20:21], v[6:7], v[16:17] op_sel_hi:[0,1,1]
.Ltmp488:
	.loc	2 50 67                         ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v16, v6, v7
	.loc	2 50 33 is_stmt 0               ; argsort.py:50:33 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v6, v7
	.loc	2 52 67 is_stmt 1               ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v7, v18, v21
	.loc	2 52 77 is_stmt 0               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_nop 0
	v_cndmask_b32_e32 v7, 0, v7, vcc
	.loc	2 53 20 is_stmt 1               ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v7, v7, v19
	.loc	2 44 54                         ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v15, v7, v15
	.loc	2 50 77                         ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e32 v6, 0, v16, vcc
.Ltmp489:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v16, v15
.Ltmp490:
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v14, v7, v14
	.loc	2 51 15                         ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v6, v6, v20
.Ltmp491:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v16, v16 row_shr:8 row_mask:0xf bank_mask:0xc
	s_nop 1
	v_mov_b32_dpp v16, v15 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp492:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v18, v15, v16
.Ltmp493:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v15, v14
	s_nop 1
	v_mov_b32_dpp v15, v15 row_shr:8 row_mask:0xf bank_mask:0xc
	s_nop 1
	v_mov_b32_dpp v15, v14 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp494:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v19, v14, v15
.Ltmp495:
	.loc	2 42 53                         ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[14:15], v[6:7], v[4:5] op_sel_hi:[0,1]
.Ltmp496:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v16, v14
.Ltmp497:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v17, v15
.Ltmp498:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	s_nop 0
	v_mov_b32_dpp v16, v16 row_shr:8 row_mask:0xf bank_mask:0xc
.Ltmp499:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v17, v17 row_shr:8 row_mask:0xf bank_mask:0xc
.Ltmp500:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	s_nop 0
	v_mov_b32_dpp v16, v14 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp501:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v17, v15 row_shl:8 row_mask:0xf bank_mask:0x3
.Ltmp502:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[4:5], v[6:7], v[4:5], v[16:17] op_sel_hi:[0,1,1]
.Ltmp503:
	.loc	2 50 67                         ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v14, v4, v5
	.loc	2 50 33 is_stmt 0               ; argsort.py:50:33 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v4, v5
	.loc	2 52 67 is_stmt 1               ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v5, v18, v19
	.loc	2 52 77 is_stmt 0               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_nop 0
	v_cndmask_b32_e32 v5, 0, v5, vcc
	.loc	2 50 77 is_stmt 1               ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e32 v4, 0, v14, vcc
	.loc	2 53 20                         ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v5, v5, v7
	.loc	2 51 15                         ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v4, v4, v6
	.loc	2 44 54                         ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v6, v5, v13
.Ltmp504:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v7, v6
	s_nop 1
	v_mov_b32_dpp v7, v7 row_shr:4 row_mask:0xf bank_mask:0xa
	s_nop 1
	v_mov_b32_dpp v7, v6 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp505:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v14, v6, v7
.Ltmp506:
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v6, v5, v12
.Ltmp507:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v7, v6
	s_nop 1
	v_mov_b32_dpp v7, v7 row_shr:4 row_mask:0xf bank_mask:0xa
	s_nop 1
	v_mov_b32_dpp v7, v6 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp508:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v15, v6, v7
.Ltmp509:
	.loc	2 42 53                         ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[6:7], v[4:5], v[2:3] op_sel_hi:[0,1]
.Ltmp510:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v12, v6
.Ltmp511:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v13, v7
.Ltmp512:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	s_nop 0
	v_mov_b32_dpp v12, v12 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp513:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v13, v13 row_shr:4 row_mask:0xf bank_mask:0xa
.Ltmp514:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	s_nop 0
	v_mov_b32_dpp v12, v6 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp515:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v13, v7 row_shl:4 row_mask:0xf bank_mask:0x5
.Ltmp516:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[2:3], v[4:5], v[2:3], v[12:13] op_sel_hi:[0,1,1]
.Ltmp517:
	.loc	2 50 67                         ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v6, v2, v3
	.loc	2 50 33 is_stmt 0               ; argsort.py:50:33 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v2, v3
	.loc	2 52 67 is_stmt 1               ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v3, v14, v15
	.loc	2 52 77 is_stmt 0               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_nop 0
	v_cndmask_b32_e32 v3, 0, v3, vcc
	.loc	2 50 77 is_stmt 1               ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e32 v2, 0, v6, vcc
	.loc	2 53 20                         ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v3, v3, v5
	.loc	2 51 15                         ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v2, v2, v4
	.loc	2 44 54                         ; argsort.py:44:54 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v4, v3, v10
.Ltmp518:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v5, v4
	s_nop 1
	v_mov_b32_dpp v5, v5 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp519:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:44:65 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v6, v4, v5
.Ltmp520:
	.loc	2 45 53                         ; argsort.py:45:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_mul_lo_u32 v4, v3, v8
.Ltmp521:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_e32 v5, v4
	s_nop 1
	v_mov_b32_dpp v5, v5 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp522:
	.loc	3 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:45:59 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_add_u32_e32 v7, v4, v5
.Ltmp523:
	.loc	2 42 53                         ; argsort.py:42:53 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_pk_mul_f32 v[4:5], v[2:3], v[0:1] op_sel_hi:[0,1]
.Ltmp524:
	.loc	3 293 36                        ; standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	s_nop 1
	v_mov_b32_dpp v4, v4 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp525:
	.loc	3 293 36 is_stmt 0              ; standard.py:293:36 @[ argsort.py:43:58 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ]
	v_mov_b32_dpp v5, v5 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
.Ltmp526:
	.loc	3 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ argsort.py:42:64 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ] ] ]
	v_pk_fma_f32 v[0:1], v[2:3], v[0:1], v[4:5] op_sel_hi:[0,1,1]
.Ltmp527:
	.loc	2 50 67                         ; argsort.py:50:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v4, v0, v1
	.loc	2 50 33 is_stmt 0               ; argsort.py:50:33 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cmp_ngt_f32_e32 vcc, v0, v1
	.loc	2 52 67 is_stmt 1               ; argsort.py:52:67 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v0, v6, v7
	.loc	2 52 77 is_stmt 0               ; argsort.py:52:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	s_nop 0
	v_cndmask_b32_e32 v0, 0, v0, vcc
	.loc	2 53 20 is_stmt 1               ; argsort.py:53:20 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v3, v0, v3
	.loc	2 50 77                         ; argsort.py:50:77 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_cndmask_b32_e32 v1, 0, v4, vcc
.Ltmp528:
	.loc	1 124 53                        ; fused_router_kernel.py:124:53
	ds_bpermute_b32 v0, v11, v3
	.loc	1 127 63                        ; fused_router_kernel.py:127:63
	v_add_u32_e32 v3, s50, v3
.Ltmp529:
	.loc	2 51 15                         ; argsort.py:51:15 @[ argsort.py:72:79 @[ argsort.py:109:77 @[ fused_router_kernel.py:108:104 ] ] ]
	v_xor_b32_e32 v1, v1, v2
.Ltmp530:
	.loc	1 127 63                        ; fused_router_kernel.py:127:63
	ds_bpermute_b32 v3, v11, v3
	.loc	1 120 46                        ; fused_router_kernel.py:120:46
	v_mul_f32_e32 v1, s16, v1
	ds_bpermute_b32 v2, v11, v1
	.loc	1 124 53                        ; fused_router_kernel.py:124:53
	v_lshl_add_u32 v4, s17, 5, v9
	s_and_b64 vcc, s[2:3], s[0:1]
	s_waitcnt lgkmcnt(2)
	v_ashrrev_i32_e32 v1, 31, v0
	v_cndmask_b32_e32 v4, v31, v4, vcc
	buffer_store_dwordx2 v[0:1], v4, s[28:31], 0 offen
	.loc	1 129 38                        ; fused_router_kernel.py:129:38
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v0, 2, v3
	v_cndmask_b32_e32 v0, v31, v0, vcc
	.loc	1 130 44                        ; fused_router_kernel.py:130:44
	v_mov_b32_e32 v1, 1
	.loc	1 129 38                        ; fused_router_kernel.py:129:38
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v2, v0, s[8:11], 0 offen
	.loc	1 130 44                        ; fused_router_kernel.py:130:44
	buffer_store_dword v1, v0, s[40:43], 0 offen
	.loc	1 56 4                          ; fused_router_kernel.py:56:4
	s_endpgm
.Ltmp531:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel fused_scaling_group_sum_routing_kernel
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 328
		.amdhsa_user_sgpr_count 15
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 13
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 50
		.amdhsa_next_free_sgpr 52
		.amdhsa_accum_offset 52
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
	.size	fused_scaling_group_sum_routing_kernel, .Lfunc_end0-fused_scaling_group_sum_routing_kernel
	.cfi_endproc
                                        ; -- End function
	.set fused_scaling_group_sum_routing_kernel.num_vgpr, 50
	.set fused_scaling_group_sum_routing_kernel.num_agpr, 0
	.set fused_scaling_group_sum_routing_kernel.numbered_sgpr, 52
	.set fused_scaling_group_sum_routing_kernel.num_named_barrier, 0
	.set fused_scaling_group_sum_routing_kernel.private_seg_size, 0
	.set fused_scaling_group_sum_routing_kernel.uses_vcc, 1
	.set fused_scaling_group_sum_routing_kernel.uses_flat_scratch, 0
	.set fused_scaling_group_sum_routing_kernel.has_dyn_sized_stack, 0
	.set fused_scaling_group_sum_routing_kernel.has_recursion, 0
	.set fused_scaling_group_sum_routing_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6696
; TotalNumSgprs: 58
; NumVgprs: 50
; NumAgprs: 0
; TotalNumVgprs: 50
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 6
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 50
; AccumOffset: 52
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 15
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 12
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
	.byte	1                               ; Abbrev [1] 0xb:0x107 DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0xe1 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x8f DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	108                             ; DW_AT_call_line
	.byte	104                             ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x4d:0x82 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.byte	109                             ; DW_AT_call_line
	.byte	77                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x59:0x75 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.byte	72                              ; DW_AT_call_line
	.byte	79                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x65:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.byte	44                              ; DW_AT_call_line
	.byte	65                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x71:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges3                 ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	293                             ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0x7f:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges4                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.byte	45                              ; DW_AT_call_line
	.byte	59                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8b:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges5                 ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	293                             ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0x99:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges6                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.byte	42                              ; DW_AT_call_line
	.byte	64                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa5:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges7                 ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	293                             ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0xb3:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges8                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.byte	43                              ; DW_AT_call_line
	.byte	58                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xbf:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges9                 ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	293                             ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0xd0:0x40 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges10                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	68                              ; DW_AT_call_line
	.byte	35                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0xdc:0x19 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges11                ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.byte	61                              ; DW_AT_call_line
	.byte	19                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xe8:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges12                ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.byte	191                             ; DW_AT_call_line
	.byte	40                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0xf5:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges13                ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.byte	63                              ; DW_AT_call_line
	.byte	19                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x101:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges14                ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	293                             ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp291-.Lfunc_begin0
	.quad	.Ltmp292-.Lfunc_begin0
	.quad	.Ltmp293-.Lfunc_begin0
	.quad	.Ltmp341-.Lfunc_begin0
	.quad	.Ltmp342-.Lfunc_begin0
	.quad	.Ltmp344-.Lfunc_begin0
	.quad	.Ltmp417-.Lfunc_begin0
	.quad	.Ltmp418-.Lfunc_begin0
	.quad	.Ltmp419-.Lfunc_begin0
	.quad	.Ltmp420-.Lfunc_begin0
	.quad	.Ltmp476-.Lfunc_begin0
	.quad	.Ltmp477-.Lfunc_begin0
	.quad	.Ltmp486-.Lfunc_begin0
	.quad	.Ltmp487-.Lfunc_begin0
	.quad	.Ltmp528-.Lfunc_begin0
	.quad	.Ltmp529-.Lfunc_begin0
	.quad	.Ltmp530-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp291-.Lfunc_begin0
	.quad	.Ltmp292-.Lfunc_begin0
	.quad	.Ltmp293-.Lfunc_begin0
	.quad	.Ltmp341-.Lfunc_begin0
	.quad	.Ltmp342-.Lfunc_begin0
	.quad	.Ltmp344-.Lfunc_begin0
	.quad	.Ltmp417-.Lfunc_begin0
	.quad	.Ltmp418-.Lfunc_begin0
	.quad	.Ltmp419-.Lfunc_begin0
	.quad	.Ltmp420-.Lfunc_begin0
	.quad	.Ltmp476-.Lfunc_begin0
	.quad	.Ltmp477-.Lfunc_begin0
	.quad	.Ltmp486-.Lfunc_begin0
	.quad	.Ltmp487-.Lfunc_begin0
	.quad	.Ltmp528-.Lfunc_begin0
	.quad	.Ltmp529-.Lfunc_begin0
	.quad	.Ltmp530-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp85-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp96-.Lfunc_begin0
	.quad	.Ltmp97-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	.Ltmp100-.Lfunc_begin0
	.quad	.Ltmp102-.Lfunc_begin0
	.quad	.Ltmp103-.Lfunc_begin0
	.quad	.Ltmp105-.Lfunc_begin0
	.quad	.Ltmp106-.Lfunc_begin0
	.quad	.Ltmp116-.Lfunc_begin0
	.quad	.Ltmp117-.Lfunc_begin0
	.quad	.Ltmp119-.Lfunc_begin0
	.quad	.Ltmp120-.Lfunc_begin0
	.quad	.Ltmp122-.Lfunc_begin0
	.quad	.Ltmp123-.Lfunc_begin0
	.quad	.Ltmp125-.Lfunc_begin0
	.quad	.Ltmp126-.Lfunc_begin0
	.quad	.Ltmp132-.Lfunc_begin0
	.quad	.Ltmp133-.Lfunc_begin0
	.quad	.Ltmp135-.Lfunc_begin0
	.quad	.Ltmp136-.Lfunc_begin0
	.quad	.Ltmp138-.Lfunc_begin0
	.quad	.Ltmp139-.Lfunc_begin0
	.quad	.Ltmp144-.Lfunc_begin0
	.quad	.Ltmp145-.Lfunc_begin0
	.quad	.Ltmp148-.Lfunc_begin0
	.quad	.Ltmp149-.Lfunc_begin0
	.quad	.Ltmp161-.Lfunc_begin0
	.quad	.Ltmp162-.Lfunc_begin0
	.quad	.Ltmp164-.Lfunc_begin0
	.quad	.Ltmp165-.Lfunc_begin0
	.quad	.Ltmp167-.Lfunc_begin0
	.quad	.Ltmp168-.Lfunc_begin0
	.quad	.Ltmp170-.Lfunc_begin0
	.quad	.Ltmp171-.Lfunc_begin0
	.quad	.Ltmp183-.Lfunc_begin0
	.quad	.Ltmp184-.Lfunc_begin0
	.quad	.Ltmp186-.Lfunc_begin0
	.quad	.Ltmp187-.Lfunc_begin0
	.quad	.Ltmp189-.Lfunc_begin0
	.quad	.Ltmp190-.Lfunc_begin0
	.quad	.Ltmp192-.Lfunc_begin0
	.quad	.Ltmp193-.Lfunc_begin0
	.quad	.Ltmp199-.Lfunc_begin0
	.quad	.Ltmp200-.Lfunc_begin0
	.quad	.Ltmp202-.Lfunc_begin0
	.quad	.Ltmp203-.Lfunc_begin0
	.quad	.Ltmp205-.Lfunc_begin0
	.quad	.Ltmp206-.Lfunc_begin0
	.quad	.Ltmp216-.Lfunc_begin0
	.quad	.Ltmp217-.Lfunc_begin0
	.quad	.Ltmp219-.Lfunc_begin0
	.quad	.Ltmp220-.Lfunc_begin0
	.quad	.Ltmp222-.Lfunc_begin0
	.quad	.Ltmp223-.Lfunc_begin0
	.quad	.Ltmp228-.Lfunc_begin0
	.quad	.Ltmp229-.Lfunc_begin0
	.quad	.Ltmp232-.Lfunc_begin0
	.quad	.Ltmp233-.Lfunc_begin0
	.quad	.Ltmp245-.Lfunc_begin0
	.quad	.Ltmp246-.Lfunc_begin0
	.quad	.Ltmp248-.Lfunc_begin0
	.quad	.Ltmp249-.Lfunc_begin0
	.quad	.Ltmp251-.Lfunc_begin0
	.quad	.Ltmp252-.Lfunc_begin0
	.quad	.Ltmp254-.Lfunc_begin0
	.quad	.Ltmp255-.Lfunc_begin0
	.quad	.Ltmp267-.Lfunc_begin0
	.quad	.Ltmp268-.Lfunc_begin0
	.quad	.Ltmp270-.Lfunc_begin0
	.quad	.Ltmp271-.Lfunc_begin0
	.quad	.Ltmp273-.Lfunc_begin0
	.quad	.Ltmp274-.Lfunc_begin0
	.quad	.Ltmp276-.Lfunc_begin0
	.quad	.Ltmp277-.Lfunc_begin0
	.quad	.Ltmp283-.Lfunc_begin0
	.quad	.Ltmp284-.Lfunc_begin0
	.quad	.Ltmp286-.Lfunc_begin0
	.quad	.Ltmp287-.Lfunc_begin0
	.quad	.Ltmp288-.Lfunc_begin0
	.quad	.Ltmp289-.Lfunc_begin0
	.quad	.Ltmp348-.Lfunc_begin0
	.quad	.Ltmp350-.Lfunc_begin0
	.quad	.Ltmp362-.Lfunc_begin0
	.quad	.Ltmp363-.Lfunc_begin0
	.quad	.Ltmp364-.Lfunc_begin0
	.quad	.Ltmp366-.Lfunc_begin0
	.quad	.Ltmp374-.Lfunc_begin0
	.quad	.Ltmp376-.Lfunc_begin0
	.quad	.Ltmp388-.Lfunc_begin0
	.quad	.Ltmp390-.Lfunc_begin0
	.quad	.Ltmp402-.Lfunc_begin0
	.quad	.Ltmp404-.Lfunc_begin0
	.quad	.Ltmp415-.Lfunc_begin0
	.quad	.Ltmp416-.Lfunc_begin0
	.quad	.Ltmp421-.Lfunc_begin0
	.quad	.Ltmp422-.Lfunc_begin0
	.quad	.Ltmp424-.Lfunc_begin0
	.quad	.Ltmp426-.Lfunc_begin0
	.quad	.Ltmp438-.Lfunc_begin0
	.quad	.Ltmp440-.Lfunc_begin0
	.quad	.Ltmp452-.Lfunc_begin0
	.quad	.Ltmp454-.Lfunc_begin0
	.quad	.Ltmp462-.Lfunc_begin0
	.quad	.Ltmp463-.Lfunc_begin0
	.quad	.Ltmp464-.Lfunc_begin0
	.quad	.Ltmp465-.Lfunc_begin0
	.quad	.Ltmp468-.Lfunc_begin0
	.quad	.Ltmp469-.Lfunc_begin0
	.quad	.Ltmp478-.Lfunc_begin0
	.quad	.Ltmp479-.Lfunc_begin0
	.quad	.Ltmp484-.Lfunc_begin0
	.quad	.Ltmp485-.Lfunc_begin0
	.quad	.Ltmp489-.Lfunc_begin0
	.quad	.Ltmp490-.Lfunc_begin0
	.quad	.Ltmp491-.Lfunc_begin0
	.quad	.Ltmp493-.Lfunc_begin0
	.quad	.Ltmp504-.Lfunc_begin0
	.quad	.Ltmp506-.Lfunc_begin0
	.quad	.Ltmp518-.Lfunc_begin0
	.quad	.Ltmp520-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp85-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp105-.Lfunc_begin0
	.quad	.Ltmp106-.Lfunc_begin0
	.quad	.Ltmp125-.Lfunc_begin0
	.quad	.Ltmp126-.Lfunc_begin0
	.quad	.Ltmp138-.Lfunc_begin0
	.quad	.Ltmp139-.Lfunc_begin0
	.quad	.Ltmp148-.Lfunc_begin0
	.quad	.Ltmp149-.Lfunc_begin0
	.quad	.Ltmp170-.Lfunc_begin0
	.quad	.Ltmp171-.Lfunc_begin0
	.quad	.Ltmp192-.Lfunc_begin0
	.quad	.Ltmp193-.Lfunc_begin0
	.quad	.Ltmp205-.Lfunc_begin0
	.quad	.Ltmp206-.Lfunc_begin0
	.quad	.Ltmp222-.Lfunc_begin0
	.quad	.Ltmp223-.Lfunc_begin0
	.quad	.Ltmp232-.Lfunc_begin0
	.quad	.Ltmp233-.Lfunc_begin0
	.quad	.Ltmp254-.Lfunc_begin0
	.quad	.Ltmp255-.Lfunc_begin0
	.quad	.Ltmp276-.Lfunc_begin0
	.quad	.Ltmp277-.Lfunc_begin0
	.quad	.Ltmp288-.Lfunc_begin0
	.quad	.Ltmp289-.Lfunc_begin0
	.quad	.Ltmp349-.Lfunc_begin0
	.quad	.Ltmp350-.Lfunc_begin0
	.quad	.Ltmp365-.Lfunc_begin0
	.quad	.Ltmp366-.Lfunc_begin0
	.quad	.Ltmp375-.Lfunc_begin0
	.quad	.Ltmp376-.Lfunc_begin0
	.quad	.Ltmp389-.Lfunc_begin0
	.quad	.Ltmp390-.Lfunc_begin0
	.quad	.Ltmp403-.Lfunc_begin0
	.quad	.Ltmp404-.Lfunc_begin0
	.quad	.Ltmp421-.Lfunc_begin0
	.quad	.Ltmp422-.Lfunc_begin0
	.quad	.Ltmp425-.Lfunc_begin0
	.quad	.Ltmp426-.Lfunc_begin0
	.quad	.Ltmp439-.Lfunc_begin0
	.quad	.Ltmp440-.Lfunc_begin0
	.quad	.Ltmp453-.Lfunc_begin0
	.quad	.Ltmp454-.Lfunc_begin0
	.quad	.Ltmp468-.Lfunc_begin0
	.quad	.Ltmp469-.Lfunc_begin0
	.quad	.Ltmp484-.Lfunc_begin0
	.quad	.Ltmp485-.Lfunc_begin0
	.quad	.Ltmp492-.Lfunc_begin0
	.quad	.Ltmp493-.Lfunc_begin0
	.quad	.Ltmp505-.Lfunc_begin0
	.quad	.Ltmp506-.Lfunc_begin0
	.quad	.Ltmp519-.Lfunc_begin0
	.quad	.Ltmp520-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges4:
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp84-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp87-.Lfunc_begin0
	.quad	.Ltmp97-.Lfunc_begin0
	.quad	.Ltmp98-.Lfunc_begin0
	.quad	.Ltmp100-.Lfunc_begin0
	.quad	.Ltmp101-.Lfunc_begin0
	.quad	.Ltmp103-.Lfunc_begin0
	.quad	.Ltmp104-.Lfunc_begin0
	.quad	.Ltmp106-.Lfunc_begin0
	.quad	.Ltmp107-.Lfunc_begin0
	.quad	.Ltmp117-.Lfunc_begin0
	.quad	.Ltmp118-.Lfunc_begin0
	.quad	.Ltmp120-.Lfunc_begin0
	.quad	.Ltmp121-.Lfunc_begin0
	.quad	.Ltmp123-.Lfunc_begin0
	.quad	.Ltmp124-.Lfunc_begin0
	.quad	.Ltmp126-.Lfunc_begin0
	.quad	.Ltmp127-.Lfunc_begin0
	.quad	.Ltmp133-.Lfunc_begin0
	.quad	.Ltmp134-.Lfunc_begin0
	.quad	.Ltmp136-.Lfunc_begin0
	.quad	.Ltmp137-.Lfunc_begin0
	.quad	.Ltmp139-.Lfunc_begin0
	.quad	.Ltmp140-.Lfunc_begin0
	.quad	.Ltmp145-.Lfunc_begin0
	.quad	.Ltmp146-.Lfunc_begin0
	.quad	.Ltmp149-.Lfunc_begin0
	.quad	.Ltmp150-.Lfunc_begin0
	.quad	.Ltmp162-.Lfunc_begin0
	.quad	.Ltmp163-.Lfunc_begin0
	.quad	.Ltmp165-.Lfunc_begin0
	.quad	.Ltmp166-.Lfunc_begin0
	.quad	.Ltmp168-.Lfunc_begin0
	.quad	.Ltmp169-.Lfunc_begin0
	.quad	.Ltmp171-.Lfunc_begin0
	.quad	.Ltmp172-.Lfunc_begin0
	.quad	.Ltmp184-.Lfunc_begin0
	.quad	.Ltmp185-.Lfunc_begin0
	.quad	.Ltmp187-.Lfunc_begin0
	.quad	.Ltmp188-.Lfunc_begin0
	.quad	.Ltmp190-.Lfunc_begin0
	.quad	.Ltmp191-.Lfunc_begin0
	.quad	.Ltmp193-.Lfunc_begin0
	.quad	.Ltmp194-.Lfunc_begin0
	.quad	.Ltmp200-.Lfunc_begin0
	.quad	.Ltmp201-.Lfunc_begin0
	.quad	.Ltmp203-.Lfunc_begin0
	.quad	.Ltmp204-.Lfunc_begin0
	.quad	.Ltmp206-.Lfunc_begin0
	.quad	.Ltmp207-.Lfunc_begin0
	.quad	.Ltmp217-.Lfunc_begin0
	.quad	.Ltmp218-.Lfunc_begin0
	.quad	.Ltmp220-.Lfunc_begin0
	.quad	.Ltmp221-.Lfunc_begin0
	.quad	.Ltmp223-.Lfunc_begin0
	.quad	.Ltmp224-.Lfunc_begin0
	.quad	.Ltmp229-.Lfunc_begin0
	.quad	.Ltmp230-.Lfunc_begin0
	.quad	.Ltmp233-.Lfunc_begin0
	.quad	.Ltmp234-.Lfunc_begin0
	.quad	.Ltmp246-.Lfunc_begin0
	.quad	.Ltmp247-.Lfunc_begin0
	.quad	.Ltmp249-.Lfunc_begin0
	.quad	.Ltmp250-.Lfunc_begin0
	.quad	.Ltmp252-.Lfunc_begin0
	.quad	.Ltmp253-.Lfunc_begin0
	.quad	.Ltmp255-.Lfunc_begin0
	.quad	.Ltmp256-.Lfunc_begin0
	.quad	.Ltmp268-.Lfunc_begin0
	.quad	.Ltmp269-.Lfunc_begin0
	.quad	.Ltmp271-.Lfunc_begin0
	.quad	.Ltmp272-.Lfunc_begin0
	.quad	.Ltmp274-.Lfunc_begin0
	.quad	.Ltmp275-.Lfunc_begin0
	.quad	.Ltmp277-.Lfunc_begin0
	.quad	.Ltmp278-.Lfunc_begin0
	.quad	.Ltmp284-.Lfunc_begin0
	.quad	.Ltmp285-.Lfunc_begin0
	.quad	.Ltmp287-.Lfunc_begin0
	.quad	.Ltmp288-.Lfunc_begin0
	.quad	.Ltmp289-.Lfunc_begin0
	.quad	.Ltmp290-.Lfunc_begin0
	.quad	.Ltmp351-.Lfunc_begin0
	.quad	.Ltmp353-.Lfunc_begin0
	.quad	.Ltmp367-.Lfunc_begin0
	.quad	.Ltmp369-.Lfunc_begin0
	.quad	.Ltmp377-.Lfunc_begin0
	.quad	.Ltmp379-.Lfunc_begin0
	.quad	.Ltmp391-.Lfunc_begin0
	.quad	.Ltmp393-.Lfunc_begin0
	.quad	.Ltmp405-.Lfunc_begin0
	.quad	.Ltmp407-.Lfunc_begin0
	.quad	.Ltmp416-.Lfunc_begin0
	.quad	.Ltmp417-.Lfunc_begin0
	.quad	.Ltmp422-.Lfunc_begin0
	.quad	.Ltmp423-.Lfunc_begin0
	.quad	.Ltmp427-.Lfunc_begin0
	.quad	.Ltmp429-.Lfunc_begin0
	.quad	.Ltmp441-.Lfunc_begin0
	.quad	.Ltmp443-.Lfunc_begin0
	.quad	.Ltmp455-.Lfunc_begin0
	.quad	.Ltmp457-.Lfunc_begin0
	.quad	.Ltmp469-.Lfunc_begin0
	.quad	.Ltmp470-.Lfunc_begin0
	.quad	.Ltmp472-.Lfunc_begin0
	.quad	.Ltmp474-.Lfunc_begin0
	.quad	.Ltmp480-.Lfunc_begin0
	.quad	.Ltmp481-.Lfunc_begin0
	.quad	.Ltmp485-.Lfunc_begin0
	.quad	.Ltmp486-.Lfunc_begin0
	.quad	.Ltmp493-.Lfunc_begin0
	.quad	.Ltmp495-.Lfunc_begin0
	.quad	.Ltmp507-.Lfunc_begin0
	.quad	.Ltmp509-.Lfunc_begin0
	.quad	.Ltmp521-.Lfunc_begin0
	.quad	.Ltmp523-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges5:
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp87-.Lfunc_begin0
	.quad	.Ltmp106-.Lfunc_begin0
	.quad	.Ltmp107-.Lfunc_begin0
	.quad	.Ltmp126-.Lfunc_begin0
	.quad	.Ltmp127-.Lfunc_begin0
	.quad	.Ltmp139-.Lfunc_begin0
	.quad	.Ltmp140-.Lfunc_begin0
	.quad	.Ltmp149-.Lfunc_begin0
	.quad	.Ltmp150-.Lfunc_begin0
	.quad	.Ltmp171-.Lfunc_begin0
	.quad	.Ltmp172-.Lfunc_begin0
	.quad	.Ltmp193-.Lfunc_begin0
	.quad	.Ltmp194-.Lfunc_begin0
	.quad	.Ltmp206-.Lfunc_begin0
	.quad	.Ltmp207-.Lfunc_begin0
	.quad	.Ltmp223-.Lfunc_begin0
	.quad	.Ltmp224-.Lfunc_begin0
	.quad	.Ltmp233-.Lfunc_begin0
	.quad	.Ltmp234-.Lfunc_begin0
	.quad	.Ltmp255-.Lfunc_begin0
	.quad	.Ltmp256-.Lfunc_begin0
	.quad	.Ltmp277-.Lfunc_begin0
	.quad	.Ltmp278-.Lfunc_begin0
	.quad	.Ltmp289-.Lfunc_begin0
	.quad	.Ltmp290-.Lfunc_begin0
	.quad	.Ltmp352-.Lfunc_begin0
	.quad	.Ltmp353-.Lfunc_begin0
	.quad	.Ltmp368-.Lfunc_begin0
	.quad	.Ltmp369-.Lfunc_begin0
	.quad	.Ltmp378-.Lfunc_begin0
	.quad	.Ltmp379-.Lfunc_begin0
	.quad	.Ltmp392-.Lfunc_begin0
	.quad	.Ltmp393-.Lfunc_begin0
	.quad	.Ltmp406-.Lfunc_begin0
	.quad	.Ltmp407-.Lfunc_begin0
	.quad	.Ltmp422-.Lfunc_begin0
	.quad	.Ltmp423-.Lfunc_begin0
	.quad	.Ltmp428-.Lfunc_begin0
	.quad	.Ltmp429-.Lfunc_begin0
	.quad	.Ltmp442-.Lfunc_begin0
	.quad	.Ltmp443-.Lfunc_begin0
	.quad	.Ltmp456-.Lfunc_begin0
	.quad	.Ltmp457-.Lfunc_begin0
	.quad	.Ltmp473-.Lfunc_begin0
	.quad	.Ltmp474-.Lfunc_begin0
	.quad	.Ltmp485-.Lfunc_begin0
	.quad	.Ltmp486-.Lfunc_begin0
	.quad	.Ltmp494-.Lfunc_begin0
	.quad	.Ltmp495-.Lfunc_begin0
	.quad	.Ltmp508-.Lfunc_begin0
	.quad	.Ltmp509-.Lfunc_begin0
	.quad	.Ltmp522-.Lfunc_begin0
	.quad	.Ltmp523-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges6:
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	.Ltmp58-.Lfunc_begin0
	.quad	.Ltmp59-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp77-.Lfunc_begin0
	.quad	.Ltmp78-.Lfunc_begin0
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp88-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp91-.Lfunc_begin0
	.quad	.Ltmp92-.Lfunc_begin0
	.quad	.Ltmp94-.Lfunc_begin0
	.quad	.Ltmp95-.Lfunc_begin0
	.quad	.Ltmp98-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	.Ltmp108-.Lfunc_begin0
	.quad	.Ltmp109-.Lfunc_begin0
	.quad	.Ltmp111-.Lfunc_begin0
	.quad	.Ltmp112-.Lfunc_begin0
	.quad	.Ltmp114-.Lfunc_begin0
	.quad	.Ltmp115-.Lfunc_begin0
	.quad	.Ltmp118-.Lfunc_begin0
	.quad	.Ltmp119-.Lfunc_begin0
	.quad	.Ltmp128-.Lfunc_begin0
	.quad	.Ltmp129-.Lfunc_begin0
	.quad	.Ltmp130-.Lfunc_begin0
	.quad	.Ltmp131-.Lfunc_begin0
	.quad	.Ltmp141-.Lfunc_begin0
	.quad	.Ltmp142-.Lfunc_begin0
	.quad	.Ltmp146-.Lfunc_begin0
	.quad	.Ltmp147-.Lfunc_begin0
	.quad	.Ltmp151-.Lfunc_begin0
	.quad	.Ltmp152-.Lfunc_begin0
	.quad	.Ltmp154-.Lfunc_begin0
	.quad	.Ltmp155-.Lfunc_begin0
	.quad	.Ltmp157-.Lfunc_begin0
	.quad	.Ltmp158-.Lfunc_begin0
	.quad	.Ltmp160-.Lfunc_begin0
	.quad	.Ltmp161-.Lfunc_begin0
	.quad	.Ltmp173-.Lfunc_begin0
	.quad	.Ltmp174-.Lfunc_begin0
	.quad	.Ltmp176-.Lfunc_begin0
	.quad	.Ltmp177-.Lfunc_begin0
	.quad	.Ltmp179-.Lfunc_begin0
	.quad	.Ltmp180-.Lfunc_begin0
	.quad	.Ltmp182-.Lfunc_begin0
	.quad	.Ltmp183-.Lfunc_begin0
	.quad	.Ltmp195-.Lfunc_begin0
	.quad	.Ltmp196-.Lfunc_begin0
	.quad	.Ltmp197-.Lfunc_begin0
	.quad	.Ltmp198-.Lfunc_begin0
	.quad	.Ltmp208-.Lfunc_begin0
	.quad	.Ltmp209-.Lfunc_begin0
	.quad	.Ltmp211-.Lfunc_begin0
	.quad	.Ltmp212-.Lfunc_begin0
	.quad	.Ltmp225-.Lfunc_begin0
	.quad	.Ltmp226-.Lfunc_begin0
	.quad	.Ltmp230-.Lfunc_begin0
	.quad	.Ltmp231-.Lfunc_begin0
	.quad	.Ltmp235-.Lfunc_begin0
	.quad	.Ltmp236-.Lfunc_begin0
	.quad	.Ltmp238-.Lfunc_begin0
	.quad	.Ltmp239-.Lfunc_begin0
	.quad	.Ltmp241-.Lfunc_begin0
	.quad	.Ltmp242-.Lfunc_begin0
	.quad	.Ltmp244-.Lfunc_begin0
	.quad	.Ltmp245-.Lfunc_begin0
	.quad	.Ltmp257-.Lfunc_begin0
	.quad	.Ltmp258-.Lfunc_begin0
	.quad	.Ltmp260-.Lfunc_begin0
	.quad	.Ltmp261-.Lfunc_begin0
	.quad	.Ltmp263-.Lfunc_begin0
	.quad	.Ltmp264-.Lfunc_begin0
	.quad	.Ltmp266-.Lfunc_begin0
	.quad	.Ltmp267-.Lfunc_begin0
	.quad	.Ltmp279-.Lfunc_begin0
	.quad	.Ltmp280-.Lfunc_begin0
	.quad	.Ltmp281-.Lfunc_begin0
	.quad	.Ltmp282-.Lfunc_begin0
	.quad	.Ltmp344-.Lfunc_begin0
	.quad	.Ltmp345-.Lfunc_begin0
	.quad	.Ltmp354-.Lfunc_begin0
	.quad	.Ltmp355-.Lfunc_begin0
	.quad	.Ltmp356-.Lfunc_begin0
	.quad	.Ltmp357-.Lfunc_begin0
	.quad	.Ltmp358-.Lfunc_begin0
	.quad	.Ltmp359-.Lfunc_begin0
	.quad	.Ltmp360-.Lfunc_begin0
	.quad	.Ltmp361-.Lfunc_begin0
	.quad	.Ltmp370-.Lfunc_begin0
	.quad	.Ltmp371-.Lfunc_begin0
	.quad	.Ltmp372-.Lfunc_begin0
	.quad	.Ltmp373-.Lfunc_begin0
	.quad	.Ltmp380-.Lfunc_begin0
	.quad	.Ltmp381-.Lfunc_begin0
	.quad	.Ltmp382-.Lfunc_begin0
	.quad	.Ltmp383-.Lfunc_begin0
	.quad	.Ltmp384-.Lfunc_begin0
	.quad	.Ltmp385-.Lfunc_begin0
	.quad	.Ltmp386-.Lfunc_begin0
	.quad	.Ltmp387-.Lfunc_begin0
	.quad	.Ltmp394-.Lfunc_begin0
	.quad	.Ltmp395-.Lfunc_begin0
	.quad	.Ltmp396-.Lfunc_begin0
	.quad	.Ltmp397-.Lfunc_begin0
	.quad	.Ltmp398-.Lfunc_begin0
	.quad	.Ltmp399-.Lfunc_begin0
	.quad	.Ltmp400-.Lfunc_begin0
	.quad	.Ltmp401-.Lfunc_begin0
	.quad	.Ltmp408-.Lfunc_begin0
	.quad	.Ltmp409-.Lfunc_begin0
	.quad	.Ltmp410-.Lfunc_begin0
	.quad	.Ltmp411-.Lfunc_begin0
	.quad	.Ltmp412-.Lfunc_begin0
	.quad	.Ltmp413-.Lfunc_begin0
	.quad	.Ltmp418-.Lfunc_begin0
	.quad	.Ltmp419-.Lfunc_begin0
	.quad	.Ltmp430-.Lfunc_begin0
	.quad	.Ltmp431-.Lfunc_begin0
	.quad	.Ltmp432-.Lfunc_begin0
	.quad	.Ltmp433-.Lfunc_begin0
	.quad	.Ltmp434-.Lfunc_begin0
	.quad	.Ltmp435-.Lfunc_begin0
	.quad	.Ltmp436-.Lfunc_begin0
	.quad	.Ltmp437-.Lfunc_begin0
	.quad	.Ltmp444-.Lfunc_begin0
	.quad	.Ltmp445-.Lfunc_begin0
	.quad	.Ltmp446-.Lfunc_begin0
	.quad	.Ltmp447-.Lfunc_begin0
	.quad	.Ltmp448-.Lfunc_begin0
	.quad	.Ltmp449-.Lfunc_begin0
	.quad	.Ltmp450-.Lfunc_begin0
	.quad	.Ltmp451-.Lfunc_begin0
	.quad	.Ltmp458-.Lfunc_begin0
	.quad	.Ltmp459-.Lfunc_begin0
	.quad	.Ltmp460-.Lfunc_begin0
	.quad	.Ltmp461-.Lfunc_begin0
	.quad	.Ltmp466-.Lfunc_begin0
	.quad	.Ltmp467-.Lfunc_begin0
	.quad	.Ltmp470-.Lfunc_begin0
	.quad	.Ltmp471-.Lfunc_begin0
	.quad	.Ltmp482-.Lfunc_begin0
	.quad	.Ltmp483-.Lfunc_begin0
	.quad	.Ltmp487-.Lfunc_begin0
	.quad	.Ltmp488-.Lfunc_begin0
	.quad	.Ltmp496-.Lfunc_begin0
	.quad	.Ltmp497-.Lfunc_begin0
	.quad	.Ltmp498-.Lfunc_begin0
	.quad	.Ltmp499-.Lfunc_begin0
	.quad	.Ltmp500-.Lfunc_begin0
	.quad	.Ltmp501-.Lfunc_begin0
	.quad	.Ltmp502-.Lfunc_begin0
	.quad	.Ltmp503-.Lfunc_begin0
	.quad	.Ltmp510-.Lfunc_begin0
	.quad	.Ltmp511-.Lfunc_begin0
	.quad	.Ltmp512-.Lfunc_begin0
	.quad	.Ltmp513-.Lfunc_begin0
	.quad	.Ltmp514-.Lfunc_begin0
	.quad	.Ltmp515-.Lfunc_begin0
	.quad	.Ltmp516-.Lfunc_begin0
	.quad	.Ltmp517-.Lfunc_begin0
	.quad	.Ltmp524-.Lfunc_begin0
	.quad	.Ltmp525-.Lfunc_begin0
	.quad	.Ltmp526-.Lfunc_begin0
	.quad	.Ltmp527-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges7:
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp78-.Lfunc_begin0
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp98-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	.Ltmp118-.Lfunc_begin0
	.quad	.Ltmp119-.Lfunc_begin0
	.quad	.Ltmp130-.Lfunc_begin0
	.quad	.Ltmp131-.Lfunc_begin0
	.quad	.Ltmp146-.Lfunc_begin0
	.quad	.Ltmp147-.Lfunc_begin0
	.quad	.Ltmp160-.Lfunc_begin0
	.quad	.Ltmp161-.Lfunc_begin0
	.quad	.Ltmp182-.Lfunc_begin0
	.quad	.Ltmp183-.Lfunc_begin0
	.quad	.Ltmp197-.Lfunc_begin0
	.quad	.Ltmp198-.Lfunc_begin0
	.quad	.Ltmp230-.Lfunc_begin0
	.quad	.Ltmp231-.Lfunc_begin0
	.quad	.Ltmp244-.Lfunc_begin0
	.quad	.Ltmp245-.Lfunc_begin0
	.quad	.Ltmp266-.Lfunc_begin0
	.quad	.Ltmp267-.Lfunc_begin0
	.quad	.Ltmp281-.Lfunc_begin0
	.quad	.Ltmp282-.Lfunc_begin0
	.quad	.Ltmp360-.Lfunc_begin0
	.quad	.Ltmp361-.Lfunc_begin0
	.quad	.Ltmp372-.Lfunc_begin0
	.quad	.Ltmp373-.Lfunc_begin0
	.quad	.Ltmp386-.Lfunc_begin0
	.quad	.Ltmp387-.Lfunc_begin0
	.quad	.Ltmp400-.Lfunc_begin0
	.quad	.Ltmp401-.Lfunc_begin0
	.quad	.Ltmp410-.Lfunc_begin0
	.quad	.Ltmp411-.Lfunc_begin0
	.quad	.Ltmp418-.Lfunc_begin0
	.quad	.Ltmp419-.Lfunc_begin0
	.quad	.Ltmp436-.Lfunc_begin0
	.quad	.Ltmp437-.Lfunc_begin0
	.quad	.Ltmp450-.Lfunc_begin0
	.quad	.Ltmp451-.Lfunc_begin0
	.quad	.Ltmp460-.Lfunc_begin0
	.quad	.Ltmp461-.Lfunc_begin0
	.quad	.Ltmp487-.Lfunc_begin0
	.quad	.Ltmp488-.Lfunc_begin0
	.quad	.Ltmp502-.Lfunc_begin0
	.quad	.Ltmp503-.Lfunc_begin0
	.quad	.Ltmp516-.Lfunc_begin0
	.quad	.Ltmp517-.Lfunc_begin0
	.quad	.Ltmp526-.Lfunc_begin0
	.quad	.Ltmp527-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges8:
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp59-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp77-.Lfunc_begin0
	.quad	.Ltmp78-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp92-.Lfunc_begin0
	.quad	.Ltmp93-.Lfunc_begin0
	.quad	.Ltmp95-.Lfunc_begin0
	.quad	.Ltmp96-.Lfunc_begin0
	.quad	.Ltmp109-.Lfunc_begin0
	.quad	.Ltmp110-.Lfunc_begin0
	.quad	.Ltmp112-.Lfunc_begin0
	.quad	.Ltmp113-.Lfunc_begin0
	.quad	.Ltmp115-.Lfunc_begin0
	.quad	.Ltmp116-.Lfunc_begin0
	.quad	.Ltmp129-.Lfunc_begin0
	.quad	.Ltmp130-.Lfunc_begin0
	.quad	.Ltmp142-.Lfunc_begin0
	.quad	.Ltmp143-.Lfunc_begin0
	.quad	.Ltmp152-.Lfunc_begin0
	.quad	.Ltmp153-.Lfunc_begin0
	.quad	.Ltmp155-.Lfunc_begin0
	.quad	.Ltmp156-.Lfunc_begin0
	.quad	.Ltmp158-.Lfunc_begin0
	.quad	.Ltmp159-.Lfunc_begin0
	.quad	.Ltmp174-.Lfunc_begin0
	.quad	.Ltmp175-.Lfunc_begin0
	.quad	.Ltmp177-.Lfunc_begin0
	.quad	.Ltmp178-.Lfunc_begin0
	.quad	.Ltmp180-.Lfunc_begin0
	.quad	.Ltmp181-.Lfunc_begin0
	.quad	.Ltmp196-.Lfunc_begin0
	.quad	.Ltmp197-.Lfunc_begin0
	.quad	.Ltmp209-.Lfunc_begin0
	.quad	.Ltmp210-.Lfunc_begin0
	.quad	.Ltmp212-.Lfunc_begin0
	.quad	.Ltmp213-.Lfunc_begin0
	.quad	.Ltmp214-.Lfunc_begin0
	.quad	.Ltmp215-.Lfunc_begin0
	.quad	.Ltmp226-.Lfunc_begin0
	.quad	.Ltmp227-.Lfunc_begin0
	.quad	.Ltmp236-.Lfunc_begin0
	.quad	.Ltmp237-.Lfunc_begin0
	.quad	.Ltmp239-.Lfunc_begin0
	.quad	.Ltmp240-.Lfunc_begin0
	.quad	.Ltmp242-.Lfunc_begin0
	.quad	.Ltmp243-.Lfunc_begin0
	.quad	.Ltmp258-.Lfunc_begin0
	.quad	.Ltmp259-.Lfunc_begin0
	.quad	.Ltmp261-.Lfunc_begin0
	.quad	.Ltmp262-.Lfunc_begin0
	.quad	.Ltmp264-.Lfunc_begin0
	.quad	.Ltmp265-.Lfunc_begin0
	.quad	.Ltmp280-.Lfunc_begin0
	.quad	.Ltmp281-.Lfunc_begin0
	.quad	.Ltmp345-.Lfunc_begin0
	.quad	.Ltmp347-.Lfunc_begin0
	.quad	.Ltmp355-.Lfunc_begin0
	.quad	.Ltmp356-.Lfunc_begin0
	.quad	.Ltmp357-.Lfunc_begin0
	.quad	.Ltmp358-.Lfunc_begin0
	.quad	.Ltmp359-.Lfunc_begin0
	.quad	.Ltmp360-.Lfunc_begin0
	.quad	.Ltmp371-.Lfunc_begin0
	.quad	.Ltmp372-.Lfunc_begin0
	.quad	.Ltmp381-.Lfunc_begin0
	.quad	.Ltmp382-.Lfunc_begin0
	.quad	.Ltmp383-.Lfunc_begin0
	.quad	.Ltmp384-.Lfunc_begin0
	.quad	.Ltmp385-.Lfunc_begin0
	.quad	.Ltmp386-.Lfunc_begin0
	.quad	.Ltmp395-.Lfunc_begin0
	.quad	.Ltmp396-.Lfunc_begin0
	.quad	.Ltmp397-.Lfunc_begin0
	.quad	.Ltmp398-.Lfunc_begin0
	.quad	.Ltmp399-.Lfunc_begin0
	.quad	.Ltmp400-.Lfunc_begin0
	.quad	.Ltmp409-.Lfunc_begin0
	.quad	.Ltmp410-.Lfunc_begin0
	.quad	.Ltmp413-.Lfunc_begin0
	.quad	.Ltmp414-.Lfunc_begin0
	.quad	.Ltmp431-.Lfunc_begin0
	.quad	.Ltmp432-.Lfunc_begin0
	.quad	.Ltmp433-.Lfunc_begin0
	.quad	.Ltmp434-.Lfunc_begin0
	.quad	.Ltmp435-.Lfunc_begin0
	.quad	.Ltmp436-.Lfunc_begin0
	.quad	.Ltmp445-.Lfunc_begin0
	.quad	.Ltmp446-.Lfunc_begin0
	.quad	.Ltmp447-.Lfunc_begin0
	.quad	.Ltmp448-.Lfunc_begin0
	.quad	.Ltmp449-.Lfunc_begin0
	.quad	.Ltmp450-.Lfunc_begin0
	.quad	.Ltmp459-.Lfunc_begin0
	.quad	.Ltmp460-.Lfunc_begin0
	.quad	.Ltmp467-.Lfunc_begin0
	.quad	.Ltmp468-.Lfunc_begin0
	.quad	.Ltmp471-.Lfunc_begin0
	.quad	.Ltmp472-.Lfunc_begin0
	.quad	.Ltmp474-.Lfunc_begin0
	.quad	.Ltmp475-.Lfunc_begin0
	.quad	.Ltmp483-.Lfunc_begin0
	.quad	.Ltmp484-.Lfunc_begin0
	.quad	.Ltmp497-.Lfunc_begin0
	.quad	.Ltmp498-.Lfunc_begin0
	.quad	.Ltmp499-.Lfunc_begin0
	.quad	.Ltmp500-.Lfunc_begin0
	.quad	.Ltmp501-.Lfunc_begin0
	.quad	.Ltmp502-.Lfunc_begin0
	.quad	.Ltmp511-.Lfunc_begin0
	.quad	.Ltmp512-.Lfunc_begin0
	.quad	.Ltmp513-.Lfunc_begin0
	.quad	.Ltmp514-.Lfunc_begin0
	.quad	.Ltmp515-.Lfunc_begin0
	.quad	.Ltmp516-.Lfunc_begin0
	.quad	.Ltmp525-.Lfunc_begin0
	.quad	.Ltmp526-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges9:
	.quad	.Ltmp56-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp214-.Lfunc_begin0
	.quad	.Ltmp215-.Lfunc_begin0
	.quad	.Ltmp346-.Lfunc_begin0
	.quad	.Ltmp347-.Lfunc_begin0
	.quad	.Ltmp474-.Lfunc_begin0
	.quad	.Ltmp475-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges10:
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp50-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp52-.Lfunc_begin0
	.quad	.Ltmp294-.Lfunc_begin0
	.quad	.Ltmp295-.Lfunc_begin0
	.quad	.Ltmp296-.Lfunc_begin0
	.quad	.Ltmp297-.Lfunc_begin0
	.quad	.Ltmp298-.Lfunc_begin0
	.quad	.Ltmp300-.Lfunc_begin0
	.quad	.Ltmp301-.Lfunc_begin0
	.quad	.Ltmp302-.Lfunc_begin0
	.quad	.Ltmp303-.Lfunc_begin0
	.quad	.Ltmp306-.Lfunc_begin0
	.quad	.Ltmp307-.Lfunc_begin0
	.quad	.Ltmp308-.Lfunc_begin0
	.quad	.Ltmp309-.Lfunc_begin0
	.quad	.Ltmp312-.Lfunc_begin0
	.quad	.Ltmp313-.Lfunc_begin0
	.quad	.Ltmp316-.Lfunc_begin0
	.quad	.Ltmp317-.Lfunc_begin0
	.quad	.Ltmp320-.Lfunc_begin0
	.quad	.Ltmp321-.Lfunc_begin0
	.quad	.Ltmp324-.Lfunc_begin0
	.quad	.Ltmp325-.Lfunc_begin0
	.quad	.Ltmp327-.Lfunc_begin0
	.quad	.Ltmp328-.Lfunc_begin0
	.quad	.Ltmp329-.Lfunc_begin0
	.quad	.Ltmp330-.Lfunc_begin0
	.quad	.Ltmp339-.Lfunc_begin0
	.quad	.Ltmp340-.Lfunc_begin0
	.quad	.Ltmp341-.Lfunc_begin0
	.quad	.Ltmp342-.Lfunc_begin0
	.quad	.Ltmp343-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges11:
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp294-.Lfunc_begin0
	.quad	.Ltmp295-.Lfunc_begin0
	.quad	.Ltmp296-.Lfunc_begin0
	.quad	.Ltmp297-.Lfunc_begin0
	.quad	.Ltmp298-.Lfunc_begin0
	.quad	.Ltmp300-.Lfunc_begin0
	.quad	.Ltmp301-.Lfunc_begin0
	.quad	.Ltmp302-.Lfunc_begin0
	.quad	.Ltmp303-.Lfunc_begin0
	.quad	.Ltmp306-.Lfunc_begin0
	.quad	.Ltmp307-.Lfunc_begin0
	.quad	.Ltmp308-.Lfunc_begin0
	.quad	.Ltmp309-.Lfunc_begin0
	.quad	.Ltmp312-.Lfunc_begin0
	.quad	.Ltmp313-.Lfunc_begin0
	.quad	.Ltmp316-.Lfunc_begin0
	.quad	.Ltmp317-.Lfunc_begin0
	.quad	.Ltmp319-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges12:
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp294-.Lfunc_begin0
	.quad	.Ltmp295-.Lfunc_begin0
	.quad	.Ltmp298-.Lfunc_begin0
	.quad	.Ltmp299-.Lfunc_begin0
	.quad	.Ltmp304-.Lfunc_begin0
	.quad	.Ltmp305-.Lfunc_begin0
	.quad	.Ltmp310-.Lfunc_begin0
	.quad	.Ltmp311-.Lfunc_begin0
	.quad	.Ltmp314-.Lfunc_begin0
	.quad	.Ltmp315-.Lfunc_begin0
	.quad	.Ltmp318-.Lfunc_begin0
	.quad	.Ltmp319-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges13:
	.quad	.Ltmp39-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.quad	.Ltmp322-.Lfunc_begin0
	.quad	.Ltmp323-.Lfunc_begin0
	.quad	.Ltmp325-.Lfunc_begin0
	.quad	.Ltmp327-.Lfunc_begin0
	.quad	.Ltmp328-.Lfunc_begin0
	.quad	.Ltmp329-.Lfunc_begin0
	.quad	.Ltmp330-.Lfunc_begin0
	.quad	.Ltmp338-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges14:
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.quad	.Ltmp325-.Lfunc_begin0
	.quad	.Ltmp326-.Lfunc_begin0
	.quad	.Ltmp331-.Lfunc_begin0
	.quad	.Ltmp332-.Lfunc_begin0
	.quad	.Ltmp333-.Lfunc_begin0
	.quad	.Ltmp334-.Lfunc_begin0
	.quad	.Ltmp335-.Lfunc_begin0
	.quad	.Ltmp336-.Lfunc_begin0
	.quad	.Ltmp337-.Lfunc_begin0
	.quad	.Ltmp338-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"fused_router_kernel.py"        ; string offset=7
.Linfo_string2:
	.asciz	"/workspace/code/Primus-Turbo/primus_turbo/triton/moe" ; string offset=30
.Linfo_string3:
	.asciz	"fused_scaling_group_sum_routing_kernel" ; string offset=83
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
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     global_buffer
      - .offset:         72
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         76
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         80
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         84
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         86
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         88
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         90
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         92
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         94
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         128
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         136
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 328
    .max_flat_workgroup_size: 256
    .name:           fused_scaling_group_sum_routing_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         fused_scaling_group_sum_routing_kernel.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     50
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
