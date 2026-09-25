	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	k_tdm_0
	.p2align	8
	.type	k_tdm_0,@function
k_tdm_0:
	global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
	v_nop
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[10:11], s[0:1], 0x0 nv
	s_mov_b32 s9, 0
	s_wait_xcnt 0x0
	s_mov_b32 s1, 0x800000
	s_mov_b32 s8, 1
	s_movk_i32 s5, 0x80
	s_mov_b32 s4, 32
	s_mov_b32 s2, 0x200000
	s_mov_b32 s0, 0x10000
	s_mov_b32 s3, s1
	s_mov_b32 s6, s9
	s_mov_b32 s7, s9
	s_wait_kmcnt 0x0
	s_bitset1_b32 s11, 31
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[8:11], s[0:7]
	s_wait_tensorcnt 0x0
	s_endpgm
.Lfunc_end0:
	.size	k_tdm_0, .Lfunc_end0-k_tdm_0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel k_tdm_0
		.amdhsa_group_segment_fixed_size 8192
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 81
		.amdhsa_next_free_sgpr 12
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-k_tdm_0)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text

	.set .Lk_tdm_0.num_vgpr, 1
	.set .Lk_tdm_0.num_agpr, 0
	.set .Lk_tdm_0.numbered_sgpr, 12
	.set .Lk_tdm_0.num_named_barrier, 0
	.set .Lk_tdm_0.private_seg_size, 0
	.set .Lk_tdm_0.uses_vcc, 0
	.set .Lk_tdm_0.uses_flat_scratch, 0
	.set .Lk_tdm_0.has_dyn_sized_stack, 0
	.set .Lk_tdm_0.has_recursion, 0
	.set .Lk_tdm_0.has_indirect_call, 0
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           16
        .value_kind:     by_value
    .group_segment_fixed_size: 8192
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 32
    .name:           k_tdm_0
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 32
      - 1
      - 1
    .sgpr_count:     12
    .sgpr_spill_count: 0
    .symbol:         k_tdm_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     1
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
