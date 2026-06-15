#!/usr/bin/env python3
"""Convert llvm-objdump disassembly to re-assembleable .s file.

Reads a disassembly (from llvm-objdump -d), resolves numeric branch targets
to labels, and emits a proper .s with AMDGPU metadata that llvm-mc can assemble.

Usage:
    python3 disasm_to_asm.py input.s metadata.yaml output.s

    input.s     = llvm-objdump -d output
    metadata    = extracted from llvm-readobj --notes (YAML portion)
    output.s    = assembleable .s file
"""

import re
import sys
import json


def parse_disasm(lines):
    """Parse llvm-objdump lines into list of (addr, instruction, comment)."""
    instrs = []
    func_name = None
    base_addr = None

    for line in lines:
        line = line.rstrip()

        # Function label: "0000000000001800 <_grouped_variable_k_gemm_kernel>:"
        m = re.match(r'^([0-9a-fA-F]+)\s+<(\w+)>:', line)
        if m:
            base_addr = int(m.group(1), 16)
            func_name = m.group(2)
            continue

        # Instruction: "\ts_load_dwordx2 s[2:3], s[0:1], 0x0   // 000000001800: C006..."
        # Match instruction with // addr: encoding [<label>]
        m = re.match(r'^\t(.+?)\s*//\s*([0-9a-fA-F]+):\s*([0-9a-fA-F]+(?:\s+[0-9a-fA-F]+)*)', line)
        if not m:
            # Try without comment (shouldn't happen in objdump output)
            m2 = re.match(r'^\t(\S.+)$', line)
            if m2 and base_addr is not None:
                instr = m2.group(1).rstrip()
                addr = instrs[-1][0] + instrs[-1][2] if instrs else base_addr
                instrs.append((addr, instr, 4))
            continue
        if base_addr is not None:
            instr = m.group(1).rstrip()
            addr = int(m.group(2), 16)
            enc_words = m.group(3).strip().split()
            isize = len(enc_words) * 4
            instrs.append((addr, instr, isize))

    return func_name, base_addr, instrs


def resolve_branches(instrs, base_addr):
    """Convert numeric branch targets to labels."""
    branch_targets = set()
    branch_re = re.compile(
        r'^(s_branch|s_cbranch_scc[01]|s_cbranch_vccnz|s_cbranch_vccz|'
        r's_cbranch_execnz|s_cbranch_execz)\s+(\d+)'
    )

    for addr, instr, isize in instrs:
        m = branch_re.match(instr)
        if m:
            offset = int(m.group(2))
            if offset > 32768:
                offset = offset - 65536
            target_addr = addr + 4 + offset * 4
            branch_targets.add(target_addr)

    label_map = {}
    for i, addr in enumerate(sorted(branch_targets)):
        label_map[addr] = f".L{i}"

    new_instrs = []
    for addr, instr, isize in instrs:
        m = branch_re.match(instr)
        if m:
            op = m.group(1)
            offset = int(m.group(2))
            if offset > 32768:
                offset = offset - 65536
            target_addr = addr + 4 + offset * 4
            if target_addr in label_map:
                instr = f"{op} {label_map[target_addr]}"
        new_instrs.append((addr, instr, isize))

    return new_instrs, label_map


def emit_asm(func_name, instrs, label_map, metadata_yaml, group_segment_size=0):
    """Generate the .s file content."""
    lines = []

    lines.append(f".amdgcn_target \"amdgcn-amd-amdhsa--gfx950\"")
    lines.append("")
    lines.append(f".text")
    lines.append(f".globl {func_name}")
    lines.append(f".p2align 8")
    lines.append(f".type {func_name},@function")
    lines.append(f"{func_name}:")

    # Reverse label_map: addr -> label
    addr_to_label = {addr: label for addr, label in label_map.items()}

    for addr, instr, isize in instrs:
        if addr in addr_to_label:
            lines.append(f"{addr_to_label[addr]}:")
        lines.append(f"\t{instr}")

    lines.append(f".Lfunc_end:")
    lines.append(f".size {func_name}, .Lfunc_end-{func_name}")
    lines.append("")

    # Emit metadata
    lines.append(metadata_yaml)

    return "\n".join(lines) + "\n"


def build_metadata_asm(func_name, meta):
    """Build the .amdhsa_kernel metadata section from parsed YAML."""
    kernel = meta["amdhsa.kernels"][0]
    vgprs = kernel.get('.vgpr_count', 0)
    sgprs = kernel.get('.sgpr_count', 0)
    agprs = kernel.get('.agpr_count', 0)
    wf_size = kernel.get('.wavefront_size', 64)

    # Round up vgprs to multiple of 4 for accum_offset
    vgpr_aligned = ((vgprs + 3) // 4) * 4

    if agprs > 0:
        next_free = vgpr_aligned + agprs
        accum_off = vgpr_aligned
    else:
        next_free = vgpr_aligned
        accum_off = vgpr_aligned

    lines = []
    lines.append(".rodata")
    lines.append(".p2align 6")
    lines.append(f".amdhsa_kernel {func_name}")
    lines.append(f"  .amdhsa_group_segment_fixed_size {kernel.get('.group_segment_fixed_size', 0)}")
    lines.append(f"  .amdhsa_private_segment_fixed_size {kernel.get('.private_segment_fixed_size', 0)}")
    lines.append(f"  .amdhsa_kernarg_size {kernel.get('.kernarg_segment_size', 0)}")
    lines.append(f"  .amdhsa_next_free_vgpr {next_free}")
    lines.append(f"  .amdhsa_next_free_sgpr {sgprs}")
    lines.append(f"  .amdhsa_accum_offset {accum_off}")
    lines.append(f"  .amdhsa_float_round_mode_32 3")
    lines.append(f"  .amdhsa_float_round_mode_16_64 3")
    lines.append(f"  .amdhsa_float_denorm_mode_32 3")
    lines.append(f"  .amdhsa_float_denorm_mode_16_64 3")
    lines.append(f"  .amdhsa_ieee_mode 1")
    lines.append(f"  .amdhsa_dx10_clamp 1")
    lines.append(f"  .amdhsa_user_sgpr_kernarg_segment_ptr 1")
    lines.append(f"  .amdhsa_system_sgpr_workgroup_id_x 1")
    lines.append(f".end_amdhsa_kernel")
    lines.append("")

    args = kernel.get('.args', [])
    lines.append('.amdgpu_metadata')
    lines.append('---')
    lines.append('amdhsa.kernels:')
    lines.append(f'  - .name: {func_name}')
    lines.append(f'    .symbol: {func_name}.kd')
    lines.append(f'    .kernarg_segment_size: {kernel.get(".kernarg_segment_size", 0)}')
    lines.append(f'    .group_segment_fixed_size: {kernel.get(".group_segment_fixed_size", 0)}')
    lines.append(f'    .private_segment_fixed_size: {kernel.get(".private_segment_fixed_size", 0)}')
    lines.append(f'    .kernarg_segment_align: {kernel.get(".kernarg_segment_align", 8)}')
    lines.append(f'    .wavefront_size: {wf_size}')
    lines.append(f'    .sgpr_count: {sgprs}')
    lines.append(f'    .vgpr_count: {vgprs}')
    lines.append(f'    .agpr_count: {agprs}')
    lines.append(f'    .max_flat_workgroup_size: {kernel.get(".max_flat_workgroup_size", 512)}')
    lines.append(f'    .sgpr_spill_count: 0')
    lines.append(f'    .vgpr_spill_count: 0')
    lines.append(f'    .uses_dynamic_stack: false')
    lines.append(f'    .uniform_work_group_size: 1')
    if args:
        lines.append(f'    .args:')
        for arg in args:
            lines.append(f'      - .offset: {arg[".offset"]}')
            lines.append(f'        .size: {arg[".size"]}')
            lines.append(f'        .value_kind: {arg[".value_kind"]}')
            if ".address_space" in arg:
                lines.append(f'        .address_space: {arg[".address_space"]}')
    lines.append(f'amdhsa.target: amdgcn-amd-amdhsa--gfx950')
    lines.append(f'amdhsa.version:')
    lines.append(f'  - 1')
    lines.append(f'  - 2')
    lines.append('...')
    lines.append('.end_amdgpu_metadata')

    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input_disasm.s output.s [--metadata metadata.yaml]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Parse metadata from inline YAML if provided
    meta_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--metadata" and i + 1 < len(sys.argv):
            meta_path = sys.argv[i + 1]

    with open(input_path) as f:
        disasm_lines = f.readlines()

    func_name, base_addr, instrs = parse_disasm(disasm_lines)
    if not func_name:
        print("ERROR: Could not find function in disassembly")
        sys.exit(1)

    print(f"Function: {func_name}")
    print(f"Base addr: 0x{base_addr:x}")
    print(f"Instructions: {len(instrs)}")

    instrs, label_map = resolve_branches(instrs, base_addr)
    print(f"Labels: {len(label_map)}")

    # Default metadata for variable-K wgrad kernel
    meta = {
        "amdhsa.kernels": [{
            ".name": func_name,
            ".symbol": f"{func_name}.kd",
            ".agpr_count": 0,
            ".group_segment_fixed_size": 0,
            ".kernarg_segment_align": 8,
            ".kernarg_segment_size": 96,
            ".max_flat_workgroup_size": 512,
            ".private_segment_fixed_size": 0,
            ".sgpr_count": 59,
            ".sgpr_spill_count": 0,
            ".vgpr_count": 221,
            ".vgpr_spill_count": 0,
            ".wavefront_size": 64,
            ".uniform_work_group_size": 1,
            ".uses_dynamic_stack": False,
            ".args": [
                {".address_space": "global", ".offset": 0, ".size": 8, ".value_kind": "global_buffer"},
                {".address_space": "global", ".offset": 8, ".size": 8, ".value_kind": "global_buffer"},
                {".address_space": "global", ".offset": 16, ".size": 8, ".value_kind": "global_buffer"},
                {".address_space": "global", ".offset": 24, ".size": 8, ".value_kind": "global_buffer"},
                {".address_space": "global", ".offset": 32, ".size": 8, ".value_kind": "global_buffer"},
                {".address_space": "global", ".offset": 40, ".size": 8, ".value_kind": "global_buffer"},
                {".offset": 48, ".size": 4, ".value_kind": "by_value"},
                {".offset": 52, ".size": 4, ".value_kind": "by_value"},
                {".offset": 56, ".size": 4, ".value_kind": "by_value"},
                {".offset": 60, ".size": 4, ".value_kind": "by_value"},
                {".offset": 64, ".size": 4, ".value_kind": "by_value"},
                {".offset": 68, ".size": 4, ".value_kind": "by_value"},
                {".offset": 72, ".size": 4, ".value_kind": "by_value"},
                {".address_space": "global", ".offset": 80, ".size": 8, ".value_kind": "global_buffer"},
                {".address_space": "global", ".offset": 88, ".size": 8, ".value_kind": "global_buffer"},
            ],
        }],
    }

    if meta_path:
        import yaml
        with open(meta_path) as f:
            meta = yaml.safe_load(f)

    metadata_str = build_metadata_asm(func_name, meta)
    output = emit_asm(func_name, instrs, label_map, metadata_str)

    with open(output_path, "w") as f:
        f.write(output)

    print(f"Wrote {output_path} ({len(output)} bytes)")


if __name__ == "__main__":
    main()
