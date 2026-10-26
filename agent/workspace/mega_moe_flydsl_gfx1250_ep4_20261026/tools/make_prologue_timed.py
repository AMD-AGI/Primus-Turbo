"""Write prologue_timed.py: dispatch_prologue_kernel.py with block-0 realtime stamps at phase boundaries.

DBG[k] (i64, 100 MHz realtime): see STAMPS below. The DBG address is baked into the kernel, so run with the
FlyDSL cache off.
"""
src = open("/home/xiaompen/Turbo/primus_turbo/flydsl/mega/bf16/dispatch_prologue_kernel.py").read()


def sub(old, new, count=1):
    global src
    assert src.count(old) == count, (old, src.count(old))
    src = src.replace(old, new)


def stamp(k, indent=8):
    pad = " " * indent
    return (f"{pad}if (thread_index == fx.Int32(0)) & (block_index == fx.Int32(0)):\n"
            f"{pad}    st(fx.Int64(DBG.data_ptr()), fx.Int32({k}), read_clock())\n")


sub("from primus_turbo.flydsl.utils.prims import atomic_add, ld, st",
    "from primus_turbo.flydsl.utils.prims import atomic_add, ld, read_clock, st\n"
    "DBG = torch.zeros(64, dtype=torch.int64, device='cuda')")
sub("""        num_tile_blocks_resource = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
""", """        num_tile_blocks_resource = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
""" + stamp(0))
sub("""        grid_sync(workspace, thread_index, block_index, grid_blocks, rank, "dispatch_prologue/A:histogram")
""", stamp(1) + """        grid_sync(workspace, thread_index, block_index, grid_blocks, rank, "dispatch_prologue/A:histogram")
""" + stamp(2))
sub("""            "dispatch_prologue/B1:all-entered",
        )
""", """            "dispatch_prologue/B1:all-entered",
        )
""" + stamp(3))
sub("""                    push_expert_idx = push_expert_idx + fx.Int32(block_threads)
        xgmi_barrier(""", """                    push_expert_idx = push_expert_idx + fx.Int32(block_threads)
""" + stamp(4) + """        xgmi_barrier(""")
sub("""            "dispatch_prologue/B3:all-gather-landed",
        )
""", """            "dispatch_prologue/B3:all-gather-landed",
        )
""" + stamp(5))
sub("""                    running_pool_offset = running_pool_offset + padded_count
            fx.gpu.barrier()
""", """                    running_pool_offset = running_pool_offset + padded_count
            fx.gpu.barrier()
""" + stamp(12, 12))
sub("""                expert_idx = expert_idx + fx.Int32(block_threads)
            fx.gpu.barrier()
""", """                expert_idx = expert_idx + fx.Int32(block_threads)
            fx.gpu.barrier()
""" + stamp(13, 12))
sub("""                comm_task_idx = comm_task_idx + fx.Int32(block_threads)
            fx.gpu.barrier()
""", """                comm_task_idx = comm_task_idx + fx.Int32(block_threads)
            fx.gpu.barrier()
""" + stamp(14, 12))
sub("""                        comm_task_counter = comm_task_counter + 1
            if thread_index < fx.Int32(num_experts_per_rank):""", """                        comm_task_counter = comm_task_counter + 1
""" + stamp(15, 12) + """            if thread_index < fx.Int32(num_experts_per_rank):""")
sub("""        grid_sync(workspace, thread_index, block_index, grid_blocks, rank, "dispatch_prologue/C:table-built")
""", stamp(6) + """        grid_sync(workspace, thread_index, block_index, grid_blocks, rank, "dispatch_prologue/C:table-built")
""" + stamp(7))
sub("""        grid_sync(workspace, thread_index, block_index, grid_blocks, rank, "dispatch_prologue/D:scatter-done")
""", stamp(8) + """        grid_sync(workspace, thread_index, block_index, grid_blocks, rank, "dispatch_prologue/D:scatter-done")
""" + stamp(9))
sub("""            "dispatch_prologue/E:origins-landed",
        )
""", """            "dispatch_prologue/E:origins-landed",
        )
""" + stamp(11))
sub("""            reset_idx = reset_idx + fx.Int32(grid_stride)
""", """            reset_idx = reset_idx + fx.Int32(grid_stride)
""" + stamp(10))
open("/home/xiaompen/mega_scratch/ep4c/prologue_timed.py", "w").write(src)
print("wrote prologue_timed.py")
