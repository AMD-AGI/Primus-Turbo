import torch
import torch.distributed as dist
from primus_turbo.pytorch.cco.pyhip_runtime_wrapper import HIPRuntimeLibrary, hipMemcpyKindEnum


class SymmetricMemory:
    def __init__(self, group: dist.ProcessGroup, alloc_size: int, signal_pad_size: int = 1024):
        if alloc_size <= 0:
            raise ValueError(
                f"requested alloc size must be greater than 0, got {alloc_size}")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "SymmetricMemory requires CUDA/HIP device support.")

        self.lib = HIPRuntimeLibrary()
        self.group = group
        self.rank = group.rank()
        self.num_ranks = group.size()
        self.buffer_size = alloc_size

        # set device to rank
        torch.cuda.set_device(self.rank)

        # allocate memory and rendezvous on memory
        buffer_ptr = self.lib.hipMalloc(alloc_size)
        signal_pad_ptr = self.lib.hipMalloc(signal_pad_size)
        self.lib.hipMemset(buffer_ptr, 0, alloc_size)
        self.lib.hipMemset(signal_pad_ptr, 0, signal_pad_size)

        def rendezvous(ptr):
            mem_handle = self.lib.hipIpcGetMemHandle(ptr)
            mem_handle_bytes = self.lib.mem_handle_to_bytes(
                mem_handle)
            mem_handle_list = [None] * self.num_ranks
            dist.all_gather_object(
                mem_handle_list, mem_handle_bytes, group=self.group)

            ptr_list = [None] * self.num_ranks
            for rank in range(self.num_ranks):
                if rank == self.rank:
                    ptr_list[rank] = ptr
                else:
                    ptr_list[rank] = self.lib.hipIpcOpenMemHandle(
                        mem_handle_list[rank]).value
            return ptr_list

        self.buffer_ptrs = rendezvous(buffer_ptr)
        self.signal_pad_ptrs = rendezvous(signal_pad_ptr)

        def _ptr_to_int(ptr):
            assert ptr is not None, "ptr is None"
            if hasattr(ptr, "value"):
                return 0 if ptr.value is None else int(ptr.value)
            return int(ptr)

        self.buffer_ptrs_dev = torch.tensor(
            [_ptr_to_int(ptr) for ptr in self.buffer_ptrs],
            dtype=torch.int64,
            device="cuda",
        )
        self.signal_pad_ptrs_dev = torch.tensor(
            [_ptr_to_int(ptr) for ptr in self.signal_pad_ptrs],
            dtype=torch.int64,
            device="cuda",
        )

        self.is_destroyed = False

        # barrier on buffer and signal pad
        self.group.barrier()

    def destroy(self):
        if self.is_destroyed:
            return
        self.is_destroyed = True
        for rank in range(self.num_ranks):
            if rank == self.rank:
                self.lib.hipFree(self.buffer_ptrs[rank])
                self.lib.hipFree(self.signal_pad_ptrs[rank])
            else:
                self.lib.hipIpcCloseMemHandle(self.buffer_ptrs[rank])
                self.lib.hipIpcCloseMemHandle(self.signal_pad_ptrs[rank])

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            # Avoid throwing from destructor path.
            pass


if __name__ == "__main__":
    import os

    rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(rank)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group("nccl",  world_size=world_size, rank=rank)
    group = dist.new_group(list(range(world_size)))
    symm_mem = SymmetricMemory(group, 1024)

    def self_test_ring_write(symm: SymmetricMemory, base_value: int = 1000):
        """Write to next rank's buffer_ptr via IPC and verify local receive."""
        if symm.num_ranks < 2:
            if symm.rank == 0:
                print("Skip IPC ring-write self-test: world_size < 2")
            return

        bytes_needed = torch.tensor([], dtype=torch.int32).element_size()
        if symm.buffer_size < bytes_needed:
            raise ValueError(
                f"buffer_size={symm.buffer_size} is too small for self-test ({bytes_needed} bytes)"
            )

        dst_rank = (symm.rank + 1) % symm.num_ranks
        src_rank = (symm.rank - 1 + symm.num_ranks) % symm.num_ranks
        stream_ptr = torch.cuda.current_stream().cuda_stream

        send = torch.tensor([base_value + symm.rank],
                            dtype=torch.int32, device="cuda")
        recv = torch.zeros(1, dtype=torch.int32, device="cuda")

        # Write local marker to next rank's buffer[0:4].
        symm.lib.hipMemcpyAsync(
            symm.buffer_ptrs[dst_rank],
            int(send.data_ptr()),
            send.numel() * send.element_size(),
            hipMemcpyKindEnum.hipMemcpyDeviceToDevice,
            stream_ptr,
        )
        torch.cuda.synchronize()
        symm.group.barrier()

        # Read local buffer[0:4] back to tensor and verify from previous rank.
        symm.lib.hipMemcpyAsync(
            int(recv.data_ptr()),
            symm.buffer_ptrs[symm.rank],
            recv.numel() * recv.element_size(),
            hipMemcpyKindEnum.hipMemcpyDeviceToDevice,
            stream_ptr,
        )
        torch.cuda.synchronize()

        expected = base_value + src_rank
        got = int(recv.item())
        ok = got == expected

        gathered = [None] * symm.num_ranks
        dist.all_gather_object(
            gathered, (symm.rank, got, expected, ok), group=symm.group)
        if symm.rank == 0:
            print("SymmetricMemory IPC ring-write self-test results:")
            for item in gathered:
                print(
                    f"  rank={item[0]} got={item[1]} expected={item[2]} ok={item[3]}")

        if not all(item[3] for item in gathered):
            raise AssertionError(
                f"IPC ring-write self-test failed: {gathered}")

    self_test_ring_write(symm_mem)

    symm_mem.destroy()
