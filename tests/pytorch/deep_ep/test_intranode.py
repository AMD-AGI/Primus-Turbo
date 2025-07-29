import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

import primus_turbo.pytorch as pt
from tests.pytorch.ref.deep_ep_ref import (
    calc_diff,
    check_data,
    get_dispatch_layout_ref,
    inplace_unique,
    per_token_cast_back,
    per_token_cast_to_fp8,
)


@instantiate_parametrized_tests
class DeepEPIntranodeTestCase(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        buffer = pt.deep_ep.Buffer(dist.group.WORLD, int(1e9))
        return buffer

    @skip_if_lt_x_gpu(2)
    @parametrize("num_tokens", [4096])
    @parametrize("hidden", [4096])
    @parametrize("num_topk", [8])
    @parametrize("num_experts", [128])
    @parametrize("num_sms", [24])
    def test_intranode(self, num_tokens: int, hidden: int, num_topk: int, num_experts: int, num_sms: int):
        # Random data
        buffer = self._init_process()
        group = dist.group.WORLD
        rank = self.rank
        num_ranks = group.size()
        torch.manual_seed(42 + rank)

        x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * rank
        x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        x_e4m3 = per_token_cast_to_fp8(x)
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs() + 1
        topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
        topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device="cuda") * rank
        topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device="cuda")
        rank_idx = topk_idx // (num_experts // num_ranks)
        rank_idx.masked_fill_(topk_idx == -1, -1)
        inplace_unique(rank_idx, num_ranks)

        invalid_index = -1
        num_experts_per_rank = num_experts // num_ranks
        invalid_index = num_experts_per_rank

        num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(
            topk_idx, num_experts
        )

        ref_num_tokens_per_rank, _, ref_num_tokens_per_expert, ref_is_token_in_rank = get_dispatch_layout_ref(
            topk_idx, num_experts, num_ranks=num_ranks
        )
        ref_gbl_num_tokens_per_rank = ref_num_tokens_per_rank.clone()
        ref_gbl_num_tokens_per_expert = ref_num_tokens_per_expert.clone()
        dist.all_reduce(ref_gbl_num_tokens_per_rank, group=group)
        dist.all_reduce(ref_gbl_num_tokens_per_expert, group=group)

        assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
        assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
        assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)

        # Config
        nvl_buffer_size = 256
        config = pt.deep_ep.Config(num_sms, 8, nvl_buffer_size)

        for previous_mode in (False, True):
            for async_mode in (False, True):
                for current_x in (x_pure_rand, x, x_e4m3):
                    for with_topk in (False, True):
                        if rank == 0:
                            print(
                                f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...',
                                flush=True,
                                end="",
                            )

                        dispatch_args = {
                            "x": current_x,
                            "num_tokens_per_rank": num_tokens_per_rank,
                            "is_token_in_rank": is_token_in_rank,
                            "num_tokens_per_expert": num_tokens_per_expert,
                            "config": config,
                            "async_finish": async_mode,
                        }
                        if with_topk:
                            dispatch_args.update(
                                {
                                    "topk_idx": topk_idx,
                                    "topk_weights": (
                                        topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
                                    ),
                                }
                            )
                        if previous_mode:
                            dispatch_args.update({"previous_event": buffer.capture()})
                        (
                            recv_x,
                            recv_topk_idx,
                            recv_topk_weights,
                            recv_num_tokens_per_expert_list,
                            handle,
                            event,
                        ) = buffer.dispatch(**dispatch_args)
                        event.current_stream_wait() if async_mode else ()
                        recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

                        # Checks
                        rank_prefix_matrix = handle[0]
                        assert ref_gbl_num_tokens_per_rank[rank].item() == recv_x.size(
                            0
                        ), f"{ref_gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}"
                        assert (
                            ref_gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()
                            == recv_num_tokens_per_expert_list
                        )
                        if current_x is not x_pure_rand:
                            check_data(recv_x, rank_prefix_matrix, num_ranks=num_ranks, rank=rank)
                        if with_topk:
                            # Check `topk_idx`
                            assert (
                                recv_topk_idx.eq(invalid_index)
                                | ((recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks)))
                            ).sum().item() == recv_topk_idx.numel()
                            for i, count in enumerate(recv_num_tokens_per_expert_list):
                                assert recv_topk_idx.eq(i).sum().item() == count

                            # Check `topk_weights`
                            if current_x is not x_pure_rand:
                                recv_topk_weights[recv_topk_idx.eq(invalid_index)] = recv_topk_weights.amax(
                                    dim=1, keepdim=True
                                ).expand_as(recv_topk_weights)[recv_topk_idx.eq(invalid_index)]
                                check_data(
                                    recv_topk_weights, rank_prefix_matrix, num_ranks=num_ranks, rank=rank
                                )

                        # Test cached dispatch (must without top-k staffs)
                        # NOTES: handle must be refreshed
                        if not with_topk:
                            dispatch_args = {
                                "x": current_x,
                                "handle": handle,
                                "config": config,
                                "async_finish": async_mode,
                            }
                            if previous_mode:
                                dispatch_args.update({"previous_event": buffer.capture()})
                            recv_x, _, _, _, _, event = buffer.dispatch(**dispatch_args)
                            event.current_stream_wait() if async_mode else ()
                            recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
                            if current_x is not x_pure_rand:
                                check_data(recv_x, rank_prefix_matrix, num_ranks=num_ranks, rank=rank)

                        # Test combine
                        combine_args = {
                            "x": recv_x,
                            "handle": handle,
                            "config": config,
                            "async_finish": async_mode,
                        }
                        if with_topk:
                            combine_args.update({"topk_weights": recv_topk_weights})
                        if previous_mode:
                            dispatch_args.update({"previous_event": buffer.capture()})
                        combined_x, combined_topk_weights, event = buffer.combine(**combine_args)
                        event.current_stream_wait() if async_mode else ()
                        check_x = combined_x.float() / is_token_in_rank.sum(dim=1).unsqueeze(1)
                        ref_x = x_pure_rand if current_x is x_pure_rand else x
                        assert calc_diff(check_x, ref_x) < 5e-6
                        if with_topk:
                            check_topk_weights = (
                                combined_topk_weights
                                if (current_x is x_pure_rand)
                                else (combined_topk_weights / is_token_in_rank.sum(dim=1).unsqueeze(1))
                            )
                            ref_topk_weights = (
                                topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
                            )
                            assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9

                        if rank == 0:
                            print(" passed", flush=True)


if __name__ == "__main__":
    run_tests()
