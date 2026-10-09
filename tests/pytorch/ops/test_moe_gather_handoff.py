###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU contract tests: run directly without importing the GPU extension package."""

import gc
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

_PATH = Path(__file__).resolve().parents[3] / "primus_turbo/pytorch/ops/moe_gather.py"
_SPEC = importlib.util.spec_from_file_location("moe_gather_under_test", _PATH)
gather = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gather)


class TestGatherHandoff(unittest.TestCase):
    def setUp(self):
        gather._PERMUTED_ACTIVATION_SEAM_TABLE.clear()
        gather._BACKWARD_GATHER_OUTPUTS.clear()
        self.source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        self.indices = torch.tensor([2, 0, 2, 1], dtype=torch.int32)
        self.rowmap = torch.zeros((3, 65), dtype=torch.int32)
        self.plan = gather._BackwardGatherPlan(self.source, self.indices, self.rowmap)

    def tearDown(self):
        gather._PERMUTED_ACTIVATION_SEAM_TABLE.clear()
        gather._BACKWARD_GATHER_OUTPUTS.clear()

    def wrapper(self):
        state = gather._BackwardGatherState(self.source, self.plan)
        return gather._BackwardGatherTensor(state)

    def materialize(self, source, *_args):
        return source.index_select(0, self.indices.long())

    def test_forward_handoff_is_single_consumer(self):
        placeholder = torch.empty(1, 4).expand(4, 4)
        probs = torch.ones(4)
        gather.register_permuted_activation_seam(placeholder, self.source, self.indices, probs)
        payload = gather.lookup_permuted_activation_seam(placeholder.detach())
        self.assertIs(payload[0], self.source)
        self.assertIs(payload[1], self.indices)
        self.assertIs(payload[2], probs)
        with self.assertRaisesRegex(RuntimeError, "no gather metadata"):
            gather.lookup_permuted_activation_seam(placeholder)

    def test_duplicate_registration_and_layout_mismatch_are_rejected(self):
        placeholder = torch.empty(1, 4).expand(4, 4)
        gather.register_permuted_activation_seam(placeholder, self.source, self.indices, None)
        with self.assertRaisesRegex(RuntimeError, "Unconsumed"):
            gather.register_permuted_activation_seam(placeholder, self.source, self.indices, None)
        with self.assertRaisesRegex(RuntimeError, "layout mismatch"):
            gather.lookup_permuted_activation_seam(placeholder.t())

    def test_ordinary_dense_tensor_has_no_handoff(self):
        self.assertIsNone(gather.lookup_permuted_activation_seam(self.source))

    def test_forward_plan_is_disabled_on_cpu(self):
        with patch.dict(
            "os.environ",
            GPTOSS_FUSED_BACKWARD_PERMUTE_QUANT="1",
            PRIMUS_TP="1",
            PRIMUS_EP="1",
            MOE_SKIP_IDENTITY_SORT="1",
        ):
            self.assertIsNone(gather._new_backward_gather_plan(self.source, self.indices, self.rowmap))

    def test_routing_mutation_is_rejected(self):
        self.indices[0] = 0
        with self.assertRaisesRegex(RuntimeError, "routing metadata was mutated"):
            self.plan.validate()

    def test_source_mutation_is_rejected_before_materialization(self):
        wrapper = self.wrapper()
        self.source.add_(1)
        with self.assertRaisesRegex(RuntimeError, "source gradient was mutated"):
            wrapper._gather_state.validate_source()

    def test_detach_alias_and_identity_view_stay_lazy(self):
        wrapper = self.wrapper()
        aliases = [wrapper.detach(), torch.ops.aten.alias.default(wrapper), wrapper.view(4, 4)]
        self.assertTrue(all(alias._gather_state is wrapper._gather_state for alias in aliases))
        self.assertIsNone(wrapper._gather_state.materialized)

    def test_dense_fallback_preserves_shared_alias_mutations(self):
        with patch.object(
            torch.ops.te_moe, "unpermute_mask_map_bwd_no_probs", side_effect=self.materialize, create=True
        ) as op:
            wrapper = self.wrapper()
            alias = wrapper.detach()
            expected = self.materialize(self.source)
            self.assertIs(wrapper.add_(2), wrapper)
            expected.add_(2)
            torch.testing.assert_close(alias.clone(), expected, rtol=0, atol=0)
            flat = alias.view(-1)
            flat[::3].zero_()
            expected.view(-1)[::3].zero_()
            torch.testing.assert_close(wrapper.clone(), expected, rtol=0, atol=0)
            self.assertEqual(op.call_count, 1)
            self.assertIs(torch.add(alias, 1, out=alias), alias)
            torch.testing.assert_close(wrapper.clone(), expected + 1, rtol=0, atol=0)

    def test_copy_returns_original_wrapper_and_updates_alias(self):
        with patch.object(
            torch.ops.te_moe, "unpermute_mask_map_bwd_no_probs", side_effect=self.materialize, create=True
        ):
            wrapper = self.wrapper()
            alias = wrapper.detach()
            self.assertIs(wrapper.copy_(torch.full((4, 4), 7.0)), wrapper)
            torch.testing.assert_close(alias.clone(), torch.full((4, 4), 7.0))

    def test_metadata_mutation_fails_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, "Metadata mutation"):
            self.wrapper().transpose_(0, 1)

    def test_materialized_or_mismatched_plan_cannot_bypass_changes(self):
        with patch.object(
            torch.ops.te_moe, "unpermute_mask_map_bwd_no_probs", side_effect=self.materialize, create=True
        ):
            wrapper = self.wrapper()
            with torch.no_grad():
                actual, kwargs = gather.resolve_backward_gather(wrapper, None)
            self.assertEqual(kwargs, {})
            actual.add_(5)
            with torch.no_grad():
                resolved, kwargs = gather.resolve_backward_gather(wrapper, self.plan)
            self.assertIs(resolved, actual)
            self.assertEqual(kwargs, {})
            torch.testing.assert_close(resolved, self.materialize(self.source) + 5)

    def test_output_registration_does_not_retain_autograd_owner(self):
        # Production eligibility uses hidden=2880; ownership behavior is shape-independent.
        owner = type("Owner", (), {})()
        out = torch.empty(self.plan.shape)
        gather.enroll_backward_gather_output(out, self.plan, owner)
        self.assertIn(out.data_ptr(), gather._BACKWARD_GATHER_OUTPUTS)
        del owner
        gc.collect()
        self.assertNotIn(out.data_ptr(), gather._BACKWARD_GATHER_OUTPUTS)

    def test_ineligible_unpermute_consumes_registration_without_fusing(self):
        owner = type("Owner", (), {})()
        out = torch.empty(self.plan.shape)
        gather.enroll_backward_gather_output(out, self.plan, owner)
        plan = gather.claim_backward_gather_output(out, self.rowmap, torch.ones(4), None, 3, 32, 4)
        self.assertIsNone(plan)
        self.assertNotIn(out.data_ptr(), gather._BACKWARD_GATHER_OUTPUTS)

    def test_higher_order_backward_uses_dense_fallback(self):
        self.assertIsNone(gather.make_backward_gather(self.source, self.plan))


if __name__ == "__main__":
    unittest.main()
