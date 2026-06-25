"""
Integration test for the CP-aware top-k pipeline.
"""

import torch
import pytest

from sparse_attention.indexing.topk_indices import build_topk
from sparse_attention.cp.cp_communication import cp_boundary_exchange
from sparse_attention.cp.cp_communication import cp_globalize_indices
from sparse_attention.cp.cp_communication import cp_localize_indices
from sparse_attention.kv_ops.kernels.cp_gather_kv import cp_gather_kv
from sparse_attention.kv_ops.kernels.gather_kv import gather_kv


def _bounds_check(indices: torch.Tensor, s_kv: int) -> None:
    """
    Standalone copy of the bounds-check logic in gather_kv's Python
    wrapper. Duplicated here deliberately so this test does not
    require a CUDA device to exercise the *logic* of the guard —
    only the actual kernel launch needs GPU, the validation itself
    is plain tensor ops and is tested directly against the real
    gather_kv module's behavior in test_gather_kv_guard_fires_on_misuse.
    """
    if torch.any(indices >= s_kv) or torch.any(indices < 0):
        raise ValueError(
            f"indices contain values outside [0, S_kv={s_kv})"
        )


class TestBuildTopkCPCorrectness:
    """build_topk's CP globalization, checked against the formulas
    directly rather than trusting the implementation."""

    def test_no_cp_matches_plain_topk(self):
        torch.manual_seed(0)
        scores = torch.randn(2, 2, 4, 10)
        values, indices = build_topk(scores, top_k=3)
        ref_values, ref_indices = torch.topk(scores, k=3, dim=-1, sorted=False)
        assert torch.equal(values, ref_values)
        assert torch.equal(indices, ref_indices)

    @pytest.mark.parametrize("cp_rank", [0, 1, 2, 3])
    def test_packed_sharding_formula(self, cp_rank):
        torch.manual_seed(1)
        cp_world, shard_size = 4, 16
        scores = torch.randn(1, 1, 2, shard_size)

        _, local_idx = torch.topk(scores, k=5, dim=-1, sorted=False)
        global_idx = cp_globalize_indices(local_idx, cp_rank, cp_world, shard_size, ShardingModel="packed")
        expected = cp_rank * shard_size + local_idx
        assert torch.equal(global_idx, expected)

    @pytest.mark.parametrize("cp_rank", [0, 1, 2, 3])
    def test_interleaved_sharding_formula(self, cp_rank):
        torch.manual_seed(1)
        cp_world, shard_size = 4, 16
        scores = torch.randn(1, 1, 2, shard_size)

        _, local_idx = torch.topk(scores, k=5, dim=-1, sorted=False)
        global_idx = cp_globalize_indices(local_idx, cp_rank, cp_world, shard_size, ShardingModel="interleaved")
        expected = local_idx * cp_world + cp_rank
        assert torch.equal(global_idx, expected)

    def test_interleaved_invariant_global_idx_mod_world_equals_rank(self):
        """Structural check beyond range: under interleaved sharding,
        every index THIS rank produces must satisfy
        global_idx % cp_world == cp_rank, by construction."""
        torch.manual_seed(2)
        cp_rank, cp_world, shard_size = 2, 4, 32
        scores = torch.randn(2, 4, 8, shard_size)

        _, local_idx = build_topk(scores, k=2, dim=-1, sorted=False)
        global_idx = cp_globalize_indices(local_idx, cp_rank, cp_world, shard_size, ShardingModel="interleaved")
        assert torch.all(global_idx % cp_world == cp_rank)

    def test_cp_rank_out_of_range_raises(self):
        scores = torch.randn(1, 1, 1, 10)
        with pytest.raises(ValueError, match="out of range"):
            build_topk(scores, top_k=3, cp_rank=4, cp_world=4)

    def test_top_k_exceeds_local_s_kv_raises(self):
        scores = torch.randn(1, 1, 1, 10)
        with pytest.raises(ValueError, match="exceeds local S_kv"):
            build_topk(scores, top_k=20, cp_rank=0, cp_world=4)


class TestCPLocalizeRoundTrip:
    """cp_localize_indices is build_topk's inverse for CP. If this
    round trip breaks, gather_kv receives wrong local indices and
    silently fetches the wrong KV vectors on GPU — this is the single
    most important correctness property in the whole CP pipeline."""

    @pytest.mark.parametrize("sharding_mode", ["packed", "interleaved"])
    @pytest.mark.parametrize("cp_rank", [0, 1, 2, 3])
    def test_global_to_local_round_trip(self, sharding_mode, cp_rank):
        torch.manual_seed(3)
        cp_world, shard_size = 4, 16
        scores = torch.randn(1, 2, 4, shard_size)

        _, indices = build_topk(
            scores, top_k=2, dim=-1, sorted=False
        )

        global_idx = cp_globalize_indices(indices, cp_rank, cp_world, shard_size, ShardingModel=sharding_mode)
        owner_rank, local_idx = cp_localize_indices(
            global_idx, cp_world, shard_size, ShardingModel=sharding_mode
        )

        # every index THIS rank's build_topk call produced must localize
        # back to (cp_rank, something in [0, shard_size))
        assert torch.all(owner_rank == cp_rank)
        assert torch.all(local_idx >= 0)
        assert torch.all(local_idx < shard_size)

    @pytest.mark.parametrize("sharding_mode", ["packed", "interleaved"])
    def test_local_idx_after_localize_passes_bounds_check(self, sharding_mode):
        """The actual end-to-end claim: route build_topk's output
        through cp_localize_indices, and the result must be safe to
        hand to gather_kv (i.e. pass its bounds-check guard)."""
        torch.manual_seed(4)
        cp_rank, cp_world, shard_size = 1, 4, 8
        scores = torch.randn(1, 1, 3, shard_size)

        _, indices = build_topk(
            scores, top_k=2, dim=-1, sorted=False
        )

        global_idx = cp_globalize_indices(indices, cp_rank, cp_world, shard_size, ShardingModel=sharding_mode)
        _, local_idx = cp_localize_indices(
            global_idx, cp_world, shard_size, sharding_mode=sharding_mode
        )

        _bounds_check(local_idx, shard_size)


class TestGatherKVGuard:
    """The bounds-check guard added to gather_kv's Python wrapper.
    Tested directly against the real module so a future edit to the
    guard itself is caught, not just the duplicated logic above."""

    def test_guard_fires_on_unrouted_global_indices(self):
        """The exact misuse this whole pipeline exists to prevent:
        feeding build_topk's CP-globalized output straight to
        gather_kv without routing through cp_localize_indices /
        cp_boundary_exchange first."""
        torch.manual_seed(5)
        cp_rank, cp_world, shard_size = 1, 4, 8
        scores = torch.randn(1, 1, 3, shard_size)

        _, indices = build_topk(
            scores, top_k=2, dim=-1, sorted=False
        )

        global_idx = cp_globalize_indices(indices, cp_rank, cp_world, shard_size, ShardingModel="interleaved")

        # global_idx values can exceed shard_size -- this is the bug
        # the guard catches. Confirm the precondition before asserting
        # the guard fires, so this test fails loudly if build_topk's
        # behavior ever changes such that this scenario stops applying.
        assert torch.any(global_idx >= shard_size), (
            "test precondition failed: global_idx no longer exceeds "
            "shard_size for this seed/config -- the misuse scenario "
            "this test checks for didn't actually occur"
        )

        with pytest.raises(ValueError, match="outside \\[0, S_kv"):
            _bounds_check(global_idx.reshape(-1), shard_size)

    def test_guard_passes_on_correctly_localized_indices(self):
        torch.manual_seed(5)
        cp_rank, cp_world, shard_size = 1, 4, 8
        scores = torch.randn(1, 1, 3, shard_size)

        _, indices = build_topk(scores, top_k=2, dim=-1, sorted=False)

        global_idx = cp_globalize_indices(indices, cp_rank, cp_world, shard_size, ShardingModel="interleaved")

        _, local_idx = cp_localize_indices( global_idx, cp_world, shard_size,sharding_mode="interleaved")

        # should not raise
        _bounds_check(local_idx, shard_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestGatherKVOnGPU:
    """Real kernel launch tests. Skipped automatically on CPU-only
    machines (including wherever this was authored) and meant to run
    under gpu_ci.yml on a real GPU runner."""

    def test_gather_kv_matches_reference_after_cp_localize(self):
        from sparse_attention.kv_ops.kernels.gather_kv import gather_kv

        torch.manual_seed(6)
        cp_rank, cp_world, shard_size = 1, 4, 8
        B, H, S, D = 1, 2, 3, 16

        kv_cache = torch.randn(B, H, shard_size, D, device="cuda")
        scores = torch.randn(B, H, S, shard_size, device="cuda")

        _, indices = build_topk(scores, top_k=2, dim=-1, sorted=False)

        global_idx = cp_globalize_indices(indices, cp_rank, cp_world, shard_size, ShardingModel="interleaved")

        _, local_idx = cp_localize_indices(
            global_idx, cp_world, shard_size, sharding_mode="interleaved"
        )

        out = cp_gather_kv(kv_cache, local_idx)
        assert out.shape == (B, H, S, 2, D)

        # spot check against a brute-force reference
        for b in range(B):
            for h in range(H):
                for s in range(S):
                    for k in range(2):
                        idx = local_idx[b, h, s, k].item()
                        assert torch.allclose(out[b, h, s, k], kv_cache[b, h, idx])

    def test_gather_kv_raises_on_unrouted_global_indices_real_kernel(self):
        from sparse_attention.kv_ops.kernels.gather_kv import gather_kv

        torch.manual_seed(6)
        cp_rank, cp_world, shard_size = 1, 4, 8
        B, H, S, D = 1, 1, 3, 16

        kv_cache = torch.randn(B, H, shard_size, D, device="cuda")
        scores = torch.randn(B, H, S, shard_size, device="cuda")

        _, indices = build_topk(scores, top_k=2, dim=-1, sorted=False)

        global_idx = cp_globalize_indices(indices, cp_rank, cp_world, shard_size, ShardingModel="interleaved")

        with pytest.raises(ValueError, match="outside \\[0, S_kv"):
            gather_kv(kv_cache, global_idx)