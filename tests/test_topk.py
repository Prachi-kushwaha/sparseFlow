import torch
import pytest

from sparse_attention.indexing.topk_indices import build_topk

class TestTopk:

    def test_build_topk(self):
        torch.manual_seed(0)
        scores = torch.randn(2,2,4,10)
        values, indices = build_topk(scores, k=3, dim=-1, sorted=False)
        ref_val, ref_indices = torch.topk(scores, k=3, dim=-1, sorted=False)
        assert torch.equal(values, ref_val)
        assert torch.equal(indices, ref_indices)

