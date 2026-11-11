"""CPU-only MoE routing trace tests; no model weights/NPU needed."""
import json
import pytest
import torch
from accuracy_checker.moe_routing_trace import MoERoutingCollector, compare_traces, read_jsonl


def test_capture_compare_and_load(tmp_path):
    ref = MoERoutingCollector("ref", sample_id="prompt-1")
    quant = MoERoutingCollector("quant", sample_id="prompt-1")
    ref.record(12, torch.tensor([[7, 12, 31, 46], [1, 2, 3, 4]]),
               torch.tensor([[.42, .29, .17, .12], [.4, .3, .2, .1]]))
    quant.record(12, torch.tensor([[7, 12, 31, 52], [1, 2, 3, 4]]),
                 torch.tensor([[.39, .27, .19, .15], [.4, .3, .2, .1]]))
    p = tmp_path / "ref.jsonl"
    ref.write_jsonl(p)
    result = compare_traces(read_jsonl(p), quant.records)
    assert result["summary"]["paired_tokens"] == 2
    assert result["summary"]["expert_set_change_rate"] == .5
    assert result["token_details"][0]["topk_overlap"] == .75
    assert result["token_details"][0]["jaccard"] == .6
    assert result["token_details"][0]["expert_replacements"] == 1
    assert result["ref_load"]["12"]["expert_token_counts"][46] == 1
    assert result["quant_load"]["12"]["expert_token_counts"][52] == 1


def test_order_change_is_not_expert_set_change():
    a = MoERoutingCollector("ref")
    b = MoERoutingCollector("quant")
    a.record(1, torch.tensor([[1, 2]]), torch.tensor([[.2, .8]]))
    b.record(1, torch.tensor([[2, 1]]), torch.tensor([[.8, .2]]))
    out = compare_traces(a.records, b.records)
    assert out["summary"]["expert_set_change_rate"] == 0
    assert out["summary"]["mean_weight_l1"] == pytest.approx(0)


def test_mask_batch_and_alignment():
    a = MoERoutingCollector("ref", "sample")
    a.record(2, torch.tensor([[[0, 1], [1, 2]], [[2, 3], [3, 4]]]),
             torch.ones(2, 2, 2) / 2,
             token_positions=torch.tensor([[10, 11], [10, 11]]),
             valid_mask=torch.tensor([[True, False], [True, True]]))
    assert len(a.records) == 3
    assert {x["sample_id"] for x in a.records} == {"sample:0", "sample:1"}
    b = MoERoutingCollector("quant", "sample")
    b.record(2, torch.tensor([[[0, 1]], [[2, 3]]]), torch.ones(2, 1, 2)/2,
             token_positions=torch.tensor([[10], [10]]))
    r = compare_traces(a.records, b.records)
    assert r["summary"]["paired_tokens"] == 2
    assert len(r["ref_only_keys"]) == 1


def test_duplicate_key_rejected():
    a = MoERoutingCollector("ref")
    a.record(0, torch.tensor([[0]]), torch.tensor([[1.0]]))
    a.record(0, torch.tensor([[0]]), torch.tensor([[1.0]]))
    with pytest.raises(ValueError, match="duplicate"):
        compare_traces(a.records, [])


def test_invalid_shapes_rejected():
    a = MoERoutingCollector("ref")
    with pytest.raises(ValueError):
        a.record(0, torch.tensor([[0, 1]]), torch.tensor([[1.]]))
