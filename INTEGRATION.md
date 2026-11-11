# MoE Routing Tracer — integration guide

`accuracy_checker/moe_routing_trace.py` is standalone and runs on CPU, CUDA, and Ascend NPU torch tensors.  The tracer consumes actual selected IDs and aggregation weights rather than recomputing model-specific router functions.

## 1. Minimal integration into `layer1_block_compare.py`

The repository's `_moe_forward_chunked` calls `_compute_moe_topk_routing` around lines 2248-2250. Right **after** that call add:

```python
# Optional; install a collector by assigning comparator.moe_trace_collector.
collector = getattr(self, "moe_trace_collector", None)
if collector is not None:
    collector.record(
        layer_idx=layer_idx,
        indices=topk_indices,
        weights=topk_scores,
        # IMPORTANT: supply stable sample IDs and absolute positions if the
        # caller runs multiple chunks or autoregressive decode.
    )
```

Do **not** record from `gate` logits before `_compute_moe_topk_routing`: GLM/DeepSeek sigmoid + noaux_tc and Kimi precomputed indices differ from plain softmax top-k.

This minimal integration is only safe for one full-sequence, nonchunked sample per collector. For production L1, instrument the comparator's model-side execution to bind separate ref and quant collectors **and** pass proper `sample_id`, `token_positions` and padding mask. Never silently use the default ID across multiple prompts or chunks; the comparison explicitly rejects duplicate keys.

## 2. Create collectors around the actual model execution

```python
from accuracy_checker.moe_routing_trace import MoERoutingCollector
ref = MoERoutingCollector(side="ref", sample_id="prompt-001")
quant = MoERoutingCollector(side="quant", sample_id="prompt-001")
# attach to comparator/ref & quant execution contexts before each run
# ... run respective paths without modifying routing outputs ...
ref.write_jsonl("out/moe_ref.jsonl")
quant.write_jsonl("out/moe_quant.jsonl")
```

If two models execute interleaved on one comparator object, a single mutable global collector is **not sufficient**: bind the recorder to each forward closure or use a scoped context. Read the original comparator's execution orchestration before wiring the final CLI.

## 3. Compare

```bash
python compare_moe_routing.py out/moe_ref.jsonl out/moe_quant.jsonl --out out/moe_routing_report.json
python -m pytest tests/test_moe_routing_trace.py -q
```

## Limits

* No automatic router hooking or CLI flag in this first patch. This is the testable metrics/serialization core.
* It records routed experts only, not shared experts.
* Route-set differences are correlations, not evidence that router calculation is itself the original cause.
* Capturing every token incurs CPU synchronization; use sampled or batch-sharded captures in large Ascend NPU runs.
* Group-filtered top-k margins need access to pre-selection scores and eligible-group masks; the current tracer does not invent these margins.
