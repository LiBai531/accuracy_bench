"""MoE expert-routing trace and BF16-vs-quant attribution metrics.

No model-specific routing formula is recomputed here: pass the *actual* chosen
expert IDs and mixture weights from the MoE forward path.  CPU-only comparisons
work on traces collected from CUDA, Ascend NPU, or CPU.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch


def _cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(device="cpu").contiguous()


@dataclass
class MoERoutingCollector:
    """Collect per-token routed experts without changing the model output.

    Pass the same sample ID and token positions to reference and quant models.
    For full-sequence prefill, positions default to [0, ..., T-1].  For chunked
    prefill/decode, explicitly supply absolute token_positions, otherwise trace
    rows from different chunks can collide and cannot be paired correctly.
    """

    side: str  # 'ref' or 'quant'
    sample_id: str = "0"
    records: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.side not in ("ref", "quant"):
            raise ValueError("side must be 'ref' or 'quant'")

    def record(
        self,
        layer_idx: int,
        indices: torch.Tensor,
        weights: torch.Tensor,
        *,
        sample_id: Optional[str] = None,
        token_positions: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Record selected IDs and *actual* aggregation weights.

        indices, weights: [T,K] or [B,T,K].  A [B,T,K] batch uses
        sample IDs '<sample_id>:<batch-index>' to avoid token collisions.
        valid_mask and token_positions have the corresponding [T]/[B,T]
        leading shape; token_positions are absolute sequence positions.
        """
        if indices.shape != weights.shape or indices.ndim not in (2, 3):
            raise ValueError("indices and weights must have the same [T,K] or [B,T,K] shape")
        if indices.dtype.is_floating_point:
            raise TypeError("expert indices must be integer dtype")
        if not weights.dtype.is_floating_point:
            raise TypeError("routing weights must be floating point")
        idx, w = _cpu(indices), _cpu(weights.float())
        if idx.ndim == 2:
            idx, w = idx.unsqueeze(0), w.unsqueeze(0)
        bsz, length, _ = idx.shape
        expected = (bsz, length) if indices.ndim == 3 else (length,)
        if token_positions is None:
            positions = torch.arange(length).expand(bsz, -1)
        else:
            if tuple(token_positions.shape) != expected:
                raise ValueError(f"token_positions expected shape {expected}")
            positions = _cpu(token_positions).long().reshape(bsz, length)
        if valid_mask is None:
            mask = torch.ones(bsz, length, dtype=torch.bool)
        else:
            if tuple(valid_mask.shape) != expected:
                raise ValueError(f"valid_mask expected shape {expected}")
            mask = _cpu(valid_mask).bool().reshape(bsz, length)
        prefix = str(self.sample_id if sample_id is None else sample_id)
        for b in range(bsz):
            sid = f"{prefix}:{b}" if indices.ndim == 3 else prefix
            for t in range(length):
                if not mask[b, t]:
                    continue
                ids = [int(v) for v in idx[b, t].tolist()]
                vals = [float(v) for v in w[b, t].tolist()]
                if len(set(ids)) != len(ids) or any(i < 0 for i in ids):
                    raise ValueError("negative or repeated expert ID in top-k")
                self.records.append({
                    "side": self.side,
                    "sample_id": sid,
                    "layer": int(layer_idx),
                    "token_position": int(positions[b, t]),
                    "expert_ids": ids,
                    "routing_weights": vals,
                })

    def write_jsonl(self, path: str | Path) -> None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as fh:
            for r in self.records:
                fh.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _key(r: Dict[str, Any]):
    return (str(r["sample_id"]), int(r["layer"]), int(r["token_position"]))


def _index(records: Iterable[Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
    result = {}
    for row in records:
        key = _key(row)
        if key in result:
            raise ValueError(f"duplicate trace key {key}; supply absolute token positions for chunked/decode runs")
        result[key] = row
    return result


def _comparison(r: Dict[str, Any], q: Dict[str, Any]) -> Dict[str, Any]:
    ri, qi = r["expert_ids"], q["expert_ids"]
    rw, qw = r["routing_weights"], q["routing_weights"]
    if len(ri) != len(rw) or len(qi) != len(qw):
        raise ValueError("expert IDs and routing weights have different lengths")
    a, b = set(ri), set(qi)
    intersection, union = len(a & b), len(a | b)
    ref_map = dict(zip(ri, rw))
    quant_map = dict(zip(qi, qw))
    all_ids = a | b
    return {
        "sample_id": r["sample_id"], "layer": int(r["layer"]),
        "token_position": int(r["token_position"]),
        "topk_overlap": intersection / max(len(a), len(b), 1),
        "jaccard": intersection / union if union else 1.0,
        "expert_set_changed": a != b,
        "expert_replacements": max(len(a), len(b)) - intersection,
        "weight_l1": sum(abs(float(ref_map.get(e, 0.0)) - float(quant_map.get(e, 0.0))) for e in all_ids),
        "ref_experts": ri, "quant_experts": qi,
    }


def compare_traces(ref: Iterable[Dict[str, Any]], quant: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare only exactly aligned (sample_id, layer, token_position) rows.

    Both missing and extra keys are surfaced.  The caller should reject a
    partially aligned run before interpreting its mean metrics.
    """
    r, q = _index(ref), _index(quant)
    paired = sorted(r.keys() & q.keys())
    rows = [_comparison(r[k], q[k]) for k in paired]
    by_layer = defaultdict(list)
    for row in rows:
        by_layer[row["layer"]].append(row)

    def stats(items):
        n = len(items)
        return {
            "paired_tokens": n,
            "expert_set_change_rate": sum(x["expert_set_changed"] for x in items) / n if n else None,
            "mean_topk_overlap": sum(x["topk_overlap"] for x in items) / n if n else None,
            "mean_jaccard": sum(x["jaccard"] for x in items) / n if n else None,
            "mean_weight_l1": sum(x["weight_l1"] for x in items) / n if n else None,
        }

    def loads(index):
        counts: Dict[int, Counter] = defaultdict(Counter)
        weights: Dict[int, Counter] = defaultdict(Counter)
        for row in index.values():
            layer = int(row["layer"])
            for expert, value in zip(row["expert_ids"], row["routing_weights"]):
                counts[layer][int(expert)] += 1
                weights[layer][int(expert)] += float(value)
        return {str(l): {"expert_token_counts": dict(c),
                         "expert_routing_weight_sums": dict(weights[l])}
                for l, c in sorted(counts.items())}

    return {
        "summary": stats(rows),
        "by_layer": {str(l): stats(v) for l, v in sorted(by_layer.items())},
        "ref_only_keys": [list(k) for k in sorted(r.keys() - q.keys())],
        "quant_only_keys": [list(k) for k in sorted(q.keys() - r.keys())],
        "ref_load": loads(r),
        "quant_load": loads(q),
        "token_details": rows,
    }
