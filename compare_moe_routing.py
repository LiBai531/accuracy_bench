"""Usage: python compare_moe_routing.py ref.jsonl quant.jsonl --out routing_report.json"""
import argparse
import json
from accuracy_checker.moe_routing_trace import compare_traces, read_jsonl


def main():
    p = argparse.ArgumentParser(description="Compare aligned BF16/W4A4 MoE expert routes")
    p.add_argument("reference")
    p.add_argument("quantized")
    p.add_argument("--out", default="moe_routing_report.json")
    args = p.parse_args()
    report = compare_traces(read_jsonl(args.reference), read_jsonl(args.quantized))
    with open(args.out, "w", encoding="utf8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    if report["ref_only_keys"] or report["quant_only_keys"]:
        print("WARNING: partial alignment; examine ref_only_keys / quant_only_keys before interpretation")


if __name__ == "__main__":
    main()
