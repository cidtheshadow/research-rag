"""
Runs both pipelines over the toy dataset and prints a comparison table.

    python -m eval.run_eval

Requires Ollama running locally with a model pulled (see pipeline/llm_client.py).
"""
import csv
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.corpus import Corpus
from pipeline.compressor import compress
from pipeline.fusion import require_k_paths
from pipeline.prompts import build_baseline_prompt, build_fused_prompt
from pipeline.llm_client import generate
from eval.dataset import QUESTIONS
from eval.metrics import is_correct, is_hallucinated, token_proxy

K_PATHS = 2  # corroboration requirement


def run():
    corpus = Corpus().build()
    rows = []

    for q in QUESTIONS:
        retrieved = corpus.retrieve(q["question"], top_k=6)

        # --- Baseline: dump everything retrieved into the prompt ---
        baseline_prompt = build_baseline_prompt(q["question"], retrieved)
        baseline_answer = generate(baseline_prompt)

        # --- Fused: compress + require k independent corroborating sources ---
        fused_claims = compress(retrieved)
        fused_claims = require_k_paths(fused_claims, k=K_PATHS)
        fused_prompt = build_fused_prompt(q["question"], fused_claims)
        fused_answer = generate(fused_prompt)

        rows.append({
            "question": q["question"],
            "baseline_answer": baseline_answer,
            "baseline_correct": is_correct(baseline_answer, q["correct_answer"]),
            "baseline_hallucinated": is_hallucinated(baseline_answer, q["hallucination_marker"]),
            "baseline_tokens": token_proxy(baseline_prompt),
            "fused_answer": fused_answer,
            "fused_correct": is_correct(fused_answer, q["correct_answer"]),
            "fused_hallucinated": is_hallucinated(fused_answer, q["hallucination_marker"]),
            "fused_tokens": token_proxy(fused_prompt),
        })

    # --- Print + save ---
    print(f"\n{'Question':<45} {'Base OK':<8} {'Base Hall':<10} {'Fused OK':<9} {'Fused Hall':<10}")
    for r in rows:
        print(f"{r['question'][:43]:<45} {str(r['baseline_correct']):<8} "
              f"{str(r['baseline_hallucinated']):<10} {str(r['fused_correct']):<9} "
              f"{str(r['fused_hallucinated']):<10}")

    n = len(rows)
    base_acc = sum(r["baseline_correct"] for r in rows) / n
    fused_acc = sum(r["fused_correct"] for r in rows) / n
    base_hall = sum(r["baseline_hallucinated"] for r in rows) / n
    fused_hall = sum(r["fused_hallucinated"] for r in rows) / n
    base_tok = sum(r["baseline_tokens"] for r in rows) / n
    fused_tok = sum(r["fused_tokens"] for r in rows) / n

    print("\n=== Summary ===")
    print(f"Accuracy:          baseline {base_acc:.0%}  ->  fused {fused_acc:.0%}")
    print(f"Hallucination rate: baseline {base_hall:.0%}  ->  fused {fused_hall:.0%}")
    print(f"Avg tokens/prompt:  baseline {base_tok:.0f}   ->  fused {fused_tok:.0f} "
          f"({(1 - fused_tok / base_tok):.0%} reduction)")

    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nWrote results.csv")


if __name__ == "__main__":
    run()
