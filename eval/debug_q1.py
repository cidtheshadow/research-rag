"""
Debug: show exactly what the compressor/fusion pipeline did for q1,
so we can tell whether the hallucination came from bad clustering
or from the model ignoring correct evidence.

Run: python -m eval.debug_q1
"""
from pipeline.corpus import Corpus
from pipeline.compressor import yagni_filter, cluster_by_claim, synthesize
from pipeline.fusion import require_k_paths
from pipeline.prompts import build_fused_prompt

corpus = Corpus().build()
question = "What year was the Eiffel Tower completed?"
retrieved = corpus.retrieve(question, top_k=6)

print("=== RETRIEVED ===")
for r in retrieved:
    print(f"  score={r['score']:.3f}  source={r['source']:15s}  text={r['text']}")

filtered = yagni_filter(retrieved)
print(f"\n=== AFTER YAGNI FILTER ({len(filtered)}/{len(retrieved)} kept) ===")
for r in filtered:
    print(f"  source={r['source']:15s}  text={r['text']}")

clusters = cluster_by_claim(filtered)
print(f"\n=== CLUSTERS ({len(clusters)}) ===")
for i, c in enumerate(clusters):
    print(f"  Cluster {i}: {[s['source'] for s in c]}")
    for s in c:
        print(f"      - {s['text']}")

fused = synthesize(clusters)
print("\n=== SYNTHESIZED CLAIMS ===")
for c in fused:
    print(f"  sources={c['supporting_sources']}  paths={c['num_independent_paths']}")
    print(f"      -> \"{c['claim_text']}\"")

final = require_k_paths(fused, k=2)
print(f"\n=== AFTER k=2 CORROBORATION FILTER ({len(final)} claims survived) ===")
for c in final:
    print(f"  sources={c['supporting_sources']}")
    print(f"      -> \"{c['claim_text']}\"")

prompt = build_fused_prompt(question, final)
print("\n=== FINAL PROMPT SENT TO LLM ===")
print(prompt)