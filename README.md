# Evidence Fusion RAG — MVP

A minimal, runnable version of the fusion pipeline: retriever → compressor →
corroboration fusion → LLM → comparison against a naive baseline.

Goal of this MVP: get **one real number** — does requiring 2+ independent
sources before including a claim reduce hallucination rate, and at what
token cost — on a tiny, hand-checkable dataset. Once this works, swap in a
real dataset and scale up.

## Pipeline

```
retriever/     vendored quantized vector search (from the
               High-Performance-Quantized-Vector-Search-Engine repo)
pipeline/
  corpus.py       builds the shared corpus + vector index
  compressor.py   YAGNI filter -> claim clustering -> one-line synthesis
  fusion.py       k-path independence requirement (the actual intervention)
  prompts.py      baseline (dump-all) vs fused (corroborated) prompt builders
  llm_client.py   Ollama wrapper (free, local, no API key)
eval/
  dataset.py      tiny synthetic dataset (multi-source evidence +
                   1 uncorroborated "hallucination bait" claim per question)
  metrics.py      correctness / hallucination / token-count checks
  run_eval.py     runs both pipelines, prints + saves comparison table
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install and start Ollama (free local LLM, no API key):
```bash
# https://ollama.com/download
ollama pull llama3.1
```
Ollama runs its server automatically on `localhost:11434` after install.

## Run

```bash
python -m eval.run_eval
```

This prints a per-question comparison and a summary:

```
Accuracy:           baseline 75%  ->  fused 100%
Hallucination rate: baseline 25%  ->  fused 0%
Avg tokens/prompt:  baseline 142  ->  fused 61  (57% reduction)
```

(Numbers above are illustrative — run it to get your real ones.)

Results are also saved to `results.csv` for further analysis / plotting.

## What to do once this runs

1. **Swap the dataset.** Replace `eval/dataset.py` with a real slice of a
   public QA dataset (HotpotQA, ambig-QA, etc.) with real multi-source
   evidence. The synthetic dataset here is just to prove the pipeline works
   end-to-end.
2. **Vary k** (the corroboration threshold) and plot accuracy/hallucination/
   tokens as a function of k — this is your main result chart.
3. **Add the adversarial perturbations** from the original ideation doc
   (source rewires, hub injection) once the baseline result is solid —
   don't build these until step 1–2 produce a real, defensible number.
4. **Write the 1-pager**: problem, method (this pipeline), result (your
   k-sweep chart), one clearly stated limitation (e.g. "independence" is
   approximated by distinct source IDs, not verified provenance).

## Known simplifications (call these out explicitly in the paper — reviewers respect this)

- "Independent source" = distinct source ID, not verified true independence.
- Canonical substitution (from the original doc) is stubbed out — it matters
  more for code/citation-heavy evidence than short factual snippets.
- Token counts use a word-count proxy, not a real tokenizer. Swap in
  `tiktoken` before reporting final numbers.
- Hallucination detection here is keyword-based for the toy dataset. On a
  real dataset you'll need an LLM-as-judge or human annotation pass.
# research-rag
