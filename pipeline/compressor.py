"""
Three-stage evidence compressor (renamed from the vague "Ponytail-style" label
to something a reviewer can understand without a glossary).

Stage 1 - YAGNI filter:      drop retrieved snippets that are weakly relevant
                              to the question (low retrieval score = noise).
Stage 2 - Clustering:        group snippets that make the *same claim* using
                              embedding similarity, regardless of exact wording.
Stage 3 - One-line synthesis: collapse each cluster into a single representative
                              line, tagged with every source that supports it.

Canonical substitution (swapping verbose text for standard references) is left
as a documented extension point -- it matters more for code/citation-heavy
evidence than for short factual snippets, so it's out of scope for this MVP.

--- Known failure mode this version fixes ---
Pure embedding similarity conflates TOPIC similarity with CLAIM agreement:
"completed in 1889" and "actually finished in 1900" are about the same event
and phrase similarly, so they can cosine-cluster together even though they
directly contradict each other. If that happens, a false claim can inherit
corroboration credit from the true sources it got merged with, and -- worse --
get picked as the cluster's representative if it happens to have the highest
individual retrieval score (persuasive contrarian phrasing like "contrary to
popular belief" often scores higher against a direct question than a plain
factual statement does).

Fix: (1) a numeric/date guard blocks clustering two snippets whose extracted
numbers disagree, regardless of embedding similarity; (2) the representative
line is chosen by majority vote within the cluster, not by max retrieval score.

This guard is itself limited -- it only catches disagreements that show up as
different numbers/dates. Two snippets disagreeing on a non-numeric fact
("Shakespeare" vs "Marlowe") would still be vulnerable to the same failure
mode. See paper Limitations for discussion.
"""
import re
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer

_SIM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

YAGNI_SCORE_THRESHOLD = 0.15   # drop weakly-relevant retrieved snippets
CLUSTER_SIM_THRESHOLD = 0.55   # snippets above this cosine sim = "same claim"

_NUMBER_RE = re.compile(r"\b\d+\b")


def yagni_filter(snippets: list, threshold: float = YAGNI_SCORE_THRESHOLD) -> list:
    return [s for s in snippets if s["score"] >= threshold]


def _cosine_sim_matrix(vecs: np.ndarray) -> np.ndarray:
    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    return norm @ norm.T


def _extract_numbers(text: str) -> set:
    return set(_NUMBER_RE.findall(text))


def _numbers_compatible(a: str, b: str) -> bool:
    """Two snippets can only be the 'same claim' if they don't cite
    conflicting numbers/dates. Snippets with no numbers are always compatible
    (nothing to conflict on)."""
    nums_a, nums_b = _extract_numbers(a), _extract_numbers(b)
    if not nums_a or not nums_b:
        return True
    return nums_a == nums_b


def cluster_by_claim(snippets: list, sim_threshold: float = CLUSTER_SIM_THRESHOLD) -> list:
    """
    Greedy clustering: snippets whose embeddings are similar enough AND whose
    cited numbers/dates don't conflict are treated as making the same claim.
    Returns a list of clusters, each cluster = list of snippet dicts.
    """
    if not snippets:
        return []

    texts = [s["text"] for s in snippets]
    vecs = _SIM_MODEL.encode(texts).astype(np.float32)
    sims = _cosine_sim_matrix(vecs)

    assigned = [-1] * len(snippets)
    clusters = []
    for i in range(len(snippets)):
        if assigned[i] != -1:
            continue
        cluster_idx = len(clusters)
        assigned[i] = cluster_idx
        members = [i]
        for j in range(i + 1, len(snippets)):
            if assigned[j] == -1 and sims[i, j] >= sim_threshold \
                    and _numbers_compatible(texts[i], texts[j]):
                assigned[j] = cluster_idx
                members.append(j)
        clusters.append([snippets[m] for m in members])
    return clusters


def _majority_representative(cluster: list) -> dict:
    """
    Pick the representative snippet by majority vote on normalized claim
    content (currently: the set of numbers cited, as a cheap proxy for
    'which version of the claim most sources agree on'), falling back to
    highest retrieval score to break ties or when no numbers are present.
    """
    if len(cluster) == 1:
        return cluster[0]

    number_signatures = [tuple(sorted(_extract_numbers(s["text"]))) for s in cluster]
    counts = Counter(number_signatures)
    top_signature, top_count = counts.most_common(1)[0]

    # No majority (all distinct, or no numbers at all) -> fall back to top score.
    if top_count == 1:
        return max(cluster, key=lambda s: s["score"])

    majority_members = [s for s, sig in zip(cluster, number_signatures) if sig == top_signature]
    return max(majority_members, key=lambda s: s["score"])


def synthesize(clusters: list) -> list:
    """
    Stage 3: one representative line per cluster (chosen by majority vote,
    not top score alone), with provenance metadata attached for the audit trace.
    """
    fused = []
    for cluster in clusters:
        sources = sorted(set(s["source"] for s in cluster))
        representative = _majority_representative(cluster)["text"]
        fused.append({
            "claim_text": representative,
            "supporting_sources": sources,
            "num_independent_paths": len(sources),
        })
    return fused


def compress(snippets: list) -> list:
    """Full pipeline: YAGNI filter -> cluster -> synthesize."""
    filtered = yagni_filter(snippets)
    clusters = cluster_by_claim(filtered)
    return synthesize(clusters)