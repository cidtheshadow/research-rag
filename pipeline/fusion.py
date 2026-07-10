"""
Fusion module: the actual hallucination-reduction intervention.

"Independent path" here means: a distinct `source` id, as tagged in the
dataset/corpus. This is a deliberately simple definition -- two snippets
from the same source are never counted as corroborating each other.
(Worth stating explicitly in the paper: source-id independence is a proxy
for true independence and doesn't catch e.g. two blogs copying one wire story.)
"""


def require_k_paths(fused_claims: list, k: int = 2) -> list:
    """Keep only claims supported by >= k independent sources."""
    return [c for c in fused_claims if c["num_independent_paths"] >= k]
