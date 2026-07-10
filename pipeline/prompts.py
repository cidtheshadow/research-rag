def build_baseline_prompt(question: str, snippets: list) -> str:
    """Naive RAG baseline: dump every retrieved snippet into the prompt."""
    evidence_block = "\n".join(f"- {s['text']}" for s in snippets)
    return (
        "Answer the question using only the evidence below. "
        "Be concise (one short sentence).\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"Question: {question}\nAnswer:"
    )


def build_fused_prompt(question: str, fused_claims: list) -> str:
    """Compressed + corroborated evidence, with provenance shown for the audit trace."""
    lines = []
    for c in fused_claims:
        srcs = ", ".join(c["supporting_sources"])
        lines.append(f"- {c['claim_text']} (corroborated by: {srcs})")
    evidence_block = "\n".join(lines) if lines else "(no evidence met the corroboration bar)"
    return (
        "Answer the question using only the evidence below. "
        "Be concise (one short sentence).\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"Question: {question}\nAnswer:"
    )
