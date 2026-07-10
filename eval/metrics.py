def is_correct(answer: str, correct_answer: str) -> bool:
    return correct_answer.lower() in answer.lower()


def is_hallucinated(answer: str, hallucination_marker: str) -> bool:
    """Did the answer repeat the single-source 'bait' claim?"""
    return hallucination_marker.lower() in answer.lower()


def token_proxy(text: str) -> int:
    """Cheap token-count proxy (no tokenizer dependency). Swap for tiktoken
    if you want exact counts for the paper's token-savings number."""
    return len(text.split())
