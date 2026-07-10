"""
Tiny synthetic benchmark.

Each question has:
  - a correct answer (ground truth)
  - 2+ CORROBORATING evidence snippets (different sources, agree with each other)
  - 1 UNCORROBORATED snippet containing a plausible but WRONG claim
    (this is the "hallucination bait" -- a single unreliable source)
  - a few NOISE snippets on unrelated topics (tests retrieval + YAGNI filtering)

This lets you directly measure: does requiring >=2 independent sources
keep the hallucination bait out of the prompt, without losing the real answer?

Swap this out for a real dataset (HotpotQA slice, etc.) once the pipeline works.
"""

QUESTIONS = [
    {
        "id": "q1",
        "question": "What year was the Eiffel Tower completed?",
        "correct_answer": "1889",
        "hallucination_marker": "1900",
        "evidence": [
            {"text": "The Eiffel Tower was completed in 1889 for the World's Fair in Paris.", "source": "encyclopedia_a"},
            {"text": "Construction on the Eiffel Tower finished in 1889, ahead of the Exposition Universelle.", "source": "travel_guide_b"},
            {"text": "Contrary to popular belief, the Eiffel Tower was actually finished in 1900.", "source": "forum_post_x"},
        ],
    },
    {
        "id": "q2",
        "question": "What is the boiling point of water at sea level in Celsius?",
        "correct_answer": "100",
        "hallucination_marker": "90",
        "evidence": [
            {"text": "At standard atmospheric pressure, water boils at 100 degrees Celsius.", "source": "textbook_a"},
            {"text": "Sea-level boiling point of water is 100C, a common reference point in chemistry.", "source": "science_site_b"},
            {"text": "Some altitude charts show water boiling at 90C at sea level under certain rare conditions.", "source": "blog_post_x"},
        ],
    },
    {
        "id": "q3",
        "question": "Who wrote the play 'Hamlet'?",
        "correct_answer": "Shakespeare",
        "hallucination_marker": "Marlowe",
        "evidence": [
            {"text": "Hamlet is a tragedy written by William Shakespeare.", "source": "literature_db_a"},
            {"text": "William Shakespeare authored Hamlet around 1600.", "source": "encyclopedia_b"},
            {"text": "Some scholars argue Hamlet was actually ghostwritten by Christopher Marlowe.", "source": "forum_post_y"},
        ],
    },
    {
        "id": "q4",
        "question": "What planet is known as the Red Planet?",
        "correct_answer": "Mars",
        "hallucination_marker": "Venus",
        "evidence": [
            {"text": "Mars is often called the Red Planet due to iron oxide on its surface.", "source": "astronomy_site_a"},
            {"text": "The Red Planet, Mars, has a reddish appearance from surface rust.", "source": "textbook_b"},
            {"text": "A viral post claimed Venus is the true Red Planet due to its atmosphere.", "source": "social_post_x"},
        ],
    },
]

# Off-topic noise documents mixed into the shared corpus to make retrieval
# and YAGNI filtering actually do something instead of being trivial.
NOISE_DOCS = [
    {"text": "The stock market closed slightly higher on Tuesday amid mixed earnings.", "source": "news_noise_1"},
    {"text": "Local bakery introduces a new sourdough recipe this weekend.", "source": "news_noise_2"},
    {"text": "New hiking trail opens in the regional national park.", "source": "news_noise_3"},
    {"text": "City council debates new zoning rules for downtown parking.", "source": "news_noise_4"},
]
