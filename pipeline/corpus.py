import numpy as np
from sentence_transformers import SentenceTransformer

from retriever import ScalarQuantizer8, CompressedIndex
from eval.dataset import QUESTIONS, NOISE_DOCS

_MODEL_NAME = "all-MiniLM-L6-v2"


class Corpus:
    """
    Wraps the quantized vector index with parallel metadata (source, text)
    so we can look up *who* said a retrieved snippet, which the fusion
    stage needs for corroboration counting.
    """

    def __init__(self):
        self.model = SentenceTransformer(_MODEL_NAME)
        self.texts = []
        self.sources = []
        self.index = None

    def build(self):
        for q in QUESTIONS:
            for ev in q["evidence"]:
                self.texts.append(ev["text"])
                self.sources.append(ev["source"])
        for doc in NOISE_DOCS:
            self.texts.append(doc["text"])
            self.sources.append(doc["source"])

        embeddings = self.model.encode(self.texts).astype(np.float32)
        quantizer = ScalarQuantizer8()
        self.index = CompressedIndex(quantizer)
        self.index.add_vectors(embeddings, self.texts)
        return self

    def retrieve(self, query: str, top_k: int = 6):
        """Returns list of dicts: {text, source, score}"""
        q_emb = self.model.encode([query]).astype(np.float32)
        matches = self.index.search_quantized_space(q_emb, top_k=top_k)
        results = []
        for idx, text, score in matches:
            results.append({
                "text": text,
                "source": self.sources[idx],
                "score": score,
            })
        return results
