"""A tiny semantic memory store with similarity search."""

import numpy as np
from sentence_transformers import SentenceTransformer


class MemoryStore:
    """Stores text memories and retrieves them by semantic similarity."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.memories: list[str] = []
        self.embeddings: np.ndarray | None = None

    def add(self, text: str) -> None:
        """Add a new memory to the store."""
        self.memories.append(text)
        self._reindex()

    def add_many(self, texts: list[str]) -> None:
        """Add multiple memories efficiently."""
        self.memories.extend(texts)
        self._reindex()

    def _reindex(self) -> None:
        """Recompute all embeddings (called after adding)."""
        if not self.memories:
            self.embeddings = None
            return
        raw = self.model.encode(self.memories)
        # Normalize for cosine similarity
        self.embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Find the top_k most similar memories to the query."""
        if self.embeddings is None:
            return []

        query_emb = self.model.encode([query])[0]
        query_emb = query_emb / np.linalg.norm(query_emb)

        similarities = self.embeddings @ query_emb
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(self.memories[i], float(similarities[i])) for i in top_indices]
    def remove(self, text: str) -> bool:
        """Remove a memory. Returns True if found and removed."""
        if text in self.memories:
            self.memories.remove(text)
            self._reindex()
            return True
        return False

def main() -> None:
    store = MemoryStore()
    store.add_many([
        "2",
        "Python is a programming language used widely in research.",
        "The MemGov paper introduces a governance layer for agent memory.",
        "Pretrained language models include GPT, Claude, and Gemini.",
        "Embeddings are vectors representing the meaning of text.",
    ])

    queries = [
        "What is 1+1?",
        "What is the research paper about?",
        "How do AI models understand language?",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        for memory, score in store.search(q, top_k=2):
            print(f"  {score:.3f}  {memory}")
    print("\nRemoving Python memory...")
    store.remove("Python is a programming language used widely in research.")
    print("\nQuery: programming languages")
    for memory, score in store.search("programming languages", top_k=3):
        print(f"  {score:.3f}  {memory}")


if __name__ == "__main__":
    main()