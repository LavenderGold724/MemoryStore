"""A tiny semantic memory store with similarity search."""

import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
from typing import Literal
from transformers import pipeline

MemoryState = Literal["active", "stale", "deleted"]


@dataclass
class Memory:
    """A memory object with content and lifecycle state."""
    text: str
    state: MemoryState = "active"
    memory_id: str = ""  # we'll use this later

    def __post_init__(self):
        if not self.memory_id:
            # Generate a simple ID if not provided
            import uuid
            self.memory_id = f"mem_{uuid.uuid4().hex[:8]}"

class MemoryStore:
    """Stores Memory objects and retrieves them by semantic similarity."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.memories: list[Memory] = []
        self.embeddings: np.ndarray | None = None
        # NLI model for contradiction detection
        # This loads lazily on first use to keep startup fast
        self._nli = None

    def _get_nli(self):
        """Lazy-load the NLI pipeline on first use."""
        if self._nli is None:
            print("Loading NLI model (one-time)...")
            self._nli = pipeline(
                "text-classification",
                model="cross-encoder/nli-deberta-v3-base"
            )
        return self._nli

    def detect_contradictions(
        self, memories: list[Memory]
    ) -> list[tuple[Memory, Memory, float]]:
        """Detect pairs of memories that contradict each other.

        Returns a list of (memory_a, memory_b, confidence) tuples.
        """
        nli = self._get_nli()
        contradictions = []

        # Check every pair
        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                m1, m2 = memories[i], memories[j]
                result = nli(f"{m1.text} [SEP] {m2.text}")[0]

                if result["label"].lower() == "contradiction":
                    contradictions.append((m1, m2, result["score"]))

        return contradictions



    def add(self, text: str, state: MemoryState = "active") -> Memory:
        """Add a memory and return the created Memory object."""
        memory = Memory(text=text, state=state)
        self.memories.append(memory)
        self._reindex()
        return memory

    def add_many(self, items: list[tuple[str, MemoryState]]) -> list[Memory]:
        """Add multiple memories. Each item is (text, state)."""
        created = [Memory(text=t, state=s) for t, s in items]
        self.memories.extend(created)
        self._reindex()
        return created

    def mark_deleted(self, memory_id: str) -> bool:
        """Mark a memory as deleted by ID."""
        for m in self.memories:
            if m.memory_id == memory_id:
                m.state = "deleted"
                return True
        return False

    def _reindex(self) -> None:
        if not self.memories:
            self.embeddings = None
            return
        texts = [m.text for m in self.memories]
        raw = self.model.encode(texts)
        self.embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    def search(self, query: str, top_k: int = 3) -> list[tuple[Memory, float]]:
        """Find top_k most similar ELIGIBLE memories to the query.

        Eligibility = state is 'active'. Deleted and stale memories
        are filtered out before ranking.
        """
        if self.embeddings is None:
            return []

        query_emb = self.model.encode([query])[0]
        query_emb = query_emb / np.linalg.norm(query_emb)

        # Compute similarities for all memories
        similarities = self.embeddings @ query_emb

        # Filter to only eligible (active) memories
        eligible_indices = [
            i for i, m in enumerate(self.memories)
            if m.state == "active"
        ]

        # Sort eligible indices by similarity (descending)
        eligible_indices.sort(key=lambda i: similarities[i], reverse=True)

        # Take top_k
        top_indices = eligible_indices[:top_k]

        return [(self.memories[i], float(similarities[i])) for i in top_indices]

def main() -> None:
    store = MemoryStore()

    store.add_many([
        ("Alice lives at 123 Main Street.", "active"),
        ("Alice lives at 999 Lake Drive.", "active"),   # active but contradicts!
        ("Alice lives at 456 Oak Avenue.", "stale"),
        ("Alice lives at 789 Pine Road.", "deleted"),
        ("Alice works as a software engineer.", "active"),
        ("Alice has a dog named Max.", "active"),
    ])

    query = "Where does Alice live?"
    print(f"\nQuery: {query}\n")

    # Get top results (eligibility filter applied)
    results = store.search(query, top_k=5)
    print("Retrieved memories:")
    for memory, score in results:
        print(f"  {score:.3f}  {memory.text}")

    # Detect contradictions
    retrieved_memories = [m for m, _ in results]
    contradictions = store.detect_contradictions(retrieved_memories)

    print("\nDetected contradictions:")
    if not contradictions:
        print("  None")
    for m1, m2, conf in contradictions:
        print(f"  [{conf:.3f}]")
        print(f"    {m1.text}")
        print(f"  contradicts")
        print(f"    {m2.text}")
if __name__ == "__main__":
    main()