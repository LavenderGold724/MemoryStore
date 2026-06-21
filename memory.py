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
    

    def search_safe(
        self, query: str, top_k: int = 3, retrieve_pool: int = 10
    ) -> tuple[list[tuple[Memory, float]], dict]:
        """Search with contradiction enforcement.

        Returns:
            - A list of (memory, score) tuples that are mutually consistent
            - A "certificate" dict explaining what was retrieved, detected, and dropped
        """
        # Step 1: Retrieve a larger pool than top_k so we have room to drop some
        candidates = self.search(query, top_k=retrieve_pool)

        if not candidates:
            return [], {"status": "empty", "retrieved": 0, "dropped": []}

        # Step 2: Detect contradictions in the candidate pool
        candidate_memories = [m for m, _ in candidates]
        contradictions = self.detect_contradictions(candidate_memories)

        # Step 3: Build a score lookup for quick access
        scores = {m.memory_id: s for m, s in candidates}

        # Step 4: For each contradiction, drop the lower-scored memory
        dropped_ids = set()
        drop_log = []

        for m1, m2, conf in contradictions:
            # Skip if one is already dropped
            if m1.memory_id in dropped_ids or m2.memory_id in dropped_ids:
                continue

            # Drop the lower-scored one
            if scores[m1.memory_id] >= scores[m2.memory_id]:
                dropped_ids.add(m2.memory_id)
                drop_log.append({
                    "kept": m1.text,
                    "dropped": m2.text,
                    "reason": "contradiction",
                    "confidence": conf,
                })
            else:
                dropped_ids.add(m1.memory_id)
                drop_log.append({
                    "kept": m2.text,
                    "dropped": m1.text,
                    "reason": "contradiction",
                    "confidence": conf,
                })

        # Step 5: Filter out dropped memories
        safe_results = [
            (m, s) for m, s in candidates
            if m.memory_id not in dropped_ids
        ]

        # Trim to requested top_k
        safe_results = safe_results[:top_k]

        # Step 6: Build the certificate
        certificate = {
            "status": "safe",
            "query": query,
            "retrieved_count": len(candidates),
            "contradictions_detected": len(contradictions),
            "dropped_count": len(dropped_ids),
            "returned_count": len(safe_results),
            "drop_log": drop_log,
        }

        return safe_results, certificate

def main() -> None:
    store = MemoryStore()

    store.add_many([
        ("Alice lives at 123 Main Street.", "active"),
        ("Alice lives at 999 Lake Drive.", "active"),     # contradicts current
        ("Alice lives at 456 Oak Avenue.", "stale"),
        ("Alice lives at 789 Pine Road.", "deleted"),
        ("Alice works as a software engineer.", "active"),
        ("Alice has a dog named Max.", "active"),
        ("Bob lives at 555 Elm Street.", "active"),
    ])

    query = "Where does Alice live?"
    print(f"\nQuery: {query}\n")

    # Use the safe search
    results, certificate = store.search_safe(query, top_k=3)

    print("Safe results:")
    for memory, score in results:
        print(f"  {score:.3f}  {memory.text}")

    print("\nCertificate:")
    import json
    print(json.dumps(certificate, indent=2))
    
if __name__ == "__main__":
    main()