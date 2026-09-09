from typing import TypedDict

import numpy as np

from .ingestion import KnowledgeBase


class TriageState(TypedDict):
    query: str
    top_k: int
    confidence_threshold: float

    query_embedding: np.ndarray

    category: str
    category_score: float


    retrieved_cases: list[dict]
    priority: str
    routed_queue: str
    confidence: float
    escalated: bool
    resolution_notes: str


def classify_node(state: TriageState, kb: KnowledgeBase) -> dict:
    """
    Compare the query embedding against every category embedding
    (cosine similarity, both sides pre-normalized -> plain dot product).
    Returns the closest category and its similarity score.
    """
    similarities = kb.category_embeddings @ state["query_embedding"]
    best_idx = int(np.argmax(similarities))

    return {
        "category": kb.category_names[best_idx],
        "category_score": float(similarities[best_idx]),
    }