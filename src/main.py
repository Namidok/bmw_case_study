from typing import TypedDict

import numpy as np

from .ingestion import KnowledgeBase

PRIORITY_TO_NUM = {"low": 1, "medium": 2, "high": 3}
NUM_TO_PRIORITY = {1: "low", 2: "medium", 3: "high"}


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

def retrieve_top_k_node(state: TriageState, kb: KnowledgeBase) -> dict:
    """
    Query ChromaDB for the K most similar past cases to this inquiry.
    Similarity = 1 - cosine_distance (collection is configured for cosine space).
    """
    results = kb.chroma_collection.query(
        query_embeddings=[state["query_embedding"].tolist()],
        n_results=state["top_k"],
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        retrieved.append({
            "text": doc,
            "category": meta["category"],
            "priority": meta["priority"],
            "routed_queue": meta["routed_queue"],
            "similarity": 1 - dist,
        })

    return {"retrieved_cases": retrieved}

def determine_priority_node(state: TriageState) -> dict:
    """
    Similarity-weighted average of the retrieved cases' priorities.
    Closer matches count more than weaker ones, rather than a plain majority vote.
    """
    retrieved = state["retrieved_cases"]
    if not retrieved:
        return {"priority": "low"}  # no evidence -> safest default

    weighted_sum = sum(
        PRIORITY_TO_NUM[c["priority"]] * c["similarity"] for c in retrieved
    )
    total_weight = sum(c["similarity"] for c in retrieved)
    avg = weighted_sum / total_weight

    rounded = max(1, min(3, round(avg)))
    return {"priority": NUM_TO_PRIORITY[rounded]}