from typing import TypedDict
from langgraph.graph import StateGraph, END

import numpy as np
import ollama
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

def route_node(state: TriageState, kb: KnowledgeBase) -> dict:
    """Direct lookup: category -> team, precomputed from historical data."""
    return {"routed_queue": kb.category_queue_map[state["category"]]}

def compute_confidence_node(state: TriageState) -> dict:
    """
    Two signals combined:
      - category_score: how close the inquiry is to its predicted category's definition
      - agreement: what fraction of retrieved cases share that same predicted category
    Low confidence on either -> flag for human review instead of guessing.
    """
    retrieved = state["retrieved_cases"]
    if retrieved:
        agreement = sum(
            1 for c in retrieved if c["category"] == state["category"]
        ) / len(retrieved)
    else:
        agreement = 0.0

    confidence = 0.5 * state["category_score"] + 0.5 * agreement
    escalated = confidence < state["confidence_threshold"]

    return {"confidence": confidence, "escalated": escalated}

def escalate_node(state: TriageState) -> dict:
    """Reached only via the low-confidence branch — skips the LLM call entirely."""
    return {"resolution_notes": "Low confidence — flagged for human review before any automated action."}


def generate_resolution_notes_node(state: TriageState) -> dict:
    """
    Reached only via the 'continue' branch — confidence was sufficient.
    The only node in the graph that calls the LLM.
    """
    context = "\n".join(f"- {c['text']}" for c in state["retrieved_cases"][:3])
    prompt = (
        f"Customer inquiry: \"{state['query']}\"\n"
        f"Category: {state['category']}, Priority: {state['priority']}\n"
        f"Similar past cases:\n{context}\n\n"
        f"Write ONLY a 1-2 sentence resolution note suggesting the next step for the assigned team. "
        f"Do not explain your reasoning or add commentary — output just the note itself."
    )
    response = ollama.chat(
        model="llama3.2:latest",
        messages=[{"role": "user", "content": prompt}],
    )
    return {"resolution_notes": response["message"]["content"].strip()}

def _confidence_router(state: TriageState) -> str:
    """The one genuine branch point: escalate early, or continue to resolution notes."""
    return "escalate" if state["escalated"] else "continue"


def build_graph(kb: KnowledgeBase):
    graph = StateGraph(TriageState)

    graph.add_node("classify", lambda s: classify_node(s, kb))
    graph.add_node("retrieve", lambda s: retrieve_top_k_node(s, kb))
    graph.add_node("prioritize", determine_priority_node)
    graph.add_node("route", lambda s: route_node(s, kb))
    graph.add_node("confidence", compute_confidence_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("resolve", generate_resolution_notes_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "prioritize")
    graph.add_edge("prioritize", "route")
    graph.add_edge("route", "confidence")

    graph.add_conditional_edges(
        "confidence",
        _confidence_router,
        {"escalate": "escalate", "continue": "resolve"},
    )
    graph.add_edge("escalate", END)
    graph.add_edge("resolve", END)

    return graph.compile()

def run_triage(query: str, kb: KnowledgeBase, top_k: int = 3, confidence_threshold: float = 0.5) -> dict:
    """Single entry point: raw text in, full triage result out."""
    graph = build_graph(kb)

    initial_state = {
        "query": query,
        "query_embedding": kb.embed(query),
        "top_k": top_k,
        "confidence_threshold": confidence_threshold,
    }

    final_state = graph.invoke(initial_state)
    return final_state