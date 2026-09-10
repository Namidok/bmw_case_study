from src.ingestion import KnowledgeBase
from src.main import classify_node, retrieve_top_k_node, determine_priority_node, route_node, compute_confidence_node, generate_resolution_notes_node

kb = KnowledgeBase()
query = "The brake fluid warning light came on and the pedal feels soft"
state = {
    "query": query,
    "query_embedding": kb.embed(query),
    "top_k": 3,
    "confidence_threshold": 0.5,
}
state.update(classify_node(state, kb))
state.update(retrieve_top_k_node(state, kb))

print("Category:", state["category"], round(state["category_score"], 3))
for c in state["retrieved_cases"]:
    print(f"  [{c['similarity']:.3f}] {c['category']} / {c['priority']} — {c['text']}")

state.update(determine_priority_node(state))
print("Priority:", state["priority"])

state.update(route_node(state, kb))
print("Routed to:", state["routed_queue"])

state.update(compute_confidence_node(state))
print("Confidence:", round(state["confidence"], 3), "| Escalated:", state["escalated"])

state.update(generate_resolution_notes_node(state))
print("Resolution notes:", state["resolution_notes"])

