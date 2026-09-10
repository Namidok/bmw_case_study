"""
How does the pipeline behave when one inquiry mixes several categories?

For each query this prints:
  - all 8 category similarities, ranked  (is the winner clear or a coin-flip?)
  - margin between top-1 and top-2       (a better "how sure" signal than the raw score)
  - category mix of the retrieved cases   (does retrieval also see it as mixed?)
  - priority, confidence, escalated

Expected pattern: a genuinely multi-intent query should show a SMALL margin and a
SPLIT retrieval mix -> agreement drops -> confidence drops -> escalated.
That is the desired behaviour: mixed inquiries are exactly what a human should see.

Run from repo root:
    python tests/test_multi_category.py            # no LLM, fast
    python tests/test_multi_category.py --llm      # also generate resolution notes
"""

import sys
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from src.ingestion import KnowledgeBase  # noqa: E402
from src.main import (  # noqa: E402
    classify_node,
    retrieve_top_k_node,
    determine_priority_node,
    route_node,
    compute_confidence_node,
    generate_resolution_notes_node,
)

QUERIES = [
    # --- controls -----------------------------------------------------------
    ("single: technical",
     "How do I reset the trip computer on the dashboard?"),
    ("single: off-topic",
     "Do you also sell bicycles?"),

    # --- two categories -----------------------------------------------------
    ("ordering + warranty",
     "I want to change the delivery address for my order, and is the paint protection "
     "package included in the warranty?"),
    ("general + billing",
     "What are your opening hours at the dealership, and can I pay my invoice online?"),
    ("service + billing",
     "I was charged for a service I cancelled last week, please refund it and rebook "
     "me for next Tuesday."),

    # --- three categories ---------------------------------------------------
    ("technical + warranty + service",
     "My infotainment screen keeps freezing every few minutes. Is this covered by the "
     "warranty, and can I book a service appointment for next week?"),
    ("billing + ordering + configurator",
     "I was charged twice for my order deposit, and the configurator won't let me change "
     "the wheel option on the order I already placed."),
    ("service + warranty + billing",
     "The car pulls to the left since the last service. I want the repair covered under "
     "warranty and a refund of the service fee I paid."),

    # --- three UNRELATED topics in one message -------------------------------
    ("technical + configurator + billing (unrelated)",
     "The brake warning light is on, my app login stopped working, and my last invoice "
     "has a charge I don't recognise."),

    # --- safety issue buried in a mixed message -------------------------------
    ("safety buried in ordering/billing",
     "Quick question about my order status and the deposit invoice, also the steering "
     "wheel locked up briefly on the motorway yesterday."),
]


def run(kb: KnowledgeBase, query: str, top_k: int, threshold: float, use_llm: bool) -> None:
    emb = kb.embed(query)
    state = {
        "query": query,
        "query_embedding": emb,
        "top_k": top_k,
        "confidence_threshold": threshold,
    }

    # full ranking over all categories (same math as classify_node)
    sims = kb.category_embeddings @ emb
    order = np.argsort(-sims)
    ranked = [(kb.category_names[i], float(sims[i])) for i in order]
    margin = ranked[0][1] - ranked[1][1]

    state.update(classify_node(state, kb))
    state.update(retrieve_top_k_node(state, kb))
    state.update(determine_priority_node(state))
    state.update(route_node(state, kb))
    state.update(compute_confidence_node(state))
    if use_llm and not state["escalated"]:
        state.update(generate_resolution_notes_node(state))

    mix = Counter(c["category"] for c in state["retrieved_cases"])

    print("  category ranking : " + "  ".join(f"{n}={s:.2f}" for n, s in ranked[:4]) + "  ...")
    print(f"  top1-top2 margin : {margin:.3f}   {'<-- narrow, classifier unsure' if margin < 0.05 else ''}")
    print(f"  retrieved mix    : {dict(mix)}")
    for c in state["retrieved_cases"]:
        print(f"      [{c['similarity']:.2f}] {c['category']:<12} {c['priority']:<7} {c['text']}")
    print(f"  -> category={state['category']}  priority={state['priority']}  "
          f"queue={state['routed_queue']}")
    print(f"  -> confidence={state['confidence']:.3f}  escalated={state['escalated']}")
    if use_llm and not state["escalated"]:
        print(f"  -> notes: {state['resolution_notes']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--llm", action="store_true", help="also generate resolution notes")
    args = p.parse_args()

    kb = KnowledgeBase()
    for label, q in QUERIES:
        print("\n" + "=" * 100)
        print(f"[{label}]")
        print(f"Q: {q}")
        run(kb, q, args.top_k, args.threshold, args.llm)


if __name__ == "__main__":
    main()
