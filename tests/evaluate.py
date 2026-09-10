import sys, csv, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.ingestion import KnowledgeBase

random.seed(0)


def retrieve_excluding_self(kb, query_embedding, case_id, top_k):
    results = kb.chroma_collection.query(
        query_embeddings=[query_embedding.tolist()], n_results=top_k + 1
    )
    retrieved = []
    for doc, meta, dist, rid in zip(
        results["documents"][0], results["metadatas"][0],
        results["distances"][0], results["ids"][0]
    ):
        if rid == case_id:
            continue
        retrieved.append({"category": meta["category"], "similarity": 1 - dist})
        if len(retrieved) == top_k:
            break
    return retrieved


def main():
    kb = KnowledgeBase()
    with open("data/past_cases.csv") as f:
        cases = list(csv.DictReader(f))

    sample = random.sample(cases, 60)
    results = []
    for c in sample:
        emb = kb.embed(c["inquiry_text"])
        sims = kb.category_embeddings @ emb
        best_idx = int(np.argmax(sims))
        pred_category = kb.category_names[best_idx]
        cat_score = float(sims[best_idx])

        retrieved = retrieve_excluding_self(kb, emb, c["case_id"], top_k=3)
        agreement = (sum(1 for r in retrieved if r["category"] == pred_category)
                     / len(retrieved)) if retrieved else 0.0

        confidence = 0.5 * cat_score + 0.5 * agreement
        correct = pred_category == c["category"]
        results.append((confidence, correct))

    print(f"Sample size: {len(results)}\n")
    print(f"{'threshold':>10} {'escalation_rate':>16} {'accuracy_on_auto':>18}")
    for thresh in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        auto = [r for r in results if r[0] >= thresh]
        esc_rate = 1 - len(auto) / len(results)
        acc = (sum(c for _, c in auto) / len(auto)) if auto else float('nan')
        print(f"{thresh:>10.2f} {esc_rate:>16.2%} {acc:>18.2%}")


if __name__ == "__main__":
    main()