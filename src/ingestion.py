import json
import csv
from pathlib import Path
from collections import defaultdict, Counter

import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = Path(__file__).resolve().parent.parent / ".chroma"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_taxonomy() -> list[dict]:
    with open(DATA_DIR / "taxonomy.json") as f:
        return json.load(f)["categories"]


def load_past_cases() -> list[dict]:
    with open(DATA_DIR / "past_cases.csv") as f:
        return list(csv.DictReader(f))


def build_category_queue_map(cases: list[dict]) -> dict[str, str]:
    """category -> routed_queue, derived from historical data. Verified 1:1 on this dataset."""
    mapping = defaultdict(set)
    for row in cases:
        mapping[row["category"]].add(row["routed_queue"])
    resolved = {}
    for cat, queues in mapping.items():
        resolved[cat] = sorted(queues)[0] if len(queues) == 1 else Counter(
            row["routed_queue"] for row in cases if row["category"] == cat
        ).most_common(1)[0][0]
    return resolved


def category_embedding_text(category: dict) -> str:
    """One embeddable string per taxonomy category: description + keywords."""
    return category["description"] + " Keywords: " + ", ".join(category["keywords"])


class KnowledgeBase:
    """
    Holds everything the triage pipeline needs, built once at startup:
      - self.model                sentence-transformers embedding model
      - self.category_names       list[str], in fixed order
      - self.category_embeddings  np.ndarray, aligned with category_names
      - self.category_queue_map   dict category -> routed_queue
      - self.chroma_collection    ChromaDB collection of embedded past cases
    """

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        taxonomy = load_taxonomy()
        self.category_names = [c["name"] for c in taxonomy]
        cat_texts = [category_embedding_text(c) for c in taxonomy]
        self.category_embeddings = self.model.encode(
            cat_texts, normalize_embeddings=True
        )

        cases = load_past_cases()
        self.category_queue_map = build_category_queue_map(cases)

        self._init_chroma(cases)

    def _init_chroma(self, cases: list[dict]) -> None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            client.delete_collection("past_cases")
        except Exception:
            pass
        self.chroma_collection = client.create_collection("past_cases")

        texts = [c["inquiry_text"] for c in cases]
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        self.chroma_collection.add(
            ids=[c["case_id"] for c in cases],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[
                {
                    "category": c["category"],
                    "priority": c["priority"],
                    "routed_queue": c["routed_queue"],
                }
                for c in cases
            ],
        )

    def embed(self, text: str):
        """Embed a single inquiry the same way category/case embeddings were built."""
        return self.model.encode([text], normalize_embeddings=True)[0]