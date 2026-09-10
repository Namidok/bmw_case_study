from src.ingestion import KnowledgeBase
from src.main import run_triage

kb = KnowledgeBase()
result = run_triage(
    "The brake fluid warning light came on and the pedal feels soft",
    kb, top_k=3, confidence_threshold=0.5
)

for k, v in result.items():
    if k not in ("query_embedding",):
        print(f"{k}: {v}")