# Smart Inquiry Triage Assistant

A prototype that reads a customer inquiry and automatically determines its category, priority, routed team, and a suggested next step — with a confidence score that flags uncertain cases for human review instead of guessing.

## How to run it

1. Clone this repo and `cd` into it.
2. Create and activate a virtual environment:
   - `python3 -m venv venv`
   - `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Install and start Ollama, then pull the model used here:
   - `ollama serve`
   - `ollama pull llama3.2:latest`
5. Run the app: `streamlit run app/app.py`

First run downloads the embedding model (~90MB, one-time, needs internet). Everything after that runs fully locally, offline.

## Models used

- **Embeddings:** `all-MiniLM-L6-v2` (sentence-transformers) — used for both classification and retrieval.
- **LLM:** `llama3.2:latest` via Ollama, local — used for exactly one thing: generating the resolution note.

## Architecture

Built as a LangGraph state machine with 5 nodes:

classify → retrieve_top_k → determine_priority → route → resolve

(confidence is checked before resolve; branches to an escalation message if low, otherwise generates a normal resolution note)

## Key design decisions

**Why the LLM only generates resolution notes, not category/priority/routing.**
Structured fields need to be reliable and explainable. Asking an LLM to classify directly risks inconsistent answers between near-identical inputs, or categories outside the taxonomy. Instead:

- **Classification** compares the inquiry's embedding against each taxonomy category's embedded description (cosine similarity, deterministic).
- **Priority** is a similarity-weighted average across the retrieved Top-K cases' priorities — closer matches count more than weaker ones, rather than a plain majority vote.
- **Routing** is a direct lookup from category to team, precomputed from historical data (verified 1:1 mapping, no conflicts in the provided dataset).
- **Resolution notes** — the one place natural language generation is actually needed — is the LLM's only job.

**Confidence score.**

confidence = 0.5 × category_score + 0.5 × agreement

where `category_score` is how close the inquiry is to its predicted category's definition, and `agreement` is what fraction of the retrieved Top-K cases share that same predicted category. Two independent signals: "how sure is the classifier" and "do similar real cases back it up." If either is low, confidence drops and the inquiry gets escalated below the configurable threshold.

One observation from testing: raw `category_score` tends to run lower than retrieval `similarity`, because it compares the inquiry against a short abstract category *definition*, while retrieval compares it against real past examples. That's expected, not a bug — it's exactly why the two signals are combined rather than relying on either alone.

**Threshold calibration.** 

The default (0.35) wasn't guessed — it comes from a sweep (`tests/evaluate.py`) that tests each historical case against the rest of the dataset. On a 60-case sample: accuracy on auto-handled cases reaches 100% at 0.35 and stays there through 0.60, while escalation rate keeps climbing meaning anything above 0.35 escalates more without improving correctness. 0.30 still let one wrong classification through automatically (98% accuracy), which is why 0.35 was chosen over the lower value.

**ChromaDB configured for cosine space explicitly**, since embeddings are pre-normalized — makes the similarity score a direct, explainable `1 - distance` rather than an opaque default metric.

## Known limitations

- Priority weighting can occasionally be pulled away from the single closest match by two weaker-but-present matches (a known trade-off of averaging vs. "always trust the closest neighbor").
- Category classification's raw confidence score is lower than intuition suggests, for the reason described above — worth recalibrating with a larger, more varied taxonomy description set.
- No persistence between app restarts — ChromaDB rebuilds from `past_cases.csv` on every startup. Fine at 300 rows; would need a persistent store at scale.
- Evaluated systematically via `tests/evaluate.py` (leave-one-out style sweep
  against the historical dataset) and `tests/test_multi_category.py` (targeted
  multi-intent and edge-case probing) — see `docs/multi_category_findings.md`
  for detailed findings.
- **Known failure mode:** 

  Single-label classification cannot split multi-intent
  queries. Example: "order status + deposit invoice + steering wheel locked up
  on the motorway" — two admin-sounding clauses drown out the one safety
  clause; confidence lands at 0.72 and the inquiry does not escalate. No
  threshold adjustment fixes this — it's structural. Planned fix: a
  lightweight safety-keyword pre-check ahead of classification, or intent
  splitting with priority = max across detected intents.

## Project structure

- `app/app.py` — Streamlit frontend (provided) + triage_inquiry wiring
- `src/ingestion.py` — Loads taxonomy + past cases, builds embeddings, ChromaDB setup
- `src/main.py` — LangGraph nodes, state definition, graph assembly
- `data/` — Provided taxonomy.json and past_cases.csv
- `tests/` — Manual sanity-check scripts used during development