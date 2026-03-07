# api

FastAPI service exposing the semantic search endpoints.

- File: `main.py`
- Purpose: load precomputed artifacts and wire together the `EmbeddingModel`, `FAISSVectorStore`, `FuzzyCluster`, `SemanticCache`, and `SearchEngine` to provide HTTP endpoints.

Endpoints:

- `POST /query` — body `{ "query": "..." }`. Returns a search result and whether it was served from the semantic cache.
- `GET /cache/stats` — returns cache metrics (`total_entries`, `hit_count`, `miss_count`, `hit_rate`).
- `DELETE /cache` — clears the semantic cache.

Notes:

- The service expects precomputed artifacts in `artifacts/` (`embeddings.npy`, `faiss.index`, `docs.pkl`, `cluster.pkl`). Use `scripts/precompute.py` to generate them.
- Run locally with: `uvicorn api.main:app --reload`.
