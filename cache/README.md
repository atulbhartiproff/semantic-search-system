# cache

Semantic caching utilities used to avoid repeating similar queries.

- File: `semantic_cache.py`
- Class: `SemanticCache`

Key behavior:

- Maintains a per-cluster cache: `cache[cluster_id] = [entries]` where each entry has `query`, `embedding`, and `result`.
- Uses cosine similarity to compare a new query embedding against stored embeddings and returns a cached result when similarity >= `similarity_threshold` (default 0.85).
- Methods: `lookup(query_embedding, cluster_id)`, `store(query, embedding, result, cluster_id)`, `stats()`, `clear()`.

Note: Threshold and storage strategy can be tuned for your workload and memory constraints.
