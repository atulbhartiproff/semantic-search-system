# retrieval

High-level search logic that ties embedding, vectorstore and clustering.

- File: `search_engine.py`
- Class: `SearchEngine`

Key behavior:

- `search(query)` performs the following steps:
	1. Embed the query via the configured embedder (`embed_query`).
	2. Query the `FAISSVectorStore` with the embedding to get top-k results.
	3. Use the `FuzzyCluster` to compute the dominant cluster for the query embedding.
	4. Returns `(embedding, results, cluster_id)`.

Usage: used by the API to get search results and decide cache lookup/store by cluster.
