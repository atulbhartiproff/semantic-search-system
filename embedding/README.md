# embedding

Embedding model wrapper using `sentence-transformers`.

- File: `embedding_model.py`
- Class: `EmbeddingModel` (wraps `SentenceTransformer("all-MiniLM-L6-v2")`).

Key behavior:

- `embed_documents(docs)` — encodes a list of documents into embeddings (shows a progress bar).
- `embed_query(query)` — encodes a single query string and returns a vector.

Dependency: requires the `sentence-transformers` package and the `all-MiniLM-L6-v2` model.
