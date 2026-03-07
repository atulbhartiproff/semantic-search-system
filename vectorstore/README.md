# vectorstore

FAISS-backed vector store wrapper.

- File: `faiss_index.py`
- Class: `FAISSVectorStore`

Key behavior:

- `__init__(dim)` creates a FAISS IndexFlatL2 index and an empty `documents` list.
- `add_documents(embeddings, docs)` adds embeddings (converted to `float32`) to the FAISS index and appends `docs` to `documents`.
- `search(query_embedding, k=5)` returns up to `k` nearest documents as a list of `{ "document": ..., "score": ... }` entries. Scores are L2 distances from the FAISS index.

Note: `query_embedding` must be a NumPy array; the class converts inputs to `float32` internally.
