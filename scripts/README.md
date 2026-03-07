# scripts

Utility and maintenance scripts for building artifacts used by the service.

- File: `precompute.py`

What it does:

- Loads the 20 Newsgroups dataset via `data.dataset_loader.load_dataset()`.
- Generates document embeddings using `EmbeddingModel`.
- Trains a `FuzzyCluster` model on embeddings.
- Builds a FAISS index and saves artifacts to `artifacts/` (`embeddings.npy`, `faiss.index`, `docs.pkl`, `cluster.pkl`).

Run:

```
python scripts/precompute.py
```

This script must run where the project root is reachable (it appends the parent directory to `sys.path`).
