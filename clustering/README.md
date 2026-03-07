# clustering

Fuzzy clustering utilities for embeddings.

- File: `fuzzy_cluster.py`
- Class: `FuzzyCluster` (wraps `sklearn.mixture.GaussianMixture`).

Key behavior:

- `fit(embeddings)` trains a GaussianMixture model on embeddings.
- `get_membership(embeddings)` returns soft membership probabilities for each cluster.
- `dominant_cluster(embedding)` returns a tuple `(cluster_id, probs)` where `cluster_id` is the index of the highest-probability cluster and `probs` are the per-cluster probabilities.

Usage: used by the `SearchEngine` to determine the dominant cluster for a query embedding.
