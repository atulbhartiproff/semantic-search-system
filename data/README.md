# data

Dataset loading and text-cleaning utilities.

- File: `dataset_loader.py`

Key behavior:

- `load_dataset()` downloads and returns the 20 Newsgroups dataset using `sklearn.datasets.fetch_20newsgroups` (with headers/footers/quotes removed) and returns `(docs, labels)`.
- `clean_text(text)` strips common mailing-list headers (From/Subject/Organization/Lines), quoted lines, and collapses whitespace.

Usage: `scripts/precompute.py` calls `load_dataset()` to obtain documents to embed and index.
