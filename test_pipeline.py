from data.dataset_loader import load_dataset
from embedding.embedding_model import get_embeddings
from vectorstore.faiss_index import build_index
from retrieval.search_engine import search

# load data
docs = load_dataset()

# embed
embeddings = get_embeddings(docs)

# build index
index = build_index(embeddings)

# query
query = "machine learning"
results = search(query, docs, index)

print("Results:")
for r in results:
    print(r)