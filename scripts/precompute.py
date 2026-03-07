import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pickle
import faiss

from data.dataset_loader import load_dataset
from embedding.embedding_model import EmbeddingModel
from clustering.fuzzy_cluster import FuzzyCluster

print("Loading dataset...")
docs, labels = load_dataset()

print("Generating embeddings...")
embedder = EmbeddingModel()
embeddings = embedder.embed_documents(docs)

print("Training clustering...")
clusterer = FuzzyCluster(n_clusters=15)
clusterer.fit(embeddings)

print("Building FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings.astype("float32"))

print("Saving artifacts...")

np.save("artifacts/embeddings.npy", embeddings)

faiss.write_index(index, "artifacts/faiss.index")

with open("artifacts/docs.pkl", "wb") as f:
    pickle.dump(docs, f)

with open("artifacts/cluster.pkl", "wb") as f:
    pickle.dump(clusterer, f)

print("Precomputation complete.")