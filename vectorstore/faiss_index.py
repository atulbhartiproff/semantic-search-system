import faiss
import numpy as np


class FAISSVectorStore:

    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []

    def add_documents(self, embeddings, docs):
        self.index.add(np.array(embeddings).astype("float32"))
        self.documents.extend(docs)

    def search(self, query_embedding, k=5):
        query_embedding = query_embedding.astype("float32").reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "document": self.documents[idx],
                "score": float(distances[0][i])
            })

        return results