from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, docs):
        return self.model.encode(docs, show_progress_bar=True)

    def embed_query(self, query):
        return self.model.encode([query])[0]