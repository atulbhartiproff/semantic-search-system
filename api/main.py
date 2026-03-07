from fastapi import FastAPI
from pydantic import BaseModel

from data.dataset_loader import load_dataset
from embedding.embedding_model import EmbeddingModel
from vectorstore.faiss_index import FAISSVectorStore
from clustering.fuzzy_cluster import FuzzyCluster
from cache.semantic_cache import SemanticCache
from retrieval.search_engine import SearchEngine
import numpy as np
import pickle
import faiss

app = FastAPI(
    title="Semantic Search Engine API",
    description="""
Semantic search with fuzzy clustering and semantic caching on the 20 Newsgroups dataset.

Features:

• Vector-based semantic search  
• Fuzzy clustering for document structure  
• Intelligent semantic cache  
• FastAPI service with live statistics  

To try it out, Go iinside each component and click on [ TRY IT OUT ] button. You can also use the /query endpoint to test the search functionality.
""",
    version="1.0.0",
    contact={
        "name": "Atul",
        "email": "atul.bharti2023@vitstudent.ac.in"
    },
)


class QueryRequest(BaseModel):
    query: str


# docs, labels = load_dataset()

# embedder = EmbeddingModel()
# doc_embeddings = embedder.embed_documents(docs)

# vectorstore = FAISSVectorStore(dim=len(doc_embeddings[0]))
# vectorstore.add_documents(doc_embeddings, docs)

# clusterer = FuzzyCluster(n_clusters=15)
# clusterer.fit(doc_embeddings)

print("Loading precomputed artifacts")

embedder = EmbeddingModel()

embeddings = np.load("artifacts/embeddings.npy")

index = faiss.read_index("artifacts/faiss.index")

with open("artifacts/docs.pkl", "rb") as f:
    docs = pickle.load(f)

with open("artifacts/cluster.pkl", "rb") as f:
    clusterer = pickle.load(f)

vectorstore = FAISSVectorStore(dim=embeddings.shape[1])
vectorstore.index = index
vectorstore.documents = docs

print("Artifacts loaded.")

cache = SemanticCache()

engine = SearchEngine(embedder, vectorstore, clusterer)


@app.post("/query")
def query_endpoint(req: QueryRequest):

    query = req.query

    embedding, results, cluster_id = engine.search(query)

    cached = cache.lookup(embedding, cluster_id)

    if cached:
        entry, sim = cached

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": float(sim),  # FIX
            "result": entry["result"],
            "dominant_cluster": int(cluster_id)  # FIX
        }

    result = results[0]["document"]

    cache.store(query, embedding, result, cluster_id)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result,
        "dominant_cluster": int(cluster_id)  # FIX
    }

@app.get("/cache/stats")
def cache_stats():

    stats = cache.stats()

    return {
        "total_entries": int(stats["total_entries"]),
        "hit_count": int(stats["hit_count"]),
        "miss_count": int(stats["miss_count"]),
        "hit_rate": float(stats["hit_rate"])
    }

@app.delete("/cache")
def clear_cache():
    cache.clear()
    return {"message": "Cache cleared"}