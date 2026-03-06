from fastapi import FastAPI
from pydantic import BaseModel

from data.dataset_loader import load_dataset
from embedding.embedding_model import EmbeddingModel
from vectorstore.faiss_index import FAISSVectorStore
from clustering.fuzzy_cluster import FuzzyCluster
from cache.semantic_cache import SemanticCache
from retrieval.search_engine import SearchEngine


app = FastAPI()


class QueryRequest(BaseModel):
    query: str


docs, labels = load_dataset()

embedder = EmbeddingModel()
doc_embeddings = embedder.embed_documents(docs)

vectorstore = FAISSVectorStore(dim=len(doc_embeddings[0]))
vectorstore.add_documents(doc_embeddings, docs)

clusterer = FuzzyCluster(n_clusters=15)
clusterer.fit(doc_embeddings)

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
            "similarity_score": sim,
            "result": entry["result"],
            "dominant_cluster": cluster_id
        }

    result = results[0]["document"]

    cache.store(query, embedding, result, cluster_id)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result,
        "dominant_cluster": cluster_id
    }


@app.get("/cache/stats")
def cache_stats():
    return cache.stats()


@app.delete("/cache")
def clear_cache():
    cache.clear()
    return {"message": "Cache cleared"}