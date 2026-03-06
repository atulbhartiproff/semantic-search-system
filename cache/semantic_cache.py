import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:

    def __init__(self, similarity_threshold=0.85):
        self.threshold = similarity_threshold
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, query_embedding, cluster_id):

        if cluster_id not in self.cache:
            return None

        for entry in self.cache[cluster_id]:
            sim = cosine_similarity(
                [query_embedding], [entry["embedding"]])[0][0]

            if sim >= self.threshold:
                self.hit_count += 1
                return entry, sim

        self.miss_count += 1
        return None

    def store(self, query, embedding, result, cluster_id):

        if cluster_id not in self.cache:
            self.cache[cluster_id] = []

        self.cache[cluster_id].append({
            "query": query,
            "embedding": embedding,
            "result": result
        })

    def stats(self):

        total_entries = sum(len(v) for v in self.cache.values())
        hits = self.hit_count
        misses = self.miss_count
        rate = hits / (hits + misses) if hits + misses > 0 else 0

        return {
            "total_entries": total_entries,
            "hit_count": hits,
            "miss_count": misses,
            "hit_rate": rate
        }

    def clear(self):
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0