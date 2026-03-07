class SearchEngine:

    def __init__(self, embedder, vectorstore, clusterer):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.clusterer = clusterer

    def search(self, query):

        embedding = self.embedder.embed_query(query)

        results = self.vectorstore.search(embedding, k=5)

        cluster_id, probs = self.clusterer.dominant_cluster(embedding)

        return embedding, results, cluster_id