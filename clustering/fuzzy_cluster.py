from sklearn.mixture import GaussianMixture
import numpy as np


class FuzzyCluster:

    def __init__(self, n_clusters=15):
        self.model = GaussianMixture(n_components=n_clusters, covariance_type="full")

    def fit(self, embeddings):
        self.model.fit(embeddings)

    def get_membership(self, embeddings):
        return self.model.predict_proba(embeddings)

    def dominant_cluster(self, embedding):
        probs = self.model.predict_proba([embedding])[0]
        return int(np.argmax(probs)), probs