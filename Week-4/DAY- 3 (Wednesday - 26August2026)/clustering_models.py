import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


class ClusteringModels:
    """
    Implements core unsupervised clustering algorithms:
    - K-Means Clustering
    - Agglomerative (Hierarchical) Clustering
    - Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit_kmeans(self, X, n_clusters=3):
        """Fits K-Means and returns the model and cluster labels."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        return kmeans, labels

    def fit_hierarchical(self, X, n_clusters=3, linkage="ward"):
        """Fits Agglomerative Hierarchical Clustering."""
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = hierarchical.fit_predict(X)
        return hierarchical, labels

    def fit_dbscan(self, X, eps=0.5, min_samples=5):
        """Fits DBSCAN density-based clustering."""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        return dbscan, labels

    def fit_all(self, X, n_clusters=3, eps=0.5, min_samples=5):
        """Fits all 3 clustering algorithms and returns cluster label dictionary."""
        _, km_labels = self.fit_kmeans(X, n_clusters=n_clusters)
        _, agg_labels = self.fit_hierarchical(X, n_clusters=n_clusters)
        _, db_labels = self.fit_dbscan(X, eps=eps, min_samples=min_samples)

        return {
            "K-Means": km_labels,
            "Hierarchical": agg_labels,
            "DBSCAN": db_labels
        }
