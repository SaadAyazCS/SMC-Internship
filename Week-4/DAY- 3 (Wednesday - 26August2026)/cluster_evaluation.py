import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)


class ClusterEvaluator:
    """
    Evaluates clustering quality:
    - Elbow Method (Inertia vs. k)
    - Silhouette Analysis (identifies optimal k)
    - Unsupervised metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin
    - Supervised external validation: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI)
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def run_elbow_and_silhouette(self, X, k_range=range(2, 10)):
        """
        Computes inertias and silhouette scores for K-Means across a range of k values.
        """
        inertias = []
        silhouette_scores = []
        k_values = list(k_range)

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            sil_score = silhouette_score(X, labels)
            silhouette_scores.append(sil_score)

        optimal_k_idx = int(np.argmax(silhouette_scores))
        optimal_k = k_values[optimal_k_idx]

        return {
            "k_values": k_values,
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "optimal_k": optimal_k,
            "best_silhouette": silhouette_scores[optimal_k_idx]
        }

    def evaluate_clustering(self, X, labels, y_true=None):
        """
        Computes comprehensive evaluation metrics for a set of cluster labels.
        """
        # Exclude noise points (-1 in DBSCAN) if calculating metrics
        mask = labels != -1
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters < 2:
            return {
                "Clusters": n_clusters,
                "Silhouette": None,
                "Calinski-Harabasz": None,
                "Davies-Bouldin": None,
                "ARI": None,
                "NMI": None
            }

        X_valid = X[mask]
        labels_valid = labels[mask]

        sil = silhouette_score(X_valid, labels_valid)
        ch = calinski_harabasz_score(X_valid, labels_valid)
        db = davies_bouldin_score(X_valid, labels_valid)

        ari = None
        nmi = None
        if y_true is not None:
            y_valid = np.array(y_true)[mask]
            ari = adjusted_rand_score(y_valid, labels_valid)
            nmi = normalized_mutual_info_score(y_valid, labels_valid)

        return {
            "Clusters": n_clusters,
            "Silhouette": round(sil, 4),
            "Calinski-Harabasz": round(ch, 2),
            "Davies-Bouldin": round(db, 4),
            "ARI": round(ari, 4) if ari is not None else "N/A",
            "NMI": round(nmi, 4) if nmi is not None else "N/A"
        }
