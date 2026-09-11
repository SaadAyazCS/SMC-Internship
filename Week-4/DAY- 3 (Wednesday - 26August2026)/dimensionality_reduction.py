import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class DimensionalityReducer:
    """
    Implements dimensionality reduction techniques:
    - Principal Component Analysis (PCA) for linear variance retention
    - t-Distributed Stochastic Neighbor Embedding (t-SNE) for non-linear manifold visualization
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def apply_pca(self, X, n_components=2):
        """
        Applies PCA, returning transformed 2D coordinates and explained variance ratios.
        """
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X)
        var_ratio = pca.explained_variance_ratio_
        return {
            "transformed_data": X_pca,
            "model": pca,
            "explained_variance_ratio": var_ratio,
            "total_variance_explained": float(np.sum(var_ratio))
        }

    def apply_tsne(self, X, n_components=2, perplexity=30.0, max_iter=1000):
        """
        Applies t-SNE for low-dimensional non-linear visualization.
        """
        # Adjust perplexity if sample size is very small
        actual_perplexity = min(perplexity, max(5, (X.shape[0] - 1) // 3))
        tsne = TSNE(
            n_components=n_components,
            perplexity=actual_perplexity,
            max_iter=max_iter,
            random_state=self.random_state
        )
        X_tsne = tsne.fit_transform(X)
        return {
            "transformed_data": X_tsne,
            "model": tsne
        }
