import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage


class UnsupervisedVisualizer:
    """
    Visualizes unsupervised learning pipelines:
    - Elbow and Silhouette Curves
    - 2D Cluster Projections (PCA / t-SNE)
    - Hierarchical Clustering Dendrogram
    - PCA Scree / Explained Variance Plots
    """

    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def plot_elbow_and_silhouette(self, eval_results, dataset_name="Dataset"):
        """Plots Inertia (Elbow) and Silhouette Score side-by-side."""
        k_values = eval_results["k_values"]
        inertias = eval_results["inertias"]
        sil_scores = eval_results["silhouette_scores"]
        optimal_k = eval_results["optimal_k"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # 1. Elbow Plot
        ax1.plot(k_values, inertias, "bo-", lw=2, markersize=8)
        ax1.set_title("Elbow Method (Inertia vs. k)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Number of Clusters (k)", fontsize=10)
        ax1.set_ylabel("Inertia (Sum of Squared Distances)", fontsize=10)
        ax1.grid(True)

        # 2. Silhouette Plot
        ax2.plot(k_values, sil_scores, "go-", lw=2, markersize=8)
        ax2.axvline(optimal_k, color="red", linestyle="--", label=f"Optimal k = {optimal_k}")
        ax2.set_title(f"Silhouette Analysis (Optimal k = {optimal_k})", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Number of Clusters (k)", fontsize=10)
        ax2.set_ylabel("Silhouette Score", fontsize=10)
        ax2.legend()
        ax2.grid(True)

        plt.suptitle(f"Cluster Optimization Analysis - {dataset_name}", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{dataset_name.lower().replace(' ', '_')}_elbow_silhouette.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {save_path}")

    def plot_clusters_2d(self, X_2d, cluster_dict, title_prefix="PCA Projection", dataset_name="Dataset"):
        """
        Plots 2D scatter plots for multiple clustering algorithms side-by-side.
        """
        n_models = len(cluster_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
        if n_models == 1:
            axes = [axes]

        for ax, (model_name, labels) in zip(axes, cluster_dict.items()):
            scatter = ax.scatter(
                X_2d[:, 0],
                X_2d[:, 1],
                c=labels,
                cmap="tab10",
                alpha=0.7,
                edgecolors="k",
                s=40
            )
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            ax.set_title(f"{model_name} (Clusters: {n_clusters})", fontsize=11, fontweight="bold")
            ax.set_xlabel("Component 1", fontsize=9)
            ax.set_ylabel("Component 2", fontsize=9)

        plt.suptitle(f"{title_prefix} - {dataset_name}", fontsize=13, fontweight="bold", y=1.03)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{dataset_name.lower().replace(' ', '_')}_{title_prefix.lower().replace(' ', '_')}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {save_path}")

    def plot_dendrogram(self, X, dataset_name="Dataset", max_samples=150):
        """Plots hierarchical clustering dendrogram."""
        # Subsample if dataset is large for readability
        if X.shape[0] > max_samples:
            idx = np.random.RandomState(42).choice(X.shape[0], max_samples, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        linked = linkage(X_sample, method="ward")
        plt.figure(figsize=(11, 5))
        dendrogram(linked, orientation="top", distance_sort="descending", show_leaf_counts=True)
        plt.title(f"Hierarchical Clustering Dendrogram (Ward Linkage) - {dataset_name}", fontsize=13, fontweight="bold")
        plt.xlabel("Sample Index", fontsize=10)
        plt.ylabel("Distance", fontsize=10)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, f"{dataset_name.lower().replace(' ', '_')}_dendrogram.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {save_path}")

    def plot_pca_variance(self, explained_var_ratio, dataset_name="Dataset"):
        """Plots PCA explained variance bar chart."""
        plt.figure(figsize=(7, 4.5))
        n_comps = len(explained_var_ratio)
        comp_labels = [f"PC{i+1}" for i in range(n_comps)]
        cumulative = np.cumsum(explained_var_ratio)

        plt.bar(comp_labels, explained_var_ratio, alpha=0.7, color="teal", label="Individual Variance")
        plt.step(comp_labels, cumulative, where="mid", color="darkred", lw=2, label="Cumulative Variance")
        plt.title(f"PCA Explained Variance Ratio - {dataset_name}", fontsize=12, fontweight="bold")
        plt.xlabel("Principal Components", fontsize=10)
        plt.ylabel("Explained Variance Ratio", fontsize=10)
        plt.ylim(0, 1.05)
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, f"{dataset_name.lower().replace(' ', '_')}_pca_variance.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {save_path}")
