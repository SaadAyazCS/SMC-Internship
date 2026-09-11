import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, make_blobs
from sklearn.preprocessing import StandardScaler

from clustering_models import ClusteringModels
from dimensionality_reduction import DimensionalityReducer
from cluster_evaluation import ClusterEvaluator
from visualization import UnsupervisedVisualizer


def load_datasets():
    """Loads 3 diverse datasets for unsupervised clustering benchmarking."""
    datasets = {}

    # 1. Iris dataset
    iris = load_iris(as_frame=True)
    datasets["Iris"] = (iris.data, iris.target)

    # 2. Wine dataset
    wine = load_wine(as_frame=True)
    datasets["Wine"] = (wine.data, wine.target)

    # 3. Synthetic Blobs
    X_blobs, y_blobs = make_blobs(n_samples=500, n_features=6, centers=4, cluster_std=1.2, random_state=42)
    feature_names = [f"feat_{i}" for i in range(6)]
    datasets["Synthetic Blobs (4 Clusters)"] = (pd.DataFrame(X_blobs, columns=feature_names), pd.Series(y_blobs))

    return datasets


def main():
    print("================================================================")
    print(" WEEK 4 - DAY 3: UNSUPERVISED LEARNING & CLUSTERING PIPELINE")
    print("================================================================")

    visualizer = UnsupervisedVisualizer(output_dir="results")
    reducer = DimensionalityReducer(random_state=42)
    evaluator = ClusterEvaluator(random_state=42)
    clustering = ClusteringModels(random_state=42)

    # [Step 1] Load Datasets
    print("\n[Step 1] Loading Unsupervised Datasets...")
    datasets = load_datasets()
    for name, (X, y) in datasets.items():
        print(f"  - {name}: {X.shape[0]} samples, {X.shape[1]} features")

    comparison_records = []

    # Process each dataset
    for name, (X, y) in datasets.items():
        print(f"\n==================================================")
        print(f"  Processing Dataset: {name}")
        print(f"==================================================")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 2. Dimensionality Reduction (PCA & t-SNE)
        pca_result = reducer.apply_pca(X_scaled, n_components=2)
        X_pca = pca_result["transformed_data"]
        print(f"  PCA: Retained {pca_result['total_variance_explained']*100:.2f}% of total variance with 2 components.")

        tsne_result = reducer.apply_tsne(X_scaled, n_components=2)
        X_tsne = tsne_result["transformed_data"]

        # Visualizations for variance & dendrogram
        visualizer.plot_pca_variance(pca_result["explained_variance_ratio"], dataset_name=name)
        visualizer.plot_dendrogram(X_scaled, dataset_name=name)

        # 3. Finding Optimal Clusters via Elbow and Silhouette
        opt_analysis = evaluator.run_elbow_and_silhouette(X_scaled, k_range=range(2, 9))
        opt_k = opt_analysis["optimal_k"]
        print(f"  Optimal k identified by Silhouette Score: {opt_k} (Silhouette = {opt_analysis['best_silhouette']:.4f})")
        visualizer.plot_elbow_and_silhouette(opt_analysis, dataset_name=name)

        # 4. Fit Clustering Algorithms with optimal k
        # Tune DBSCAN eps adaptively based on feature count
        eps_val = 0.8 if X.shape[1] <= 4 else 1.5
        cluster_assignments = clustering.fit_all(X_scaled, n_clusters=opt_k, eps=eps_val, min_samples=5)

        # 5. Visualizing Clusters on 2D projections
        visualizer.plot_clusters_2d(X_pca, cluster_assignments, title_prefix="PCA Clusters", dataset_name=name)
        visualizer.plot_clusters_2d(X_tsne, cluster_assignments, title_prefix="t-SNE Clusters", dataset_name=name)

        # 6. Evaluate each clustering algorithm
        for alg_name, labels in cluster_assignments.items():
            metrics = evaluator.evaluate_clustering(X_scaled, labels, y_true=y)
            record = {
                "Dataset": name,
                "Algorithm": alg_name,
                "Clusters Found": metrics["Clusters"],
                "Silhouette": metrics["Silhouette"],
                "Calinski-Harabasz": metrics["Calinski-Harabasz"],
                "Davies-Bouldin": metrics["Davies-Bouldin"],
                "ARI (Ground Truth)": metrics["ARI"],
                "NMI (Ground Truth)": metrics["NMI"]
            }
            comparison_records.append(record)

    # [Step 7] Summary Table & Export
    summary_df = pd.DataFrame(comparison_records)
    print("\n=========================================================================================")
    print("                      CLUSTERING BENCHMARK COMPARISON TABLE                              ")
    print("=========================================================================================")
    print(summary_df.to_string(index=False))

    csv_path = os.path.join("results", "unsupervised_comparison.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark comparison table to: {csv_path}")

    print("\n================================================================")
    print(" DAY 3 UNSUPERVISED PIPELINE COMPLETED SUCCESSFULLY!")
    print("================================================================")


if __name__ == "__main__":
    main()
