import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from classification_models import ClassificationModels
from imbalanced_handler import ImbalancedHandler
from evaluation_metrics import EvaluationMetrics
from visualization import ClassificationVisualizer
from model_comparison import ClassificationComparison


def load_datasets():
    """Loads binary, multi-class, and imbalanced datasets."""
    datasets = {}

    # 1. Binary classification: Breast Cancer
    cancer = load_breast_cancer(as_frame=True)
    datasets["Breast Cancer (Binary)"] = (cancer.data, cancer.target, True)

    # 2. Multi-class classification: Wine
    wine = load_wine(as_frame=True)
    datasets["Wine (Multi-Class)"] = (wine.data, wine.target, False)

    # 3. Synthetic Imbalanced Dataset (90:10 ratio)
    handler = ImbalancedHandler(random_state=42)
    X_imb, y_imb = handler.create_imbalanced_dataset(n_samples=1000, weights=(0.90, 0.10))
    datasets["Synthetic Imbalanced (90-10)"] = (X_imb, y_imb, True)

    return datasets


def main():
    print("================================================================")
    print(" WEEK 4 - DAY 2: SUPERVISED LEARNING (CLASSIFICATION) PIPELINE")
    print("================================================================")

    visualizer = ClassificationVisualizer(output_dir="results")

    # [Step 1] Load Datasets
    print("\n[Step 1] Loading Classification Datasets...")
    datasets = load_datasets()
    for name, (X, y, is_binary) in datasets.items():
        print(f"  - {name}: {X.shape[0]} samples, {X.shape[1]} features, Binary={is_binary}")

    # [Step 2] Train & Evaluate 8 Classifiers on Breast Cancer (Binary)
    print("\n[Step 2] Training & Evaluating 8 Classifiers on Breast Cancer...")
    X_cancer, y_cancer, _ = datasets["Breast Cancer (Binary)"]
    X_tr, X_te, y_tr, y_te = train_test_split(X_cancer, y_cancer, test_size=0.25, random_state=42, stratify=y_cancer)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    clf_models = ClassificationModels(random_state=42)
    clf_models.train_all(X_tr_scaled, y_tr)
    predictions = clf_models.predict_all(X_te_scaled)

    eval_results = {}
    print("\nSingle Split Evaluation (Breast Cancer):")
    for name, preds in predictions.items():
        metrics = EvaluationMetrics.evaluate(y_te, preds["y_pred"], preds["y_proba"])
        eval_results[name] = metrics
        auc_str = f"{metrics['ROC-AUC']:.4f}" if metrics['ROC-AUC'] is not None else "N/A"
        print(f"  {name:25s} | Acc: {metrics['Accuracy']:.4f} | Prec: {metrics['Precision']:.4f} | Rec: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | AUC: {auc_str}")

    # Visualizations for Breast Cancer
    visualizer.plot_confusion_matrices(eval_results, dataset_name="Breast Cancer")
    visualizer.plot_roc_curves(y_te, predictions, dataset_name="Breast Cancer")

    # [Step 3] Handling Imbalanced Data with SMOTE
    print("\n[Step 3] Addressing Imbalanced Data with SMOTE...")
    handler = ImbalancedHandler(random_state=42)
    X_imb, y_imb, _ = datasets["Synthetic Imbalanced (90-10)"]
    X_tr_imb, X_te_imb, y_tr_imb, y_te_imb = train_test_split(X_imb, y_imb, test_size=0.3, random_state=42, stratify=y_imb)

    X_tr_imb_scaled = scaler.fit_transform(X_tr_imb)
    X_te_imb_scaled = scaler.transform(X_te_imb)

    # Train Random Forest before SMOTE
    rf_before = ClassificationModels(random_state=42).models["Random Forest"]
    rf_before.fit(X_tr_imb_scaled, y_tr_imb)
    pred_before = rf_before.predict(X_te_imb_scaled)
    metrics_before = EvaluationMetrics.evaluate(y_te_imb, pred_before)

    # Apply SMOTE
    X_resampled, y_resampled, before_counts, after_counts = handler.apply_smote(X_tr_imb_scaled, y_tr_imb)
    print(f"  Class counts before SMOTE: {before_counts}")
    print(f"  Class counts after SMOTE : {after_counts}")

    # Train Random Forest after SMOTE
    rf_after = ClassificationModels(random_state=42).models["Random Forest"]
    rf_after.fit(X_resampled, y_resampled)
    pred_after = rf_after.predict(X_te_imb_scaled)
    metrics_after = EvaluationMetrics.evaluate(y_te_imb, pred_after)

    print("\n  Performance on Minority Class:")
    print(f"  - Before SMOTE -> Recall: {metrics_before['Recall']:.4f}, F1: {metrics_before['F1-Score']:.4f}")
    print(f"  - After SMOTE  -> Recall: {metrics_after['Recall']:.4f}, F1: {metrics_after['F1-Score']:.4f}")

    visualizer.plot_imbalance_distribution(before_counts, after_counts)

    # [Step 4] Stratified 5-Fold Cross-Validation Pipeline
    print("\n[Step 4] Running 5-Fold Stratified Cross-Validation across Datasets...")
    comparison = ClassificationComparison(n_splits=5, random_state=42, output_dir="results")

    for ds_name in ["Breast Cancer (Binary)", "Wine (Multi-Class)"]:
        X_data, y_data, is_bin = datasets[ds_name]
        comparison.run_cross_validation(ds_name, X_data, y_data, is_binary=is_bin)

    summary_df = comparison.print_summary()
    comparison.export_csv()
    visualizer.plot_model_comparison(summary_df)

    print("\n================================================================")
    print(" DAY 2 CLASSIFICATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("================================================================")


if __name__ == "__main__":
    main()
