import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes, make_friedman1
from model_comparison import ModelComparison
from hyperparameter_tuning import HyperparameterTuner
from visualization import RegressionVisualizer


def load_datasets():
    """Loads 3 standard regression datasets."""
    datasets = {}

    # Dataset 1: California Housing (sample 2000 rows for swift execution)
    california = fetch_california_housing(as_frame=True)
    sample_df = california.frame.sample(n=2000, random_state=42)
    X_cal = sample_df.drop(columns=[california.target_names[0]])
    y_cal = sample_df[california.target_names[0]]
    datasets["California Housing"] = (X_cal, y_cal)

    # Dataset 2: Diabetes dataset
    diabetes = load_diabetes(as_frame=True)
    datasets["Diabetes"] = (diabetes.data, diabetes.target)

    # Dataset 3: Synthetic Friedman-1 non-linear dataset
    X_friedman, y_friedman = make_friedman1(n_samples=1000, n_features=10, noise=1.0, random_state=42)
    X_friedman_df = pd.DataFrame(X_friedman, columns=[f"feat_{i}" for i in range(10)])
    y_friedman_series = pd.Series(y_friedman, name="target")
    datasets["Friedman-1 (Synthetic)"] = (X_friedman_df, y_friedman_series)

    return datasets


def main():
    print("================================================================")
    print(" WEEK 4 - DAY 1: SUPERVISED LEARNING (REGRESSION) PIPELINE")
    print("================================================================")

    # 1. Load Datasets
    print("\n[Step 1] Loading 3 Datasets for Model Comparison...")
    datasets = load_datasets()
    for name, (X, y) in datasets.items():
        print(f"  - {name}: {X.shape[0]} samples, {X.shape[1]} features")

    # 2. Run Model Comparison across all datasets
    print("\n[Step 2] Training & Evaluating 7 Regression Models...")
    comparison = ModelComparison(random_state=42, output_dir="results")
    visualizer = RegressionVisualizer(output_dir="results")

    dataset_eval_runs = {}
    for name, (X, y) in datasets.items():
        run_data = comparison.compare_on_dataset(name, X, y)
        dataset_eval_runs[name] = run_data
        # Generate plots for each dataset
        visualizer.plot_predictions_vs_actual(run_data["y_test"], run_data["eval_results"], dataset_name=name)
        visualizer.plot_residuals(run_data["y_test"], run_data["eval_results"], dataset_name=name)

    # 3. Print & Export Summary
    print("\n[Step 3] Generating Benchmark Comparison Report...")
    summary_df = comparison.print_summary()
    comparison.export_csv()
    visualizer.plot_model_comparison(summary_df, metric="R2")
    visualizer.plot_model_comparison(summary_df, metric="RMSE")

    # 4. Hyperparameter Tuning using GridSearchCV
    print("\n[Step 4] Running Hyperparameter Tuning via GridSearchCV on Diabetes...")
    tuner = HyperparameterTuner(cv=5, random_state=42)
    # Using Diabetes dataset training split from step 2
    diabetes_train_X = dataset_eval_runs["Diabetes"]["X_train"]
    diabetes_train_y = dataset_eval_runs["Diabetes"]["y_train"]
    tuning_results = tuner.tune_all(diabetes_train_X, diabetes_train_y)

    print("\n" + "="*50)
    print("       GRIDSEARCHCV BEST PARAMETERS & SCORES     ")
    print("="*50)
    for model_name, info in tuning_results.items():
        print(f"\nModel: {model_name}")
        print(f"  Best CV MSE Score: {info['best_cv_score']:.4f}")
        print(f"  Best Parameters  : {info['best_params']}")

    print("\n================================================================")
    print(" DAY 1 REGRESSION PIPELINE COMPLETED SUCCESSFULLY!")
    print("================================================================")


if __name__ == "__main__":
    main()
