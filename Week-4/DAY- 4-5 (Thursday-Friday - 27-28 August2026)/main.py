import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from feature_engineering import FeatureEngineer
from feature_selection import FeatureSelector
from imbalanced_data import ImbalancedDataHandler
from ml_pipeline import MLPipeline


def create_realistic_mixed_dataset(n_samples=1000, random_state=42):
    """
    Creates a realistic dataset with mixed types (numeric, categorical)
    and intentional missing values to thoroughly exercise production pipelines.
    """
    rng = np.random.RandomState(random_state)

    ages = rng.normal(38, 12, size=n_samples).clip(18, 75)
    salaries = rng.exponential(scale=45000, size=n_samples) + 20000
    credit_scores = rng.normal(650, 80, size=n_samples).clip(300, 850)
    years_employed = rng.poisson(lam=6, size=n_samples)

    education_levels = rng.choice(["High School", "Bachelors", "Masters", "PhD"], size=n_samples, p=[0.3, 0.4, 0.2, 0.1])
    departments = rng.choice(["Sales", "Engineering", "Marketing", "HR", "Finance"], size=n_samples)
    home_ownership = rng.choice(["RENT", "OWN", "MORTGAGE"], size=n_samples)

    # Introduce intentional missing values (real-world simulation)
    ages[rng.choice(n_samples, size=30, replace=False)] = np.nan
    salaries[rng.choice(n_samples, size=25, replace=False)] = np.nan
    home_ownership[rng.choice(n_samples, size=20, replace=False)] = None

    # Target variable: loan approval / promotion probability
    prob = (
        (salaries > 50000).astype(int) * 0.35 +
        (credit_scores > 620).astype(int) * 0.35 +
        (education_levels == "Masters").astype(int) * 0.15 +
        (years_employed > 4).astype(int) * 0.15
    )
    prob = np.nan_to_num(prob, nan=0.2).clip(0.05, 0.95)
    target = rng.binomial(1, prob)

    df = pd.DataFrame({
        "Age": ages,
        "Salary": salaries,
        "CreditScore": credit_scores,
        "YearsEmployed": years_employed,
        "Education": education_levels,
        "Department": departments,
        "HomeOwnership": home_ownership,
        "Target": target
    })
    return df


def main():
    print("========================================================================")
    print(" WEEK 4 - DAY 4-5: FEATURE ENGINEERING & PRODUCTION ML PIPELINE")
    print("========================================================================")

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # [Step 1] Feature Scaling Comparison
    print("\n[Step 1] Evaluating Feature Scalers (Standard, MinMax, Robust, Normalizer)...")
    numeric_demo = pd.DataFrame({
        "Income": [20000, 45000, 52000, 80000, 1000000],  # Severe outlier
        "Score": [10, 45, 55, 78, 99],
        "Ratio": [0.1, 0.4, 0.5, 0.7, 0.95]
    })
    _, scaler_summary = FeatureEngineer.compare_scalers(numeric_demo)
    print("\nScaler Distribution Summary:")
    print(scaler_summary.to_string(index=False))

    # [Step 2] Categorical Encoding Methods
    print("\n[Step 2] Comparing Categorical Encoders (One-Hot, Ordinal, Target)...")
    cat_demo = pd.DataFrame({
        "City": ["Karachi", "Lahore", "Islamabad", "Karachi", "Lahore"],
        "Priority": ["Low", "Medium", "High", "Low", "High"]
    })
    target_demo = pd.Series([0, 1, 1, 0, 1])
    encoded_dict = FeatureEngineer.encode_categorical(cat_demo, target_demo)
    print("\nOne-Hot Encoded Sample Columns:", list(encoded_dict["OneHot"].columns))
    print("Target Encoded Output:\n", encoded_dict["TargetEncoded"])

    # [Step 3] Feature Selection Paradigms
    print("\n[Step 3] Benchmarking Feature Selection (Filter vs Wrapper vs Embedded)...")
    X_synth = pd.DataFrame(np.random.RandomState(42).randn(200, 8), columns=[f"feat_{i}" for i in range(8)])
    y_synth = (X_synth["feat_0"] * 2 + X_synth["feat_1"] * -1.5 + np.random.randn(200) > 0).astype(int)

    selector = FeatureSelector(random_state=42)
    filter_features, filter_df = selector.filter_selection(X_synth, y_synth, k=3)
    wrapper_features, wrapper_df = selector.wrapper_rfe(X_synth, y_synth, n_features_to_select=3)
    embedded_features, embedded_df = selector.embedded_selection(X_synth, y_synth, max_features=3)

    print(f"  - Filter Method (SelectKBest) Top Features : {filter_features}")
    print(f"  - Wrapper Method (RFE) Top Features        : {wrapper_features}")
    print(f"  - Embedded Method (Random Forest) Features : {embedded_features}")

    # [Step 4] Handling Imbalanced Data Strategies
    print("\n[Step 4] Comparing Imbalance Mitigation Strategies...")
    imb_handler = ImbalancedDataHandler(random_state=42)
    X_imb, y_imb = imb_handler.generate_imbalanced_data(n_samples=1000, weights=(0.92, 0.08))
    X_tr_i, X_te_i, y_tr_i, y_te_i = train_test_split(X_imb, y_imb, test_size=0.3, random_state=42, stratify=y_imb)

    imbalance_df = imb_handler.compare_strategies(X_tr_i, X_te_i, y_tr_i, y_te_i)
    print("\nMinority Class Performance Comparison across Strategies:")
    print(imbalance_df.to_string(index=False))

    # [Step 5] Building & Testing the Production-Ready MLPipeline Class
    print("\n[Step 5] Constructing & Executing the Reusable MLPipeline Class...")
    df_raw = create_realistic_mixed_dataset(n_samples=1200, random_state=42)
    print(f"  Raw Dataset Created: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    print("  Missing values per column:\n", df_raw.isnull().sum()[df_raw.isnull().sum() > 0])

    X_raw = df_raw.drop(columns=["Target"])
    y_raw = df_raw["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.25, random_state=42, stratify=y_raw)

    # Instantiate reusable pipeline
    ml_pipe = MLPipeline(
        task="classification",
        estimator=RandomForestClassifier(random_state=42),
        scaler_type="standard"
    )

    # Fit pipeline
    print("\n  Fitting End-to-End Pipeline on Raw Data...")
    ml_pipe.fit(X_train, y_train)

    # Initial Evaluation
    initial_metrics = ml_pipe.evaluate(X_test, y_test)
    print("  Initial Test Metrics:")
    for k, v in initial_metrics.items():
        print(f"    - {k}: {v}")

    # Hyperparameter Tuning
    print("\n  Running GridSearchCV Hyperparameter Tuning on Pipeline...")
    param_grid = {
        "model__n_estimators": [50, 100],
        "model__max_depth": [5, 10, None]
    }
    tune_res = ml_pipe.tune_hyperparameters(param_grid, X_train, y_train, cv=3)
    print(f"  Best Parameters: {tune_res['best_params']}")
    print(f"  Best CV F1 Score: {tune_res['best_score']}")

    # Post-tuning Evaluation
    tuned_metrics = ml_pipe.evaluate(X_test, y_test)
    print("\n  Post-Tuning Test Metrics:")
    for k, v in tuned_metrics.items():
        print(f"    - {k}: {v}")

    # Feature Importance Inspection
    fi_df = ml_pipe.get_feature_importances()
    print("\nTop 5 Transformed Feature Importances:")
    print(fi_df.head(5).to_string(index=False))

    # [Step 6] Saving and Reloading the Pipeline Artifact
    save_path = os.path.join(results_dir, "production_ml_pipeline.joblib")
    ml_pipe.save(save_path)

    print("\n  Testing Pipeline Reloading & Inference...")
    loaded_pipe = MLPipeline.load(save_path, task="classification")
    sample_preds = loaded_pipe.predict(X_test.head(3))
    print("  Predictions from reloaded model for first 3 test samples:", sample_preds)

    # Export metrics summary
    summary_report = pd.DataFrame([
        {"Stage": "Initial Pipeline", **initial_metrics},
        {"Stage": "Tuned Pipeline", **tuned_metrics}
    ])
    summary_csv = os.path.join(results_dir, "feature_engineering_summary.csv")
    summary_report.to_csv(summary_csv, index=False)
    print(f"\nSaved pipeline evaluation summary to: {summary_csv}")

    print("\n========================================================================")
    print(" DAY 4-5 FEATURE ENGINEERING & PIPELINE BUILDING COMPLETED SUCCESSFULLY!")
    print("========================================================================")


if __name__ == "__main__":
    main()
