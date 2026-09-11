import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


class MLPipeline:
    """
    A reusable, production-ready Machine Learning Pipeline class.
    
    Capabilities:
    - Automated detection and preprocessing of mixed data types (Numeric & Categorical)
    - Missing value imputation and configurable feature scaling
    - Seamless integration with Scikit-learn Pipeline and ColumnTransformer
    - Supports both Classification and Regression tasks
    - Automated hyperparameter optimization via GridSearchCV
    - Model persistence (save / load via joblib)
    - Feature importance inspection
    """

    def __init__(
        self,
        task="classification",
        estimator=None,
        numeric_features=None,
        categorical_features=None,
        scaler_type="standard",
        random_state=42
    ):
        self.task = task.lower()
        self.estimator = estimator
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.scaler_type = scaler_type
        self.random_state = random_state

        if self.estimator is None:
            if self.task == "classification":
                self.estimator = RandomForestClassifier(random_state=self.random_state)
            else:
                self.estimator = RandomForestRegressor(random_state=self.random_state)

        self.preprocessor = None
        self.pipeline = None
        self.is_fitted = False

    def _get_scaler(self):
        if self.scaler_type == "minmax":
            return MinMaxScaler()
        elif self.scaler_type == "robust":
            return RobustScaler()
        return StandardScaler()

    def _auto_detect_columns(self, X):
        """Automatically identifies numeric and categorical columns if not explicitly provided."""
        if self.numeric_features is None:
            self.numeric_features = list(X.select_dtypes(include=[np.number]).columns)
        if self.categorical_features is None:
            self.categorical_features = list(X.select_dtypes(include=["object", "category", "bool"]).columns)

    def build_pipeline(self, X):
        """Constructs the ColumnTransformer and full Scikit-learn Pipeline."""
        self._auto_detect_columns(X)

        # 1. Numeric pipeline: Median imputation + Scaling
        num_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", self._get_scaler())
        ])

        # 2. Categorical pipeline: Constant imputation + One-Hot Encoding
        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        transformers = []
        if self.numeric_features:
            transformers.append(("num", num_transformer, self.numeric_features))
        if self.categorical_features:
            transformers.append(("cat", cat_transformer, self.categorical_features))

        self.preprocessor = ColumnTransformer(transformers=transformers)

        self.pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("model", self.estimator)
        ])
        return self.pipeline

    def fit(self, X, y):
        """Builds and fits the entire end-to-end pipeline on raw input data."""
        if self.pipeline is None:
            self.build_pipeline(X)
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Generates predictions for new unseen raw data."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling predict.")
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """Generates class probability estimates (classification only)."""
        if self.task != "classification":
            raise NotImplementedError("predict_proba is only available for classification tasks.")
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling predict_proba.")
        return self.pipeline.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """Evaluates pipeline performance and returns standard task metrics."""
        y_pred = self.predict(X_test)

        if self.task == "classification":
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            auc = None
            if hasattr(self.pipeline.named_steps["model"], "predict_proba"):
                try:
                    y_proba = self.predict_proba(X_test)
                    if y_proba.shape[1] == 2:
                        auc = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
                except Exception:
                    auc = None

            return {
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4),
                "ROC-AUC": round(auc, 4) if auc is not None else "N/A"
            }
        else:
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            return {
                "R2": round(r2, 4),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4)
            }

    def tune_hyperparameters(self, param_grid, X, y, cv=5, scoring=None):
        """
        Runs GridSearchCV on pipeline parameters (e.g. 'model__n_estimators', 'model__max_depth').
        """
        if self.pipeline is None:
            self.build_pipeline(X)

        if scoring is None:
            scoring = "f1_weighted" if self.task == "classification" else "r2"

        grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        grid_search.fit(X, y)
        self.pipeline = grid_search.best_estimator_
        self.is_fitted = True

        return {
            "best_params": grid_search.best_params_,
            "best_score": round(grid_search.best_score_, 4),
            "cv_results": grid_search.cv_results_
        }

    def get_feature_importances(self):
        """Extracts transformed feature names along with model feature importances."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted first.")

        model = self.pipeline.named_steps["model"]
        if not hasattr(model, "feature_importances_"):
            return None

        col_transformer = self.pipeline.named_steps["preprocessor"]
        feature_names = col_transformer.get_feature_names_out()

        df_importance = pd.DataFrame({
            "Transformed_Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        return df_importance

    def save(self, filepath):
        """Serializes and saves the trained pipeline artifact."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.pipeline, filepath)
        print(f"Pipeline artifact saved to: {filepath}")

    @classmethod
    def load(cls, filepath, task="classification"):
        """Loads a persisted pipeline artifact."""
        loaded_obj = cls(task=task)
        loaded_obj.pipeline = joblib.load(filepath)
        loaded_obj.is_fitted = True
        return loaded_obj
