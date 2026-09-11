import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class ClassificationModels:
    """
    Implements and encapsulates key traditional and ensemble classification algorithms:
    - Logistic Regression
    - Support Vector Classifier (SVC)
    - Gaussian Naive Bayes
    - Decision Tree Classifier
    - Random Forest Classifier
    - Gradient Boosting Classifier
    - XGBoost Classifier
    - LightGBM Classifier
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = self._initialize_models()
        self.fitted_models = {}

    def _initialize_models(self):
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "Support Vector Machine": SVC(probability=True, random_state=self.random_state),
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            "Gradient Boosting": GradientBoostingClassifier(random_state=self.random_state),
            "XGBoost": XGBClassifier(eval_metric="logloss", random_state=self.random_state),
            "LightGBM": LGBMClassifier(verbose=-1, random_state=self.random_state)
        }

    def train_all(self, X_train, y_train):
        """Trains all initialized classifiers on the provided training set."""
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            self.fitted_models[name] = model
        return self.fitted_models

    def predict_all(self, X_test):
        """Generates class predictions and probability estimates for all models."""
        predictions = {}
        for name, model in self.fitted_models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            predictions[name] = {
                "y_pred": y_pred,
                "y_proba": y_proba
            }
        return predictions
