import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class RegressionModels:
    """
    Implements and evaluates standard regression algorithms:
    - Linear Regression
    - Ridge Regression (L2)
    - Lasso Regression (L1)
    - ElasticNet Regression (L1 + L2)
    - Support Vector Regression (SVR)
    - Decision Tree Regressor
    - Random Forest Regressor
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = self._initialize_models()
        self.fitted_models = {}

    def _initialize_models(self):
        return {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0, random_state=self.random_state),
            "Lasso Regression": Lasso(alpha=0.1, random_state=self.random_state),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state),
            "Support Vector Regression": SVR(C=1.0, epsilon=0.2),
            "Decision Tree": DecisionTreeRegressor(random_state=self.random_state),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        }

    def train_all(self, X_train, y_train):
        """Trains all initialized regression models."""
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            self.fitted_models[name] = model
        return self.fitted_models

    def evaluate_model(self, model, X_test, y_test):
        """Calculates standard regression metrics for a given model."""
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "predictions": predictions
        }

    def evaluate_all(self, X_test, y_test):
        """Evaluates all fitted models and returns a dictionary of metrics."""
        results = {}
        for name, model in self.fitted_models.items():
            results[name] = self.evaluate_model(model, X_test, y_test)
        return results
