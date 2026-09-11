from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


class HyperparameterTuner:
    """
    Performs hyperparameter tuning for various regression algorithms using GridSearchCV.
    """

    def __init__(self, cv=5, scoring="neg_mean_squared_error", random_state=42):
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.tuning_results = {}

    def get_param_grids(self):
        """Returns search spaces for each tunable model."""
        return {
            "Ridge Regression": {
                "model": Ridge(random_state=self.random_state),
                "params": {
                    "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "solver": ["auto", "svd", "cholesky"]
                }
            },
            "Lasso Regression": {
                "model": Lasso(random_state=self.random_state),
                "params": {
                    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                    "max_iter": [1000, 2000]
                }
            },
            "ElasticNet": {
                "model": ElasticNet(random_state=self.random_state),
                "params": {
                    "alpha": [0.01, 0.1, 1.0],
                    "l1_ratio": [0.2, 0.5, 0.8]
                }
            },
            "Support Vector Regression": {
                "model": SVR(),
                "params": {
                    "C": [0.1, 1.0, 10.0],
                    "epsilon": [0.01, 0.1, 0.2],
                    "kernel": ["rbf", "linear"]
                }
            },
            "Decision Tree": {
                "model": DecisionTreeRegressor(random_state=self.random_state),
                "params": {
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            },
            "Random Forest": {
                "model": RandomForestRegressor(random_state=self.random_state),
                "params": {
                    "n_estimators": [50, 100],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5]
                }
            }
        }

    def tune_model(self, model_name, model, param_grid, X_train, y_train):
        """Runs GridSearchCV for a specific model."""
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)

        result = {
            "best_model": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "best_cv_score": -grid_search.best_score_
        }
        self.tuning_results[model_name] = result
        return result

    def tune_all(self, X_train, y_train):
        """Runs GridSearchCV for all configured models."""
        param_grids = self.get_param_grids()
        for name, config in param_grids.items():
            print(f"  Tuning {name}...")
            self.tune_model(name, config["model"], config["params"], X_train, y_train)
        return self.tuning_results
