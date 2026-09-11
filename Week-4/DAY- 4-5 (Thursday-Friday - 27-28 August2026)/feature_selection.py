import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE,
    SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class FeatureSelector:
    """
    Implements the three primary paradigms of feature selection:
    1. Filter Methods (SelectKBest with ANOVA F-test and Mutual Information)
    2. Wrapper Methods (Recursive Feature Elimination - RFE)
    3. Embedded Methods (L1-based and Tree-based feature importance)
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def filter_selection(self, X, y, k=5, method="f_classif"):
        """
        Applies filter-based feature selection using SelectKBest.
        """
        score_func = f_classif if method == "f_classif" else mutual_info_classif
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)

        selected_mask = selector.get_support()
        selected_cols = [col for col, mask in zip(X.columns, selected_mask) if mask]
        scores = selector.scores_

        score_df = pd.DataFrame({
            "Feature": X.columns,
            "Score": scores,
            "Selected": selected_mask
        }).sort_values(by="Score", ascending=False)

        return selected_cols, score_df

    def wrapper_rfe(self, X, y, n_features_to_select=5):
        """
        Applies wrapper-based Recursive Feature Elimination (RFE) using LogisticRegression.
        """
        estimator = LogisticRegression(max_iter=1000, random_state=self.random_state)
        rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)
        rfe.fit(X, y)

        selected_cols = [col for col, mask in zip(X.columns, rfe.support_) if mask]
        ranking_df = pd.DataFrame({
            "Feature": X.columns,
            "Ranking": rfe.ranking_,
            "Selected": rfe.support_
        }).sort_values(by="Ranking")

        return selected_cols, ranking_df

    def embedded_selection(self, X, y, max_features=5):
        """
        Applies embedded feature selection using RandomForest feature importances.
        """
        forest = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        selector = SelectFromModel(estimator=forest, max_features=max_features, prefit=False)
        selector.fit(X, y)

        selected_cols = [col for col, mask in zip(X.columns, selector.get_support()) if mask]
        importances = selector.estimator_.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances,
            "Selected": selector.get_support()
        }).sort_values(by="Importance", ascending=False)

        return selected_cols, importance_df
