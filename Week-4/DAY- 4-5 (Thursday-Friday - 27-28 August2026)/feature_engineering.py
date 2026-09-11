import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder
)
from category_encoders import TargetEncoder


class FeatureEngineer:
    """
    Modular suite for feature scaling, normalization, and categorical encoding.
    """

    @staticmethod
    def compare_scalers(df_numeric):
        """
        Applies and compares StandardScaler, MinMaxScaler, RobustScaler, and Normalizer
        on a numeric DataFrame.
        """
        scalers = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
            "Normalizer": Normalizer()
        }

        transformed_dfs = {}
        stats_summary = []

        for name, scaler in scalers.items():
            scaled_array = scaler.fit_transform(df_numeric)
            scaled_df = pd.DataFrame(scaled_array, columns=df_numeric.columns)
            transformed_dfs[name] = scaled_df

            # Record min, max, mean, std across all numeric values
            all_vals = scaled_array.flatten()
            stats_summary.append({
                "Scaler": name,
                "Min": round(float(np.min(all_vals)), 4),
                "Max": round(float(np.max(all_vals)), 4),
                "Mean": round(float(np.mean(all_vals)), 4),
                "Std": round(float(np.std(all_vals)), 4)
            })

        return transformed_dfs, pd.DataFrame(stats_summary)

    @staticmethod
    def encode_categorical(df_cat, target_series=None):
        """
        Demonstrates One-Hot Encoding, Ordinal Encoding, and Target Encoding.
        """
        # 1. One-Hot Encoding
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        ohe_array = ohe.fit_transform(df_cat)
        ohe_cols = ohe.get_feature_names_out(df_cat.columns)
        df_ohe = pd.DataFrame(ohe_array, columns=ohe_cols)

        # 2. Ordinal Encoding
        ord_enc = OrdinalEncoder()
        df_ord = pd.DataFrame(ord_enc.fit_transform(df_cat), columns=df_cat.columns)

        # 3. Target Encoding (if target is supplied)
        df_target = None
        if target_series is not None:
            te = TargetEncoder(cols=list(df_cat.columns))
            df_target = te.fit_transform(df_cat, target_series)

        return {
            "OneHot": df_ohe,
            "Ordinal": df_ord,
            "TargetEncoded": df_target
        }
