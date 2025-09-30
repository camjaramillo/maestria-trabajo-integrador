"""Centralized models and hyperparameter grids.

This module exposes two simple dictionaries for experiments:
- MODELS: mapping name -> estimator instance (or None if unavailable)
- PARAM_GRIDS: mapping name -> hyperparameter grid (useful for GridSearch/Optuna wrappers)

Notes:
- Estimators from optional libraries (xgboost, lightgbm, catboost) are imported under try/except
  and will be set to None if the package is not installed. Notebooks should handle None values.
- PARAM_GRIDS keys use estimator parameter names (no pipeline prefix). When using inside a
  Pipeline named 'model', prefix parameter keys with 'model__'.
"""
from typing import Dict, Any
import warnings

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

_have_xgb = True
_have_lgb = True
_have_cat = True
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
    _have_xgb = False

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None
    _have_lgb = False

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None
    _have_cat = False

# Helper to warn when a model is unavailable
def _maybe_warn(name: str, est):
    if est is None:
        warnings.warn(f"Estimator '{name}' is not available (missing package). Set to None.")
    return est

# Single dict with all models to test (instances). Set to None if not available.
MODELS: Dict[str, Any] = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(random_state=42),
    "Lasso": Lasso(random_state=42),
    "ElasticNet": ElasticNet(max_iter=10000),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "XGBoost": _maybe_warn("XGBoost", XGBRegressor(random_state=42, n_jobs=-1) if XGBRegressor is not None else None),
    "LightGBM": _maybe_warn("LightGBM", LGBMRegressor(random_state=42, n_jobs=-1) if LGBMRegressor is not None else None),
    "CatBoost": _maybe_warn("CatBoost", CatBoostRegressor(random_state=42, verbose=0) if CatBoostRegressor is not None else None),
    "Extra Trees": ExtraTreesRegressor(n_jobs=-1, random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    "MLP Regressor": MLPRegressor(random_state=42, max_iter=500)
}

# Single dict with hyperparameter grids. These are *suggested* grids adapted for
# hourly spot-price forecasting (many observations, likely temporal patterns).
# Keys are estimator parameter names (no pipeline prefix). If you use a Pipeline
# with a step named 'model', prefix keys with 'model__'.
PARAM_GRIDS: Dict[str, Dict[str, Any]] = {
    # Linear / regularized
    "ElasticNet": {
        "alpha": [0.001, 0.01, 0.1, 1, 10],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
    },
    # Tree ensembles
    "Random Forest": {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10]
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7]
    },
    # Boosting libraries
    "XGBoost": {
        "n_estimators": [200, 500, 1000],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 6, 9],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    },
    "LightGBM": {
        "n_estimators": [200, 500, 1000],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [31, 61, 127],
        "min_child_samples": [10, 20, 50]
    },
    "CatBoost": {
        "iterations": [200, 500, 1000],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [4, 6, 8]
    },
    # Robust models for time series
    "Extra Trees": {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    },
    "HistGradientBoosting": {
        "max_iter": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 7, 15]
    },
    "MLP Regressor": {
        "hidden_layer_sizes": [(64,), (128, 64), (128, 64, 32)],
        "alpha": [1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-3, 1e-4]
    },
    # Small grids for classical models
    "Decision Tree": {"max_depth": [5, 10, 20, None], "min_samples_split": [2, 5, 10]},
    "SVR": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    "KNN": {"n_neighbors": [3, 5, 7]}
}

__all__ = ["MODELS", "PARAM_GRIDS"]