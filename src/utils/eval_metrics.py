"""Central evaluation metrics and helper functions for notebooks.
Place this file under `src/utils` and import from notebooks using:

import sys
from pathlib import Path
sys.path.append(str(Path('../../src').resolve()))
from utils.eval_metrics import SCORING_FUNCS, evaluate_model

This module provides: R2, MAE, MSE, RMSE, MAPE(%) and evaluate helpers.
"""
from typing import Callable, Dict, Any
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error

def mape(y_true, y_pred, eps: float = 1e-8):
    """Mean Absolute Percentage Error expressed in percent.

    Protected against zeros in the denominator by using `eps`.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0


# Dictionary of scoring functions
SCORING_FUNCS: Dict[str, Callable[[Any, Any], float]] = {
    "R2": r2_score,
    "MAE": mean_absolute_error,
    "MSE": mean_squared_error,
    "RMSE": root_mean_squared_error,
    "MAPE(%)": mape,
}


def evaluate_predictions(y_true, y_pred, scoring_funcs: Dict[str, Callable] = SCORING_FUNCS) -> Dict[str, float]:
    """Evaluate predictions with the provided scoring functions.

    Returns a dict of metric_name -> value. Any metric that fails will return np.nan.
    """
    results = {}
    for name, func in scoring_funcs.items():
        try:
            results[name] = float(func(y_true, y_pred))
        except Exception:
            results[name] = np.nan
    return results


def evaluate_model(model, X_test, y_test, scoring_funcs: Dict[str, Callable] = SCORING_FUNCS) -> Dict[str, float]:
    """Evaluate a fitted model implementing .predict(X_test).

    If the model does not implement `predict` or raises, the returned metrics will be NaN.
    """
    try:
        y_pred = model.predict(X_test)
    except Exception:
        return {k: np.nan for k in scoring_funcs.keys()}

    return evaluate_predictions(y_test, y_pred, scoring_funcs)


__all__ = [
    "mape",
    "SCORING_FUNCS",
    "evaluate_predictions",
    "evaluate_model",
]
