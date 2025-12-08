"""
Utility functions for regression model evaluation.

Provides:
- regression_metrics: MAE, RMSE, MAPE
- print_val_test_metrics: pretty printing for val/test splits
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true, y_pred):
    """
    Compute MAE, RMSE, and MAPE (%) between true and predicted values.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.

    Returns
    -------
    mae : float
        Mean absolute error.
    rmse : float
        Root mean squared error.
    mape : float
        Mean absolute percentage error (in percent).
    """
    # Convert to numpy arrays for safety
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)

    # Older sklearn versions do not support squared=False, so compute RMSE manually
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Avoid division by zero in MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return mae, rmse, mape



def print_val_test_metrics(label, y_true_val, y_pred_val, y_true_test, y_pred_test):
    """
    Print and return metrics for validation and test sets.

    Parameters
    ----------
    label : str
        Name of the model/baseline.
    y_true_val, y_pred_val : array-like
        True and predicted values on the validation set.
    y_true_test, y_pred_test : array-like
        True and predicted values on the test set.

    Returns
    -------
    metrics_dict : dict
        Dictionary with MAE, RMSE, MAPE for val and test.
    """
    mae_v, rmse_v, mape_v = regression_metrics(y_true_val, y_pred_val)
    mae_t, rmse_t, mape_t = regression_metrics(y_true_test, y_pred_test)

    print(f"{label}")
    print("  Validation:")
    print(f"    MAE : {mae_v:,.0f} USD")
    print(f"    RMSE: {rmse_v:,.0f} USD")
    print(f"    MAPE: {mape_v:.2f}%")
    print("  Test:")
    print(f"    MAE : {mae_t:,.0f} USD")
    print(f"    RMSE: {rmse_t:,.0f} USD")
    print(f"    MAPE: {mape_t:.2f}%")

    return {
        "model": label,
        "val_MAE_USD": mae_v,
        "val_RMSE_USD": rmse_v,
        "val_MAPE_pct": mape_v,
        "test_MAE_USD": mae_t,
        "test_RMSE_USD": rmse_t,
        "test_MAPE_pct": mape_t,
    }
    
