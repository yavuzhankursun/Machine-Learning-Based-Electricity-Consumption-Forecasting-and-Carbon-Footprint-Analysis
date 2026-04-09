"""Paylasilan metrik fonksiyonlari."""
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def mape(y_true, y_pred):
    """Weighted MAPE - kucuk degerler (< 10 kWh) haric."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true > 10
    if mask.sum() == 0:
        return 0.0
    return (np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]).mean() * 100


def compute_metrics(y_true, y_pred, label=''):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2   = r2_score(y_true, y_pred)
    m    = mape(y_true, y_pred)
    if label:
        print(f"  {label:6s}: MAE={mae:8.1f}  RMSE={rmse:8.1f}  MAPE={m:6.2f}%  R2={r2:.4f}")
    return {'mae': mae, 'rmse': rmse, 'mape': m, 'r2': r2}
