"""
Random Forest Baseline Model
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mean_absolute_percentage_error(y_true, y_pred):
    """Weighted MAPE hesapla — kucuk degerlerin inflation'ini onler."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    # Cok kucuk degerleri filtrele (< 0.5 TWh) — adacik ulkelerin MAPE'yi sismesini onler
    mask = y_true > 0.5
    if mask.sum() == 0:
        return 0.0
    # Weighted MAPE: buyuk ulkeler daha fazla agirlik alir
    abs_errors = np.abs(y_true[mask] - y_pred[mask])
    return (np.sum(abs_errors) / np.sum(np.abs(y_true[mask]))) * 100


def evaluate(y_true, y_pred, label=""):
    """Model performans metriklerini hesapla."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    if label:
        print(f"  {label}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, R2={r2:.4f}")
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}


def train_random_forest(data):
    """Random Forest egit ve degerlendir."""
    print("\n=== RANDOM FOREST (Baseline) ===")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=0.7,
        random_state=42,
        n_jobs=-1
    )

    model.fit(data['X_train'], data['y_train'])

    # Tahminler
    pred_train = model.predict(data['X_train'])
    pred_val = model.predict(data['X_val'])
    pred_test = model.predict(data['X_test'])

    # Degerlendirme
    metrics_train = evaluate(data['y_train'], pred_train, "Train")
    metrics_val = evaluate(data['y_val'], pred_val, "Val")
    metrics_test = evaluate(data['y_test'], pred_test, "Test")

    # Feature importance
    importances = dict(zip(data['feature_cols'],
                           model.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print("\n  Feature Importance (Top 5):")
    for feat, imp in sorted_imp[:5]:
        print(f"    {feat}: {imp:.4f}")

    return {
        'model': model,
        'name': 'Random Forest',
        'metrics_train': metrics_train,
        'metrics_val': metrics_val,
        'metrics_test': metrics_test,
        'pred_train': pred_train,
        'pred_val': pred_val,
        'pred_test': pred_test,
        'feature_importance': importances,
    }


if __name__ == '__main__':
    import sys; sys.path.insert(0, '.')
    from data_merge import load_and_merge
    from feature_engineering import prepare_data

    df = load_and_merge()
    data = prepare_data(df)
    result = train_random_forest(data)
