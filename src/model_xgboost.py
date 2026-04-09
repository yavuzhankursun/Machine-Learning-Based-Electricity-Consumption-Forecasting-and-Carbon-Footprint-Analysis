"""
XGBoost Ana Model
"""
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from model_rf import evaluate


def tune_xgboost(data, n_iter=20):
    """Hiperparametre optimizasyonu (RandomizedSearchCV + TimeSeriesSplit)."""
    print("\n  Hiperparametre aramasi (RandomizedSearchCV)...")

    param_distributions = {
        'n_estimators': [200, 300, 500],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'min_child_weight': [5, 10, 15],
        'reg_alpha': [0.1, 0.5, 1.0],
        'reg_lambda': [1.0, 3.0, 5.0],
    }

    tscv = TimeSeriesSplit(n_splits=4)

    base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        base_model, param_distributions,
        n_iter=n_iter, cv=tscv,
        scoring='neg_mean_absolute_error',
        random_state=42, n_jobs=-1, verbose=0
    )

    search.fit(data['X_train'], data['y_train'])

    print(f"  En iyi parametreler: {search.best_params_}")
    print(f"  En iyi CV MAE: {-search.best_score_:.2f}")

    return search.best_params_


def train_xgboost(data, best_params=None):
    """XGBoost egit ve degerlendir."""
    print("\n=== XGBOOST (Ana Model) ===")

    # Hiperparametre optimizasyonu (skip edilebilir, ablation icin)
    if best_params is None:
        best_params = tune_xgboost(data, n_iter=20)

    model = xgb.XGBRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )

    # Egitim (early stopping ile)
    model.fit(
        data['X_train'], data['y_train'],
        eval_set=[(data['X_val'], data['y_val'])],
        verbose=False
    )

    print(f"  Best iteration: {model.best_iteration}")

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
        'name': 'XGBoost',
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
    result = train_xgboost(data)
