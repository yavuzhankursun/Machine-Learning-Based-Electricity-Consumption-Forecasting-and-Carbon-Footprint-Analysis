"""Random Forest - Baseline Model"""
from sklearn.ensemble import RandomForestRegressor
from metrics import compute_metrics


def train_random_forest(data):
    print("\n=== RANDOM FOREST (Baseline) ===")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features=0.7,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(data['X_train'], data['y_train'])

    pred_train = model.predict(data['X_train'])
    pred_val   = model.predict(data['X_val'])
    pred_test  = model.predict(data['X_test'])

    m_train = compute_metrics(data['y_train'], pred_train, 'Train')
    m_val   = compute_metrics(data['y_val'],   pred_val,   'Val  ')
    m_test  = compute_metrics(data['y_test'],  pred_test,  'Test ')

    importances = dict(zip(data['feature_cols'], model.feature_importances_))
    top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print("  Feature Importance (Top 5):")
    for k, v in top5:
        print(f"    {k}: {v:.4f}")

    return {
        'name'         : 'Random Forest',
        'model'        : model,
        'pred_train'   : pred_train,
        'pred_val'     : pred_val,
        'pred_test'    : pred_test,
        'metrics_train': m_train,
        'metrics_val'  : m_val,
        'metrics_test' : m_test,
        'feature_importance': importances,
    }
