"""
XGBoost + Optuna Hiperparametre Optimizasyonu

Optuna n_trials=30:
  Her trial: bir hiperparametre kombinasyonu dener ve val MAPE hesaplar.
  Optuna, Tree-structured Parzen Estimator (TPE) kullanarak
  onceki denemelerin sonuclarindan ogrenip daha umit verici bolgeleri arar.
  30 trial: random search'ten ~3-5x daha verimli, ~2 dk surede tamamlanir.
"""
import optuna
import xgboost as xgb
from metrics import compute_metrics, mape

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective(trial, data):
    params = {
        'max_depth'        : trial.suggest_int('max_depth', 3, 8),
        'n_estimators'     : trial.suggest_int('n_estimators', 100, 600),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'subsample'        : trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight' : trial.suggest_int('min_child_weight', 3, 20),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 0.0, 3.0),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 0.5, 5.0),
        'random_state': 42, 'n_jobs': -1,
    }
    m = xgb.XGBRegressor(**params, eval_metric='mae', early_stopping_rounds=20)
    m.fit(
        data['X_train'], data['y_train'],
        eval_set=[(data['X_val'], data['y_val'])],
        verbose=False,
    )
    return mape(data['y_val'], m.predict(data['X_val']))


def train_xgboost(data, best_params=None, n_trials=30):
    """
    n_trials=30: 30 Optuna deneme → en iyi hiperparametreler secilir.
    early_stopping_rounds=30: Val loss 30 round iyilesmezse egitim durur.
    """
    print(f"\n=== XGBOOST + OPTUNA (n_trials={n_trials}) ===")

    if best_params is None:
        print(f"  Optuna hiperparametre araması basliyor ({n_trials} trial)...")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda t: _objective(t, data), n_trials=n_trials)
        best_params = study.best_params
        print(f"  En iyi val MAPE: {study.best_value:.2f}%")
        print(f"  En iyi parametreler: {best_params}")

    model = xgb.XGBRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric='mae',
    )
    model.fit(
        data['X_train'], data['y_train'],
        eval_set=[(data['X_val'], data['y_val'])],
        verbose=False,
    )
    print(f"  Kullanilan iteration: {model.best_iteration}")

    pred_train = model.predict(data['X_train'])
    pred_val   = model.predict(data['X_val'])
    pred_test  = model.predict(data['X_test'])

    m_train = compute_metrics(data['y_train'], pred_train, 'Train')
    m_val   = compute_metrics(data['y_val'],   pred_val,   'Val  ')
    m_test  = compute_metrics(data['y_test'],  pred_test,  'Test ')

    importances = dict(zip(data['feature_cols'], model.feature_importances_))

    return {
        'name'         : 'XGBoost',
        'model'        : model,
        'best_params'  : best_params,
        'pred_train'   : pred_train,
        'pred_val'     : pred_val,
        'pred_test'    : pred_test,
        'metrics_train': m_train,
        'metrics_val'  : m_val,
        'metrics_test' : m_test,
        'feature_importance': importances,
    }
