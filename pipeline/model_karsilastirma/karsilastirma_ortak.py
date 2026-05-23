"""
MODEL KARSILASTIRMA — Ortak Harness (7 model + Walk-Forward CV)
================================================================
Bu modul, 3 asamada da kullanilan ortak altyapiyi saglar:

  7 MODEL:
    1. Random Forest   (feature + lag tabanli)
    2. XGBoost         (feature + lag tabanli)
    3. LightGBM        (feature + lag tabanli)
    4. CatBoost        (feature + lag tabanli)
    5. LSTM            (feature + lag tabanli, deep learning)
    6. ARIMA           (univariate — hedefin kendi gecmisi)
    7. SARIMA          (univariate — mevsimsel ARIMA baseline)

  DEGERLENDIRME — WALK-FORWARD EXPANDING WINDOW CROSS-VALIDATION:
    * TUM modeller AYNI test noktalarinda degerlendirilir → R² karsilastirilabilir.
    * train[0..t-1] → tahmin[t]; pencere her adimda genisler.
    * Gelecek bilgisi ASLA sizmaz (leakage guard).
    * MinMaxScaler her fold'da YALNIZCA train fold'a fit edilir.
    * ARIMA/SARIMA her fold'da train[0..t-1] uzerinde yeniden fit edilir,
      1 adim ileri tahmin yapar.

NOT: ARIMA/SARIMA feature kullanmaz; yalniz hedefin (y) kendi zaman serisini
     kullanir. Bu, tezdeki standart univariate baseline'dir.
"""

import warnings
import numpy as np
import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.ensemble        import RandomForestRegressor
from sklearn.preprocessing   import MinMaxScaler
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost                 import XGBRegressor
from lightgbm                import LGBMRegressor
from catboost                import CatBoostRegressor

# ── TensorFlow (LSTM) — opsiyonel ──────────────────────────────────────────────
try:
    import os as _os
    _os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf
    from tensorflow.keras.models    import Sequential
    from tensorflow.keras.layers    import LSTM as LSTM_Layer, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel('ERROR')
    TF_MEVCUT = True
except Exception:
    TF_MEVCUT = False

# ── statsmodels (ARIMA/SARIMA) ──────────────────────────────────────────────────
try:
    from statsmodels.tsa.arima.model      import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SM_MEVCUT = True
except Exception:
    SM_MEVCUT = False

# 7 modelin kanonik isim sirasi (arayuzde de bu sira kullanilir)
MODEL_SIRASI = ['RF', 'XGBoost', 'LightGBM', 'CatBoost', 'LSTM', 'ARIMA', 'SARIMA']

# Forecast asamasinda en iyi modeli yeniden kurmak icin isim → sinif haritasi
MODEL_CLS = {
    'RF':       RandomForestRegressor,
    'XGBoost':  XGBRegressor,
    'LightGBM': LGBMRegressor,
    'CatBoost': CatBoostRegressor,
}
# Feature kullanan modeller (lag + feature). ARIMA/SARIMA univariate olup bu gruba girmez.
FEATURE_MODELLER = set(MODEL_CLS) | {'LSTM'}


# ═══════════════════════════════════════════════════════════════════════════════
# METRIKLER
# ═══════════════════════════════════════════════════════════════════════════════

def mape(y_true, y_pred):
    t, p = np.array(y_true, float), np.array(y_pred, float)
    mask = t != 0
    return float(np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100) if mask.sum() else np.nan


def metrikleri_hesapla(y_true, y_pred):
    """R², MAE, RMSE, MAPE — tek dict.
    Model NaN/inf tahmin uretirse (patlama) metrik NaN dondurulur → model
    karsilastirmadan elenir (gercek degerle doldurup R² sismesi YAPILMAZ;
    bkz. code review M1)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if not np.all(np.isfinite(y_pred)):
        return {'r2': np.nan, 'mae': np.nan, 'rmse': np.nan, 'mape': np.nan}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mp   = mape(y_true, y_pred)
    r2   = float(r2_score(y_true, y_pred)) if len(y_true) > 2 else float('nan')
    return {'r2': round(r2, 4), 'mae': round(mae, 4),
            'rmse': round(rmse, 4), 'mape': round(mp, 4) if mp == mp else np.nan}


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD CV — FEATURE TABANLI MODELLER (RF / XGB / LGBM / CatBoost)
# ═══════════════════════════════════════════════════════════════════════════════

def walk_forward_feature(model_factory, X_all, y_all, min_train, fark_al=False):
    """Expanding window: her adimda train[0..t-1] ile egit, t'yi tahmin et.
    LEAKAGE GUARD: scaler SADECE train fold'a fit edilir.

    fark_al=True : Hedef olarak fark (Δy=y[t]-y[t-1]) modellenir, tahmin seviyeye
    geri cevrilir (ŷ[t]=y[t-1]+Δŷ). Tree modellerinin trend ekstrapolasyon
    kisitini cozer. R² yine SEVIYE uzerinden hesaplanir (karsilastirilabilir)."""
    preds, trues = [], []
    for t in range(min_train, len(y_all)):
        sc = MinMaxScaler()
        m = model_factory()
        if fark_al and t >= 3:
            # d[i]=y[i]-y[i-1], girdi X_all[i] (i=1..t-1) → d[t] tahmin → seviyeye al
            X_tr = X_all[1:t]
            d_tr = y_all[1:t] - y_all[0:t - 1]
            m.fit(sc.fit_transform(X_tr), d_tr)
            d_hat = m.predict(sc.transform(X_all[t:t + 1]))[0]
            p = y_all[t - 1] + d_hat
        else:
            m.fit(sc.fit_transform(X_all[:t]), y_all[:t])
            p = m.predict(sc.transform(X_all[t:t + 1]))[0]
        preds.append(float(p))
        trues.append(float(y_all[t]))
    return np.array(trues), np.array(preds)


# ── Tree model fabrikalari + Optuna ─────────────────────────────────────────────

def _tscv(n, n_splits):
    return TimeSeriesSplit(n_splits=max(2, min(n_splits, n - 2)))


def _cv_rmse_tree(factory, X, y, n_splits):
    rmses = []
    for tr_i, vl_i in _tscv(len(y), n_splits).split(X):
        sc = MinMaxScaler()
        m = factory()
        m.fit(sc.fit_transform(X[tr_i]), y[tr_i])
        rmses.append(np.sqrt(mean_squared_error(y[vl_i], m.predict(sc.transform(X[vl_i])))))
    return float(np.mean(rmses))


def tune_rf(X, y, n_trials, n_splits=3, seed=42):
    def obj(tr):
        p = dict(n_estimators=tr.suggest_int('n_estimators', 50, 400),
                 max_depth=tr.suggest_int('max_depth', 2, 10),
                 min_samples_leaf=tr.suggest_int('min_samples_leaf', 1, 5),
                 max_features=tr.suggest_categorical('max_features', ['sqrt', 'log2', 0.7]),
                 random_state=seed, n_jobs=-1)
        return _cv_rmse_tree(lambda: RandomForestRegressor(**p), X, y, n_splits)
    st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return dict(st.best_params, random_state=seed, n_jobs=-1)


def tune_xgb(X, y, n_trials, n_splits=3, seed=42):
    def obj(tr):
        p = dict(n_estimators=tr.suggest_int('n_estimators', 50, 300),
                 max_depth=tr.suggest_int('max_depth', 2, 5),
                 learning_rate=tr.suggest_float('learning_rate', 0.01, 0.2, log=True),
                 subsample=tr.suggest_float('subsample', 0.6, 1.0),
                 colsample_bytree=tr.suggest_float('colsample_bytree', 0.6, 1.0),
                 reg_alpha=tr.suggest_float('reg_alpha', 0.0, 2.0),
                 reg_lambda=tr.suggest_float('reg_lambda', 0.5, 3.0),
                 random_state=seed, verbosity=0)
        return _cv_rmse_tree(lambda: XGBRegressor(**p), X, y, n_splits)
    st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return dict(st.best_params, random_state=seed, verbosity=0)


def tune_lgbm(X, y, n_trials, n_splits=3, seed=42):
    def obj(tr):
        p = dict(n_estimators=tr.suggest_int('n_estimators', 50, 300),
                 max_depth=tr.suggest_int('max_depth', 2, 6),
                 num_leaves=tr.suggest_int('num_leaves', 7, 31),
                 learning_rate=tr.suggest_float('learning_rate', 0.01, 0.2, log=True),
                 subsample=tr.suggest_float('subsample', 0.6, 1.0),
                 colsample_bytree=tr.suggest_float('colsample_bytree', 0.6, 1.0),
                 min_child_samples=tr.suggest_int('min_child_samples', 2, 10),
                 reg_alpha=tr.suggest_float('reg_alpha', 0.0, 2.0),
                 reg_lambda=tr.suggest_float('reg_lambda', 0.0, 3.0),
                 random_state=seed, n_jobs=-1, verbosity=-1)
        return _cv_rmse_tree(lambda: LGBMRegressor(**p), X, y, n_splits)
    st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return dict(st.best_params, random_state=seed, n_jobs=-1, verbosity=-1)


def tune_catboost(X, y, n_trials, n_splits=3, seed=42):
    def obj(tr):
        p = dict(iterations=tr.suggest_int('iterations', 50, 300),
                 depth=tr.suggest_int('depth', 2, 6),
                 learning_rate=tr.suggest_float('learning_rate', 0.01, 0.2, log=True),
                 l2_leaf_reg=tr.suggest_float('l2_leaf_reg', 1.0, 6.0),
                 random_state=seed, verbose=0, allow_writing_files=False)
        return _cv_rmse_tree(lambda: CatBoostRegressor(**p), X, y, n_splits)
    st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return dict(st.best_params, random_state=seed, verbose=0, allow_writing_files=False)


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD CV — LSTM
# ═══════════════════════════════════════════════════════════════════════════════

def _lstm_olustur(n_feat, units1=64, units2=32, dropout=0.2, lr=1e-3):
    m = Sequential([
        Input(shape=(1, n_feat)),
        LSTM_Layer(units1, return_sequences=True),
        Dropout(dropout),
        LSTM_Layer(units2, return_sequences=False),
        Dropout(dropout),
        Dense(16, activation='relu'),
        Dense(1),
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    m.compile(optimizer=opt, loss='mse')
    return m


def walk_forward_lstm(X_all, y_all, min_train, units1=64, units2=32,
                      dropout=0.2, lr=1e-3, epochs=120, batch=4, patience=15,
                      fark_al=False):
    """LSTM walk-forward CV. Hedef ayri scaler ile olceklenir, tahmin geri donusturulur.
    fark_al=True ise hedef fark (Δy) modellenir, tahmin seviyeye cevrilir."""
    if not TF_MEVCUT:
        return None, None
    preds, trues = [], []
    for t in range(min_train, len(y_all)):
        sc_f, sc_h = MinMaxScaler(), MinMaxScaler()
        if fark_al and t >= 3:
            X_tr = X_all[1:t]
            tgt  = y_all[1:t] - y_all[0:t - 1]
        else:
            X_tr = X_all[:t]
            tgt  = y_all[:t]
        Xtr = sc_f.fit_transform(X_tr)
        Xte = sc_f.transform(X_all[t:t + 1])
        ytr = sc_h.fit_transform(tgt.reshape(-1, 1)).ravel()
        nf = Xtr.shape[1]
        tf.keras.utils.set_random_seed(42)
        m = _lstm_olustur(nf, units1, units2, dropout, lr)
        es = EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
        m.fit(Xtr.reshape(-1, 1, nf), ytr, epochs=epochs, batch_size=batch,
              callbacks=[es], verbose=0)
        ps = m.predict(Xte.reshape(1, 1, nf), verbose=0).ravel()[0]
        out = sc_h.inverse_transform([[ps]])[0][0]
        pred = (y_all[t - 1] + out) if (fark_al and t >= 3) else out
        preds.append(float(pred))
        trues.append(float(y_all[t]))
        del m
        tf.keras.backend.clear_session()
    return np.array(trues), np.array(preds)


def tune_lstm(X_all, y_all, n_trials, n_splits=3, seed=42):
    """Hafif LSTM Optuna (TimeSeriesSplit). Tek hedefli asamalar icin."""
    if not TF_MEVCUT:
        return dict(units1=64, units2=32, dropout=0.2, lr=1e-3, batch=4)
    tscv = _tscv(len(y_all), n_splits)
    nf = X_all.shape[1]

    def obj(tr):
        u1 = tr.suggest_categorical('units1', [32, 64, 128])
        u2 = tr.suggest_categorical('units2', [16, 32, 64])
        dr = tr.suggest_float('dropout', 0.1, 0.4)
        lr = tr.suggest_float('lr', 1e-4, 1e-2, log=True)
        ba = tr.suggest_categorical('batch', [4, 8])
        rmses = []
        for tr_i, vl_i in tscv.split(X_all):
            sc_f, sc_h = MinMaxScaler(), MinMaxScaler()
            Xtr = sc_f.fit_transform(X_all[tr_i]); Xvl = sc_f.transform(X_all[vl_i])
            ytr = sc_h.fit_transform(y_all[tr_i].reshape(-1, 1)).ravel()
            tf.keras.utils.set_random_seed(seed)
            m = _lstm_olustur(nf, u1, u2, dr, lr)
            es = EarlyStopping(monitor='loss', patience=8, restore_best_weights=True)
            m.fit(Xtr.reshape(-1, 1, nf), ytr, epochs=80, batch_size=ba,
                  callbacks=[es], verbose=0)
            p = sc_h.inverse_transform(
                m.predict(Xvl.reshape(-1, 1, nf), verbose=0).reshape(-1, 1)).ravel()
            rmses.append(np.sqrt(mean_squared_error(y_all[vl_i], p)))
            tf.keras.backend.clear_session()
        return float(np.mean(rmses))

    st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return st.best_params


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD CV — ARIMA / SARIMA (UNIVARIATE)
# ═══════════════════════════════════════════════════════════════════════════════

_ARIMA_ADAYLAR  = [(1, 1, 1), (2, 1, 1), (1, 1, 0), (0, 1, 1), (2, 1, 2), (1, 0, 0)]
# SARIMA: yillik veride mevsimsel FARK ALMA (D=1) kisa seride iraksiyor → D=0 ile
# sinirli tutuluyor (yalniz mevsimsel AR/MA terimi). Periyot kucuk (2-3 yil).
_SARIMA_ADAYLAR = [(1, 0, 0, 2), (0, 0, 1, 2), (1, 0, 1, 2), (1, 0, 0, 3), (0, 0, 1, 3)]


def _en_iyi_arima_order(y_train):
    """Ilk egitim penceresinde AIC'e gore en iyi ARIMA order'i sec."""
    best, best_aic = (1, 1, 1), np.inf
    for order in _ARIMA_ADAYLAR:
        try:
            aic = ARIMA(y_train, order=order).fit().aic
            if np.isfinite(aic) and aic < best_aic:
                best_aic, best = aic, order
        except Exception:
            continue
    return best


def _en_iyi_sarima_order(y_train, base_order):
    """Ilk egitim penceresinde AIC'e gore en iyi mevsimsel order'i sec."""
    best, best_aic = None, np.inf
    for so in _SARIMA_ADAYLAR:
        try:
            aic = SARIMAX(y_train, order=base_order, seasonal_order=so,
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=0).aic
            if np.isfinite(aic) and aic < best_aic:
                best_aic, best = aic, so
        except Exception:
            continue
    return best


def walk_forward_arima(y_all, min_train, seasonal=False):
    """ARIMA/SARIMA expanding-window walk-forward CV.
    Order, ilk egitim penceresinde AIC ile secilir; her fold parametreleri
    yeniden fit eder, 1 adim ileri tahmin yapar.
    Donus: (trues, preds, order, seasonal_order_or_None)."""
    if not SM_MEVCUT:
        return None, None, None, None
    y_all = np.asarray(y_all, float)
    base_order = _en_iyi_arima_order(y_all[:min_train])
    seas_order = _en_iyi_sarima_order(y_all[:min_train], base_order) if seasonal else None
    if seasonal and seas_order is None:
        seasonal = False  # mevsimsel fit basarisiz → ARIMA'ya dus

    preds, trues = [], []
    for t in range(min_train, len(y_all)):
        train = y_all[:t]
        # LEAKAGE-FREE GUARD: tahmin, egitim penceresinin tarihsel bandina kisitlanir.
        # SARIMA/ARIMA yillik veride bazen patlar (R² -milyon); bu makul bir sinir koyar.
        tmin, tmax = float(np.min(train)), float(np.max(train))
        rng = max(tmax - tmin, abs(tmax) * 0.1, 1e-6)
        alt, ust = tmin - 1.5 * rng, tmax + 1.5 * rng
        try:
            if seasonal:
                fit = SARIMAX(train, order=base_order, seasonal_order=seas_order,
                              enforce_stationarity=False,
                              enforce_invertibility=False).fit(disp=0)
            else:
                fit = ARIMA(train, order=base_order).fit()
            p = float(np.asarray(fit.forecast(1)).ravel()[0])
            if not np.isfinite(p):
                p = float(train[-1])
        except Exception:
            p = float(train[-1])  # naive fallback (son bilinen deger)
        p = float(np.clip(p, alt, ust))  # asiri sapmayi engelle
        preds.append(p)
        trues.append(float(y_all[t]))
    return np.array(trues), np.array(preds), base_order, seas_order


# ═══════════════════════════════════════════════════════════════════════════════
# UST SEVIYE: TUM 7 MODELI CALISTIR
# ═══════════════════════════════════════════════════════════════════════════════

def yedi_modeli_calistir(X_all, y_all, min_train, n_trials_tree=20,
                         n_trials_lstm=0, lstm_epochs=120, etiket='', logla=True,
                         fark_al=False):
    """X_all (feature+lag) ve y_all (hedef) icin 7 modeli walk-forward CV ile
    degerlendirir. AYNI test noktalari → karsilastirilabilir R².

    n_trials_lstm=0 ise LSTM sabit mimari (units 64/32) kullanir (hizli).
    fark_al=True ise feature-tabanli 5 model (RF/XGB/LGBM/CatBoost/LSTM) hedefin
    farkini modeller (trend ekstrapolasyon kisitini cozer); ARIMA/SARIMA zaten
    I=1 ile fark alir, degismez.
    Donus: {model_isim: {'r2','mae','rmse','mape','cv_n', ...}}
    """
    sonuc = {}
    n = len(y_all)

    def _log(msg):
        if logla:
            print(msg, flush=True)

    # ── Tree modelleri ──
    tree_tanim = [
        ('RF',       tune_rf,       RandomForestRegressor),
        ('XGBoost',  tune_xgb,      XGBRegressor),
        ('LightGBM', tune_lgbm,     LGBMRegressor),
        ('CatBoost', tune_catboost, CatBoostRegressor),
    ]
    for isim, tuner, cls in tree_tanim:
        try:
            params = tuner(X_all, y_all, n_trials_tree) if n_trials_tree > 0 else {}
            trues, preds = walk_forward_feature(lambda p=params, c=cls: c(**p),
                                                X_all, y_all, min_train, fark_al=fark_al)
            m = metrikleri_hesapla(trues, preds)
            m['cv_n'] = len(trues)
            sonuc[isim] = {**m, 'params': params,
                           'trues': trues.tolist(), 'preds': preds.tolist()}
            _log(f"    {isim:<9} R²={m['r2']:>8.4f}  RMSE={m['rmse']:>10.4f}  "
                 f"MAPE={m['mape'] if m['mape']==m['mape'] else float('nan'):>7.2f}%  (n={len(trues)})")
        except Exception as e:
            _log(f"    {isim:<9} HATA: {e}")
            sonuc[isim] = {'r2': np.nan, 'mae': np.nan, 'rmse': np.nan,
                           'mape': np.nan, 'cv_n': 0}

    # ── LSTM ──
    if TF_MEVCUT:
        try:
            lp = {'units1': 64, 'units2': 32, 'dropout': 0.2, 'lr': 1e-3, 'batch': 4}
            if n_trials_lstm > 0:
                lp = {**lp, **tune_lstm(X_all, y_all, n_trials_lstm)}
            trues, preds = walk_forward_lstm(
                X_all, y_all, min_train,
                units1=lp['units1'], units2=lp['units2'],
                dropout=lp['dropout'], lr=lp['lr'],
                epochs=lstm_epochs, batch=lp['batch'], fark_al=fark_al)
            m = metrikleri_hesapla(trues, preds)
            m['cv_n'] = len(trues)
            sonuc['LSTM'] = {**m, 'params': lp,
                             'trues': trues.tolist(), 'preds': preds.tolist()}
            _log(f"    {'LSTM':<9} R²={m['r2']:>8.4f}  RMSE={m['rmse']:>10.4f}  "
                 f"MAPE={m['mape'] if m['mape']==m['mape'] else float('nan'):>7.2f}%  (n={len(trues)})")
        except Exception as e:
            _log(f"    {'LSTM':<9} HATA: {e}")
            sonuc['LSTM'] = {'r2': np.nan, 'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'cv_n': 0}
    else:
        _log("    LSTM      atlandi (TensorFlow yok)")
        sonuc['LSTM'] = {'r2': np.nan, 'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'cv_n': 0}

    # ── ARIMA / SARIMA (univariate, ayni test noktalari) ──
    for isim, seasonal in [('ARIMA', False), ('SARIMA', True)]:
        if not SM_MEVCUT:
            sonuc[isim] = {'r2': np.nan, 'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'cv_n': 0}
            continue
        try:
            trues, preds, order, so = walk_forward_arima(y_all, min_train, seasonal=seasonal)
            m = metrikleri_hesapla(trues, preds)
            m['cv_n'] = len(trues)
            extra = {'order': str(order)}
            if seasonal:
                extra['seasonal_order'] = str(so)
            sonuc[isim] = {**m, **extra, 'trues': trues.tolist(), 'preds': preds.tolist()}
            ek = f"order={order}" + (f" seasonal={so}" if seasonal else "")
            _log(f"    {isim:<9} R²={m['r2']:>8.4f}  RMSE={m['rmse']:>10.4f}  "
                 f"MAPE={m['mape'] if m['mape']==m['mape'] else float('nan'):>7.2f}%  ({ek})")
        except Exception as e:
            _log(f"    {isim:<9} HATA: {e}")
            sonuc[isim] = {'r2': np.nan, 'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'cv_n': 0}

    return sonuc


def en_iyi_model(sonuc):
    """R²'si en yuksek modeli dondur (NaN'lar haric)."""
    gecerli = {k: v for k, v in sonuc.items() if v.get('r2') == v.get('r2')}
    if not gecerli:
        return None
    return max(gecerli, key=lambda k: gecerli[k]['r2'])


# ═══════════════════════════════════════════════════════════════════════════════
# FORECAST — EN IYI MODELLE GELECEK TAHMINI
# ═══════════════════════════════════════════════════════════════════════════════
#
# CV (yukaridaki yedi_modeli_calistir) modelin BASARISINI olcer (gecmis gercek
# veri uzerinde). Forecast ise GELECEGI tahmin eder. Iki ayri is:
#   * Tree/LSTM: tum veride final egitilir, sonra iteratif forecast yapilir
#     (model kendi tahminini bir sonraki yilin lag'i olarak kullanir → leakage yok).
#   * ARIMA/SARIMA: univariate, dogrudan cok-adim forecast (feature kullanmaz).

def _final_predict_fn(isim, params, X_all, y_all, lstm_epochs=200, lstm_patience=20,
                      seed=42, fark_al=False):
    """En iyi feature-tabanli modeli (RF/XGB/LGBM/CatBoost/LSTM) TUM veride egit.
    Donus: predict_fn(X_2d) -> np.array.

    fark_al=True ise model hedefin FARKINI (Δy) ogrenir; donen predict_fn de Δŷ
    uretir. Seviyeye cevirme (ŷ=y[t-1]+Δŷ) iteratif_forecast icinde yapilir."""
    X_all = np.asarray(X_all, float)
    y_all = np.asarray(y_all, float)
    if fark_al and len(y_all) >= 3:
        X_fit = X_all[1:]
        y_fit = y_all[1:] - y_all[:-1]   # Δy
    else:
        X_fit, y_fit = X_all, y_all

    if isim in MODEL_CLS:
        sc = MinMaxScaler()
        m = MODEL_CLS[isim](**(params or {}))
        m.fit(sc.fit_transform(X_fit), y_fit)

        def predict_fn(X_2d):
            return m.predict(sc.transform(np.asarray(X_2d, float)))
        return predict_fn

    if isim == 'LSTM':
        if not TF_MEVCUT:
            return None
        p = params or {}
        sc_f, sc_h = MinMaxScaler(), MinMaxScaler()
        Xs = sc_f.fit_transform(X_fit)
        ys = sc_h.fit_transform(y_fit.reshape(-1, 1)).ravel()
        nf = Xs.shape[1]
        tf.keras.utils.set_random_seed(seed)
        m = _lstm_olustur(nf, p.get('units1', 64), p.get('units2', 32),
                          p.get('dropout', 0.2), p.get('lr', 1e-3))
        es = EarlyStopping(monitor='loss', patience=lstm_patience,
                           restore_best_weights=True)
        m.fit(Xs.reshape(-1, 1, nf), ys, epochs=lstm_epochs,
              batch_size=p.get('batch', 4), callbacks=[es], verbose=0)

        def predict_fn(X_2d):
            Xt = sc_f.transform(np.asarray(X_2d, float))
            ps = m.predict(Xt.reshape(-1, 1, nf), verbose=0).ravel()
            return sc_h.inverse_transform(ps.reshape(-1, 1)).ravel()
        return predict_fn

    return None  # ARIMA/SARIMA bu fonksiyonda ele alinmaz


def _orderi_coz(o):
    """JSON'dan gelen '(1, 1, 0)' string'ini tuple'a cevir; tuple ise aynen don."""
    if o is None:
        return None
    if isinstance(o, (tuple, list)):
        return tuple(o)
    try:
        import ast
        return tuple(ast.literal_eval(str(o)))
    except Exception:
        return None


def forecast_arima_seri(y_all, order, seasonal_order, n_steps):
    """Univariate ARIMA/SARIMA ile n adim ileri forecast.
    LEAKAGE YOK: yalnizca hedefin kendi gecmis serisini kullanir.
    Tarihsel banda clip edilir (kisa yillik seride patlamayi engeller)."""
    y = np.asarray(y_all, float)
    order = _orderi_coz(order) or (1, 1, 1)
    so = _orderi_coz(seasonal_order)
    tmin, tmax = float(np.min(y)), float(np.max(y))
    rng = max(tmax - tmin, abs(tmax) * 0.1, 1e-6)
    alt, ust = tmin - 1.5 * rng, tmax + 1.5 * rng
    try:
        if so is not None and SM_MEVCUT:
            fit = SARIMAX(y, order=order, seasonal_order=so,
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=0)
        else:
            fit = ARIMA(y, order=order).fit()
        fc = np.asarray(fit.forecast(n_steps), float).ravel()
        fc = np.nan_to_num(fc, nan=y[-1], posinf=ust, neginf=alt)
        return np.clip(fc, alt, ust)
    except Exception:
        return np.full(n_steps, y[-1], float)


def iteratif_forecast(predict_fn, gecmis_hedef, fc_yillar, x_kurucu,
                      lag_sayisi=2, clip_band=True, band_kat=5.0, fark_al=False):
    """Feature-tabanli modeller (tree/LSTM) icin iteratif gelecek tahmini.

    predict_fn   : X_2d -> np.array  (_final_predict_fn'den)
    gecmis_hedef : {yil: deger}  hedefin gecmis gercek serisi (lag kaynagi)
    fc_yillar    : tahmin edilecek yillar, KRONOLOJIK sirali
    x_kurucu     : (yil, lag_listesi) -> X (1 satirlik 2D dizi)
                   Asama 1   : lambda yil, lags: [[yil, lags[0], lags[1]]]
                   Asama 2/3 : lambda yil, lags: [[lags[0], lags[1], *feat_degerleri]]
    fark_al      : True ise predict_fn Δŷ uretir; seviye ŷ=y[t-1]+Δŷ ile hesaplanir.

    Model kendi tahminini bir sonraki yilin lag'i olarak kullanir (leakage yok).
    """
    hist = dict(gecmis_hedef)
    vals = [v for v in hist.values() if np.isfinite(v)]
    vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 1.0)
    rng = max(vmax - vmin, 1e-6)
    alt, ust = vmin - band_kat * rng, vmax + band_kat * rng
    fc = {}
    for yil in fc_yillar:
        lags = []
        for k in range(1, lag_sayisi + 1):
            v = hist.get(yil - k, np.nan)
            if not np.isfinite(v):
                bilinen = [val for yr, val in sorted(hist.items())
                           if yr < yil and np.isfinite(val)]
                v = bilinen[-1] if bilinen else 0.0
            lags.append(float(v))
        X = np.nan_to_num(np.asarray(x_kurucu(yil, lags), float),
                          nan=0.0, posinf=1e9, neginf=-1e9)
        try:
            ham = float(predict_fn(X)[0])
            # fark modunda model Δŷ uretir → seviye = onceki yil (lags[0]) + Δŷ
            pred = (lags[0] + ham) if fark_al else ham
            if not np.isfinite(pred):
                pred = lags[0]
        except Exception:
            pred = lags[0]
        if clip_band:
            pred = float(np.clip(pred, alt, ust))
        hist[yil] = pred
        fc[yil] = pred
    return fc
