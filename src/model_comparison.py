"""
Model Karsilastirma ve Akademik Dogrulama Testleri
- Naive baseline
- 3 model karsilastirmasi
- Ablation study
- Residual analizi
- Overfitting kontrolu
- Korelasyon analizi (leakage kaniti)
"""
import numpy as np
import pandas as pd
from scipy import stats
from model_rf import mean_absolute_percentage_error, evaluate


def naive_baseline(data):
    """Naive model: Gecen yilin degeri = bu yilin tahmini (persistence)."""
    print("\n=== NAIVE BASELINE (Persistence Model) ===")
    # demand_lag_1 zaten gecen yilin degeri
    lag_col_idx = data['feature_cols'].index('demand_lag_1')

    pred_val = data['X_val'][:, lag_col_idx]
    pred_test = data['X_test'][:, lag_col_idx]

    metrics_val = evaluate(data['y_val'], pred_val, "Val")
    metrics_test = evaluate(data['y_test'], pred_test, "Test")

    return {
        'name': 'Naive Baseline',
        'metrics_val': metrics_val,
        'metrics_test': metrics_test,
        'pred_test': pred_test,
    }


def comparison_table(results):
    """Tum modellerin karsilastirma tablosunu olustur."""
    print("\n" + "=" * 80)
    print("MODEL KARSILASTIRMA TABLOSU")
    print("=" * 80)

    header = f"{'Model':<20} {'Train MAPE':>12} {'Val MAPE':>12} {'Test MAPE':>12} {'Test R2':>10} {'Naive+?':>8}"
    print(header)
    print("-" * 80)

    naive_test_mape = None
    for r in results:
        if r['name'] == 'Naive Baseline':
            naive_test_mape = r['metrics_test']['mape']
            break

    rows = []
    for r in results:
        train_mape = r.get('metrics_train', {}).get('mape', '-')
        val_mape = r['metrics_val']['mape']
        test_mape = r['metrics_test']['mape']
        test_r2 = r['metrics_test']['r2']

        train_str = f"{train_mape:.2f}%" if isinstance(train_mape, float) else "-"
        if r['name'] == 'Naive Baseline':
            better = "-"
        elif naive_test_mape and test_mape < naive_test_mape:
            better = "Evet"
        else:
            better = "Hayir"

        print(f"{r['name']:<20} {train_str:>12} {val_mape:>11.2f}% {test_mape:>11.2f}% {test_r2:>10.4f} {better:>8}")

        rows.append({
            'Model': r['name'],
            'Train_MAPE': train_mape if isinstance(train_mape, float) else None,
            'Val_MAPE': val_mape,
            'Test_MAPE': test_mape,
            'Test_R2': test_r2,
            'Better_Than_Naive': better,
        })

    print("=" * 80)
    return pd.DataFrame(rows)


def correlation_analysis(data):
    """Feature-hedef korelasyon analizi (leakage kaniti)."""
    print("\n=== LEAKAGE KANITI: Feature-Hedef Korelasyon Analizi ===")

    df = data['full_df']
    feature_cols = data['feature_cols']
    target = 'electricity_demand'

    correlations = {}
    for col in feature_cols:
        if col in df.columns:
            corr = df[col].corr(df[target])
            correlations[col] = corr

    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    # Autoregresif lag feature'lar — bunlar MESRU, leakage degil
    autoregressive_cols = {'demand_lag_1', 'demand_lag_2', 'demand_lag_3',
                           'demand_rolling_mean_3', 'demand_rolling_std_3',
                           'demand_yoy_growth'}

    print(f"\n  {'Feature':<45} {'Korelasyon':>12} {'Durum':>15}")
    print("  " + "-" * 72)
    leakage_found = False
    for feat, corr in sorted_corr:
        if feat in autoregressive_cols:
            status = "AUTOREGRESIF"  # Lag feature, mesru
        elif abs(corr) > 0.95:
            status = "LEAKAGE!"
            leakage_found = True
        elif abs(corr) > 0.85:
            status = "DIKKAT"
        else:
            status = "TEMIZ"
        print(f"  {feat:<45} {corr:>12.4f} {status:>15}")

    print("\n  NOT: Lag feature'lar (demand_lag_1/2/3, rolling) autoregresif bilesendedir.")
    print("  Bunlar GECMIS verileri kullanir, GELECEK bilgisi SIZDIRMAZ → Mesru.")

    if not leakage_found:
        print("  SONUC: Hicbir non-autoregresif feature hedefle r > 0.95 gostermemektedir → TEMIZ")
    else:
        print("  UYARI: Potansiyel leakage tespit edildi!")

    return correlations


def ablation_study(data, train_func, model_name="XGBoost"):
    """Ablation study: Onemli feature'lari cikararak etkiyi olc."""
    print(f"\n=== ABLATION STUDY ({model_name}) ===")
    print("  Her seferinde bir feature grubu cikariliyor...\n")

    feature_cols = data['feature_cols']

    # Tam modelin performansi
    from model_xgboost import train_xgboost
    import io, contextlib

    # Sessiz calistir — once tam model ile tuning yap
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        full_result = train_xgboost(data)
    full_mape = full_result['metrics_test']['mape']
    # Tuned parametreleri al (ablation'da tekrar tuning yapma)
    tuned_params = {k: v for k, v in full_result['model'].get_params().items()
                    if k not in ('random_state', 'n_jobs', 'early_stopping_rounds')}
    print(f"  {'Tam Model':<40} Test MAPE: {full_mape:.2f}%")

    # Feature gruplari (optimize edilmis feature seti)
    groups = {
        'Lag features': ['demand_lag_1', 'demand_lag_2', 'demand_lag_3',
                         'demand_rolling_mean_3', 'demand_rolling_std_3', 'demand_yoy_growth'],
        'Ekonomik': ['GDP_Growth', 'GDP_Per_Capita'],
        'Demografik': ['Population', 'Population_Density', 'Urbanization_Rate'],
    }
    # Sadece mevcut feature'lari test et
    groups = {k: [c for c in v if c in feature_cols] for k, v in groups.items()}
    groups = {k: v for k, v in groups.items() if v}

    ablation_results = {'Tam Model': full_mape}

    for group_name, cols_to_remove in groups.items():
        # Feature indekslerini bul
        remaining = [i for i, c in enumerate(feature_cols) if c not in cols_to_remove]
        remaining_names = [feature_cols[i] for i in remaining]

        # Yeni data dict olustur
        data_ablated = {
            'X_train': data['X_train'][:, remaining],
            'y_train': data['y_train'],
            'X_val': data['X_val'][:, remaining],
            'y_val': data['y_val'],
            'X_test': data['X_test'][:, remaining],
            'y_test': data['y_test'],
            'feature_cols': remaining_names,
        }

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            result = train_xgboost(data_ablated, best_params=tuned_params)
        ablated_mape = result['metrics_test']['mape']
        diff = ablated_mape - full_mape
        print(f"  {group_name + ' cikarildi':<40} Test MAPE: {ablated_mape:.2f}% (delta: +{diff:.2f}%)")
        ablation_results[f'{group_name} cikarildi'] = ablated_mape

    return ablation_results


def residual_analysis(y_true, y_pred, years=None, label=""):
    """Residual (artik) analizi."""
    print(f"\n=== RESIDUAL ANALIZI ({label}) ===")

    residuals = y_true - y_pred

    # Temel istatistikler
    print(f"  Ortalama residual: {np.mean(residuals):.4f}")
    print(f"  Std residual: {np.std(residuals):.4f}")
    print(f"  Medyan residual: {np.median(residuals):.4f}")

    # Normallik testi (Shapiro-Wilk, max 5000 ornekle)
    sample = residuals[:5000] if len(residuals) > 5000 else residuals
    stat, p_value = stats.shapiro(sample)
    normal = "EVET" if p_value > 0.05 else "HAYIR"
    print(f"  Shapiro-Wilk normallik testi: stat={stat:.4f}, p={p_value:.4f} → Normal: {normal}")

    # Ortalama sifirdan farkli mi? (t-testi)
    t_stat, t_p = stats.ttest_1samp(residuals, 0)
    biased = "EVET" if t_p < 0.05 else "HAYIR"
    print(f"  Bias testi (t-test, H0: mean=0): t={t_stat:.4f}, p={t_p:.4f} → Bias var: {biased}")

    return residuals


def overfitting_check(results):
    """Train vs Test MAPE farki kontrolu."""
    print("\n=== OVERFITTING KONTROLU ===")
    print(f"  {'Model':<20} {'Train MAPE':>12} {'Test MAPE':>12} {'Fark':>8} {'Durum':>12}")
    print("  " + "-" * 64)

    for r in results:
        if 'metrics_train' not in r or not r.get('metrics_train'):
            continue
        train_m = r['metrics_train']['mape']
        test_m = r['metrics_test']['mape']
        diff = test_m - train_m
        if diff < 2:
            status = "SAGLIKLI"
        elif diff < 5:
            status = "KABUL EDL."
        else:
            status = "OVERFITTING"
        print(f"  {r['name']:<20} {train_m:>11.2f}% {test_m:>11.2f}% {diff:>+7.2f}% {status:>12}")


def diebold_mariano_test(y_true, pred1, pred2, name1, name2):
    """Diebold-Mariano testi: iki model arasindaki fark anlamli mi?"""
    e1 = y_true - pred1
    e2 = y_true - pred2
    d = e1**2 - e2**2

    mean_d = np.mean(d)
    std_d = np.std(d, ddof=1) / np.sqrt(len(d))

    if std_d == 0:
        return None

    dm_stat = mean_d / std_d
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    significant = "EVET" if p_value < 0.05 else "HAYIR"
    better = name1 if mean_d > 0 else name2
    print(f"  {name1} vs {name2}: DM={dm_stat:.4f}, p={p_value:.4f} → Fark anlamli: {significant} (Daha iyi: {better})")

    return {'dm_stat': dm_stat, 'p_value': p_value, 'significant': p_value < 0.05, 'better': better}


def turkey_specific_eval(data, results):
    """Turkiye-ozel model degerlendirmesi."""
    print("\n=== TURKIYE-OZEL DEGERLENDIRME ===")
    test_df = data['test_df']
    turkey_mask = test_df['iso_code'].values == 'TUR'

    if turkey_mask.sum() == 0:
        print("  Turkiye test verisinde bulunamadi!")
        return

    y_turkey = data['y_test'][turkey_mask]
    print(f"  Turkiye test verisi: {turkey_mask.sum()} yil")
    print(f"  Gercek degerler: {y_turkey}")

    print(f"\n  {'Model':<20} {'Turkey MAPE':>12} {'Turkey MAE':>12} {'Turkey R2':>10}")
    print("  " + "-" * 54)
    for r in results:
        if r['name'] == 'Naive Baseline':
            # Naive icin lag_1 kullan
            lag_idx = data['feature_cols'].index('demand_lag_1')
            pred = data['X_test'][turkey_mask, lag_idx]
        else:
            pred = r['pred_test'][turkey_mask]
        m = evaluate(y_turkey, pred, "")
        print(f"  {r['name']:<20} {m['mape']:>11.2f}% {m['mae']:>12.2f} {m['r2']:>10.4f}")



def cross_validation_report(data):
    """TimeSeriesSplit cross-validation raporu."""
    print("\n=== CROSS-VALIDATION RAPORU (TimeSeriesSplit, 4-fold) ===")
    from sklearn.model_selection import TimeSeriesSplit
    import xgboost as xgb

    tscv = TimeSeriesSplit(n_splits=4)

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(data['X_train'])):
        X_tr = data['X_train'][train_idx]
        y_tr = data['y_train'][train_idx]
        X_vl = data['X_train'][val_idx]
        y_vl = data['y_train'][val_idx]

        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7,
            random_state=42, n_jobs=-1
        )
        model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)

        pred = model.predict(X_vl)
        metrics = evaluate(y_vl, pred, "")

        fold_results.append(metrics)

        print(f"  Fold {fold+1}: Train={len(train_idx)}, Val={len(val_idx)}, "
              f"MAPE={metrics['mape']:.2f}%, R2={metrics['r2']:.4f}")

    avg_mape = np.mean([r['mape'] for r in fold_results])
    avg_r2 = np.mean([r['r2'] for r in fold_results])
    std_mape = np.std([r['mape'] for r in fold_results])

    print(f"\n  Ortalama: MAPE={avg_mape:.2f}% (+/-{std_mape:.2f}%), R2={avg_r2:.4f}")

    return fold_results


def run_all_tests(data, results):
    """Tum akademik dogrulama testlerini calistir."""
    print("\n" + "=" * 80)
    print("AKADEMIK DOGRULAMA TESTLERI")
    print("=" * 80)

    # Test 1: Korelasyon analizi
    correlations = correlation_analysis(data)

    # Test: Cross-validation
    cv_results = cross_validation_report(data)

    # Test 2: Karsilastirma tablosu (naive dahil)
    comparison_df = comparison_table(results)

    # Test 3: Overfitting kontrolu
    overfitting_check(results)

    # Test 4: Residual analizi (en iyi model icin)
    best_model = min([r for r in results if r['name'] != 'Naive Baseline'],
                     key=lambda x: x['metrics_test']['mape'])
    residuals = residual_analysis(
        data['y_test'], best_model['pred_test'],
        label=best_model['name'])

    # Test 5: Istatistiksel anlamlilik
    print("\n=== ISTATISTIKSEL ANLAMLILIK (Diebold-Mariano) ===")
    model_results = [r for r in results if r['name'] != 'Naive Baseline']
    for i in range(len(model_results)):
        for j in range(i+1, len(model_results)):
            diebold_mariano_test(
                data['y_test'],
                model_results[i]['pred_test'],
                model_results[j]['pred_test'],
                model_results[i]['name'],
                model_results[j]['name']
            )

    # Test 6: Turkiye-ozel degerlendirme
    turkey_specific_eval(data, results)

    return {
        'comparison_df': comparison_df,
        'correlations': correlations,
        'residuals': residuals,
        'best_model': best_model,
    }
