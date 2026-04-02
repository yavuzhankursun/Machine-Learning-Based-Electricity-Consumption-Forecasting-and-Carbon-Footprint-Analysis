"""
Gorsellestirme — Tez icin grafikler
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'figures')


def setup_style():
    """Tez icin uygun grafik stili."""
    plt.rcParams.update({
        'figure.figsize': (12, 7),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 150,
    })
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_correlation_heatmap(data):
    """Feature-hedef korelasyon isi haritasi."""
    setup_style()
    df = data['full_df']
    feature_cols = data['feature_cols']
    cols = feature_cols + ['electricity_demand']
    cols_available = [c for c in cols if c in df.columns]

    corr_matrix = df[cols_available].corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('Feature-Hedef Korelasyon Matrisi (Leakage Kontrolu)')

    path = os.path.join(OUTPUT_DIR, '01_correlation_heatmap.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


def plot_feature_importance(results):
    """Feature importance karsilastirmasi (RF ve XGBoost)."""
    setup_style()

    fi_results = [r for r in results if r.get('feature_importance')]
    n = len(fi_results)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(8 * n, 8))
    if n == 1:
        axes = [axes]

    colors = ['#2196F3', '#FF9800', '#4CAF50']
    for idx, result in enumerate(fi_results):
        imp = result['feature_importance']
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        features = [x[0] for x in sorted_imp[:12]]
        values = [x[1] for x in sorted_imp[:12]]

        ax = axes[idx]
        ax.barh(range(len(features)), values, color=colors[idx % len(colors)])
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'{result["name"]} - Feature Importance')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '02_feature_importance.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


def plot_model_comparison(comparison_df):
    """Model karsilastirma grafigi."""
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = comparison_df['Model'].values
    colors = ['#9E9E9E', '#4CAF50', '#2196F3', '#FF5722'][:len(models)]

    # MAPE karsilastirma
    ax = axes[0]
    test_mapes = comparison_df['Test_MAPE'].values
    bars = ax.bar(models, test_mapes, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Test MAPE (%)')
    ax.set_title('Model Test MAPE Karsilastirmasi')
    for bar, val in zip(bars, test_mapes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax.tick_params(axis='x', rotation=20)

    # R2 karsilastirma
    ax = axes[1]
    r2_values = comparison_df['Test_R2'].values
    bars = ax.bar(models, r2_values, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Test R2')
    ax.set_title('Model Test R2 Karsilastirmasi')
    ax.set_ylim(min(0, min(r2_values) - 0.1), 1.05)
    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    ax.tick_params(axis='x', rotation=20)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '03_model_comparison.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


def plot_residual_analysis(y_true, y_pred, model_name):
    """Residual analiz grafikleri."""
    setup_style()
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Residual histogram
    ax = axes[0, 0]
    ax.hist(residuals, bins=50, color='#2196F3', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual (TWh)')
    ax.set_ylabel('Frekans')
    ax.set_title('Residual Dagilimi')

    # 2. Q-Q plot
    ax = axes[0, 1]
    from scipy import stats
    stats.probplot(residuals, dist='norm', plot=ax)
    ax.set_title('Q-Q Plot (Normallik Kontrolu)')

    # 3. Predicted vs Actual
    ax = axes[1, 0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color='#2196F3')
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Ideal (y=x)')
    ax.set_xlabel('Gercek (TWh)')
    ax.set_ylabel('Tahmin (TWh)')
    ax.set_title('Gercek vs Tahmin')
    ax.legend()

    # 4. Residual vs Predicted
    ax = axes[1, 1]
    ax.scatter(y_pred, residuals, alpha=0.3, s=10, color='#FF9800')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Tahmin (TWh)')
    ax.set_ylabel('Residual (TWh)')
    ax.set_title('Residual vs Tahmin (Heteroskedastisite Kontrolu)')

    fig.suptitle(f'{model_name} - Residual Analizi', fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '04_residual_analysis.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


def plot_turkey_forecast(history_df, carbon_df=None):
    """Turkiye enerji tuketimi tahmin grafigi."""
    setup_style()

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # 1. Elektrik tuketimi tahmini
    ax = axes[0]
    actual = history_df[history_df['type'] == 'actual']
    predicted = history_df[history_df['type'] == 'predicted']

    ax.plot(actual['year'], actual['electricity_demand_twh'],
            'o-', color='#2196F3', linewidth=2, markersize=5, label='Gercek')
    ax.plot(predicted['year'], predicted['electricity_demand_twh'],
            's--', color='#FF5722', linewidth=2, markersize=8, label='Tahmin')

    # Gercek ve tahmin arasini baglayan cizgi
    if len(actual) > 0 and len(predicted) > 0:
        ax.plot([actual['year'].iloc[-1], predicted['year'].iloc[0]],
                [actual['electricity_demand_twh'].iloc[-1],
                 predicted['electricity_demand_twh'].iloc[0]],
                '--', color='#FF5722', linewidth=1.5, alpha=0.5)

    ax.set_xlabel('Yil')
    ax.set_ylabel('Elektrik Tuketimi (TWh)')
    ax.set_title('Turkiye Elektrik Tuketimi: Gercek ve Tahmin')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Tahmin noktalarini etiketle
    for _, row in predicted.iterrows():
        ax.annotate(f'{row["electricity_demand_twh"]:.0f} TWh',
                    xy=(row['year'], row['electricity_demand_twh']),
                    textcoords="offset points", xytext=(0, 15),
                    ha='center', fontweight='bold', fontsize=11)

    # 2. Karbon ayak izi
    ax = axes[1]
    if carbon_df is not None:
        actual_c = carbon_df[carbon_df['type'] == 'actual']
        predicted_c = carbon_df[carbon_df['type'] == 'predicted']

        ax.fill_between(actual_c['year'], 0, actual_c['co2_mt'],
                       alpha=0.3, color='#2196F3')
        ax.plot(actual_c['year'], actual_c['co2_mt'],
                'o-', color='#2196F3', linewidth=2, markersize=5, label='Gercek CO2')

        ax.fill_between(predicted_c['year'], 0, predicted_c['co2_mt'],
                       alpha=0.3, color='#FF5722')
        ax.plot(predicted_c['year'], predicted_c['co2_mt'],
                's--', color='#FF5722', linewidth=2, markersize=8, label='Tahmin CO2')

        for _, row in predicted_c.iterrows():
            ax.annotate(f'{row["co2_mt"]:.0f} Mt',
                        xy=(row['year'], row['co2_mt']),
                        textcoords="offset points", xytext=(0, 15),
                        ha='center', fontweight='bold', fontsize=11)

    ax.set_xlabel('Yil')
    ax.set_ylabel('CO2 Emisyonu (Mt)')
    ax.set_title('Turkiye Karbon Ayak Izi: Gercek ve Tahmin')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '05_turkey_forecast.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


def plot_ablation(ablation_results):
    """Ablation study grafigi."""
    setup_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    names = list(ablation_results.keys())
    values = list(ablation_results.values())

    colors = ['#4CAF50'] + ['#FF9800'] * (len(names) - 1)
    bars = ax.barh(range(len(names)), values, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Test MAPE (%)')
    ax.set_title('Ablation Study: Feature Grubu Cikarildiginda MAPE Degisimi')

    # Tam model cizgisi
    ax.axvline(values[0], color='green', linestyle='--', alpha=0.5, label=f'Tam Model: {values[0]:.2f}%')
    ax.legend()

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}%', va='center', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '06_ablation_study.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


def plot_overfitting_check(results):
    """Train vs Test MAPE karsilastirmasi."""
    setup_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    models = []
    train_mapes = []
    test_mapes = []

    for r in results:
        if r.get('metrics_train') and r['name'] != 'Naive Baseline':
            models.append(r['name'])
            train_mapes.append(r['metrics_train']['mape'])
            test_mapes.append(r['metrics_test']['mape'])

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_mapes, width, label='Train MAPE', color='#2196F3', edgecolor='white')
    bars2 = ax.bar(x + width/2, test_mapes, width, label='Test MAPE', color='#FF5722', edgecolor='white')

    ax.set_ylabel('MAPE (%)')
    ax.set_title('Overfitting Kontrolu: Train vs Test MAPE')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    for bar, val in zip(bars1, train_mapes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, test_mapes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '07_overfitting_check.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


def generate_all_plots(data, results, comparison_df, ablation_results,
                       turkey_history, carbon_df):
    """Tum grafikleri olustur."""
    print("\n=== GRAFIKLER OLUSTURULUYOR ===")

    plot_correlation_heatmap(data)
    plot_feature_importance(results)
    plot_model_comparison(comparison_df)

    best = min([r for r in results if r['name'] != 'Naive Baseline'],
               key=lambda x: x['metrics_test']['mape'])
    plot_residual_analysis(data['y_test'], best['pred_test'], best['name'])
    plot_turkey_forecast(turkey_history, carbon_df)

    if ablation_results:
        plot_ablation(ablation_results)

    plot_overfitting_check(results)

    print(f"\n  Tum grafikler kaydedildi: {OUTPUT_DIR}")
