"""README görsellerini üretir: her grafik açık ve koyu tema olmak üzere iki varyant.

Kullanım:
    python3 docs/make_figures.py

Çıktı: docs/assets/*-light.png ve docs/assets/*-dark.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parent.parent
SONUC = ROOT / "pipeline" / "sonuclar" / "model_karsilastirma"
KARBON = ROOT / "pipeline" / "asama3_karbon_tahmini" / "ciktilar" / "karbon_tahmin_2026_2030.csv"
ASSETS = ROOT / "docs" / "assets"

# GitHub'ın kendi arka plan renkleriyle birebir eşleşen iki tema.
THEMES = {
    "light": {
        "bg": "#ffffff",
        "fg": "#1f2328",
        "muted": "#59636e",
        "grid": "#d1d9e0",
        "bar": "#afb8c1",
        "accent": "#0f766e",
        "accent_soft": "#5eead4",
        "warn": "#bc4c00",
    },
    "dark": {
        "bg": "#0d1117",
        "fg": "#e6edf3",
        "muted": "#9198a1",
        "grid": "#30363d",
        "bar": "#484f58",
        "accent": "#2dd4bf",
        "accent_soft": "#134e4a",
        "warn": "#f0883e",
    },
}


def style(theme: dict) -> None:
    plt.rcParams.update({
        "figure.facecolor": theme["bg"],
        "axes.facecolor": theme["bg"],
        "savefig.facecolor": theme["bg"],
        "text.color": theme["fg"],
        "axes.labelcolor": theme["fg"],
        "axes.edgecolor": theme["grid"],
        "xtick.color": theme["muted"],
        "ytick.color": theme["muted"],
        "grid.color": theme["grid"],
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "600",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
    })


def save(fig, name: str, variant: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / f"{name}-{variant}.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  ✓ {out.relative_to(ROOT)}")


def panel_r2(ax, frame: pd.DataFrame, title: str, theme: dict) -> None:
    """Tek aşamanın model × R² yatay bar paneli; en iyi model vurgulanır."""
    frame = frame.set_index("model").sort_values("r2", ascending=False)
    best = frame["r2"].idxmax()
    colors = [theme["accent"] if m == best else theme["bar"] for m in frame.index]

    positions = range(len(frame))
    ax.barh(positions, frame["r2"], color=colors, height=0.62, zorder=3)
    ax.set_yticks(list(positions), frame.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.14)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.grid(axis="x", linewidth=0.7, zorder=0)
    ax.set_title(title, loc="left", pad=14, color=theme["fg"])
    ax.set_xlabel("R²  (walk-forward CV)", color=theme["muted"], fontsize=10)

    for y, (model, row) in enumerate(frame.iterrows()):
        weight = "bold" if model == best else "normal"
        color = theme["accent"] if model == best else theme["muted"]
        ax.text(row["r2"] + 0.02, y, f"{row['r2']:.4f}",
                va="center", fontsize=10, fontweight=weight, color=color)


def fig_model_comparison(variant: str, theme: dict) -> None:
    """Aşama 2 (elektrik) ve Aşama 3 (karbon) için 7 modelin karşılaştırması."""
    style(theme)
    elektrik = pd.read_csv(SONUC / "asama2_model_karsilastirma.csv")
    karbon = pd.read_csv(SONUC / "asama3_model_karsilastirma.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    panel_r2(axes[0], elektrik, "Aşama 2 · Elektrik talebi (TWh)", theme)
    panel_r2(axes[1], karbon, "Aşama 3 · Karbon ayak izi (MtCO₂)", theme)
    fig.subplots_adjust(wspace=0.34)
    save(fig, "model-comparison", variant)


def fig_carbon_projection(variant: str, theme: dict) -> None:
    """2026-2030 karbon projeksiyonu: en iyi model vurgulu, diğerleri soluk."""
    style(theme)
    data = pd.read_csv(KARBON)
    years = data["year"]
    best_model = data["EN_IYI_MODEL"].iloc[0]

    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    others = [c for c in data.columns
              if c.endswith("_MtCO2") and c not in (f"{best_model}_MtCO2", "EN_IYI_MtCO2")]

    for index, column in enumerate(others):
        ax.plot(years, data[column], color=theme["bar"], linewidth=1.2, alpha=0.65,
                zorder=2, label=f"diğer {len(others)} model" if index == 0 else None)

    ax.plot(years, data["EN_IYI_MtCO2"], color=theme["accent"], linewidth=2.8,
            marker="o", markersize=6, zorder=4, label=f"{best_model} (en iyi)")
    ax.fill_between(years, data[others].min(axis=1), data[others].max(axis=1),
                    color=theme["accent"], alpha=0.07, zorder=1)

    for year, value in zip(years, data["EN_IYI_MtCO2"]):
        ax.annotate(f"{value:.0f}", (year, value), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9.5,
                    fontweight="bold", color=theme["accent"])

    ax.set_title("Türkiye · CO₂ salımı projeksiyonu, 2026–2030",
                 loc="left", pad=14, color=theme["fg"])
    ax.set_ylabel("MtCO₂", color=theme["muted"], fontsize=10)
    ax.set_xticks(years)
    ax.grid(axis="y", linewidth=0.7, zorder=0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, loc="upper left",
              fontsize=10, labelcolor=theme["fg"], ncol=2, columnspacing=1.6)
    save(fig, "carbon-projection", variant)


def fig_stage1_features(variant: str, theme: dict) -> None:
    """Aşama 1'de her dışsal değişken için ulaşılan en iyi R²."""
    style(theme)
    data = pd.read_csv(SONUC / "asama1_model_karsilastirma.csv")
    best = (data[data["en_iyi"]]
            .sort_values("r2")
            .loc[:, ["feature", "model", "r2"]])

    colors = [theme["warn"] if r2 < 0.5 else
              theme["accent_soft"] if r2 < 0.9 else
              theme["accent"]
              for r2 in best["r2"]]

    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    positions = range(len(best))
    ax.barh(positions, best["r2"].clip(lower=0), color=colors, height=0.66, zorder=3)
    ax.set_yticks(list(positions), best["feature"], fontsize=10)
    ax.set_xlim(0, 1.2)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.grid(axis="x", linewidth=0.7, zorder=0)
    ax.set_xlabel("En iyi R²", color=theme["muted"], fontsize=10)
    ax.set_title("Aşama 1 · Dışsal değişkenlerin tahmin edilebilirliği",
                 loc="left", pad=14, color=theme["fg"])

    for y, (_, row) in enumerate(best.iterrows()):
        label = f"{row['r2']:.3f}  ·  {row['model']}"
        ax.text(max(row["r2"], 0) + 0.02, y, label, va="center",
                fontsize=9.5, color=theme["muted"])

    save(fig, "stage1-features", variant)


def main() -> None:
    for variant, theme in THEMES.items():
        print(f"[{variant}]")
        fig_model_comparison(variant, theme)
        fig_carbon_projection(variant, theme)
        fig_stage1_features(variant, theme)


if __name__ == "__main__":
    main()
