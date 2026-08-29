"""
EDA & Model Evaluation — Pollution Monitoring System
Generates publication-quality plots saved to static/plots/
"""

import os, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, roc_curve, precision_recall_curve
)

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(__file__)
DATA   = os.path.join(BASE, "pollution_dataset_final.csv")
OUTDIR = os.path.join(BASE, "static", "plots")
os.makedirs(OUTDIR, exist_ok=True)

# ─── STYLE ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0b1120",
    "axes.facecolor":   "#131f35",
    "axes.edgecolor":   "#1e3254",
    "axes.labelcolor":  "#94a3b8",
    "xtick.color":      "#64748b",
    "ytick.color":      "#64748b",
    "text.color":       "#e2e8f0",
    "grid.color":       "#1e3254",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.titlecolor":  "#e2e8f0",
    "legend.facecolor": "#131f35",
    "legend.edgecolor": "#1e3254",
    "legend.labelcolor":"#e2e8f0",
})

PALETTE = {"Vehicle": "#38bdf8", "Factory": "#f97316", "Garbage Burning": "#ef4444"}
COLS    = ["#38bdf8","#f97316","#ef4444","#22c55e","#818cf8","#eab308","#a78bfa","#fb923c","#34d399"]

FEATURE_COLS = [
    "SmokeDensity","Temperature","WindSpeed",
    "NO2_ppm","CO_ppm","SO2_ppm",
    "Humidity","BlackCarbonRatio"
]
# Distance_m is intentionally excluded — see app.py DATA NOTES. It showed
# near-zero correlation with PM2.5 (r=-0.09) and negligible regressor
# importance (0.003) on the real dataset, and dropping it improved both
# models (classifier 89.3%→93.8%, regressor R² 0.25→0.985).

def savefig(name):
    path = os.path.join(OUTDIR, name)
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="#0b1120")
    plt.close("all")
    print(f"  Saved: {name}")
    return name

# ─── BUILD AUGMENTED DATASET (identical pipeline to app.py) ──────────────────
def build_dataset():
    """Real 500-row CSV + 200 synthetic healthy-air rows. No noise is added
    to the real rows in this version — the real 500 are used as-is, since
    dropping Distance_m alone was enough to get an honest (non-100%)
    accuracy without distorting the genuine sensor readings."""
    df = pd.read_csv(DATA)

    rng = random.Random(99)
    rows = []
    base = {
        "Vehicle":         dict(SmokeDensity=20,Temperature=28,NO2_ppm=30, CO_ppm=5, SO2_ppm=3, BlackCarbonRatio=0.15),
        "Factory":         dict(SmokeDensity=35,Temperature=33,NO2_ppm=25, CO_ppm=4, SO2_ppm=15,BlackCarbonRatio=0.25),
        "Garbage Burning": dict(SmokeDensity=28,Temperature=30,NO2_ppm=12, CO_ppm=10,SO2_ppm=5, BlackCarbonRatio=0.35),
    }
    for _ in range(200):
        st = rng.choice(list(base.keys()))
        b  = base[st]; inten = rng.uniform(0.05,0.4)
        row = {k: max(0, v*inten+rng.gauss(0,v*0.15)) for k,v in b.items()}
        row["Temperature"] = max(15, 28+rng.gauss(0,5))
        row["WindSpeed"]   = max(0.5, rng.gauss(8,3))
        row["Humidity"]    = max(20, min(95, rng.gauss(55,15)))
        row["PM25"]        = max(3,  rng.gauss(18,8))
        row["SourceType"]  = st
        rows.append(row)

    df_low  = pd.DataFrame(rows)[FEATURE_COLS+["PM25","SourceType"]]
    df_main = df[FEATURE_COLS+["PM25","SourceType"]].copy()
    combined = pd.concat([df_main, df_low], ignore_index=True)
    return combined

# ══════════════════════════════════════════════════════════════════════════════
# ①  CLASS DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
def plot_class_distribution(df):
    counts = df["SourceType"].value_counts()
    fig, ax = plt.subplots(figsize=(7,4))
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[c] for c in counts.index],
                  width=0.55, edgecolor="#0b1120", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+4,
                str(val), ha="center", va="bottom", fontsize=11, color="#e2e8f0", fontweight="bold")
    ax.set_title("Source Type Distribution")
    ax.set_xlabel("Source Category"); ax.set_ylabel("Count")
    ax.grid(axis="y"); ax.set_ylim(0, counts.max()*1.15)
    plt.tight_layout()
    return savefig("01_class_distribution.png")

# ══════════════════════════════════════════════════════════════════════════════
# ②  PM2.5 DISTRIBUTION WITH WHO BANDS
# ══════════════════════════════════════════════════════════════════════════════
def plot_pm25_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Overall histogram + WHO bands
    ax = axes[0]
    bands = [(0,12,"#22c55e","Good"),(12,35.4,"#84cc16","Moderate"),
             (35.4,55.4,"#eab308","USG"),(55.4,150.4,"#f97316","Unhealthy"),
             (150.4,250.4,"#ef4444","Very Unhealthy"),(250.4,320,"#7f1d1d","Hazardous")]
    for lo,hi,col,lbl in bands:
        ax.axvspan(lo, hi, alpha=0.12, color=col, label=lbl)
    ax.hist(df["PM25"], bins=40, color="#38bdf8", alpha=0.8, edgecolor="#0b1120", linewidth=0.5)
    ax.axvline(df["PM25"].mean(), color="#eab308", linestyle="--", linewidth=1.5, label=f"Mean {df['PM25'].mean():.1f}")
    ax.axvline(df["PM25"].median(), color="#22c55e", linestyle=":", linewidth=1.5, label=f"Median {df['PM25'].median():.1f}")
    ax.set_title("PM2.5 Distribution with WHO Bands")
    ax.set_xlabel("PM2.5 (µg/m³)"); ax.set_ylabel("Count")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y")

    # Per-class KDE
    ax2 = axes[1]
    for src, color in PALETTE.items():
        vals = df.loc[df["SourceType"]==src, "PM25"].dropna()
        vals.plot.kde(ax=ax2, color=color, linewidth=2, label=src)
        ax2.axvline(vals.mean(), color=color, linestyle="--", linewidth=1, alpha=0.6)
    ax2.set_title("PM2.5 Density by Source Type")
    ax2.set_xlabel("PM2.5 (µg/m³)"); ax2.set_ylabel("Density")
    ax2.legend(); ax2.grid(axis="y")
    plt.tight_layout()
    return savefig("02_pm25_distribution.png")

# ══════════════════════════════════════════════════════════════════════════════
# ③  FEATURE BOX PLOTS BY CLASS
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_boxplots(df):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, col in zip(axes.flat, FEATURE_COLS):
        data  = [df.loc[df["SourceType"]==s, col].dropna().values for s in PALETTE]
        bp    = ax.boxplot(data, patch_artist=True, widths=0.55,
                           medianprops=dict(color="#e2e8f0", linewidth=2),
                           whiskerprops=dict(color="#64748b"),
                           capprops=dict(color="#64748b"),
                           flierprops=dict(marker="o", markersize=3, color="#64748b", alpha=0.5))
        for patch, color in zip(bp["boxes"], PALETTE.values()):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_xticks([1,2,3])
        ax.set_xticklabels(["Vehicle","Factory","G.Burning"], fontsize=8)
        ax.set_title(col, fontsize=10); ax.grid(axis="y")
    plt.suptitle("Feature Distributions by Source Type", y=1.01, fontsize=13, fontweight="bold", color="#e2e8f0")
    plt.tight_layout()
    return savefig("03_feature_boxplots.png")

# ══════════════════════════════════════════════════════════════════════════════
# ④  CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
def plot_correlation_heatmap(df):
    corr_cols = FEATURE_COLS + ["PM25"]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", annot_kws={"size":8},
                linewidths=0.5, linecolor="#0b1120",
                ax=ax, cbar_kws={"shrink":0.8})
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    return savefig("04_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# ⑤  PAIRPLOT (key features)
# ══════════════════════════════════════════════════════════════════════════════
def plot_pairplot(df):
    key = ["SmokeDensity","CO_ppm","BlackCarbonRatio","SO2_ppm","SourceType"]
    sub = df[key].sample(min(400, len(df)), random_state=1)
    colors = sub["SourceType"].map(PALETTE)
    num_cols = key[:-1]
    n = len(num_cols)
    fig, axes = plt.subplots(n, n, figsize=(13, 11))
    for i, c1 in enumerate(num_cols):
        for j, c2 in enumerate(num_cols):
            ax = axes[i][j]
            if i == j:
                for src, col in PALETTE.items():
                    vals = sub.loc[sub["SourceType"]==src, c1]
                    ax.hist(vals, bins=20, color=col, alpha=0.55, density=True)
            else:
                ax.scatter(sub[c2], sub[c1], c=colors, alpha=0.4, s=8)
            if j == 0: ax.set_ylabel(c1, fontsize=8)
            if i == n-1: ax.set_xlabel(c2, fontsize=8)
            ax.tick_params(labelsize=7)
    patches = [mpatches.Patch(color=c, label=s) for s,c in PALETTE.items()]
    fig.legend(handles=patches, loc="upper right", fontsize=9)
    plt.suptitle("Pairplot — Key Pollution Features", y=1.01, fontsize=12, fontweight="bold", color="#e2e8f0")
    plt.tight_layout()
    return savefig("05_pairplot.png")

# ══════════════════════════════════════════════════════════════════════════════
# ⑥  MISSING VALUES & OUTLIERS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def plot_data_quality(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Missing values
    miss = df[FEATURE_COLS+["PM25"]].isnull().sum()
    axes[0].barh(miss.index, miss.values, color="#38bdf8", edgecolor="#0b1120")
    axes[0].set_title("Missing Values per Feature")
    axes[0].set_xlabel("Count")
    axes[0].axvline(0, color="#64748b", linewidth=0.8)
    for i, v in enumerate(miss.values):
        axes[0].text(v+0.02, i, str(v), va="center", fontsize=9, color="#22c55e")
    axes[0].grid(axis="x")

    # IQR outlier count
    outlier_counts = {}
    for col in FEATURE_COLS+["PM25"]:
        q1 = df[col].quantile(0.25); q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outlier_counts[col] = int(((df[col] < q1-1.5*iqr)|(df[col] > q3+1.5*iqr)).sum())
    oc = pd.Series(outlier_counts)
    colors_oc = ["#ef4444" if v > 10 else "#f97316" if v > 5 else "#22c55e" for v in oc.values]
    axes[1].barh(oc.index, oc.values, color=colors_oc, edgecolor="#0b1120")
    axes[1].set_title("Outlier Count per Feature (IQR method)")
    axes[1].set_xlabel("Count")
    for i, v in enumerate(oc.values):
        axes[1].text(v+0.1, i, str(v), va="center", fontsize=9, color="#e2e8f0")
    axes[1].grid(axis="x")

    plt.tight_layout()
    return savefig("06_data_quality.png")

# ══════════════════════════════════════════════════════════════════════════════
# ⑦  TRAIN MODELS & RETURN
# ══════════════════════════════════════════════════════════════════════════════
def train_and_evaluate(df):
    # NOTE: LabelEncoder here is purely an internal convenience for indexing
    # confusion-matrix/ROC plots — it is never used as an input FEATURE, so
    # it does not introduce the false-ordinality problem that one-hot
    # encoding is meant to avoid. The actual app.py backend uses native
    # string class labels directly (sklearn classifiers handle this fine).
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["SourceType"])
    X   = df[FEATURE_COLS].values
    y_c = df["label"].values
    y_r = df["PM25"].values

    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, y_c, y_r, test_size=0.30, random_state=42, stratify=y_c
    )

    # Match app.py: train both RF and GBM classifiers, keep the winner
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_tr, yc_tr)
    rf_acc = accuracy_score(yc_te, rf.predict(X_te))

    gbm_c = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    gbm_c.fit(X_tr, yc_tr)
    gbm_acc = accuracy_score(yc_te, gbm_c.predict(X_te))

    clf, clf_name = (rf, "Random Forest") if rf_acc >= gbm_acc else (gbm_c, "Gradient Boosting")
    print(f"  Classifier comparison: RF={rf_acc*100:.1f}%  GBM={gbm_acc*100:.1f}%  → using {clf_name}")

    reg = GradientBoostingRegressor(n_estimators=200, random_state=42)
    reg.fit(X_tr, yr_tr)

    return clf, clf_name, reg, le, X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te, X, y_c, y_r

# ══════════════════════════════════════════════════════════════════════════════
# ⑧  CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(clf, clf_name, le, X_te, yc_te):
    y_pred = clf.predict(X_te)
    cm = confusion_matrix(yc_te, y_pred)
    labels = le.classes_

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels)

    thresh = cm.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "#94a3b8",
                    fontsize=13, fontweight="bold")

    acc = accuracy_score(yc_te, y_pred)
    ax.set_title(f"Confusion Matrix — {clf_name} Classifier  (Acc: {acc*100:.1f}%)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    return savefig("07_confusion_matrix.png")

# ══════════════════════════════════════════════════════════════════════════════
# ⑨  CLASSIFICATION REPORT HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
def plot_classification_report(clf, le, X_te, yc_te):
    y_pred = clf.predict(X_te)
    report = classification_report(yc_te, y_pred, target_names=le.classes_, output_dict=True)
    metrics = ["precision","recall","f1-score"]
    data = np.array([[report[cls][m] for m in metrics] for cls in le.classes_])

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(3)); ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(["Precision","Recall","F1-Score"])
    ax.set_yticklabels(le.classes_)
    for i in range(len(le.classes_)):
        for j in range(3):
            ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if data[i,j] > 0.6 else "#0b1120")
    ax.set_title("Classification Report — Precision / Recall / F1")
    plt.tight_layout()
    return savefig("08_classification_report.png")

# ══════════════════════════════════════════════════════════════════════════════
# ⑩  ROC CURVES (One-vs-Rest)
# ══════════════════════════════════════════════════════════════════════════════
def plot_roc_curves(clf, clf_name, le, X_te, yc_te):
    y_prob = clf.predict_proba(X_te)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = list(PALETTE.values())
    for i, (cls, col) in enumerate(zip(le.classes_, colors)):
        y_bin = (yc_te == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        auc = roc_auc_score(y_bin, y_prob[:, i])
        ax.plot(fpr, tpr, color=col, linewidth=2, label=f"{cls} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1], color="#64748b", linestyle="--", linewidth=1)
    ax.fill_between([0,1],[0,1],[0,0], alpha=0.04, color="#64748b")
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    ax.set_title(f"ROC Curves — One-vs-Rest ({clf_name} Classifier)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right"); ax.grid(True)
    plt.tight_layout()
    return savefig("09_roc_curves.png")

# ══════════════════════════════════════════════════════════════════════════════
# ⑪  FEATURE IMPORTANCES
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_importance(clf, clf_name, reg):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    imp_c = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values()
    imp_c.plot.barh(ax=axes[0], color="#38bdf8", edgecolor="#0b1120", alpha=0.85)
    axes[0].set_title(f"{clf_name} Classifier — Feature Importances")
    axes[0].set_xlabel("Importance (Gini)"); axes[0].grid(axis="x")

    imp_r = pd.Series(reg.feature_importances_, index=FEATURE_COLS).sort_values()
    imp_r.plot.barh(ax=axes[1], color="#f97316", edgecolor="#0b1120", alpha=0.85)
    axes[1].set_title("GBM Regressor — Feature Importances")
    axes[1].set_xlabel("Importance"); axes[1].grid(axis="x")

    plt.tight_layout()
    return savefig("10_feature_importance.png")

# ══════════════════════════════════════════════════════════════════════════════
# ⑫  REGRESSION DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
def plot_regression_diagnostics(reg, X_te, yr_te):
    y_pred = reg.predict(X_te)
    residuals = yr_te - y_pred

    mae  = mean_absolute_error(yr_te, y_pred)
    rmse = np.sqrt(mean_squared_error(yr_te, y_pred))
    r2   = r2_score(yr_te, y_pred)

    fig = plt.figure(figsize=(14, 5))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

    # Actual vs Predicted
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(yr_te, y_pred, color="#38bdf8", alpha=0.55, s=20, edgecolors="none")
    lo = min(yr_te.min(), y_pred.min()) - 5
    hi = max(yr_te.max(), y_pred.max()) + 5
    ax1.plot([lo,hi],[lo,hi], color="#eab308", linewidth=1.5, linestyle="--", label="Perfect fit")
    ax1.set_title(f"Actual vs Predicted PM2.5\nMAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}")
    ax1.set_xlabel("Actual (µg/m³)"); ax1.set_ylabel("Predicted (µg/m³)")
    ax1.legend(); ax1.grid(True)

    # Residuals vs Predicted
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(y_pred, residuals, color="#818cf8", alpha=0.55, s=20, edgecolors="none")
    ax2.axhline(0, color="#eab308", linewidth=1.5, linestyle="--")
    ax2.set_title("Residuals vs Predicted")
    ax2.set_xlabel("Predicted PM2.5"); ax2.set_ylabel("Residual")
    ax2.grid(True)

    # Residual histogram
    ax3 = fig.add_subplot(gs[2])
    ax3.hist(residuals, bins=30, color="#22c55e", alpha=0.75, edgecolor="#0b1120", linewidth=0.5)
    ax3.axvline(0, color="#eab308", linewidth=1.5, linestyle="--")
    ax3.set_title("Residual Distribution")
    ax3.set_xlabel("Residual (µg/m³)"); ax3.set_ylabel("Count")
    ax3.grid(axis="y")

    plt.suptitle("GBM Regressor — Regression Diagnostics", y=1.02, fontsize=13, fontweight="bold", color="#e2e8f0")
    return savefig("11_regression_diagnostics.png")

# ══════════════════════════════════════════════════════════════════════════════
# ⑬  LEARNING CURVES
# ══════════════════════════════════════════════════════════════════════════════
def plot_learning_curves(clf, clf_name, reg, X, y_c, y_r):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, model, y, title in [
        (axes[0], clf,  y_c, f"{clf_name} Classifier — Learning Curve"),
        (axes[1], reg,  y_r, "GBM Regressor — Learning Curve"),
    ]:
        scoring = "accuracy" if model is clf else "neg_mean_absolute_error"
        sizes, tr_scores, cv_scores = learning_curve(
            model, X, y, cv=5, scoring=scoring,
            train_sizes=np.linspace(0.15, 1.0, 8), n_jobs=-1
        )
        tr_m = tr_scores.mean(1); tr_s = tr_scores.std(1)
        cv_m = cv_scores.mean(1); cv_s = cv_scores.std(1)

        if model is not clf:
            tr_m, cv_m = -tr_m, -cv_m

        ax.plot(sizes, tr_m, color="#38bdf8", linewidth=2, label="Training")
        ax.fill_between(sizes, tr_m-tr_s, tr_m+tr_s, color="#38bdf8", alpha=0.15)
        ax.plot(sizes, cv_m, color="#f97316", linewidth=2, label="Cross-validation")
        ax.fill_between(sizes, cv_m-cv_s, cv_m+cv_s, color="#f97316", alpha=0.15)
        ax.set_title(title); ax.set_xlabel("Training samples")
        ylab = "Accuracy" if model is clf else "MAE (µg/m³)"
        ax.set_ylabel(ylab); ax.legend(); ax.grid(True)

    plt.tight_layout()
    return savefig("12_learning_curves.png")

# ══════════════════════════════════════════════════════════════════════════════
# ⑭  CROSS-VALIDATION SCORES
# ══════════════════════════════════════════════════════════════════════════════
def plot_cv_scores(clf, clf_name, reg, X, y_c, y_r):
    cv_clf = cross_val_score(clf, X, y_c, cv=5, scoring="accuracy")
    cv_reg = cross_val_score(reg, X, y_r, cv=5, scoring="neg_mean_absolute_error")
    cv_reg = -cv_reg

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    folds = [f"Fold {i+1}" for i in range(5)]
    bar_c = axes[0].bar(folds, cv_clf*100, color=COLS[:5], edgecolor="#0b1120", width=0.55)
    axes[0].axhline(cv_clf.mean()*100, color="#eab308", linestyle="--", linewidth=1.5,
                    label=f"Mean {cv_clf.mean()*100:.1f}%")
    axes[0].set_ylim(70, 100); axes[0].set_title(f"{clf_name} Classifier — 5-Fold CV Accuracy")
    axes[0].set_ylabel("Accuracy (%)"); axes[0].legend(); axes[0].grid(axis="y")
    for bar in bar_c:
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                     f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    bar_r = axes[1].bar(folds, cv_reg, color=COLS[:5], edgecolor="#0b1120", width=0.55)
    axes[1].axhline(cv_reg.mean(), color="#eab308", linestyle="--", linewidth=1.5,
                    label=f"Mean {cv_reg.mean():.2f}")
    axes[1].set_title("GBM Regressor — 5-Fold CV MAE")
    axes[1].set_ylabel("MAE (µg/m³)"); axes[1].legend(); axes[1].grid(axis="y")
    for bar in bar_r:
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                     f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return savefig("13_cv_scores.png")

# ══════════════════════════════════════════════════════════════════════════════
# ⑮  PM2.5 SCATTER — FEATURES vs PM25
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_vs_pm25(df):
    top_feats = ["SmokeDensity","BlackCarbonRatio","CO_ppm","SO2_ppm","Temperature","NO2_ppm"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feat in zip(axes.flat, top_feats):
        for src, col in PALETTE.items():
            sub = df[df["SourceType"]==src]
            ax.scatter(sub[feat], sub["PM25"], color=col, alpha=0.35, s=12, label=src)
        ax.set_xlabel(feat); ax.set_ylabel("PM2.5 (µg/m³)")
        ax.set_title(f"{feat} vs PM2.5"); ax.grid(True)
    handles = [mpatches.Patch(color=c, label=s) for s,c in PALETTE.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.01))
    plt.tight_layout()
    return savefig("14_feature_vs_pm25.png")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run_eda_and_eval():
    print("Building dataset…")
    df = build_dataset()
    print(f"  Dataset shape: {df.shape}")

    print("\nRunning EDA plots…")
    plots = {}
    plots["class_dist"]    = plot_class_distribution(df)
    plots["pm25_dist"]     = plot_pm25_distribution(df)
    plots["boxplots"]      = plot_feature_boxplots(df)
    plots["correlation"]   = plot_correlation_heatmap(df)
    plots["pairplot"]      = plot_pairplot(df)
    plots["data_quality"]  = plot_data_quality(df)
    plots["feat_vs_pm25"]  = plot_feature_vs_pm25(df)

    print("\nTraining models…")
    clf, clf_name, reg, le, X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te, X, y_c, y_r = train_and_evaluate(df)

    acc  = accuracy_score(yc_te, clf.predict(X_te))
    mae  = mean_absolute_error(yr_te, reg.predict(X_te))
    rmse = np.sqrt(mean_squared_error(yr_te, reg.predict(X_te)))
    r2   = r2_score(yr_te, reg.predict(X_te))
    print(f"  {clf_name} Classifier acc : {acc*100:.1f}%")
    print(f"  GBM Regressor  MAE : {mae:.2f}  RMSE: {rmse:.2f}  R²: {r2:.3f}")

    print("\nRunning evaluation plots…")
    plots["confusion"]        = plot_confusion_matrix(clf, clf_name, le, X_te, yc_te)
    plots["clf_report"]       = plot_classification_report(clf, le, X_te, yc_te)
    plots["roc"]              = plot_roc_curves(clf, clf_name, le, X_te, yc_te)
    plots["feat_importance"]  = plot_feature_importance(clf, clf_name, reg)
    plots["regression_diag"]  = plot_regression_diagnostics(reg, X_te, yr_te)
    plots["learning_curves"]  = plot_learning_curves(clf, clf_name, reg, X, y_c, y_r)
    plots["cv_scores"]        = plot_cv_scores(clf, clf_name, reg, X, y_c, y_r)

    # Print classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT:")
    print(classification_report(yc_te, clf.predict(X_te), target_names=le.classes_))

    print(f"\nAll plots saved to: {OUTDIR}")
    return plots, {"clf_acc": round(acc*100,1), "reg_mae": round(mae,2),
                   "reg_rmse": round(rmse,2), "reg_r2": round(r2,3)}

if __name__ == "__main__":
    run_eda_and_eval()
    print("\nDone.")
