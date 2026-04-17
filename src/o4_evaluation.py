"""
04_evaluation.py
─────────────────
Berechnet MAE, RMSE, R² und erstellt alle Evaluations-Plots.
Gibt außerdem eine Tabelle mit ursprünglichem vs. empfohlenem Preis zurück.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


FEAT_LABELS = {
    "log_act_price":    "Listenpreis (log)",
    "disc_pct":         "Rabatttiefe (%)",
    "rating_num":       "Produktbewertung",
    "log_rating_count": "Anzahl Bewertungen (log)",
    "cat_enc":          "Produktkategorie",
    "price_ratio":      "Preis-Ratio",
    "inv_proxy":        "Lagerbestand-Proxy",
}



def compute_metrics(model, name, X_tr, y_tr, X_te, y_te):
    """Berechnet MAE, RMSE, R² und CV-MAE (zurück auf ₹-Skala)."""
    pred = model.predict(X_te)
    y_r  = np.expm1(y_te)
    p_r  = np.expm1(pred)

    mae  = mean_absolute_error(y_r, p_r)
    rmse = np.sqrt(mean_squared_error(y_r, p_r))
    r2   = r2_score(y_r, p_r)
    cv   = -cross_val_score(model, X_tr, y_tr, cv=3,
                            scoring="neg_mean_absolute_error").mean()
    cv_inr = np.expm1(cv) - 1

    print(f"\n  ── {name} ──")
    print(f"     MAE  (Test):    ₹{mae:,.0f}")
    print(f"     RMSE (Test):    ₹{rmse:,.0f}")
    print(f"     R²   (Test):    {r2:.4f}")
    print(f"     CV-MAE (Train): ₹{cv_inr:,.0f}")

    return {"Model": name, "MAE": mae, "RMSE": rmse,
            "R2": r2, "CV_MAE": cv_inr,
            "y_true": y_r.values, "y_pred": p_r}


# ── Preisempfehlungs-Tabelle ──────────────────────────────────────────────────

def build_price_table(df, X_test, y_test, best_model, features, n=50):
    
    idx      = X_test.index[:n]
    pred_log = best_model.predict(X_test.loc[idx])
    pred_inr = np.expm1(pred_log)
    orig_inr = np.expm1(y_test.loc[idx].values)

    table = pd.DataFrame({
        "Produktname":         df.loc[idx, "product_name"].values,
        "Ursprünglicher Preis (₹)": orig_inr.round(0).astype(int),
        "Empfohlener Preis (₹)":    pred_inr.round(0).astype(int),
        "Differenz (₹)":            (pred_inr - orig_inr).round(0).astype(int),
        "Änderung (%)":             ((pred_inr - orig_inr) / orig_inr * 100).round(1),
    })
    return table


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_model_comparison(results, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Abbildung 1: Modellvergleich – Random Forest vs. XGBoost",
                 fontsize=13, fontweight="bold")

    metrics = ["MAE",   "RMSE",  "R2"]
    labels  = ["MAE (₹)", "RMSE (₹)", "R² Score"]
    colors  = ["#2196F3", "#FF9800"]

    for i, (m, lbl) in enumerate(zip(metrics, labels)):
        vals   = [r[m] for r in results]
        models = [r["Model"] for r in results]
        bars   = axes[i].bar(models, vals, color=colors,
                             width=0.4, edgecolor="white")
        axes[i].set_title(lbl, fontweight="bold")
        axes[i].set_ylabel(lbl)
        for bar, v in zip(bars, vals):
            axes[i].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() * 1.01,
                         f"{v:.3f}" if m == "R2" else f"₹{v:,.0f}",
                         ha="center", va="bottom", fontsize=9)
        axes[i].set_ylim(0, max(vals) * 1.2)

    plt.tight_layout()
    path = os.path.join(out_dir, "plot1_modellvergleich.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_actual_vs_predicted(best_res, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(best_res["y_true"], best_res["y_pred"],
               alpha=0.4, s=20, color="#1565C0")
    lim = max(best_res["y_true"].max(), best_res["y_pred"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfekte Vorhersage")
    ax.set_xlabel("Tatsächlicher Preis (₹)")
    ax.set_ylabel("Empfohlener Preis (₹)")
    ax.set_title(
        f"Abbildung 2: Actual vs. Predicted – {best_res['Model']}\n"
        f"R² = {best_res['R2']:.4f}  |  MAE = ₹{best_res['MAE']:,.0f}",
        fontweight="bold")
    ax.legend()
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    plt.tight_layout()
    path = os.path.join(out_dir, "plot2_actual_vs_predicted.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(rf_model, xgb_model, features, xgb_label, out_dir):
    fi_rf  = pd.Series(rf_model.feature_importances_,  index=features)\
               .rename(index=FEAT_LABELS).sort_values()
    fi_xgb = pd.Series(xgb_model.feature_importances_, index=features)\
               .rename(index=FEAT_LABELS).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Abbildung 3: Feature Importance – Einflussfaktoren auf den Preis",
                 fontweight="bold")

    for ax, fi, col, title in zip(axes,
                                   [fi_xgb, fi_rf],
                                   ["#FF9800", "#2196F3"],
                                   [xgb_label, "Random Forest"]):
        bars = ax.barh(fi.index, fi.values, color=col, edgecolor="white")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Importance")
        for bar, v in zip(bars, fi.values):
            ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "plot3_feature_importance.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_shap(xgb_model, X_test, features, out_dir):
    if not (SHAP_AVAILABLE and XGBOOST_AVAILABLE):
        print("  SHAP übersprungen (pip install shap xgboost)")
        return False
    try:
        explainer   = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test,
                          feature_names=[FEAT_LABELS[f] for f in features],
                          show=False, plot_type="dot")
        plt.title("Abbildung 4: SHAP-Werte – XGBoost Entscheidungserklärung",
                  fontweight="bold")
        plt.tight_layout()
        path = os.path.join(out_dir, "plot4_shap.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")
        return True
    except Exception as e:
        print(f"  SHAP Fehler: {e}")
        return False
