
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
np.random.seed(42)

from src.o1_data_pipeline      import load_and_clean
from src.o2_feature_engineering import build_features
from src.o3_train_models        import train
from src.o4_evaluation          import (compute_metrics, build_price_table,
                                        plot_model_comparison,
                                        plot_actual_vs_predicted,
                                        plot_feature_importance, plot_shap)
from src.o5_simulation          import run_simulation

OUT_DIR    = "outputs"
MODEL_DIR  = "models"
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 65)
print("  DYNAMIC PRICING – TECHNISCHER TEIL  (Gattas, 2025)")
print("=" * 65)

# ── 1. Daten laden ────────────────────────────────────────────────────────────
print("\n[1/5] Datenpipeline …")
raw = load_and_clean("data/Air_Conditioners.csv", "data/amazon.csv")

# ── 2. Features bauen ─────────────────────────────────────────────────────────
print("\n[2/5] Feature Engineering …")
df, le, features, target = build_features(raw)

# ── 3. Modelle trainieren ─────────────────────────────────────────────────────
print("\n[3/5] Modelltraining …")
rf, xgb_model, X_train, X_test, y_train, y_test, xgb_label = train(
    df, features, target, models_dir=MODEL_DIR
)

# ── 4. Evaluation ─────────────────────────────────────────────────────────────
print("\n[4/5] Evaluation …")
rf_res  = compute_metrics(rf,        "Random Forest", X_train, y_train, X_test, y_test)
xgb_res = compute_metrics(xgb_model, xgb_label,       X_train, y_train, X_test, y_test)

winner     = xgb_label if xgb_res["MAE"] < rf_res["MAE"] else "Random Forest"
best_model = xgb_model if winner == xgb_label else rf
best_res   = xgb_res   if winner == xgb_label else rf_res
print(f"\n  ✓ Bestes Modell: {winner}")

results = [rf_res, xgb_res]
plot_model_comparison(results, OUT_DIR)
plot_actual_vs_predicted(best_res, OUT_DIR)
plot_feature_importance(rf, xgb_model, features, xgb_label, OUT_DIR)
shap_ok = plot_shap(xgb_model, X_test, features, OUT_DIR)

price_table = build_price_table(df, X_test, y_test, best_model, features, n=50)
csv_path = os.path.join(OUT_DIR, "preisempfehlungen.csv")
price_table.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")
print(price_table.head(10).to_string(index=False))

# ── 5. Simulation ─────────────────────────────────────────────────────────────
print("\n[5/5] Simulation …")
sim_df, summary = run_simulation(X_test, y_test, best_model, out_dir=OUT_DIR)

print("\n" + "=" * 65)
print("  ERGEBNISZUSAMMENFASSUNG")
print("=" * 65)
print(f"""
  Bestes Modell : {winner}
  MAE           : ₹{best_res['MAE']:,.0f}
  RMSE          : ₹{best_res['RMSE']:,.0f}
  R²            : {best_res['R2']:.4f}

  Umsatz-Uplift : +{summary['Umsatz-Uplift (%)']:.1f} %
  Margin-Uplift : +{summary['Margin-Uplift (%)']:.1f} %

  Outputs:
    outputs/preisempfehlungen.csv
    outputs/simulation_ergebnisse.csv
    outputs/plot1_modellvergleich.png
    outputs/plot2_actual_vs_predicted.png
    outputs/plot3_feature_importance.png
    {'outputs/plot4_shap.png' if shap_ok else '(plot4_shap – shap fehlt)'}
    outputs/plot5_umsatz.png
    outputs/plot6_gewinn.png
    outputs/plot7_stabilitaet.png

  Dashboard starten:
    streamlit run dashboard.py
""")
print("  ✓ Fertig!")
print("=" * 65)
