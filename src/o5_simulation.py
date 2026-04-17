
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_simulation(X_test, y_test, best_model, n_products=300,
                   n_steps=30, cost_rate=0.60, out_dir="../outputs"):
  
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(0)

    # ── Produkte auswählen ────────────────────────────────────────────────
    base_idx  = np.random.choice(len(X_test), n_products, replace=False)
    X_sim     = X_test.iloc[base_idx].copy().reset_index(drop=True)
    y_sim     = np.expm1(y_test.iloc[base_idx].values)   # echter Preis ₹

    static_price = np.expm1(X_sim["log_act_price"].values) * 0.85
    cost         = np.expm1(X_sim["log_act_price"].values) * cost_rate

    # ── Arrays für Ergebnisse ─────────────────────────────────────────────
    static_rev     = np.zeros(n_steps)
    dynamic_rev    = np.zeros(n_steps)
    static_margin  = np.zeros(n_steps)
    dynamic_margin = np.zeros(n_steps)

    # ── Simulation ────────────────────────────────────────────────────────
    for t in range(n_steps):
        # Marktvolatilität: ±3% tägliche Preisschwankung
        noise = np.random.normal(0, 0.03, n_products)
        # Saisonalität: 14-Tage-Zyklus (Wochenend-/Sale-Effekte)
        demand_shock = 1 + 0.15 * np.sin(2 * np.pi * t / 14)

        # Dynamic: Modell reagiert auf veränderte Marktbedingungen
        X_step = X_sim.copy()
        X_step["disc_pct"]    = (X_step["disc_pct"]    * (1 + noise)).clip(0, 90)
        X_step["price_ratio"] = (X_step["price_ratio"] * (1 + noise * 0.5)).clip(0.3, 1.0)
        dynamic_pred = np.expm1(best_model.predict(X_step))

        # Nachfragefunktion: qty ∝ (Preis / Marktpreis)^(-1.5)
        # → Wenn Preis steigt, sinkt die Nachfrage (Preiselastizität)
        qty_static  = demand_shock * (static_price  / y_sim) ** (-1.5)
        qty_dynamic = demand_shock * (dynamic_pred  / y_sim) ** (-1.5)
        qty_static  = np.clip(qty_static,  0.3, 5)
        qty_dynamic = np.clip(qty_dynamic, 0.3, 5)

        # Umsatz = Preis × Menge
        static_rev[t]     = (static_price  * qty_static).sum()
        dynamic_rev[t]    = (dynamic_pred  * qty_dynamic).sum()

        # Marge = (Preis - Einkaufspreis) × Menge
        static_margin[t]  = ((static_price  - cost) * qty_static).sum()
        dynamic_margin[t] = ((dynamic_pred  - cost) * qty_dynamic).sum()

    # ── Ergebnisse zusammenfassen ─────────────────────────────────────────
    steps = np.arange(1, n_steps + 1)

    results_df = pd.DataFrame({
        "Tag":                    steps,
        "Umsatz_Statisch (₹)":   static_rev.round(0).astype(int),
        "Umsatz_Dynamisch (₹)":  dynamic_rev.round(0).astype(int),
        "Marge_Statisch (₹)":    static_margin.round(0).astype(int),
        "Marge_Dynamisch (₹)":   dynamic_margin.round(0).astype(int),
    })

    total_rev_static   = static_rev.sum()
    total_rev_dynamic  = dynamic_rev.sum()
    total_mar_static   = static_margin.sum()
    total_mar_dynamic  = dynamic_margin.sum()
    rev_uplift         = (total_rev_dynamic  - total_rev_static)  / total_rev_static  * 100
    margin_uplift      = (total_mar_dynamic  - total_mar_static)  / total_mar_static  * 100

    summary = {
        "Gesamtumsatz Statisch (₹)":   total_rev_static,
        "Gesamtumsatz Dynamisch (₹)":  total_rev_dynamic,
        "Umsatz-Uplift (%)":           rev_uplift,
        "Gesamtmarge Statisch (₹)":    total_mar_static,
        "Gesamtmarge Dynamisch (₹)":   total_mar_dynamic,
        "Margin-Uplift (%)":           margin_uplift,
    }

    print(f"\n[Simulation] Gesamtumsatz  Statisch:  ₹{total_rev_static:>12,.0f}")
    print(f"[Simulation] Gesamtumsatz  Dynamisch: ₹{total_rev_dynamic:>12,.0f}")
    print(f"[Simulation] Umsatz-Uplift:            +{rev_uplift:.1f} %")
    print(f"[Simulation] Gesamtmarge   Statisch:  ₹{total_mar_static:>12,.0f}")
    print(f"[Simulation] Gesamtmarge   Dynamisch: ₹{total_mar_dynamic:>12,.0f}")
    print(f"[Simulation] Margin-Uplift:            +{margin_uplift:.1f} %")

    # ── CSV speichern ─────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "simulation_ergebnisse.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"[Simulation] ✓ CSV gespeichert: {csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    _plot_revenue(steps, static_rev, dynamic_rev, rev_uplift, out_dir)
    _plot_margin(steps, static_margin, dynamic_margin, margin_uplift, out_dir)
    _plot_stability(steps, X_sim, y_sim, best_model, static_price,
                    n_products, n_steps, out_dir)

    return results_df, summary


# ── Private Plot-Funktionen ───────────────────────────────────────────────────

def _plot_revenue(steps, static_rev, dynamic_rev, uplift, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Abbildung 5: Umsatzentwicklung – Dynamic vs. Statisches Pricing",
                 fontweight="bold")

    axes[0].plot(steps, dynamic_rev / 1e6, color="#E53935", lw=2,
                 label="Dynamic Pricing")
    axes[0].plot(steps, static_rev  / 1e6, color="#1E88E5", lw=2,
                 linestyle="--", label="Statisches Pricing")
    axes[0].fill_between(steps, static_rev / 1e6, dynamic_rev / 1e6,
                          alpha=0.15, color="#E53935")
    axes[0].set_xlabel("Tag"); axes[0].set_ylabel("Tagesumsatz (₹ Mio.)")
    axes[0].set_title("Täglicher Umsatz", fontweight="bold")
    axes[0].legend()

    axes[1].plot(steps, np.cumsum(dynamic_rev) / 1e6, color="#E53935", lw=2,
                 label="Dynamic Pricing")
    axes[1].plot(steps, np.cumsum(static_rev)  / 1e6, color="#1E88E5", lw=2,
                 linestyle="--", label="Statisches Pricing")
    axes[1].set_xlabel("Tag"); axes[1].set_ylabel("Kumulierter Umsatz (₹ Mio.)")
    axes[1].set_title(f"Kumulierter Umsatz  (+{uplift:.1f} % Uplift)", fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "plot5_umsatz.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def _plot_margin(steps, static_margin, dynamic_margin, uplift, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Abbildung 6: Gewinnprognose – Dynamic vs. Statisches Pricing",
                 fontweight="bold")

    axes[0].plot(steps, dynamic_margin / 1e6, color="#E53935", lw=2,
                 label="Dynamic Pricing")
    axes[0].plot(steps, static_margin  / 1e6, color="#1E88E5", lw=2,
                 linestyle="--", label="Statisches Pricing")
    axes[0].fill_between(steps, static_margin / 1e6, dynamic_margin / 1e6,
                          alpha=0.15, color="#E53935")
    axes[0].set_xlabel("Tag"); axes[0].set_ylabel("Tagesgewinn (₹ Mio.)")
    axes[0].set_title("Tägliche Marge", fontweight="bold")
    axes[0].legend()

    axes[1].plot(steps, np.cumsum(dynamic_margin) / 1e6, color="#E53935", lw=2,
                 label="Dynamic Pricing")
    axes[1].plot(steps, np.cumsum(static_margin)  / 1e6, color="#1E88E5", lw=2,
                 linestyle="--", label="Statisches Pricing")
    axes[1].set_xlabel("Tag"); axes[1].set_ylabel("Kumulierter Gewinn (₹ Mio.)")
    axes[1].set_title(f"Kumulierter Gewinn  (+{uplift:.1f} % Uplift)", fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "plot6_gewinn.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def _plot_stability(steps, X_sim, y_sim, best_model, static_price,
                    n_products, n_steps, out_dir):
    np.random.seed(1)
    dyn_std = []
    for t in range(n_steps):
        noise = np.random.normal(0, 0.03, n_products)
        X_step = X_sim.copy()
        X_step["disc_pct"]    = (X_step["disc_pct"]    * (1 + noise)).clip(0, 90)
        X_step["price_ratio"] = (X_step["price_ratio"] * (1 + noise * 0.5)).clip(0.3, 1.0)
        dyn_std.append(np.expm1(best_model.predict(X_step)).std())

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, dyn_std,
            color="#E53935", lw=2, label="Dynamic Pricing – Preis-Std.")
    ax.axhline(static_price.std(), color="#1E88E5", lw=2,
               linestyle="--", label="Statisches Pricing – Preis-Std.")
    ax.set_xlabel("Tag"); ax.set_ylabel("Standardabweichung (₹)")
    ax.set_title("Abbildung 7: Preisvolatilität – Systemstabilität",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "plot7_stabilitaet.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")
