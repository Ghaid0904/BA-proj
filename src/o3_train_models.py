

import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    from sklearn.ensemble import GradientBoostingRegressor


def train(df, features, target, models_dir="../models"):
    
    os.makedirs(models_dir, exist_ok=True)

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[Training] Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Random Forest ──────────────────────────────────────────────────────
    print("[Training] Random Forest – GridSearchCV läuft …")
    rf_params = {
        "n_estimators":      [100, 200],
        "max_depth":         [10, 20, None],
        "min_samples_split": [2, 5],
    }
    rf_gs = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_params, cv=3, scoring="neg_mean_absolute_error",
        n_jobs=-1, verbose=0
    )
    rf_gs.fit(X_train, y_train)
    rf_best = rf_gs.best_estimator_
    print(f"[Training] RF  beste Params: {rf_gs.best_params_}")

    # ── XGBoost ────────────────────────────────────────────────────────────
    if XGBOOST_AVAILABLE:
        print("[Training] XGBoost – GridSearchCV läuft …")
        xgb_params = {
            "n_estimators":     [200, 400],
            "learning_rate":    [0.05, 0.1],
            "max_depth":        [4, 6],
            "subsample":        [0.8],
            "colsample_bytree": [0.8],
        }
        xgb_gs = GridSearchCV(
            xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
            xgb_params, cv=3, scoring="neg_mean_absolute_error",
            n_jobs=-1, verbose=0
        )
        xgb_gs.fit(X_train, y_train)
        xgb_best = xgb_gs.best_estimator_
        xgb_label = "XGBoost"
        print(f"[Training] XGB beste Params: {xgb_gs.best_params_}")
    else:
        print("[Training] Gradient Boosting (sklearn Fallback) …")
        xgb_params_fb = {
            "n_estimators":  [200, 300],
            "learning_rate": [0.05, 0.1],
            "max_depth":     [4, 6],
        }
        xgb_gs = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            xgb_params_fb, cv=3, scoring="neg_mean_absolute_error",
            n_jobs=-1, verbose=0
        )
        xgb_gs.fit(X_train, y_train)
        xgb_best = xgb_gs.best_estimator_
        xgb_label = "Gradient Boosting"

    with open(os.path.join(models_dir, "rf_model.pkl"),  "wb") as f:
        pickle.dump(rf_best, f)
    with open(os.path.join(models_dir, "xgb_model.pkl"), "wb") as f:
        pickle.dump(xgb_best, f)
    print(f"[Training] ✓ Modelle gespeichert in '{models_dir}/'")

    return rf_best, xgb_best, X_train, X_test, y_train, y_test, xgb_label


def load_models(models_dir="../models"):
    """Lädt gespeicherte Modelle (damit man nicht neu trainieren muss)."""
    with open(os.path.join(models_dir, "rf_model.pkl"),  "rb") as f:
        rf_best = pickle.load(f)
    with open(os.path.join(models_dir, "xgb_model.pkl"), "rb") as f:
        xgb_best = pickle.load(f)
    return rf_best, xgb_best
