
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


FEATURES = [
    "log_act_price",     # Listenpreis (log) → exogen
    "disc_pct",          # Rabatttiefe %     → endogen 
    "rating_num",        # Produktbewertung  → endogen
    "log_rating_count",  # Popularität (log) → endogen
    "cat_enc",           # Kategorie         → exogen
    "inv_proxy",         # Lagerbestand-Proxy→ endogen
]

TARGET = "log_disc_price"   # Zielvariable: log(Verkaufspreis)


def build_features(data: pd.DataFrame):
    
    df = data.copy()

    df["log_act_price"]    = np.log1p(df["act_price"])
    df["log_disc_price"]   = np.log1p(df["disc_price"])
    df["log_rating_count"] = np.log1p(df["rating_count_num"])

    le = LabelEncoder()
    df["cat_enc"] = le.fit_transform(df["main_cat"])
    df["inv_proxy"] = 1 / (df["rating_count_num"] + 1) * 1000

    print(f"[Features] ✓ {len(FEATURES)} Features gebaut: {FEATURES}")
    return df, le, FEATURES, TARGET


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_pipeline_01 import load_and_clean

    raw = load_and_clean("../data/Air_Conditioners.csv", "../data/amazon.csv")
    df, le, feats, target = build_features(raw)
    print(df[feats + [target]].describe())
