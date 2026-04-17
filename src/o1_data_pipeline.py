

import pandas as pd
import numpy as np



def clean_price(s):
    """'₹32,999' → 32999.0"""
    if pd.isna(s):
        return np.nan
    try:
        return float(str(s).replace("₹", "").replace(",", "").strip())
    except ValueError:
        return np.nan

def clean_count(s):
    """'2,255' → 2255.0  |  Text → NaN"""
    if pd.isna(s):
        return np.nan
    try:
        return float(str(s).replace(",", "").strip())
    except ValueError:
        return np.nan

def clean_rating(s):
    """'4.2' → 4.2  |  Text → NaN"""
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return np.nan



def load_and_clean(ac_path: str, amz_path: str) -> pd.DataFrame:
    

    # ── Air Conditioners ──────────────────────────────────────────────────
    ac = pd.read_csv(ac_path)

    ac["disc_price"]       = ac["discount_price"].apply(clean_price)
    ac["act_price"]        = ac["actual_price"].apply(clean_price)
    ac["rating_count_num"] = ac["no_of_ratings"].apply(clean_count)
    ac["rating_num"]       = pd.to_numeric(ac["ratings"], errors="coerce")
    ac["disc_pct"]         = (
        (ac["act_price"] - ac["disc_price"]) / ac["act_price"] * 100
    )
    ac["main_cat"]         = "Appliances"
    ac["product_name"]     = ac["name"].str[:80]   # kürzen für Tabelle

    ac_clean = ac[["product_name", "disc_price", "act_price",
                   "disc_pct", "rating_num", "rating_count_num", "main_cat"]].copy()

    # ── Amazon Multi-Category ─────────────────────────────────────────────
    amz = pd.read_csv(amz_path)

    amz["disc_price"]       = amz["discounted_price"].apply(clean_price)
    amz["act_price"]        = amz["actual_price"].apply(clean_price)
    amz["disc_pct"]         = amz["discount_percentage"].apply(
                                  lambda s: clean_rating(str(s).replace("%", ""))
                              )
    amz["rating_num"]       = amz["rating"].apply(clean_rating)
    amz["rating_count_num"] = amz["rating_count"].apply(clean_rating)
    amz["main_cat"]         = amz["category"].str.split("|").str[0]
    amz["product_name"]     = amz["product_name"].str[:80]

    amz_clean = amz[["product_name", "disc_price", "act_price",
                     "disc_pct", "rating_num", "rating_count_num", "main_cat"]].copy()


    data = pd.concat([ac_clean, amz_clean], ignore_index=True)
    data.dropna(subset=["disc_price", "act_price", "disc_pct",
                        "rating_num", "rating_count_num"], inplace=True)
    data = data[data["disc_price"] > 0].reset_index(drop=True)

    print(f"[Pipeline] ✓ {len(data):,} Produkte geladen "
          f"({len(ac_clean)} AC + {len(amz_clean)} Amazon)")
    return data


if __name__ == "__main__":
    df = load_and_clean("../data/Air_Conditioners.csv", "../data/amazon.csv")
    print(df.head())
    print(df.dtypes)
