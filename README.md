# Dynamic Pricing – Bachelorarbeit
**Autor:** Ghaid Gattas | **Betreuer:** Mag. Dr. Christian Buchta

---

## 📁 Projektstruktur

```
dynamic_pricing_project/
│
├── data/
│   ├── Air_Conditioners.csv
│   └── amazon.csv
│
├── src/
│   ├── 01_data_pipeline.py        # Daten laden & bereinigen
│   ├── 02_feature_engineering.py  # Features bauen
│   ├── 03_train_models.py         # RF & XGBoost trainieren
│   ├── 04_evaluation.py           # Metriken & Plots
│   └── 05_simulation.py           # Dynamic vs. Statisch
│
├── models/                        # Gespeicherte Modelle (nach main.py)
├── outputs/                       # Plots & CSVs (nach main.py)
│
├── main.py                        # Kompletten technischen Teil ausführen
├── dashboard.py                   # Streamlit Dashboard
└── README.md
```

---

## ⚙️ Setup (einmalig)

```bash
pip3 install pandas numpy scikit-learn xgboost shap matplotlib seaborn streamlit
```

---

## 🚀 Ausführen

### Schritt 1 – Technischen Teil ausführen
```bash
python3 main.py
```
Generiert alle Plots in `outputs/` und beide CSV-Dateien.

### Schritt 2 – Dashboard starten
```bash
streamlit run dashboard.py
```
Öffnet automatisch im Browser unter `http://localhost:8501`

---

## 📊 Output

| Datei | Beschreibung |
|---|---|
| `outputs/preisempfehlungen.csv` | Tabelle: Produktname, ursprünglicher Preis, empfohlener Preis |
| `outputs/simulation_ergebnisse.csv` | 30-Tage-Simulation: Umsatz & Marge täglich |
| `outputs/plot1_modellvergleich.png` | MAE / RMSE / R² Vergleich |
| `outputs/plot2_actual_vs_predicted.png` | Actual vs. Predicted |
| `outputs/plot3_feature_importance.png` | Wichtigste Einflussfaktoren |
| `outputs/plot4_shap.png` | SHAP-Erklärbarkeit |
| `outputs/plot5_umsatz.png` | Umsatzentwicklung |
| `outputs/plot6_gewinn.png` | Gewinnprognose |
| `outputs/plot7_stabilitaet.png` | Preisvolatilität |
