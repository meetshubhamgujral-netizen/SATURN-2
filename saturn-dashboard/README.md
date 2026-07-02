# 🪐 Saturn — AI Market Understanding Dashboard

A premium, self-service **Business Intelligence dashboard** built with Streamlit for **Saturn**, an
upcoming UAE brand in premium men's accessories (ties, pocket squares, dress socks, handkerchiefs).

Upload a UAE consumer-survey dataset and move through a full analytics pipeline — **Data Cleaning →
Descriptive → Diagnostic → Latent Analysis → Predictive → Regression Lab → Association Rules →
Business Insights** — in an interface designed to feel like commercial BI software (Power BI /
Tableau) rather than a typical Streamlit app.

> All analytics are computed **live from the loaded dataset**. No external data is used.

---

## ✨ Features

| Module | What it does |
|---|---|
| 🏠 **Home** | Hero banner, feature overview, quick stats, call-to-action. |
| 📁 **Upload Dataset** | CSV / Excel / JSON upload, automatic type detection, missing & duplicate snapshot, preview. A bundled Saturn sample loads in one click. |
| 🧹 **Data Cleaning** | Quality report (missing, duplicates, invalid Likert, speeders, mixed types), missing-value heatmap, imputation (mean/median/mode/ffill/bfill/custom), duplicate removal, outlier detection (IQR / Z-score) with remove/cap, encoding (one-hot / label), scaler comparison (Standard / MinMax / Robust / Normalizer), a **0–100 data-quality score**, and exports (CSV / Excel / PDF / transformation log). |
| 📊 **Descriptive** | KPI cards, full statistics (mean, median, mode, variance, std, quartiles, range, skewness, kurtosis), frequency & percentage tables, cross-tabulation, and an interactive chart gallery (bar, pie, donut, histogram, box, violin, scatter, treemap, sunburst). |
| 🔍 **Diagnostic** | Pearson & Spearman correlation matrix, interactive heatmap, ranked strongest relationships, scatter matrix, and plain-English interpretation with multicollinearity flags. |
| 🧠 **Latent Analysis** | Factorability checks (**KMO** overall & per-variable, **Bartlett's** test of sphericity), **PCA** with scree plot, explained-variance curve, loadings heatmap and an interactive **biplot**, plus **Exploratory Factor Analysis** with varimax/promax/oblimin rotation, communalities and automatic factor naming. |
| 🤖 **Predictive** | Auto-detects **classification vs regression** for the chosen target. Classification (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN) with accuracy/precision/recall/F1, confusion matrix, ROC, feature importance. Regression (Linear, Decision Tree, Random Forest, Gradient Boosting) with MAE/MSE/RMSE/R², prediction & residual plots. **K-Means segmentation** with elbow method, silhouette analysis, cluster profiles and 2D/3D plots. |
| 📈 **Regression Lab** | **Linear (OLS), Ridge (L2) and Lasso (L1)** side by side inside a scaling pipeline. Coefficient-shrinkage comparison, **regularisation paths**, **5-fold cross-validated α tuning**, prediction/residual diagnostics and automatic **Lasso feature selection** in plain English. |
| 🔗 **Association Rules** | **Market-basket analysis (apriori)** over the multi-select answers (products, occasions, brands, trial-triggers). Tunable support / confidence / lift / itemset size, frequent-itemset & rule tables, a support-vs-confidence rule landscape and a **rule network** graph. Mix several columns for cross-category rules. |
| 📝 **Summary & Insights** | Auto-generated executive summary, demographics, shopping & spending behaviour, purchase intent, key drivers, opportunities, risks and actionable recommendations — exportable as TXT / Excel / PDF. |
| ⚙️ **Settings** | Light / dark theme toggle, reset-to-raw, app info. |

---

## 🗂️ Project structure

```
saturn-dashboard/
├── app.py                       # Main entry point + sidebar navigation
├── requirements.txt
├── README.md
├── .gitignore
├── generate_synthetic_data.py   # Recreates the sample dataset
├── data/
│   ├── saturn_survey_raw.csv     # Bundled sample (3,030 rows, ~10% noisy)
│   └── _data_dictionary.csv
├── utils/                       # Reusable logic
│   ├── theme.py                  # CSS, palette, KPI cards, plotly theming
│   ├── data_loader.py            # Loading, schema, type detection
│   ├── cleaning.py               # Quality report, imputation, outliers, scaling, score
│   ├── descriptive.py            # Stats, frequency, crosstab, charts
│   ├── diagnostic.py             # Correlations + auto-insights
│   ├── latent.py                 # KMO/Bartlett, PCA, factor analysis
│   ├── predictive.py             # Classification / regression / clustering
│   ├── regression_reg.py         # Linear / Ridge / Lasso + CV tuning
│   ├── assoc.py                  # Apriori frequent itemsets + association rules
│   └── insights.py               # Business-insight generation
├── components/                  # One render() per page
│   ├── home.py  upload.py  cleaning.py  descriptive.py  diagnostic.py
│   ├── latent.py  predictive.py  regression_reg.py  assoc.py
│   ├── insights.py  settings.py
│   └── _common.py
├── assets/   models/   reports/   exports/
```

---

## 🚀 Run locally

```bash
# 1. Clone
git clone https://github.com/<your-username>/saturn-dashboard.git
cd saturn-dashboard

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch
streamlit run app.py
```

The app opens at `http://localhost:8501`. Click **Upload Dataset → Load Saturn Sample** to explore
everything immediately, or upload your own survey export.

---

## ☁️ Deploy on Streamlit Community Cloud

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Saturn AI Market Understanding Dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/saturn-dashboard.git
   git push -u origin main
   ```
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **Create app → Deploy a public app from GitHub**.
4. Select your repository, set:
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically and builds the app.

No code changes are required — the bundled dataset ships in `data/`, so the app is fully functional
on first load.

---

## 📊 The bundled dataset

`data/saturn_survey_raw.csv` contains **3,030 synthetic UAE respondents** answering all 30 survey
questions. It is intentionally **~10% "dirty"** so the Data Cleaning page has real work to do:

- Missing values, Likert **straight-lining** and **out-of-range** ratings (0, 6, "N/A")
- **Logical contradictions** (e.g. "Never" wears accessories yet buys monthly)
- **Whitespace / casing** noise (`  dubai `, `MALE`)
- **Speeder** responses (completion < 25s) and **mixed-type** numeric cells (`"five"`, `"3 "`)
- **Junk** open-ended text and **30 exact duplicate rows**

Underneath the noise is genuine structure (income → spend, formality → purchase frequency,
importance ratings → purchase intent) so the diagnostic and predictive pages surface real findings.

Regenerate it any time with:

```bash
python generate_synthetic_data.py
```

---

## 🛠️ Built with

Python · Streamlit · Plotly · pandas · NumPy · scikit-learn · SciPy · openpyxl · fpdf2 ·
streamlit-option-menu

> Latent Analysis and Association Rules ship with self-contained numpy/scipy implementations, so no
> extra packages are required. If `factor-analyzer` and `mlxtend` happen to be installed the app will
> use them automatically, but they are intentionally left out of `requirements.txt` for a lean,
> reliable install.
