"""Dataset loading, schema constants, automatic type detection."""
import io
import os
import json
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(HERE, "data", "saturn_survey_raw.csv")

# ---- Schema knowledge of the Saturn survey ----
META_COLS = ["respondent_id", "submission_time", "duration_seconds"]
LIKERT_COLS = [
    "imp_fabric_quality", "imp_design", "imp_pricing", "imp_brand_rep",
    "imp_comfort", "imp_durability", "imp_variety", "imp_sustainability",
]
MULTISELECT_COLS = ["occasions", "accessories_purchased", "brands_purchased", "convince_try"]
OPEN_ENDED = ["most_important_feature"]

# Friendly labels for charts / tables
LABELS = {
    "age_group": "Age Group", "gender": "Gender", "emirate": "Emirate",
    "nationality": "Nationality", "income": "Monthly Income (AED)", "occupation": "Occupation",
    "formal_attire_freq": "Formal Attire Frequency", "occasions": "Occasions",
    "accessories_purchased": "Accessories Purchased", "purchase_freq": "Purchase Frequency",
    "buy_location": "Buying Location", "shopping_channel": "Shopping Channel",
    "spend_tie": "Spend on Tie", "spend_socks": "Spend on Socks",
    "main_influence": "Main Purchase Influence",
    "imp_fabric_quality": "Importance: Fabric Quality", "imp_design": "Importance: Design",
    "imp_pricing": "Importance: Pricing", "imp_brand_rep": "Importance: Brand Reputation",
    "imp_comfort": "Importance: Comfort", "imp_durability": "Importance: Durability",
    "imp_variety": "Importance: Variety", "imp_sustainability": "Importance: Sustainability",
    "brands_purchased": "Brands Purchased", "loyalty_factor": "Loyalty Factor",
    "willing_try_new": "Willing to Try New Brand", "convince_try": "What Would Convince",
    "purchase_likelihood_6m": "Purchase Likelihood (6m)",
    "saturn_consideration": "Saturn Consideration", "most_important_feature": "Most Important Feature",
}

# Ordered categories (for nicer axis ordering where relevant)
ORDERED = {
    "age_group": ["Under 18", "18–24", "25–34", "35–44", "45–54", "55+"],
    "income": ["Under 5,000", "5,000–10,000", "10,001–20,000", "20,001–35,000", "Above 35,000"],
    "formal_attire_freq": ["Daily", "3–5 times/week", "Once a week", "Occasionally", "Rarely"],
    "purchase_freq": ["Monthly", "Every 3 months", "Every 6 months", "Once a year", "Only when needed"],
    "spend_tie": ["Under AED 50", "AED 50–100", "AED 101–200", "AED 201–300", "Above AED 300"],
    "spend_socks": ["Under AED 25", "AED 25–50", "AED 51–75", "Above AED 75"],
    "willing_try_new": ["Definitely", "Probably", "Not Sure", "Probably Not", "Definitely Not"],
    "purchase_likelihood_6m": ["Very Likely", "Likely", "Neutral", "Unlikely", "Very Unlikely"],
    "saturn_consideration": ["Definitely Would", "Probably Would", "Not Sure",
                             "Probably Would Not", "Definitely Would Not"],
}


def label(col):
    return LABELS.get(col, col.replace("_", " ").title())


@st.cache_data(show_spinner=False)
def load_default():
    return pd.read_csv(DEFAULT_PATH)


def load_uploaded(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    if name.endswith(".json"):
        data = json.load(file)
        return pd.json_normalize(data)
    raise ValueError("Unsupported file type. Upload CSV, Excel or JSON.")


def coerce_likert(df):
    """Return a copy where Likert columns are numeric (invalid -> NaN)."""
    out = df.copy()
    for c in LIKERT_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "duration_seconds" in out.columns:
        out["duration_seconds"] = pd.to_numeric(out["duration_seconds"], errors="coerce")
    return out


def numeric_columns(df):
    """Likert + any genuinely numeric columns (post coercion)."""
    d = coerce_likert(df)
    cols = []
    for c in d.columns:
        if c == "respondent_id":
            continue
        if pd.api.types.is_numeric_dtype(d[c]) and d[c].notna().sum() > 0:
            cols.append(c)
    return cols


def categorical_columns(df):
    num = set(numeric_columns(df))
    return [c for c in df.columns
            if c not in num and c not in META_COLS and c not in MULTISELECT_COLS
            and c not in OPEN_ENDED]


def detect_types(df):
    """Classify every column for the upload-summary page."""
    d = coerce_likert(df)
    info = []
    for c in df.columns:
        if c in MULTISELECT_COLS:
            kind = "Multi-select"
        elif c in OPEN_ENDED:
            kind = "Text (open-ended)"
        elif c in ("submission_time",):
            kind = "Datetime"
        elif pd.api.types.is_numeric_dtype(d[c]) and c != "respondent_id":
            kind = "Numeric"
        elif df[c].nunique(dropna=True) <= 12:
            kind = "Categorical"
        else:
            kind = "Identifier/Text"
        info.append({
            "Column": c,
            "Type": kind,
            "Unique": int(df[c].nunique(dropna=True)),
            "Missing": int(df[c].isna().sum()),
            "Missing %": round(100 * df[c].isna().mean(), 1),
            "Example": _example(df[c]),
        })
    return pd.DataFrame(info)


def _example(s):
    vals = s.dropna()
    return str(vals.iloc[0]) if len(vals) else "—"


def memory_usage_mb(df):
    return df.memory_usage(deep=True).sum() / 1024 ** 2


def to_excel_bytes(df_dict):
    """df_dict: {sheet_name: dataframe} -> xlsx bytes."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        for sheet, d in df_dict.items():
            d.to_excel(xl, sheet_name=sheet[:31], index=False)
    return buf.getvalue()


def explode_multiselect(df, col):
    """Return a long Series of individual selections for a multi-select column."""
    s = df[col].dropna().astype(str)
    parts = s.str.split(";").explode().str.strip()
    return parts[parts != ""]
