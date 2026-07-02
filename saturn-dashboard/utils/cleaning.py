"""Data-cleaning engine: quality report, imputation, duplicates, outliers,
encoding, scaling and an overall data-quality score."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer,
    LabelEncoder, OneHotEncoder,
)

from utils.data_loader import LIKERT_COLS, META_COLS, coerce_likert, numeric_columns


# --------------------------------------------------------------------------- #
# Dataset health
# --------------------------------------------------------------------------- #
def quality_report(df):
    d = coerce_likert(df)
    n_rows, n_cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    total_cells = n_rows * n_cols
    dup_rows = int(df.duplicated().sum())
    constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    high_card = [c for c in df.columns
                 if df[c].nunique(dropna=True) > 0.5 * n_rows
                 and not pd.api.types.is_numeric_dtype(df[c])]
    # mixed-type columns: mostly numeric values but containing some text
    mixed = []
    for c in df.columns:
        if c in ("submission_time", "respondent_id"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        vals = df[c].dropna().astype(str)
        if len(vals) == 0:
            continue
        numlike = pd.to_numeric(vals, errors="coerce").notna()
        if 0 < numlike.mean() < 1:
            mixed.append(c)
    # invalid likert (outside 1..5 after coercion)
    invalid_likert = 0
    for c in LIKERT_COLS:
        if c in d.columns:
            v = d[c]
            invalid_likert += int(((v < 1) | (v > 5)).sum())
    # speeders
    speeders = 0
    if "duration_seconds" in d.columns:
        speeders = int((d["duration_seconds"] < 30).sum())
    return {
        "rows": n_rows, "cols": n_cols,
        "missing_cells": missing_cells,
        "missing_pct": round(100 * missing_cells / total_cells, 2) if total_cells else 0,
        "dup_rows": dup_rows,
        "dup_pct": round(100 * dup_rows / n_rows, 2) if n_rows else 0,
        "constant_cols": constant,
        "high_card_cols": high_card,
        "mixed_cols": mixed,
        "invalid_likert": invalid_likert,
        "speeders": speeders,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2),
    }


def missing_summary(df):
    miss = df.isna().sum()
    pct = 100 * df.isna().mean()
    out = pd.DataFrame({"Column": df.columns,
                        "Missing": miss.values,
                        "Missing %": pct.round(2).values})
    return out[out["Missing"] > 0].sort_values("Missing", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Fixes
# --------------------------------------------------------------------------- #
def fix_invalid_likert(df):
    """Set Likert values outside 1..5 (and non-numeric) to NaN. Returns (df, n)."""
    d = df.copy()
    n = 0
    for c in LIKERT_COLS:
        if c in d.columns:
            num = pd.to_numeric(d[c], errors="coerce")
            bad = num.isna() & d[c].notna()
            oob = num.notna() & ((num < 1) | (num > 5))
            n += int(bad.sum() + oob.sum())
            num[(num < 1) | (num > 5)] = np.nan
            d[c] = num
    return d, n


def strip_whitespace_case(df, cols):
    """Trim whitespace and title-case selected categorical columns."""
    d = df.copy()
    for c in cols:
        if c in d.columns and not pd.api.types.is_numeric_dtype(d[c]):
            d[c] = (d[c].astype(str).str.strip()
                    .replace({"nan": np.nan})
                    .map(lambda x: _smart_title(x) if isinstance(x, str) else x))
    return d


def _smart_title(x):
    # keep known acronyms / city spellings reasonable
    return x.title() if x.isupper() or x.islower() else x


def impute(df, columns, method, custom_value=None):
    """Impute missing values for given columns with the chosen strategy."""
    d = df.copy()
    for c in columns:
        if c not in d.columns:
            continue
        s = pd.to_numeric(d[c], errors="coerce") if c in LIKERT_COLS else d[c]
        if method == "Mean" and pd.api.types.is_numeric_dtype(s):
            d[c] = s.fillna(round(s.mean()))
        elif method == "Median" and pd.api.types.is_numeric_dtype(s):
            d[c] = s.fillna(round(s.median()))
        elif method == "Mode":
            mode = d[c].mode(dropna=True)
            if len(mode):
                d[c] = d[c].fillna(mode.iloc[0])
        elif method == "Forward Fill":
            d[c] = d[c].ffill()
        elif method == "Backward Fill":
            d[c] = d[c].bfill()
        elif method == "Custom Value":
            d[c] = d[c].fillna(custom_value)
    return d


def drop_missing_rows(df, columns=None):
    return df.dropna(subset=columns) if columns else df.dropna()


def drop_missing_columns(df, threshold_pct):
    keep = [c for c in df.columns if df[c].isna().mean() * 100 <= threshold_pct]
    return df[keep]


def remove_duplicates(df):
    before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    return out, before - len(out)


# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #
def outlier_bounds(series, method="IQR", z=3.0):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if method == "IQR":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mean, std = s.mean(), s.std()
    return mean - z * std, mean + z * std


def count_outliers(series, method="IQR", z=3.0):
    lo, hi = outlier_bounds(series, method, z)
    s = pd.to_numeric(series, errors="coerce")
    return int(((s < lo) | (s > hi)).sum()), (lo, hi)


def handle_outliers(df, column, action, method="IQR", z=3.0):
    d = df.copy()
    lo, hi = outlier_bounds(d[column], method, z)
    s = pd.to_numeric(d[column], errors="coerce")
    mask = (s < lo) | (s > hi)
    if action == "Remove":
        d = d[~mask.fillna(False)].reset_index(drop=True)
    elif action == "Cap":
        s = s.clip(lo, hi)
        d[column] = s
    return d, int(mask.sum())


# --------------------------------------------------------------------------- #
# Encoding & scaling
# --------------------------------------------------------------------------- #
def encode(df, columns, method):
    d = df.copy()
    if method == "Label Encoding":
        for c in columns:
            le = LabelEncoder()
            d[c] = le.fit_transform(d[c].astype(str))
        return d
    # One-hot
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    arr = enc.fit_transform(d[columns].astype(str))
    names = enc.get_feature_names_out(columns)
    ohe = pd.DataFrame(arr, columns=names, index=d.index).astype(int)
    d = pd.concat([d.drop(columns=columns), ohe], axis=1)
    return d


SCALERS = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
    "Normalizer": Normalizer,
}


def scale(df, columns, scaler_name):
    d = coerce_likert(df)[columns].dropna()
    scaler = SCALERS[scaler_name]()
    scaled = pd.DataFrame(scaler.fit_transform(d), columns=columns, index=d.index)
    return d, scaled


# --------------------------------------------------------------------------- #
# Quality score
# --------------------------------------------------------------------------- #
def quality_score(df):
    """Heuristic 0-100 score combining completeness, uniqueness, validity."""
    rep = quality_report(df)
    completeness = 100 - rep["missing_pct"]
    uniqueness = 100 - rep["dup_pct"]
    total_cells = rep["rows"] * rep["cols"]
    validity = 100 - (100 * (rep["invalid_likert"] + len(rep["mixed_cols"]) * 5) / max(total_cells, 1))
    validity = max(validity, 0)
    consistency = 100 - min(rep["speeders"] / max(rep["rows"], 1) * 100, 100)
    score = 0.40 * completeness + 0.25 * uniqueness + 0.20 * validity + 0.15 * consistency
    return round(max(0, min(score, 100)), 1), {
        "Completeness": round(completeness, 1),
        "Uniqueness": round(uniqueness, 1),
        "Validity": round(validity, 1),
        "Consistency": round(consistency, 1),
    }
