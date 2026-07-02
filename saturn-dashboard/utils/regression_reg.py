"""Regularised regression lab — Ordinary Least Squares, Ridge (L2) and Lasso (L1).

Every model is fitted inside a StandardScaler pipeline so the coefficients are
directly comparable and the regularisation strength means the same thing across
features. Provides coefficient-shrinkage comparison, regularisation paths,
cross-validated alpha tuning and automatic Lasso feature selection.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, RidgeCV, LassoCV)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.data_loader import (coerce_likert, numeric_columns, label, ORDERED)

# Ordered intent variables that can be scored 5→1 and used as a numeric target
INTENT_TARGETS = ["purchase_likelihood_6m", "saturn_consideration",
                  "willing_try_new"]
ALPHA_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


# --------------------------------------------------------------------------- #
# Target / feature preparation
# --------------------------------------------------------------------------- #
def target_catalogue(df):
    """Return {display_label: ('num'|'ord', column)} of usable regression targets."""
    cat = {}
    for c in numeric_columns(df):
        cat[f"{label(c)}  (numeric)"] = ("num", c)
    for c in INTENT_TARGETS:
        if c in df.columns:
            cat[f"{label(c)}  (intent 5→1)"] = ("ord", c)
    return cat


def build_target(df, kind, col):
    if kind == "num":
        return pd.to_numeric(coerce_likert(df)[col], errors="coerce").rename(col)
    order = ORDERED.get(col, list(df[col].dropna().unique()))
    mapping = {v: len(order) - i for i, v in enumerate(order)}  # best -> high
    return df[col].map(mapping).rename(col)


def build_xy(df, y, feature_cols):
    """One-hot categorical + numeric features aligned to a numeric target.

    Returns (X, y, labels) where labels maps each engineered column to a clean,
    unambiguous display name (e.g. 'income_Under 5,000' -> 'Monthly Income (AED) = Under 5,000').
    """
    d = coerce_likert(df).copy()
    num_all = set(numeric_columns(df))
    use = [c for c in feature_cols if c != y.name]
    num = [c for c in use if c in num_all]
    cat = [c for c in use if c not in num_all]
    X = pd.get_dummies(pd.concat([d[num], d[cat].astype(str)], axis=1),
                       columns=cat, drop_first=True)
    data = pd.concat([X, y.rename("__t__")], axis=1).apply(pd.to_numeric, errors="coerce").dropna()
    X = data.drop(columns="__t__")
    yv = data["__t__"]
    X = X.fillna(X.mean(numeric_only=True)).fillna(0)
    return X, yv, feature_labels(X.columns, df)


def feature_labels(columns, df):
    """Map engineered column names to unambiguous, human-readable labels."""
    out = {}
    for name in columns:
        if name in df.columns:                 # pass-through numeric feature
            out[name] = label(name)
            continue
        matched = None                          # longest matching categorical prefix
        for c in df.columns:
            if name.startswith(c + "_") and (matched is None or len(c) > len(matched)):
                matched = c
        if matched:
            out[name] = f"{label(matched)} = {name[len(matched) + 1:]}"
        else:
            out[name] = label(name)
    return out


# --------------------------------------------------------------------------- #
# Alpha suggestions (cross-validated)
# --------------------------------------------------------------------------- #
def suggest_alphas(X, y):
    Xs = StandardScaler().fit_transform(X)
    r = RidgeCV(alphas=np.logspace(-3, 2, 40)).fit(Xs, y)
    l = LassoCV(alphas=np.logspace(-3, 2, 40), max_iter=5000, cv=5).fit(Xs, y)
    return round(float(r.alpha_), 4), round(float(l.alpha_), 4)


# --------------------------------------------------------------------------- #
# Fit & compare
# --------------------------------------------------------------------------- #
def fit_models(X, y, alpha_ridge, alpha_lasso, labels=None, test_size=0.25):
    labels = labels or {c: label(c) for c in X.columns}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
    specs = {
        "Linear (OLS)": LinearRegression(),
        f"Ridge (α={alpha_ridge:g})": Ridge(alpha=alpha_ridge),
        f"Lasso (α={alpha_lasso:g})": Lasso(alpha=alpha_lasso, max_iter=10000),
    }
    n, p = Xte.shape
    rows, fitted, preds, coefs = [], {}, {}, {}
    for name, est in specs.items():
        pipe = make_pipeline(StandardScaler(), est)
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        r2 = r2_score(yte, pred)
        adj = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)
        reg = pipe.steps[-1][1]
        nz = int(np.sum(np.abs(reg.coef_) > 1e-8))
        rows.append({
            "Model": name,
            "R²": round(r2, 3),
            "Adj. R²": round(adj, 3),
            "RMSE": round(np.sqrt(mean_squared_error(yte, pred)), 3),
            "MAE": round(mean_absolute_error(yte, pred), 3),
            "Non-zero coefs": nz,
        })
        fitted[name] = pipe
        preds[name] = pred
        coefs[name] = pd.Series(reg.coef_, index=X.columns)
    results = pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)
    coef_df = pd.DataFrame(coefs)
    coef_df.index = [labels.get(i, label(i)) for i in coef_df.index]
    return {"results": results, "fitted": fitted, "preds": preds, "coef_df": coef_df,
            "yte": yte, "features": list(X.columns), "best": results.iloc[0]["Model"]}


def coef_compare_fig(coef_df, top=12):
    order = coef_df.abs().max(axis=1).sort_values(ascending=False).head(top).index
    sub = coef_df.loc[order]
    long = sub.reset_index(names="Feature").melt(
        id_vars="Feature", var_name="Model", value_name="Coefficient")
    fig = px.bar(long, x="Coefficient", y="Feature", color="Model",
                 orientation="h", barmode="group",
                 color_discrete_sequence=["#1E3A8A", "#0EA5A3", "#C026D3"])
    fig.add_vline(x=0, line_color="#94A3B8")
    fig.update_layout(title="Standardised coefficients — OLS vs Ridge vs Lasso",
                      yaxis=dict(autorange="reversed"))
    return fig


# --------------------------------------------------------------------------- #
# Regularisation paths
# --------------------------------------------------------------------------- #
def _path(X, y, kind, labels=None, alphas=np.logspace(-3, 2, 60), top=8):
    labels = labels or {c: label(c) for c in X.columns}
    Xs = StandardScaler().fit_transform(X)
    Est = Ridge if kind == "ridge" else Lasso
    coefs = []
    for a in alphas:
        m = Est(alpha=a, max_iter=10000) if kind == "lasso" else Est(alpha=a)
        m.fit(Xs, y)
        coefs.append(m.coef_)
    C = np.array(coefs)  # (n_alpha, n_feat)
    imp = np.abs(C).max(axis=0)
    idx = np.argsort(imp)[::-1][:top]
    names = [labels.get(X.columns[i], label(X.columns[i])) for i in idx]
    fig = go.Figure()
    for j, i in enumerate(idx):
        fig.add_scatter(x=alphas, y=C[:, i], mode="lines", name=names[j],
                        line=dict(width=2))
    fig.add_hline(y=0, line_color="#94A3B8", line_dash="dot")
    fig.update_layout(title=f"{kind.title()} regularisation path",
                      xaxis_title="α (regularisation strength)",
                      yaxis_title="Standardised coefficient",
                      xaxis_type="log")
    return fig


def ridge_path_fig(X, y, labels=None):
    return _path(X, y, "ridge", labels)


def lasso_path_fig(X, y, labels=None):
    return _path(X, y, "lasso", labels)


# --------------------------------------------------------------------------- #
# Cross-validation curves
# --------------------------------------------------------------------------- #
def cv_curve_fig(X, y, kind, alphas=None):
    alphas = alphas or np.logspace(-3, 2, 30)
    Xs = StandardScaler().fit_transform(X)
    Est = Ridge if kind == "ridge" else Lasso
    means, stds = [], []
    for a in alphas:
        m = Est(alpha=a, max_iter=10000) if kind == "lasso" else Est(alpha=a)
        sc = cross_val_score(m, Xs, y, cv=5, scoring="neg_root_mean_squared_error")
        means.append(-sc.mean())
        stds.append(sc.std())
    means, stds = np.array(means), np.array(stds)
    best_i = int(np.argmin(means))
    accent = "#0EA5A3" if kind == "ridge" else "#C026D3"
    fig = go.Figure()
    fig.add_scatter(x=alphas, y=means, mode="lines+markers", name="CV RMSE",
                    line=dict(color=accent, width=3))
    fig.add_scatter(x=np.concatenate([alphas, alphas[::-1]]),
                    y=np.concatenate([means + stds, (means - stds)[::-1]]),
                    fill="toself", fillcolor="rgba(14,165,163,0.12)",
                    line=dict(color="rgba(0,0,0,0)"), name="±1 SD", showlegend=False)
    fig.add_vline(x=alphas[best_i], line_dash="dash", line_color="#DC2A52",
                  annotation_text=f"best α = {alphas[best_i]:.3g}",
                  annotation_position="top")
    fig.update_layout(title=f"{kind.title()} — 5-fold CV RMSE vs α",
                      xaxis_title="α", yaxis_title="Cross-validated RMSE",
                      xaxis_type="log")
    return fig, round(float(alphas[best_i]), 4)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def pred_vs_actual_fig(yte, pred, name):
    fig = px.scatter(x=yte, y=pred, opacity=0.55,
                     labels={"x": "Actual", "y": "Predicted"},
                     color_discrete_sequence=["#0EA5A3"])
    lo, hi = float(min(np.min(yte), np.min(pred))), float(max(np.max(yte), np.max(pred)))
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                    line=dict(dash="dash", color="#94A3B8"), name="Ideal")
    fig.update_layout(title=f"Predicted vs Actual — {name}", showlegend=False)
    return fig


def residual_fig(yte, pred, name):
    resid = np.asarray(yte) - np.asarray(pred)
    fig = px.scatter(x=pred, y=resid, opacity=0.55,
                     labels={"x": "Predicted", "y": "Residual"},
                     color_discrete_sequence=["#C026D3"])
    fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
    fig.update_layout(title=f"Residuals — {name}")
    return fig


def lasso_selection(coef_df):
    """Plain-English read-out of Lasso's automatic feature selection."""
    lasso_col = [c for c in coef_df.columns if c.startswith("Lasso")]
    if not lasso_col:
        return []
    s = coef_df[lasso_col[0]]
    kept = s[s.abs() > 1e-8].sort_values(key=abs, ascending=False)
    dropped = s[s.abs() <= 1e-8]
    lines = [f"Lasso retained **{len(kept)}** of **{len(s)}** predictors, "
             f"zeroing out **{len(dropped)}** as redundant."]
    if len(kept):
        top = kept.head(5)
        drivers = ", ".join(f"**{i}** ({v:+.2f})" for i, v in top.items())
        lines.append(f"Strongest surviving drivers: {drivers}.")
    if len(dropped):
        names = ", ".join(dropped.index[:6])
        more = "…" if len(dropped) > 6 else ""
        lines.append(f"Eliminated (little unique signal): {names}{more}.")
    return lines
