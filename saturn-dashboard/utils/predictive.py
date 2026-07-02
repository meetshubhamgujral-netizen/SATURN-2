"""Predictive analytics: automatic classification / regression model comparison,
plus K-Means clustering with elbow and silhouette diagnostics."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans

from utils.data_loader import (
    coerce_likert, numeric_columns, categorical_columns, label, LIKERT_COLS,
)


# --------------------------------------------------------------------------- #
# Task detection & feature preparation
# --------------------------------------------------------------------------- #
def detect_task(df, target):
    s = coerce_likert(df)[target] if target in LIKERT_COLS else df[target]
    s = s.dropna()
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 12:
        return "Regression"
    return "Classification"


def build_xy(df, target, feature_cols):
    d = coerce_likert(df).copy()
    use = [c for c in feature_cols if c != target]
    num = [c for c in use if c in numeric_columns(df)]
    cat = [c for c in use if c not in num]
    X_num = d[num].copy()
    X = pd.get_dummies(pd.concat([X_num, d[cat].astype(str)], axis=1),
                       columns=cat, drop_first=True)
    y = d[target]
    data = pd.concat([X, y.rename("__target__")], axis=1).dropna()
    X = data.drop(columns="__target__")
    y = data["__target__"]
    # impute any residual numeric NaN with column means (safety)
    X = X.fillna(X.mean(numeric_only=True)).fillna(0)
    return X, y


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def run_classification(X, y, test_size=0.25):
    y = y.astype(str)
    strat = y if y.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=strat)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=220, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=15),
    }
    rows, fitted = [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(yte, pred), 3),
            "Precision": round(precision_score(yte, pred, average="weighted", zero_division=0), 3),
            "Recall": round(recall_score(yte, pred, average="weighted", zero_division=0), 3),
            "F1 Score": round(f1_score(yte, pred, average="weighted", zero_division=0), 3),
        })
        fitted[name] = m
    results = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    best = results.iloc[0]["Model"]
    return {
        "results": results, "best": best, "model": fitted[best],
        "Xtr": Xtr, "Xte": Xte, "ytr": ytr, "yte": yte,
        "pred": fitted[best].predict(Xte), "classes": sorted(y.unique()),
        "features": list(X.columns),
    }


def confusion_fig(yte, pred, classes):
    cm = confusion_matrix(yte, pred, labels=classes)
    fig = px.imshow(cm, x=classes, y=classes, text_auto=True, aspect="auto",
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual", color="Count"))
    fig.update_layout(title="Confusion Matrix")
    return fig


def roc_fig(model, Xte, yte, classes):
    """ROC for binary targets; macro-OvR note for multiclass."""
    if not hasattr(model, "predict_proba"):
        return None, None
    proba = model.predict_proba(Xte)
    if len(classes) == 2:
        pos = classes[1]
        fpr, tpr, _ = roc_curve((yte == pos).astype(int), proba[:, 1])
        a = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {a:.3f}",
                        line=dict(color="#DC2A52", width=3))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                        line=dict(dash="dash", color="#94A3B8"))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate")
        return fig, a
    # multiclass macro AUC (no single curve)
    try:
        a = roc_auc_score(label_binarize(yte, classes=classes), proba,
                          average="macro", multi_class="ovr")
    except Exception:
        a = None
    return None, a


def feature_importance_fig(model, feature_names, top=12):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        return None
    s = pd.Series(imp, index=feature_names).sort_values(ascending=True).tail(top)
    fig = px.bar(x=s.values, y=[label(i.split("_")[0]) if "_" in i else label(i) for i in s.index],
                 orientation="h", color=s.values, color_continuous_scale="Tealgrn",
                 labels={"x": "Importance", "y": ""})
    fig.update_layout(title="Top Feature Importances", coloraxis_showscale=False)
    return fig


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def run_regression(X, y, test_size=0.25):
    y = pd.to_numeric(y, errors="coerce")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=220, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    rows, fitted = [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        rmse = np.sqrt(mean_squared_error(yte, pred))
        rows.append({
            "Model": name,
            "MAE": round(mean_absolute_error(yte, pred), 3),
            "MSE": round(mean_squared_error(yte, pred), 3),
            "RMSE": round(rmse, 3),
            "R² Score": round(r2_score(yte, pred), 3),
        })
        fitted[name] = m
    results = pd.DataFrame(rows).sort_values("R² Score", ascending=False).reset_index(drop=True)
    best = results.iloc[0]["Model"]
    pred = fitted[best].predict(Xte)
    return {"results": results, "best": best, "model": fitted[best],
            "yte": yte, "pred": pred, "features": list(X.columns)}


def pred_vs_actual_fig(yte, pred):
    fig = px.scatter(x=yte, y=pred, opacity=0.6,
                     labels={"x": "Actual", "y": "Predicted"},
                     color_discrete_sequence=["#7C3AED"])
    lo, hi = float(min(yte.min(), pred.min())), float(max(yte.max(), pred.max()))
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                    line=dict(dash="dash", color="#94A3B8"), name="Ideal")
    fig.update_layout(title="Predicted vs Actual", showlegend=False)
    return fig


def residual_fig(yte, pred):
    resid = yte - pred
    fig = px.scatter(x=pred, y=resid, opacity=0.6,
                     labels={"x": "Predicted", "y": "Residual"},
                     color_discrete_sequence=["#F2772E"])
    fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
    fig.update_layout(title="Residual Plot")
    return fig


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def _scaled(df, cols):
    d = coerce_likert(df)[cols].dropna()
    return d, StandardScaler().fit_transform(d)


def elbow(df, cols, k_range=range(2, 11)):
    _, Xs = _scaled(df, cols)
    wcss = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        wcss.append(km.inertia_)
    fig = px.line(x=list(k_range), y=wcss, markers=True,
                  labels={"x": "Number of clusters (k)", "y": "WCSS (inertia)"},
                  color_discrete_sequence=["#1E3A8A"])
    fig.update_layout(title="Elbow Method")
    # crude elbow = max second-difference
    diffs = np.diff(wcss, 2)
    best_k = list(k_range)[int(np.argmax(diffs)) + 1] if len(diffs) else list(k_range)[0]
    return fig, best_k


def silhouette(df, cols, k_range=range(2, 9)):
    _, Xs = _scaled(df, cols)
    scores = []
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xs)
        scores.append(silhouette_score(Xs, labels))
    fig = px.bar(x=list(k_range), y=[round(s, 3) for s in scores],
                 labels={"x": "k", "y": "Silhouette score"},
                 color=scores, color_continuous_scale="Emrld")
    fig.update_layout(title="Silhouette Analysis", coloraxis_showscale=False)
    best_k = list(k_range)[int(np.argmax(scores))]
    return fig, best_k, round(max(scores), 3)


def run_kmeans(df, cols, k):
    d, Xs = _scaled(df, cols)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    d = d.copy()
    d["Cluster"] = [f"Cluster {i}" for i in labels]
    profile = d.groupby("Cluster")[cols].mean().round(2)
    sizes = d["Cluster"].value_counts().sort_index()
    profile.insert(0, "Size", sizes)
    sil = round(silhouette_score(Xs, labels), 3) if k > 1 else np.nan
    return d, profile.reset_index(), sil, Xs


def cluster_scatter(d, cols):
    x, y = cols[0], cols[1] if len(cols) > 1 else cols[0]
    fig = px.scatter(d, x=x, y=y, color="Cluster",
                     labels={x: label(x), y: label(y)},
                     color_discrete_sequence=["#1E3A8A", "#C8A24B", "#0F9D7E",
                                              "#7C3AED", "#F2772E", "#DC2A52"])
    fig.update_layout(title="Cluster Scatter (2D)")
    return fig


def cluster_scatter_3d(d, cols):
    if len(cols) < 3:
        return None
    fig = px.scatter_3d(d, x=cols[0], y=cols[1], z=cols[2], color="Cluster",
                        color_discrete_sequence=["#1E3A8A", "#C8A24B", "#0F9D7E",
                                                 "#7C3AED", "#F2772E", "#DC2A52"])
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(title="Cluster Scatter (3D)")
    return fig
