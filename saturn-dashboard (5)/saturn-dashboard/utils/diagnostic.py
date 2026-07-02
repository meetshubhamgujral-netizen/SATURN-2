"""Diagnostic analytics: correlation matrices and automatic relationship flags."""
import numpy as np
import pandas as pd
import plotly.express as px

from utils.data_loader import coerce_likert, numeric_columns, label


def corr_matrix(df, method="pearson"):
    d = coerce_likert(df)[numeric_columns(df)]
    d = d.dropna()
    return d.corr(method=method)


def heatmap(corr):
    disp = corr.copy()
    disp.index = [label(c) for c in disp.index]
    disp.columns = [label(c) for c in disp.columns]
    fig = px.imshow(disp, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(title=f"Correlation Matrix ({len(corr)} variables)")
    return fig


def top_relationships(corr, k=8):
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i, j]))
    pairs = [p for p in pairs if not np.isnan(p[2])]
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    rows = []
    for a, b, r in pairs[:k]:
        rows.append({
            "Variable A": label(a),
            "Variable B": label(b),
            "Correlation": round(r, 2),
            "Strength": _strength(r),
            "Direction": "Positive" if r > 0 else "Negative",
        })
    return pd.DataFrame(rows)


def _strength(r):
    a = abs(r)
    if a >= 0.7:
        return "Very strong"
    if a >= 0.5:
        return "Strong"
    if a >= 0.3:
        return "Moderate"
    if a >= 0.15:
        return "Weak"
    return "Negligible"


def scatter_matrix(df, cols):
    d = coerce_likert(df)[cols].dropna()
    fig = px.scatter_matrix(d, dimensions=cols,
                            labels={c: label(c) for c in cols},
                            color_discrete_sequence=["#1E3A8A"], opacity=0.5)
    fig.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=3))
    fig.update_layout(title="Scatter Matrix")
    return fig


def narrative(corr):
    """Plain-English bullets describing the strongest relationships."""
    tr = top_relationships(corr, k=6)
    lines = []
    for _, row in tr.iterrows():
        verb = "rises with" if row["Direction"] == "Positive" else "falls as"
        lines.append(
            f"**{row['Variable A']}** {verb} **{row['Variable B']}** "
            f"({row['Strength'].lower()}, r = {row['Correlation']})."
        )
    # Multicollinearity flag
    high = tr[tr["Correlation"].abs() >= 0.7]
    if len(high):
        lines.append(
            "⚠️ Possible **multicollinearity** between: "
            + "; ".join(f"{r['Variable A']} & {r['Variable B']}" for _, r in high.iterrows())
            + " — consider dropping one before modelling."
        )
    return lines
