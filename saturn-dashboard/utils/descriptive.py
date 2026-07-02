"""Descriptive analytics: summary statistics, frequency tables, cross-tabs,
and Plotly chart builders."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from utils.data_loader import (
    coerce_likert, numeric_columns, ORDERED, label, explode_multiselect,
)
from utils.theme import COLORWAY


def describe_numeric(df):
    d = coerce_likert(df)
    cols = numeric_columns(df)
    rows = []
    for c in cols:
        s = d[c].dropna()
        if len(s) < 2:
            continue
        rows.append({
            "Variable": label(c),
            "Count": int(s.count()),
            "Mean": round(s.mean(), 2),
            "Median": round(s.median(), 2),
            "Mode": round(s.mode().iloc[0], 2) if len(s.mode()) else np.nan,
            "Std": round(s.std(), 2),
            "Variance": round(s.var(), 2),
            "Min": round(s.min(), 2),
            "Q1": round(s.quantile(0.25), 2),
            "Q3": round(s.quantile(0.75), 2),
            "Max": round(s.max(), 2),
            "Range": round(s.max() - s.min(), 2),
            "Skewness": round(stats.skew(s), 2),
            "Kurtosis": round(stats.kurtosis(s), 2),
        })
    return pd.DataFrame(rows)


def frequency_table(df, col):
    sample = str(df[col].dropna().iloc[0]) if df[col].notna().any() else ""
    if ";" in sample:
        s = explode_multiselect(df, col)
    else:
        s = df[col].dropna().astype(str)
    vc = s.value_counts()
    out = pd.DataFrame({label(col): vc.index, "Count": vc.values})
    out["Percentage"] = (100 * out["Count"] / out["Count"].sum()).round(1)
    order = ORDERED.get(col)
    if order:
        out["__o"] = out[label(col)].map({v: i for i, v in enumerate(order)})
        out = out.sort_values("__o").drop(columns="__o").reset_index(drop=True)
    return out


def crosstab(df, a, b, normalize=False):
    ct = pd.crosstab(df[a].astype(str), df[b].astype(str),
                     normalize="index" if normalize else False)
    for col, order in ((a, ORDERED.get(a)), (b, ORDERED.get(b))):
        pass
    if ORDERED.get(a):
        ct = ct.reindex([v for v in ORDERED[a] if v in ct.index])
    if ORDERED.get(b):
        ct = ct[[v for v in ORDERED[b] if v in ct.columns]]
    return (ct * 100).round(1) if normalize else ct


# --------------------------------------------------------------------------- #
# Chart builders (all return a plotly figure; theming applied by caller)
# --------------------------------------------------------------------------- #
def bar(df, col, horizontal=True):
    ft = frequency_table(df, col)
    name = label(col)
    fig = px.bar(ft, x="Count" if horizontal else name,
                 y=name if horizontal else "Count",
                 orientation="h" if horizontal else "v",
                 text="Count", color="Count", color_continuous_scale="Blues")
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, title=f"Distribution — {name}")
    if horizontal:
        fig.update_yaxes(categoryorder="total ascending")
    return fig


def pie(df, col, donut=False):
    ft = frequency_table(df, col)
    name = label(col)
    fig = px.pie(ft, names=name, values="Count", hole=0.55 if donut else 0,
                 color_discrete_sequence=COLORWAY)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(title=f"{'Donut' if donut else 'Pie'} — {name}", showlegend=False)
    return fig


def histogram(df, col, bins=20):
    d = coerce_likert(df)
    fig = px.histogram(d, x=col, nbins=bins, color_discrete_sequence=[COLORWAY[0]])
    fig.update_layout(title=f"Histogram — {label(col)}", bargap=0.05)
    return fig


def box(df, col, by=None):
    d = coerce_likert(df)
    fig = px.box(d, y=col, x=by, color=by, color_discrete_sequence=COLORWAY,
                 points="outliers")
    fig.update_layout(title=f"Box Plot — {label(col)}" + (f" by {label(by)}" if by else ""))
    return fig


def violin(df, col, by=None):
    d = coerce_likert(df)
    fig = px.violin(d, y=col, x=by, color=by, box=True, points=False,
                    color_discrete_sequence=COLORWAY)
    fig.update_layout(title=f"Violin — {label(col)}" + (f" by {label(by)}" if by else ""))
    return fig


def scatter(df, x, y, color=None):
    d = coerce_likert(df)
    fig = px.scatter(d, x=x, y=y, color=color, color_discrete_sequence=COLORWAY,
                     opacity=0.65, color_continuous_scale="Viridis")
    fig.update_layout(title=f"{label(y)} vs {label(x)}")
    return fig


def treemap(df, col):
    ft = frequency_table(df, col)
    name = label(col)
    fig = px.treemap(ft, path=[name], values="Count",
                     color="Count", color_continuous_scale="Tealgrn")
    fig.update_layout(title=f"Treemap — {name}")
    return fig


def sunburst(df, a, b):
    grp = df.groupby([df[a].astype(str), df[b].astype(str)]).size().reset_index(name="Count")
    grp.columns = [label(a), label(b), "Count"]
    fig = px.sunburst(grp, path=[label(a), label(b)], values="Count",
                      color="Count", color_continuous_scale="Purp")
    fig.update_layout(title=f"Sunburst — {label(a)} → {label(b)}")
    return fig


def stacked_crosstab(df, a, b, normalize=True):
    ct = crosstab(df, a, b, normalize=normalize)
    fig = go.Figure()
    for i, col in enumerate(ct.columns):
        fig.add_bar(name=str(col), x=ct.index.astype(str), y=ct[col],
                    marker_color=COLORWAY[i % len(COLORWAY)])
    fig.update_layout(barmode="stack",
                      title=f"{label(a)} vs {label(b)}"
                            + (" (% within row)" if normalize else " (counts)"),
                      yaxis_title="%" if normalize else "Count",
                      xaxis_title=label(a))
    return fig


def crosstab_heatmap(df, a, b, normalize=True):
    ct = crosstab(df, a, b, normalize=normalize)
    fig = px.imshow(ct, text_auto=True, aspect="auto",
                    color_continuous_scale="Blues",
                    labels=dict(x=label(b), y=label(a), color="%" if normalize else "Count"))
    fig.update_layout(title=f"Heatmap — {label(a)} vs {label(b)}")
    return fig
