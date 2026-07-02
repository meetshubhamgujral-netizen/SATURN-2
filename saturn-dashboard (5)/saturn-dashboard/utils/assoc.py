"""Association-rule mining (market-basket analysis) over the survey's multi-select
answers — occasions, accessories purchased, brands purchased and trial triggers.

Uses `mlxtend` (apriori + association_rules) when available, otherwise a compact,
fully vectorised apriori implementation so the page always runs.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import explode_multiselect, MULTISELECT_COLS, label

try:
    from mlxtend.frequent_patterns import apriori as _apriori
    from mlxtend.frequent_patterns import association_rules as _assoc_rules
    _HAS_MLX = True
except Exception:                                              # pragma: no cover
    _HAS_MLX = False

PREFIX = {"occasions": "Occasion", "accessories_purchased": "Product",
          "brands_purchased": "Brand", "convince_try": "Trigger"}


def basket_columns(df):
    return [c for c in MULTISELECT_COLS if c in df.columns]


# --------------------------------------------------------------------------- #
# Transaction encoding
# --------------------------------------------------------------------------- #
def build_onehot(df, cols):
    """One-hot 'transaction' matrix. Items are prefixed when >1 column is mixed."""
    multi = len(cols) > 1
    frames = []
    for c in cols:
        s = df[c].dropna().astype(str)
        parts = s.str.split(";").apply(lambda xs: [x.strip() for x in xs if x.strip()])
        tag = PREFIX.get(c, label(c))
        dummies = (parts.apply(lambda items: pd.Series(
            {(f"{tag}: {i}" if multi else i): True for i in items}))
            .fillna(False))
        frames.append(dummies)
    onehot = pd.concat(frames, axis=1).fillna(False)
    onehot = onehot.loc[onehot.any(axis=1)]                    # drop empty baskets
    # collapse any duplicate columns (same item from overlapping cols) — version-safe
    if onehot.columns.duplicated().any():
        onehot = onehot.T.groupby(level=0).any().T
    return onehot.astype(bool)


# --------------------------------------------------------------------------- #
# Frequent itemsets
# --------------------------------------------------------------------------- #
def frequent_itemsets(onehot, min_support=0.05, max_len=3):
    if _HAS_MLX:
        fi = _apriori(onehot, min_support=min_support, use_colnames=True, max_len=max_len)
        if fi.empty:
            return fi
        fi["length"] = fi["itemsets"].apply(len)
        return fi.sort_values("support", ascending=False).reset_index(drop=True)
    return _apriori_manual(onehot, min_support, max_len)


def _apriori_manual(onehot, min_support, max_len):
    M = onehot.values
    cols = list(onehot.columns)
    n = len(M)
    # 1-itemsets
    supp = M.mean(axis=0)
    freq = [({cols[i]}, supp) for i, supp in enumerate(supp) if supp >= min_support]
    results = [(frozenset(s), sup) for s, sup in freq]
    current = [frozenset(s) for s, _ in freq]
    idx = {c: i for i, c in enumerate(cols)}
    k = 2
    while current and k <= max_len:
        cand = set()
        cl = sorted(current, key=lambda x: sorted(x))
        for a in range(len(cl)):
            for b in range(a + 1, len(cl)):
                union = cl[a] | cl[b]
                if len(union) == k:
                    cand.add(frozenset(union))
        nxt = []
        for c in cand:
            colidx = [idx[i] for i in c]
            sup = M[:, colidx].all(axis=1).mean()
            if sup >= min_support:
                results.append((c, sup))
                nxt.append(c)
        current = nxt
        k += 1
    fi = pd.DataFrame({"support": [s for _, s in results],
                       "itemsets": [i for i, _ in results]})
    fi["length"] = fi["itemsets"].apply(len)
    return fi.sort_values("support", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
def association_rules(fi, min_confidence=0.3, min_lift=1.0):
    if fi.empty or (fi["length"] >= 2).sum() == 0:
        return pd.DataFrame()
    if _HAS_MLX:
        rules = _assoc_rules(fi, metric="confidence", min_threshold=min_confidence)
        rules = rules[rules["lift"] >= min_lift].copy()
    else:
        rules = _rules_manual(fi, min_confidence, min_lift)
    if rules.empty:
        return rules
    rules["antecedents_str"] = rules["antecedents"].apply(_fmt)
    rules["consequents_str"] = rules["consequents"].apply(_fmt)
    keep = ["antecedents_str", "consequents_str", "support", "confidence",
            "lift", "leverage", "conviction"]
    out = rules[keep].rename(columns={"antecedents_str": "Antecedent (if)",
                                      "consequents_str": "Consequent (then)"})
    for c in ["support", "confidence", "lift", "leverage", "conviction"]:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).round(3)
    return out.sort_values("lift", ascending=False).reset_index(drop=True)


def _rules_manual(fi, min_confidence, min_lift):
    sup = {frozenset(k): v for k, v in zip(fi["itemsets"], fi["support"])}
    from itertools import combinations
    rows = []
    for items, s in sup.items():
        if len(items) < 2:
            continue
        items = frozenset(items)
        for r in range(1, len(items)):
            for ante in combinations(items, r):
                ante = frozenset(ante)
                cons = items - ante
                if ante not in sup or cons not in sup:
                    continue
                conf = s / sup[ante]
                lift = conf / sup[cons]
                if conf < min_confidence or lift < min_lift:
                    continue
                lev = s - sup[ante] * sup[cons]
                conv = (1 - sup[cons]) / (1 - conf) if conf < 1 else np.inf
                rows.append({"antecedents": ante, "consequents": cons,
                             "support": s, "confidence": conf, "lift": lift,
                             "leverage": lev, "conviction": conv})
    return pd.DataFrame(rows)


def _fmt(fs):
    return ", ".join(sorted(fs))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def item_freq_fig(onehot, top=15):
    s = onehot.mean(axis=0).sort_values(ascending=True).tail(top) * 100
    fig = px.bar(x=s.values, y=s.index, orientation="h",
                 color=s.values, color_continuous_scale="Magenta",
                 labels={"x": "Support (% of baskets)", "y": ""})
    fig.update_layout(title="Most frequent items", coloraxis_showscale=False)
    return fig


def itemset_bar(fi, top=12):
    d = fi[fi["length"] >= 2].head(top).copy()
    if d.empty:
        d = fi.head(top).copy()
    d["label"] = d["itemsets"].apply(lambda s: " + ".join(sorted(s)))
    d = d.sort_values("support")
    fig = px.bar(d, x="support", y="label", orientation="h",
                 color="support", color_continuous_scale="Purp",
                 labels={"support": "Support", "label": ""})
    fig.update_layout(title="Top frequent itemsets", coloraxis_showscale=False)
    return fig


def scatter_fig(rules):
    d = rules.copy()
    d["conf_pct"] = d["confidence"] * 100
    fig = px.scatter(d, x="support", y="conf_pct", size="lift", color="lift",
                     color_continuous_scale="Plasma", size_max=26,
                     hover_data=["Antecedent (if)", "Consequent (then)"],
                     labels={"support": "Support", "conf_pct": "Confidence (%)",
                             "lift": "Lift"})
    fig.update_layout(title="Rule landscape — support vs confidence (size / colour = lift)")
    return fig


def network_fig(rules, top=12):
    """Directed antecedent→consequent graph with a simple circular layout."""
    d = rules.head(top)
    nodes = []
    for _, r in d.iterrows():
        nodes += r["Antecedent (if)"].split(", ") + r["Consequent (then)"].split(", ")
    nodes = list(dict.fromkeys(nodes))
    if not nodes:
        return go.Figure()
    ang = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    pos = {n: (np.cos(a), np.sin(a)) for n, a in zip(nodes, ang)}

    edge_traces = []
    lift_max = d["lift"].max() or 1
    for _, r in d.iterrows():
        for a in r["Antecedent (if)"].split(", "):
            for c in r["Consequent (then)"].split(", "):
                x0, y0 = pos[a]
                x1, y1 = pos[c]
                w = 1 + 4 * (r["lift"] / lift_max)
                edge_traces.append(go.Scatter(
                    x=[x0, x1], y=[y0, y1], mode="lines",
                    line=dict(width=w, color="rgba(192,38,211,0.45)"),
                    hoverinfo="text",
                    text=f"{a} → {c}<br>lift {r['lift']:.2f}, conf {r['confidence']:.0%}",
                    showlegend=False))

    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    node_trace = go.Scatter(
        x=xs, y=ys, mode="markers+text", text=nodes, textposition="top center",
        marker=dict(size=16, color="#4F46E5", line=dict(width=1, color="#fff")),
        hoverinfo="text", showlegend=False)

    fig = go.Figure(edge_traces + [node_trace])
    fig.update_layout(title=f"Rule network — top {len(d)} rules by lift",
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def narrative(rules, top=5):
    lines = []
    for _, r in rules.head(top).iterrows():
        lines.append(
            f"Respondents choosing **{r['Antecedent (if)']}** are "
            f"**{r['lift']:.1f}×** more likely to also choose "
            f"**{r['Consequent (then)']}** "
            f"(confidence {r['confidence']:.0%}, support {r['support']:.0%}).")
    return lines
