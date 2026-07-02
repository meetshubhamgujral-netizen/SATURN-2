"""Latent-structure analytics: factorability (KMO, Bartlett), Principal Component
Analysis and Exploratory Factor Analysis with rotation + automatic interpretation.

Primary path uses `factor_analyzer` (nice rotated solutions & KMO/Bartlett). If that
package is unavailable the module falls back to a self-contained numpy / scipy /
scikit-learn implementation, so the page always runs.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis

from utils.data_loader import coerce_likert, label

# Optional premium backend --------------------------------------------------- #
try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import (
        calculate_kmo, calculate_bartlett_sphericity)
    _HAS_FA = True
except Exception:                                              # pragma: no cover
    _HAS_FA = False


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def prep(df, cols):
    """Return (clean dataframe, standardised ndarray) for the chosen columns."""
    d = coerce_likert(df)[cols].apply(pd.to_numeric, errors="coerce").dropna()
    Xs = StandardScaler().fit_transform(d) if len(d) else np.empty((0, len(cols)))
    return d, Xs


# --------------------------------------------------------------------------- #
# Factorability
# --------------------------------------------------------------------------- #
def factorability(d):
    """KMO (overall + per-variable) and Bartlett's test of sphericity."""
    if _HAS_FA:
        try:
            chi2, p = calculate_bartlett_sphericity(d)
            kmo_per, kmo_all = calculate_kmo(d)
        except Exception:
            return _factorability_manual(d)
        dof = d.shape[1] * (d.shape[1] - 1) / 2
        return {"kmo": float(kmo_all), "kmo_per": pd.Series(kmo_per, index=d.columns),
                "chi2": float(chi2), "dof": dof, "p": float(p)}
    return _factorability_manual(d)


def _factorability_manual(d):
    from scipy.stats import chi2 as chi2_dist
    R = np.corrcoef(d.values, rowvar=False)
    p = R.shape[0]
    n = len(d)
    # Bartlett
    det = np.linalg.det(R)
    det = max(det, 1e-12)
    stat = -(n - 1 - (2 * p + 5) / 6) * np.log(det)
    dof = p * (p - 1) / 2
    pval = float(chi2_dist.sf(stat, dof))
    # KMO from partial correlations
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R)
    d_diag = np.sqrt(np.outer(np.diag(R_inv), np.diag(R_inv)))
    partial = -R_inv / d_diag
    np.fill_diagonal(partial, 0)
    r2 = R ** 2
    np.fill_diagonal(r2, 0)
    p2 = partial ** 2
    kmo_num = r2.sum()
    kmo_den = kmo_num + p2.sum()
    kmo_all = kmo_num / kmo_den if kmo_den else np.nan
    per = r2.sum(axis=0) / (r2.sum(axis=0) + p2.sum(axis=0))
    return {"kmo": float(kmo_all), "kmo_per": pd.Series(per, index=d.columns),
            "chi2": float(stat), "dof": dof, "p": pval}


def kmo_verdict(kmo):
    if kmo >= 0.9:
        return "Marvellous"
    if kmo >= 0.8:
        return "Meritorious"
    if kmo >= 0.7:
        return "Middling"
    if kmo >= 0.6:
        return "Mediocre"
    if kmo >= 0.5:
        return "Miserable"
    return "Unacceptable"


# --------------------------------------------------------------------------- #
# PCA
# --------------------------------------------------------------------------- #
def pca_fit(d, Xs):
    p = Xs.shape[1]
    pca = PCA(n_components=p, random_state=42).fit(Xs)
    eig = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    names = [f"PC{i+1}" for i in range(p)]
    # loadings = eigenvector * sqrt(eigenvalue)  (correlation of item with PC)
    loadings = pca.components_.T * np.sqrt(eig)
    load_df = pd.DataFrame(loadings, index=[label(c) for c in d.columns], columns=names)
    scores = pd.DataFrame(pca.transform(Xs), columns=names, index=d.index)
    kaiser = int((eig > 1).sum())
    var_df = pd.DataFrame({
        "Component": names,
        "Eigenvalue": np.round(eig, 3),
        "Variance %": np.round(evr * 100, 2),
        "Cumulative %": np.round(np.cumsum(evr) * 100, 2),
    })
    return {"pca": pca, "eig": eig, "evr": evr, "names": names,
            "loadings": load_df, "scores": scores, "kaiser": max(kaiser, 1),
            "var_df": var_df}


def scree_fig(res):
    eig, names = res["eig"], res["names"]
    fig = go.Figure()
    fig.add_bar(x=names, y=eig, name="Eigenvalue",
                marker_color="#4F46E5", opacity=0.55)
    fig.add_scatter(x=names, y=eig, mode="lines+markers", name="Trend",
                    line=dict(color="#4F46E5", width=3))
    fig.add_hline(y=1, line_dash="dash", line_color="#DC2A52",
                  annotation_text="Kaiser cut-off (λ = 1)", annotation_position="top right")
    fig.update_layout(title="Scree Plot — eigenvalues by component",
                      yaxis_title="Eigenvalue", xaxis_title="Component")
    return fig


def variance_fig(res):
    var_df = res["var_df"]
    fig = go.Figure()
    fig.add_bar(x=var_df["Component"], y=var_df["Variance %"], name="Variance %",
                marker_color="#0EA5A3")
    fig.add_scatter(x=var_df["Component"], y=var_df["Cumulative %"],
                    mode="lines+markers", name="Cumulative %",
                    line=dict(color="#C8A24B", width=3), yaxis="y")
    fig.add_hline(y=80, line_dash="dot", line_color="#94A3B8",
                  annotation_text="80% explained", annotation_position="bottom right")
    fig.update_layout(title="Explained Variance", yaxis_title="% of total variance",
                      xaxis_title="Component")
    return fig


def loadings_heatmap(load_df, k=None, title="Component Loadings"):
    disp = load_df.iloc[:, :k] if k else load_df
    fig = px.imshow(disp, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    labels=dict(color="Loading"))
    fig.update_layout(title=title)
    return fig


def biplot(res, color_series=None, k1="PC1", k2="PC2"):
    scores, load_df = res["scores"], res["loadings"]
    sc = scores[[k1, k2]].copy()
    if color_series is not None:
        sc["_c"] = color_series.reindex(sc.index).astype(str).values
        fig = px.scatter(sc, x=k1, y=k2, color="_c", opacity=0.55,
                         labels={"_c": ""})
    else:
        fig = px.scatter(sc, x=k1, y=k2, opacity=0.5,
                         color_discrete_sequence=["#4F46E5"])
    # loading vectors (scaled to the score cloud)
    scale = 0.9 * max(sc[k1].abs().max(), sc[k2].abs().max())
    L = load_df[[k1, k2]]
    lmax = np.abs(L.values).max() or 1
    for name, row in L.iterrows():
        x, y = row[k1] / lmax * scale, row[k2] / lmax * scale
        fig.add_annotation(x=x, y=y, ax=0, ay=0, xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True,
                           arrowhead=2, arrowwidth=1.6, arrowcolor="#DC2A52")
        fig.add_annotation(x=x, y=y, text=name, showarrow=False,
                           font=dict(size=11, color="#DC2A52"))
    fig.update_layout(title=f"PCA Biplot — {k1} vs {k2}", showlegend=color_series is not None)
    return fig


# --------------------------------------------------------------------------- #
# Exploratory Factor Analysis
# --------------------------------------------------------------------------- #
def fa_fit(d, n_factors, rotation="varimax"):
    cols = [label(c) for c in d.columns]
    rot = None if rotation == "none" else rotation
    if _HAS_FA:
        try:
            fa = FactorAnalyzer(n_factors=n_factors, rotation=rot, method="minres")
            fa.fit(d)
            load = fa.loadings_
            comm = fa.get_communalities()
            uniq = fa.get_uniquenesses()
            ss, prop, cum = fa.get_factor_variance()
            ev, _ = fa.get_eigenvalues()
        except Exception:
            return _fa_fit_manual(d, n_factors, rotation)
    else:
        return _fa_fit_manual(d, n_factors, rotation)

    names = [f"Factor {i+1}" for i in range(n_factors)]
    load_df = pd.DataFrame(load, index=cols, columns=names)
    comm_df = pd.DataFrame({"Variable": cols,
                            "Communality": np.round(comm, 3),
                            "Uniqueness": np.round(uniq, 3)})
    var_df = pd.DataFrame({"Factor": names,
                           "SS Loadings": np.round(ss, 3),
                           "Variance %": np.round(np.array(prop) * 100, 2),
                           "Cumulative %": np.round(np.array(cum) * 100, 2)})
    return {"loadings": load_df, "communalities": comm_df, "variance": var_df,
            "names": names, "eigenvalues": np.array(ev)}


def _varimax(Phi, gamma=1.0, q=100, tol=1e-6):
    """Kaiser varimax rotation (fallback implementation)."""
    p, k = Phi.shape
    if k < 2:
        return Phi
    R = np.eye(k)
    d = 0.0
    for _ in range(q):
        d_old = d
        Lm = Phi @ R
        diag = np.diag(np.diag(Lm.T @ Lm))
        u, s, vt = np.linalg.svd(Phi.T @ (Lm ** 3 - (gamma / p) * (Lm @ diag)))
        R = u @ vt
        d = np.sum(s)
        if d_old and d / d_old < 1 + tol:
            break
    return Phi @ R


def _fa_fit_manual(d, n_factors, rotation="varimax"):
    cols = [label(c) for c in d.columns]
    Xs = StandardScaler().fit_transform(d)
    fa = FactorAnalysis(n_components=n_factors, random_state=42, rotation=None).fit(Xs)
    load = fa.components_.T
    if rotation in ("varimax", "promax"):
        load = _varimax(load)
    names = [f"Factor {i+1}" for i in range(n_factors)]
    load_df = pd.DataFrame(load, index=cols, columns=names)
    comm = (load ** 2).sum(axis=1)
    uniq = 1 - comm
    ss = (load ** 2).sum(axis=0)
    prop = ss / load.shape[0]
    cum = np.cumsum(prop)
    comm_df = pd.DataFrame({"Variable": cols,
                            "Communality": np.round(comm, 3),
                            "Uniqueness": np.round(uniq, 3)})
    var_df = pd.DataFrame({"Factor": names,
                           "SS Loadings": np.round(ss, 3),
                           "Variance %": np.round(prop * 100, 2),
                           "Cumulative %": np.round(cum * 100, 2)})
    R = np.corrcoef(Xs, rowvar=False)
    ev = np.sort(np.linalg.eigvalsh(R))[::-1]
    return {"loadings": load_df, "communalities": comm_df, "variance": var_df,
            "names": names, "eigenvalues": ev}


def fa_loadings_bar(load_df, factor):
    s = load_df[factor].reindex(load_df[factor].abs().sort_values().index)
    fig = px.bar(x=s.values, y=s.index, orientation="h",
                 color=s.values, color_continuous_scale="RdBu_r",
                 range_color=[-1, 1], labels={"x": "Loading", "y": ""})
    fig.update_layout(title=f"{factor} — variable loadings", coloraxis_showscale=False)
    return fig


def communalities_fig(comm_df):
    s = comm_df.sort_values("Communality")
    fig = px.bar(s, x="Communality", y="Variable", orientation="h",
                 color="Communality", color_continuous_scale="Purples",
                 range_color=[0, 1])
    fig.update_layout(title="Communalities — variance explained per variable",
                      coloraxis_showscale=False)
    return fig


def fa_narrative(load_df, thresh=0.4):
    """Name each latent factor from its dominant variables."""
    lines = []
    for f in load_df.columns:
        loads = load_df[f]
        strong = loads[loads.abs() >= thresh].sort_values(key=abs, ascending=False)
        if strong.empty:
            strong = loads.reindex(loads.abs().sort_values(ascending=False).index).head(2)
        items = ", ".join(f"**{v}** ({loads[v]:+.2f})" for v in strong.index[:4])
        lead = strong.index[0].replace("Importance: ", "") if len(strong) else "—"
        lines.append(f"**{f}** is driven mainly by {items} — a *{lead.lower()}* dimension.")
    return lines


def pca_narrative(res, k):
    lines = []
    for i in range(min(k, len(res["names"]))):
        pc = res["names"][i]
        var = res["var_df"].loc[i, "Variance %"]
        col = res["loadings"][pc]
        top = col.reindex(col.abs().sort_values(ascending=False).index).head(3)
        items = ", ".join(f"**{v}** ({col[v]:+.2f})" for v in top.index)
        lines.append(f"**{pc}** explains **{var:.1f}%** of variance, led by {items}.")
    cum = res["var_df"].loc[k - 1, "Cumulative %"] if k <= len(res["var_df"]) else 100
    lines.append(f"The first **{k}** components together retain **{cum:.1f}%** of the "
                 f"total information in the ratings.")
    return lines
