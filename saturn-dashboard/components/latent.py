import numpy as np
import pandas as pd
import streamlit as st

from utils.theme import kpi_card, kpi_row, section, PALETTE
from utils import latent as L
from utils.data_loader import numeric_columns, categorical_columns, label, LIKERT_COLS
from components._common import chart, require_data

ACC = PALETTE["indigo"]


def render():
    require_data()
    section("Latent Analysis", "Uncover the hidden dimensions behind the rating questions — "
            "principal components and rotated factors reveal the few underlying motivations "
            "that drive many correlated answers.", ACC)

    df = st.session_state["df"]
    nums = numeric_columns(df)
    if len(nums) < 3:
        st.info("Latent analysis needs at least three numeric (Likert) variables. "
                "Coerce the importance ratings on the Cleaning page first.")
        return

    default = [c for c in LIKERT_COLS if c in nums] or nums
    cols = st.multiselect("Variables to analyse (importance ratings recommended)", nums,
                          default=default, format_func=label)
    if len(cols) < 3:
        st.caption("Select at least three variables.")
        return

    d, Xs = L.prep(df, cols)
    if len(d) < 10:
        st.warning("Not enough complete rows for a stable solution after dropping missing values.")
        return

    fac = L.factorability(d)
    pca = L.pca_fit(d, Xs)

    verdict = L.kmo_verdict(fac["kmo"])
    sig = "significant ✅" if fac["p"] < 0.05 else "not significant ⚠️"
    kpi_row([
        kpi_card("Variables", f"{len(cols)}", "🧬", PALETTE["indigo"], f"{len(d):,} complete rows"),
        kpi_card("KMO", f"{fac['kmo']:.2f}", "🧪", PALETTE["royal"], f"{verdict} adequacy"),
        kpi_card("Bartlett p", f"{fac['p']:.1e}", "📐", PALETTE["emerald"], sig),
        kpi_card("Suggested factors", f"{pca['kaiser']}", "🧭", PALETTE["crimson"], "Kaiser λ > 1"),
    ])

    st.markdown(
        f'<div class="glass" style="margin-top:6px">🧪 <b>Factorability.</b> The '
        f'Kaiser–Meyer–Olkin measure is <b>{fac["kmo"]:.2f}</b> ({verdict.lower()}), and '
        f'Bartlett’s test of sphericity is <b>{sig.split()[0]}</b> '
        f'(χ² = {fac["chi2"]:.0f}, p = {fac["p"]:.1e}) — the ratings are '
        f'{"suitably" if fac["kmo"] >= 0.5 and fac["p"] < 0.05 else "weakly"} '
        f'inter-correlated for latent-variable extraction.</div>',
        unsafe_allow_html=True)

    st.write("")
    tabs = st.tabs(["🧭 Principal Components (PCA)", "🧩 Factor Analysis (EFA)", "🧪 Diagnostics"])

    # ============================================================ PCA
    with tabs[0]:
        section("How many components?", "The scree plot and cumulative variance show how much "
                "information each component retains.", ACC)
        c1, c2 = st.columns(2)
        with c1:
            chart(L.scree_fig(pca), "Latent Analysis", height=380)
        with c2:
            chart(L.variance_fig(pca), "Latent Analysis", height=380)

        st.dataframe(pca["var_df"], use_container_width=True, hide_index=True, height=300)

        st.write("")
        section("Component Loadings", "How strongly each rating aligns with each component "
                "(red = positive, blue = negative).", ACC)
        keep = st.slider("Components to display", 2, len(cols), min(pca["kaiser"] + 1, len(cols)))
        chart(L.loadings_heatmap(pca["loadings"], keep), "Latent Analysis", height=460)

        st.write("")
        section("PCA Biplot", "Respondents projected onto two components, with rating vectors "
                "overlaid. Optionally colour points by a segment.", ACC)
        cc1, cc2, cc3 = st.columns([1, 1, 1])
        cats = categorical_columns(df)
        color_by = cc1.selectbox("Colour points by", ["(none)"] + cats,
                                 format_func=lambda x: label(x) if x != "(none)" else x)
        pcx = cc2.selectbox("X axis", pca["names"], index=0)
        pcy = cc3.selectbox("Y axis", pca["names"], index=1)
        cser = None if color_by == "(none)" else df[color_by]
        chart(L.biplot(pca, cser, pcx, pcy), "Latent Analysis", height=520)

        section("Plain-English Read-out", "", ACC)
        for line in L.pca_narrative(pca, keep):
            st.markdown(f"- {line}")

        st.download_button("⬇️ Download component loadings (CSV)",
                           pca["loadings"].to_csv().encode(),
                           "saturn_pca_loadings.csv", "text/csv")

    # ============================================================ Factor Analysis
    with tabs[1]:
        section("Exploratory Factor Analysis", "Extract latent constructs and rotate them for a "
                "clean, interpretable structure.", ACC)
        c1, c2 = st.columns([1, 1])
        n_factors = c1.slider("Number of factors", 1, max(len(cols) - 1, 1),
                              min(max(pca["kaiser"], 2), len(cols) - 1))
        rotation = c2.selectbox("Rotation", ["varimax", "promax", "oblimin", "none"], index=0,
                                help="Varimax (orthogonal) is the most common; promax/oblimin allow "
                                     "correlated factors.")
        fa = L.fa_fit(d, n_factors, rotation)

        section("Factor Loadings", "Which ratings define each latent factor.", ACC)
        chart(L.loadings_heatmap(fa["loadings"], title="Rotated Factor Loadings"),
              "Latent Analysis", height=440)

        pick = st.selectbox("Inspect one factor", fa["names"])
        f1, f2 = st.columns([1, 1])
        with f1:
            chart(L.fa_loadings_bar(fa["loadings"], pick), "Latent Analysis", height=380)
        with f2:
            chart(L.communalities_fig(fa["communalities"]), "Latent Analysis", height=380)

        st.write("")
        v1, v2 = st.columns([1.1, 1])
        with v1:
            section("Variance Explained", "", ACC)
            st.dataframe(fa["variance"], use_container_width=True, hide_index=True)
        with v2:
            section("Communalities", "Shared variance per variable.", ACC)
            st.dataframe(fa["communalities"], use_container_width=True, hide_index=True, height=260)

        section("Interpreting the Factors", "Automatic naming from dominant loadings.", ACC)
        for line in L.fa_narrative(fa["loadings"]):
            st.markdown(f"- {line}")

        st.download_button("⬇️ Download factor loadings (CSV)",
                           fa["loadings"].to_csv().encode(),
                           "saturn_factor_loadings.csv", "text/csv")

    # ============================================================ Diagnostics
    with tabs[2]:
        section("Sampling Adequacy per Variable", "KMO for each rating — values below 0.5 are weak "
                "candidates and could be dropped.", ACC)
        kdf = (fac["kmo_per"].rename("KMO").reset_index()
               .rename(columns={"index": "variable"}))
        kdf["Variable"] = [label(c) for c in kdf["variable"]]
        kdf["KMO"] = kdf["KMO"].round(3)
        import plotly.express as px
        f = px.bar(kdf.sort_values("KMO"), x="KMO", y="Variable", orientation="h",
                   color="KMO", color_continuous_scale="Teal", range_color=[0, 1])
        f.add_vline(x=0.5, line_dash="dash", line_color="#DC2A52")
        f.update_layout(title="KMO by variable (0.5 threshold)", coloraxis_showscale=False)
        chart(f, "Latent Analysis", height=420)

        st.dataframe(kdf[["Variable", "KMO"]].sort_values("KMO", ascending=False),
                     use_container_width=True, hide_index=True, height=320)
        st.caption("KMO overall ≥ 0.6 and a significant Bartlett test indicate the correlation "
                   "structure is suitable for PCA / factor analysis.")
