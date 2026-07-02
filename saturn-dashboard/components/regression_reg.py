import numpy as np
import pandas as pd
import streamlit as st

from utils.theme import kpi_card, kpi_row, section, PALETTE
from utils import regression_reg as RG
from utils.data_loader import (numeric_columns, categorical_columns, label,
                               LIKERT_COLS)
from components._common import chart, require_data

ACC = PALETTE["teal"]


def render():
    require_data()
    section("Regression Lab", "Linear, Ridge and Lasso regression side by side. Watch how L2 and L1 "
            "penalties shrink coefficients, tune the strength with cross-validation, and let Lasso "
            "select features automatically.", ACC)

    df = st.session_state["df"]
    nums = numeric_columns(df)
    cats = categorical_columns(df)

    catalogue = RG.target_catalogue(df)
    if not catalogue:
        st.info("No numeric target available. Coerce the Likert ratings on the Cleaning page first.")
        return

    keys = list(catalogue.keys())
    default_key = next((k for k in keys if "Saturn Consideration" in k), keys[0])
    c1, c2 = st.columns([1, 1])
    tgt_label = c1.selectbox("Target variable (what to predict)", keys,
                             index=keys.index(default_key))
    kind, tcol = catalogue[tgt_label]
    y = RG.build_target(df, kind, tcol)

    feat_pool = [c for c in (nums + cats) if c != tcol]
    default_feats = [c for c in (LIKERT_COLS + ["income", "occupation", "formal_attire_freq",
                     "age_group", "shopping_channel"]) if c in feat_pool]
    feats = c2.multiselect("Predictor features", feat_pool,
                           default=default_feats[:12], format_func=label)

    if len(feats) < 2:
        st.caption("Select at least two predictor features.")
        return

    X, yv, labels = RG.build_xy(df, y, feats)
    if len(yv) < 30:
        st.warning("Too few complete rows for a reliable fit after dropping missing values.")
        return

    # ---- Regularisation strength ----
    section("Regularisation Strength (α)", "Higher α = stronger penalty = more shrinkage. "
            "Use the cross-validated suggestion or explore manually.", ACC)
    if st.button("✨  Suggest best α via cross-validation", use_container_width=True):
        with st.spinner("Cross-validating…"):
            ar, al = RG.suggest_alphas(X, yv)
            st.session_state["reg_alpha"] = (ar, al)

    sug = st.session_state.get("reg_alpha")
    a1, a2 = st.columns(2)
    ridge_alpha = a1.select_slider("Ridge α (L2)", options=RG.ALPHA_GRID,
                                   value=_closest(sug[0]) if sug else 1.0)
    lasso_alpha = a2.select_slider("Lasso α (L1)", options=RG.ALPHA_GRID,
                                   value=_closest(sug[1]) if sug else 0.1)
    if sug:
        st.caption(f"Cross-validation suggests Ridge α ≈ **{sug[0]:g}** and Lasso α ≈ **{sug[1]:g}** "
                   f"(sliders snapped to the nearest preset).")

    res = RG.fit_models(X, yv, ridge_alpha, lasso_alpha, labels)

    st.write("")
    section("Model Comparison", "Ranked by R² on a held-out 25% test split.", ACC)
    st.dataframe(res["results"], use_container_width=True, hide_index=True)
    best = res["results"].iloc[0]
    lasso_row = res["results"][res["results"]["Model"].str.startswith("Lasso")].iloc[0]
    kpi_row([
        kpi_card("Best Model", best["Model"].split(" (")[0], "🏆", PALETTE["teal"]),
        kpi_card("R² Score", f"{best['R²']:.3f}", "📈", PALETTE["emerald"]),
        kpi_card("RMSE", f"{best['RMSE']:.3f}", "📏", PALETTE["orange"]),
        kpi_card("Lasso keeps", f"{int(lasso_row['Non-zero coefs'])}/{len(res['features'])}",
                 "✂️", PALETTE["crimson"], "features selected"),
    ])

    st.write("")
    tabs = st.tabs(["⚖️ Coefficients", "🛤️ Regularisation Paths", "🎚️ α Tuning (CV)", "🔬 Fit Diagnostics"])

    # -------------------------------------------------- Coefficients
    with tabs[0]:
        section("Coefficient Shrinkage", "Standardised coefficients across the three models — "
                "Ridge pulls values toward zero, Lasso drives some exactly to zero.", ACC)
        top = st.slider("Features to show", 5, min(20, len(res["coef_df"])),
                        min(12, len(res["coef_df"])))
        chart(RG.coef_compare_fig(res["coef_df"], top), "Regression Lab", height=460)
        st.dataframe(res["coef_df"].round(3), use_container_width=True, height=320)
        st.download_button("⬇️ Download coefficients (CSV)",
                           res["coef_df"].to_csv().encode(),
                           "saturn_regression_coefficients.csv", "text/csv")

    # -------------------------------------------------- Paths
    with tabs[1]:
        section("Regularisation Paths", "Each line is one feature’s coefficient as α increases. "
                "Ridge shrinks smoothly; Lasso snaps features to zero one by one.", ACC)
        p1, p2 = st.columns(2)
        with p1:
            chart(RG.ridge_path_fig(X, yv, labels), "Regression Lab", height=420)
        with p2:
            chart(RG.lasso_path_fig(X, yv, labels), "Regression Lab", height=420)
        st.caption("Where a Lasso line hits zero and stays there, that predictor has been dropped "
                   "from the model at that penalty level.")

    # -------------------------------------------------- CV
    with tabs[2]:
        section("Cross-Validated α", "5-fold CV RMSE across a grid of penalties — the dashed line "
                "marks the α that minimises error.", ACC)
        cc1, cc2 = st.columns(2)
        with cc1:
            fig_r, best_r = RG.cv_curve_fig(X, yv, "ridge")
            chart(fig_r, "Regression Lab", height=400)
            st.caption(f"Ridge CV-optimal α ≈ **{best_r:g}**.")
        with cc2:
            fig_l, best_l = RG.cv_curve_fig(X, yv, "lasso")
            chart(fig_l, "Regression Lab", height=400)
            st.caption(f"Lasso CV-optimal α ≈ **{best_l:g}**.")

        section("Lasso Feature Selection", "What survived and what was eliminated at the current "
                f"Lasso α = {lasso_alpha:g}.", ACC)
        for line in RG.lasso_selection(res["coef_df"]):
            st.markdown(f"- {line}")

    # -------------------------------------------------- Diagnostics
    with tabs[3]:
        model_names = list(res["fitted"].keys())
        which = st.selectbox("Model to inspect", model_names,
                             index=model_names.index(res["best"]))
        pred = res["preds"][which]
        d1, d2 = st.columns(2)
        with d1:
            chart(RG.pred_vs_actual_fig(res["yte"], pred, which.split(" (")[0]),
                  "Regression Lab", height=420)
        with d2:
            chart(RG.residual_fig(res["yte"], pred, which.split(" (")[0]),
                  "Regression Lab", height=420)
        out = pd.DataFrame({"Actual": np.asarray(res["yte"]), "Predicted": np.round(pred, 3)})
        st.download_button("⬇️ Download predictions (CSV)", out.to_csv(index=False).encode(),
                           "saturn_regression_predictions.csv", "text/csv")


def _closest(val):
    return min(RG.ALPHA_GRID, key=lambda a: abs(a - val))
