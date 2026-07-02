import numpy as np
import pandas as pd
import streamlit as st

from utils.theme import kpi_card, kpi_row, section, PALETTE
from utils import predictive as P
from utils.data_loader import (numeric_columns, categorical_columns, label,
                               LIKERT_COLS, ORDERED)
from components._common import chart, require_data

ACC = PALETTE["crimson"]


def render():
    require_data()
    section("Predictive Analytics", "Train models on the fly. The app auto-detects whether your "
            "target needs classification or regression — plus K-Means customer segmentation.", ACC)

    df = st.session_state["df"]
    nums = numeric_columns(df)
    cats = categorical_columns(df)

    mode = st.tabs(["🎯 Classification / Regression", "🧩 Customer Segmentation (K-Means)"])

    # =============================================================== Supervised
    with mode[0]:
        good_targets = [c for c in (cats + nums)
                        if c in df.columns and df[c].nunique(dropna=True) > 1]
        default_t = "saturn_consideration" if "saturn_consideration" in good_targets else good_targets[0]
        c1, c2 = st.columns([1, 1])
        target = c1.selectbox("Target variable (what to predict)", good_targets,
                              index=good_targets.index(default_t), format_func=label)
        detected = P.detect_task(df, target)
        is_num_target = target in nums
        with c2:
            if is_num_target:
                task = st.radio(
                    "Task type", ["Classification", "Regression"],
                    index=0 if detected == "Classification" else 1, horizontal=True,
                    help=f"Auto-detected: {detected}. A 1–5 rating can be modelled either "
                         f"as 5 classes or as a continuous score — your choice.")
            else:
                task = "Classification"
                st.markdown(
                    f'<div class="glass" style="text-align:center">Detected task<br>'
                    f'<span style="font-family:Poppins;font-size:22px;font-weight:800;'
                    f'color:{ACC}">{task}</span></div>', unsafe_allow_html=True)

        feat_pool = [c for c in (nums + cats) if c != target]
        default_feats = [c for c in (LIKERT_COLS + ["income", "occupation", "formal_attire_freq",
                         "age_group", "shopping_channel"]) if c in feat_pool]
        feats = st.multiselect("Predictor features", feat_pool,
                               default=default_feats[:12], format_func=label)

        if st.button("🚀  Train models", use_container_width=True):
            if len(feats) < 2:
                st.warning("Select at least two predictor features.")
            else:
                with st.spinner("Training models…"):
                    X, y = P.build_xy(df, target, feats)
                    st.session_state["pred_cache"] = {"task": task, "target": target}
                    if task == "Classification":
                        st.session_state["pred_cache"]["res"] = P.run_classification(X, y)
                    else:
                        st.session_state["pred_cache"]["res"] = P.run_regression(X, y)

        cache = st.session_state.get("pred_cache")
        if cache and cache.get("target") == target:
            if cache["task"] == "Classification":
                _show_classification(cache["res"])
            else:
                _show_regression(cache["res"])

    # ============================================================== Clustering
    with mode[1]:
        section("K-Means Customer Segmentation", "Group respondents by their importance ratings "
                "to reveal distinct consumer personas.", ACC)
        feat_cols = st.multiselect("Clustering features", nums,
                                   default=[c for c in LIKERT_COLS if c in nums][:6],
                                   format_func=label)
        if len(feat_cols) < 2:
            st.caption("Select at least two features.")
            return

        cc1, cc2 = st.columns(2)
        with cc1:
            efig, ek = P.elbow(df, feat_cols)
            chart(efig, "Predictive Analytics", height=360)
            st.caption(f"Elbow suggests **k ≈ {ek}** (largest bend in WCSS).")
        with cc2:
            sfig, sk, sscore = P.silhouette(df, feat_cols)
            chart(sfig, "Predictive Analytics", height=360)
            st.caption(f"Best silhouette at **k = {sk}** (score {sscore}).")

        k = st.slider("Number of clusters (k)", 2, 8, int(sk))
        d_lab, profile, sil, _ = P.run_kmeans(df, feat_cols, k)
        kpi_row([
            kpi_card("Clusters", f"{k}", "🧩", PALETTE["crimson"]),
            kpi_card("Silhouette", f"{sil}", "📐", PALETTE["emerald"]),
            kpi_card("Respondents", f"{len(d_lab):,}", "👥", PALETTE["royal"]),
        ])
        st.write("")
        section("Cluster Profiles", "Mean importance score per segment (higher = matters more).", ACC)
        st.dataframe(profile, use_container_width=True, hide_index=True)

        v1, v2 = st.columns(2)
        with v1:
            chart(P.cluster_scatter(d_lab, feat_cols), "Predictive Analytics", height=440)
        with v2:
            fig3d = P.cluster_scatter_3d(d_lab, feat_cols)
            if fig3d is not None:
                chart(fig3d, "Predictive Analytics", height=440)
            else:
                st.caption("Select at least three features for a 3D plot.")

        st.download_button("⬇️ Download cluster assignments",
                           d_lab.to_csv(index=False).encode(),
                           "saturn_clusters.csv", "text/csv")


# --------------------------------------------------------------------------- #
def _show_classification(res):
    st.write("")
    section("Model Comparison", "Ranked by weighted F1 score.", ACC)
    st.dataframe(res["results"], use_container_width=True, hide_index=True)
    best = res["results"].iloc[0]
    kpi_row([
        kpi_card("Best Model", best["Model"], "🏆", PALETTE["crimson"]),
        kpi_card("Accuracy", f"{best['Accuracy']:.3f}", "🎯", PALETTE["royal"]),
        kpi_card("F1 Score", f"{best['F1 Score']:.3f}", "⚖️", PALETTE["emerald"]),
        kpi_card("Recall", f"{best['Recall']:.3f}", "🔁", PALETTE["purple"]),
    ])
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        chart(P.confusion_fig(res["yte"], res["pred"], res["classes"]),
              "Predictive Analytics", height=420)
    with c2:
        roc, auc_val = P.roc_fig(res["model"], res["Xte"], res["yte"], res["classes"])
        if roc is not None:
            chart(roc, "Predictive Analytics", height=420)
        else:
            msg = f"Macro OvR AUC ≈ {auc_val:.3f}" if auc_val else "AUC not available"
            st.markdown(f'<div class="glass" style="height:380px;display:flex;align-items:center;'
                        f'justify-content:center;text-align:center">ROC curve shown for binary targets.'
                        f'<br>Multiclass target → {msg}.</div>', unsafe_allow_html=True)
    fi = P.feature_importance_fig(res["model"], res["features"])
    if fi is not None:
        chart(fi, "Predictive Analytics", height=420)


def _show_regression(res):
    st.write("")
    section("Model Comparison", "Ranked by R² score.", ACC)
    st.dataframe(res["results"], use_container_width=True, hide_index=True)
    best = res["results"].iloc[0]
    kpi_row([
        kpi_card("Best Model", best["Model"], "🏆", PALETTE["crimson"]),
        kpi_card("R² Score", f"{best['R² Score']:.3f}", "📈", PALETTE["emerald"]),
        kpi_card("RMSE", f"{best['RMSE']:.3f}", "📏", PALETTE["orange"]),
        kpi_card("MAE", f"{best['MAE']:.3f}", "📐", PALETTE["royal"]),
    ])
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        chart(P.pred_vs_actual_fig(res["yte"], res["pred"]), "Predictive Analytics", height=420)
    with c2:
        chart(P.residual_fig(res["yte"], res["pred"]), "Predictive Analytics", height=420)
    fi = P.feature_importance_fig(res["model"], res["features"])
    if fi is not None:
        chart(fi, "Predictive Analytics", height=420)
