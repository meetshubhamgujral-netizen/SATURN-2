import numpy as np
import pandas as pd
import streamlit as st

from utils.theme import kpi_card, kpi_row, section, PALETTE
from utils import assoc as A
from utils.data_loader import label
from components._common import chart, require_data

ACC = PALETTE["fuchsia"]


def render():
    require_data()
    section("Association Rules", "Market-basket analysis on the multi-select answers. Discover which "
            "products, occasions, brands and trial-triggers travel together — measured by support, "
            "confidence and lift.", ACC)

    df = st.session_state["df"]
    baskets = A.basket_columns(df)
    if not baskets:
        st.info("No multi-select columns found in this dataset (e.g. accessories purchased, "
                "brands purchased, occasions, trial triggers).")
        return

    default_basket = ["accessories_purchased"] if "accessories_purchased" in baskets else baskets[:1]
    cols = st.multiselect("Basket columns (pick one, or mix several for cross-category rules)",
                          baskets, default=default_basket, format_func=label)
    if not cols:
        st.caption("Select at least one basket column.")
        return

    onehot = A.build_onehot(df, cols)
    if onehot.empty or onehot.shape[1] < 2:
        st.warning("Not enough items to mine rules from this selection.")
        return

    # ---- Controls ----
    section("Mining Parameters", "Support = how often; confidence = how reliable; lift = how much "
            "stronger than chance.", ACC)
    c1, c2, c3, c4 = st.columns(4)
    min_sup = c1.slider("Min support", 0.01, 0.50, 0.05, 0.01,
                        help="Minimum fraction of baskets an itemset must appear in.")
    min_conf = c2.slider("Min confidence", 0.05, 1.0, 0.30, 0.05,
                         help="Minimum P(consequent | antecedent).")
    min_lift = c3.slider("Min lift", 1.0, 5.0, 1.0, 0.1,
                         help="1.0 = independent; > 1 = positive association.")
    max_len = c4.slider("Max itemset size", 2, 4, 3)

    fi = A.frequent_itemsets(onehot, min_sup, max_len)
    rules = A.association_rules(fi, min_conf, min_lift) if not fi.empty else pd.DataFrame()

    kpi_row([
        kpi_card("Transactions", f"{len(onehot):,}", "🧺", PALETTE["fuchsia"]),
        kpi_card("Distinct items", f"{onehot.shape[1]}", "🏷️", PALETTE["royal"]),
        kpi_card("Frequent itemsets", f"{len(fi)}", "📦", PALETTE["emerald"]),
        kpi_card("Rules found", f"{len(rules)}", "🔗", PALETTE["crimson"]),
    ])

    if fi.empty:
        st.warning("No itemsets meet the minimum support. Lower the support threshold.")
        return

    st.write("")
    tabs = st.tabs(["🎯 Item Frequency", "📦 Frequent Itemsets", "🔗 Rules", "🕸️ Rule Network"])

    # -------------------------------------------------- Item frequency
    with tabs[0]:
        section("Most Frequent Items", "How often each individual item appears across all baskets.", ACC)
        chart(A.item_freq_fig(onehot), "Association Rules", height=460)
        freq = (onehot.mean().sort_values(ascending=False) * 100).round(1)
        ft = freq.reset_index()
        ft.columns = ["Item", "Support %"]
        st.dataframe(ft, use_container_width=True, hide_index=True, height=300)

    # -------------------------------------------------- Frequent itemsets
    with tabs[1]:
        section("Frequent Itemsets", "Combinations that co-occur often enough to matter.", ACC)
        show = fi.copy()
        show["Itemset"] = show["itemsets"].apply(lambda s: " + ".join(sorted(s)))
        show["Support"] = show["support"].round(3)
        show = show.rename(columns={"length": "Size"})[["Itemset", "Size", "Support"]]
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(show, use_container_width=True, hide_index=True, height=420)
        with c2:
            chart(A.itemset_bar(fi), "Association Rules", height=420)

    # -------------------------------------------------- Rules
    with tabs[2]:
        if rules.empty:
            st.warning("No rules meet the current confidence / lift thresholds. "
                       "Try lowering min confidence or min lift.")
        else:
            section("Association Rules", "Sorted by lift — the strongest, most surprising "
                    "co-occurrences first.", ACC)
            sort_by = st.radio("Sort by", ["lift", "confidence", "support"], horizontal=True,
                               format_func=str.capitalize)
            rsorted = rules.sort_values(sort_by, ascending=False).reset_index(drop=True)
            st.dataframe(rsorted, use_container_width=True, hide_index=True, height=360)

            st.write("")
            section("Rule Landscape", "Every rule plotted by support and confidence; "
                    "bigger, brighter points have higher lift.", ACC)
            chart(A.scatter_fig(rsorted), "Association Rules", height=460)

            section("Plain-English Read-out", "Top rules translated into business language.", ACC)
            for line in A.narrative(rsorted):
                st.markdown(f"- {line}")

            st.download_button("⬇️ Download rules (CSV)", rsorted.to_csv(index=False).encode(),
                               "saturn_association_rules.csv", "text/csv")

    # -------------------------------------------------- Network
    with tabs[3]:
        if rules.empty:
            st.caption("No rules to visualise — relax the thresholds on the Rules tab.")
        else:
            section("Rule Network", "Arrows point from antecedent to consequent; thicker links "
                    "mean higher lift.", ACC)
            top_n = st.slider("Number of top rules to graph", 3, min(25, len(rules)),
                              min(12, len(rules)))
            chart(A.network_fig(rules.sort_values("lift", ascending=False), top_n),
                  "Association Rules", height=560)
            st.caption("Clusters of tightly linked items suggest natural product bundles or "
                       "co-marketing opportunities for Saturn.")
