import streamlit as st
from utils.theme import kpi_card, kpi_row, section, PALETTE


def render():
    accent = PALETTE["royal"]
    st.markdown(f"""
    <div class="hero">
      <div class="logo">🪐 SATURN</div>
      <h1>AI Market Understanding Dashboard</h1>
      <p>A premium consumer-intelligence platform for Saturn — an upcoming UAE brand in
      premium ties, pocket squares, dress socks and handkerchiefs. Upload a survey dataset and
      move seamlessly from cleaning to descriptive, diagnostic and predictive analytics,
      ending in board-ready business insights.</p>
      <div>
        <span class="badge">UAE Consumer Survey</span>
        <span class="badge">Self-service BI</span>
        <span class="badge">ML-powered Segmentation</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    df = st.session_state.get("df")
    if df is not None:
        kpi_row([
            kpi_card("Respondents", f"{len(df):,}", "👥", PALETTE["royal"]),
            kpi_card("Variables", f"{df.shape[1]}", "🧬", PALETTE["purple"]),
            kpi_card("Survey Sections", "6", "🗂️", PALETTE["emerald"]),
            kpi_card("Status", "Loaded", "✅", PALETTE["gold"]),
        ])
    else:
        kpi_row([
            kpi_card("Products", "4", "👔", PALETTE["royal"], "Ties · Squares · Socks · Hankies"),
            kpi_card("Market", "UAE", "🇦🇪", PALETTE["purple"]),
            kpi_card("Survey Qs", "30", "📝", PALETTE["emerald"]),
            kpi_card("Dataset", "Not loaded", "📁", PALETTE["gold"]),
        ])

    st.write("")
    section("What this dashboard does", "Ten connected modules, one analytics pipeline.", accent)
    c1, c2, c3 = st.columns(3)
    feats = [
        ("🧹", "Data Cleaning", "Quality report, missing-value handling, duplicate & outlier treatment, encoding, scaling and a 0–100 quality score."),
        ("📊", "Descriptive Analytics", "KPIs, full statistics, frequency tables, cross-tabs and a rich library of interactive Plotly charts."),
        ("🔍", "Diagnostic Analytics", "Pearson & Spearman correlations, heatmaps and automatic plain-English relationship insights."),
        ("🧠", "Latent Analysis", "KMO & Bartlett factorability, PCA scree & biplots, and rotated factor analysis that names the hidden motivations."),
        ("🤖", "Predictive Analytics", "Auto classification / regression model comparison plus K-Means segmentation with elbow & silhouette."),
        ("📈", "Regression Lab", "Linear, Ridge & Lasso with live coefficient shrinkage, cross-validated α tuning and automatic feature selection."),
        ("🔗", "Association Rules", "Market-basket mining (apriori) over multi-select answers — support, confidence, lift and a rule network."),
        ("📝", "Business Insights", "Auto-generated executive summary, opportunities, risks and actionable recommendations."),
        ("⚙️", "Export & Settings", "Download the clean dataset, Excel and PDF reports; toggle light / dark mode."),
    ]
    for col, group in zip((c1, c2, c3), [feats[0:3], feats[3:6], feats[6:9]]):
        with col:
            for ico, title, desc in group:
                st.markdown(
                    f'<div class="glass" style="margin-bottom:14px;">'
                    f'<div style="font-size:24px">{ico}</div>'
                    f'<div style="font-family:Poppins;font-weight:700;font-size:17px;margin:6px 0 4px;">{title}</div>'
                    f'<div style="font-size:13.5px;opacity:.85;line-height:1.5">{desc}</div></div>',
                    unsafe_allow_html=True)

    st.write("")
    section("Get started", "", accent)
    a, b = st.columns([1, 3])
    with a:
        if st.button("📁  Upload / Load Dataset", use_container_width=True):
            st.session_state["nav_to"] = "Upload Dataset"
            st.rerun()
    with b:
        st.markdown(
            '<div class="glass">A ready-made <b>Saturn UAE survey</b> (3,030 responses with realistic '
            'noise) is bundled with the app — load it on the Upload page to explore every module instantly.</div>',
            unsafe_allow_html=True)
