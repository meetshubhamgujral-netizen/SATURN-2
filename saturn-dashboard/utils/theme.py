"""Premium look-and-feel: CSS injection, palette, KPI cards, plotly styling."""
import streamlit as st

# --------------------------------------------------------------------------- #
# Palette (luxury fashion inspired)
# --------------------------------------------------------------------------- #
PALETTE = {
    "royal": "#1E3A8A",
    "navy": "#0B1437",
    "gold": "#C8A24B",
    "emerald": "#0F9D7E",
    "orange": "#F2772E",
    "purple": "#7C3AED",
    "crimson": "#DC2A52",
    "sky": "#3B82F6",
    "indigo": "#4F46E5",    # Latent Analysis
    "teal": "#0EA5A3",      # Regression Lab
    "fuchsia": "#C026D3",   # Association Rules
}

# Per-page accent colours (each analytics page is visually distinct)
PAGE_ACCENT = {
    "Home": PALETTE["royal"],
    "Upload Dataset": PALETTE["sky"],
    "Data Cleaning": PALETTE["emerald"],
    "Descriptive Analytics": PALETTE["purple"],
    "Diagnostic Analytics": PALETTE["orange"],
    "Latent Analysis": PALETTE["indigo"],
    "Predictive Analytics": PALETTE["crimson"],
    "Regression Lab": PALETTE["teal"],
    "Association Rules": PALETTE["fuchsia"],
    "Summary & Insights": PALETTE["gold"],
    "Settings": "#64748B",
}

# Categorical colourway for charts
COLORWAY = [
    PALETTE["royal"], PALETTE["gold"], PALETTE["emerald"], PALETTE["purple"],
    PALETTE["orange"], PALETTE["crimson"], PALETTE["sky"], "#94A3B8",
]


def plotly_layout(fig, dark: bool, accent: str = PALETTE["royal"], height: int = 420):
    """Apply a consistent premium plotly theme to a figure."""
    font_color = "#E2E8F0" if dark else "#1E293B"
    grid = "rgba(148,163,184,0.18)" if dark else "rgba(100,116,139,0.16)"
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Poppins, sans-serif", color=font_color, size=13),
        colorway=COLORWAY,
        margin=dict(l=10, r=10, t=50, b=10),
        height=height,
        title=dict(font=dict(size=16, color=accent)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=grid, zeroline=False)
    fig.update_yaxes(gridcolor=grid, zeroline=False)
    return fig


def inject_css(dark: bool = True):
    """Global stylesheet — gradients, glassmorphism, fonts, KPI cards."""
    if dark:
        bg = "linear-gradient(135deg,#0B1437 0%,#101a4d 45%,#1E3A8A 100%)"
        text = "#E6ECFF"
        card_bg = "rgba(255,255,255,0.06)"
        card_border = "rgba(255,255,255,0.12)"
        sub = "#A9B6E0"
        panel = "rgba(255,255,255,0.04)"
    else:
        bg = "linear-gradient(135deg,#F4F7FF 0%,#EAF0FF 50%,#E3ECFF 100%)"
        text = "#13204A"
        card_bg = "rgba(255,255,255,0.75)"
        card_border = "rgba(30,58,138,0.14)"
        sub = "#5B6B95"
        panel = "rgba(255,255,255,0.6)"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@500;600;700;800&display=swap');

    .stApp {{
        background: {bg};
        background-attachment: fixed;
        color: {text};
        font-family: 'Inter', sans-serif;
    }}
    h1,h2,h3,h4 {{ font-family: 'Poppins', sans-serif; letter-spacing:-0.3px; color:{text}; }}
    .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {"rgba(7,12,38,0.92)" if dark else "rgba(255,255,255,0.85)"};
        border-right: 1px solid {card_border};
        backdrop-filter: blur(8px);
    }}

    /* Glassmorphism card */
    .glass {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 20px;
        padding: 22px 24px;
        backdrop-filter: blur(14px);
        box-shadow: 0 10px 30px rgba(2,8,40,0.18);
        transition: transform .18s ease, box-shadow .18s ease;
    }}
    .glass:hover {{ transform: translateY(-3px); box-shadow: 0 16px 40px rgba(2,8,40,0.28); }}

    /* KPI card */
    .kpi {{
        position: relative; overflow: hidden;
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 18px; padding: 18px 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 24px rgba(2,8,40,0.16);
        transition: transform .18s ease;
        height: 100%;
    }}
    .kpi:hover {{ transform: translateY(-4px) scale(1.01); }}
    .kpi .ico {{ font-size: 26px; }}
    .kpi .lbl {{ font-size: 12.5px; color:{sub}; font-weight:600; text-transform:uppercase; letter-spacing:.6px; margin-top:6px; }}
    .kpi .val {{ font-family:'Poppins'; font-size: 30px; font-weight:800; line-height:1.05; margin-top:2px; color:{text}; }}
    .kpi .sub {{ font-size: 12px; color:{sub}; margin-top:4px; }}
    .kpi .bar {{ position:absolute; left:0; top:0; height:100%; width:5px; }}

    /* Hero */
    .hero {{
        border-radius: 26px; padding: 46px 40px; position:relative; overflow:hidden;
        background: linear-gradient(120deg, rgba(30,58,138,0.92), rgba(124,58,237,0.78));
        box-shadow: 0 20px 60px rgba(2,8,40,0.35);
    }}
    .hero h1 {{ color:#fff; font-size: 44px; margin:0 0 8px 0; }}
    .hero p {{ color:#E6ECFF; font-size: 17px; max-width: 760px; margin:0; }}
    .hero .logo {{
        display:inline-flex; align-items:center; gap:12px; margin-bottom:18px;
        font-family:'Poppins'; font-weight:800; font-size:22px; color:#fff;
    }}
    .badge {{
        display:inline-block; padding:6px 14px; border-radius:999px; font-size:12.5px; font-weight:600;
        background: rgba(200,162,75,0.18); color:{PALETTE['gold']}; border:1px solid rgba(200,162,75,0.4);
        margin-right:8px; margin-top:10px;
    }}

    .pill {{
        display:inline-block; padding:5px 12px; border-radius:999px; font-size:12px; font-weight:600;
        background:{panel}; border:1px solid {card_border}; color:{sub}; margin:2px 4px 2px 0;
    }}
    .section-title {{ font-family:'Poppins'; font-weight:700; font-size:22px; margin: 4px 0 2px 0; }}
    .section-sub {{ color:{sub}; font-size:13.5px; margin-bottom:14px; }}

    .stButton>button {{
        border-radius: 12px; font-weight:600; border:1px solid {card_border};
        background: linear-gradient(120deg,{PALETTE['royal']},{PALETTE['purple']}); color:#fff;
        padding: 10px 22px; transition: all .15s ease;
    }}
    .stButton>button:hover {{ filter:brightness(1.08); transform: translateY(-1px); }}
    .stDownloadButton>button {{ border-radius:12px; font-weight:600; }}

    [data-testid="stMetricValue"] {{ font-family:'Poppins'; font-weight:800; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{ border-radius:10px 10px 0 0; padding:8px 16px; }}
    .dataframe tbody tr:hover {{ background: rgba(124,58,237,0.06); }}
    </style>
    """, unsafe_allow_html=True)


def kpi_card(label, value, icon="📊", accent=PALETTE["royal"], sub=""):
    """Return HTML for one colourful KPI card."""
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi">
      <div class="bar" style="background:{accent}"></div>
      <div class="ico">{icon}</div>
      <div class="lbl">{label}</div>
      <div class="val">{value}</div>
      {sub_html}
    </div>
    """


def kpi_row(cards):
    """Render a responsive row of KPI cards (list of HTML strings)."""
    cols = st.columns(len(cards))
    for col, html in zip(cols, cards):
        col.markdown(html, unsafe_allow_html=True)


def section(title, subtitle="", accent=PALETTE["royal"]):
    st.markdown(
        f'<div class="section-title" style="color:{accent}">{title}</div>'
        f'<div class="section-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )
