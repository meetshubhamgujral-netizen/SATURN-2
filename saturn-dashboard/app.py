"""
Saturn — AI Market Understanding Dashboard
Main Streamlit entry point. Run locally with:  streamlit run app.py
"""
import streamlit as st

from utils.theme import inject_css, PALETTE
from components import (home, upload, cleaning, descriptive, diagnostic, latent,
                        predictive, regression_reg, assoc, insights, settings)

# --------------------------------------------------------------------------- #
# Page config (must be the first Streamlit call)
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Saturn — AI Market Understanding Dashboard",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Session-state defaults
# --------------------------------------------------------------------------- #
for k, v in {"dark": True, "df": None, "df_raw": None, "log": []}.items():
    st.session_state.setdefault(k, v)

inject_css(st.session_state["dark"])

PAGES = ["Home", "Upload Dataset", "Data Cleaning", "Descriptive Analytics",
         "Diagnostic Analytics", "Latent Analysis", "Predictive Analytics",
         "Regression Lab", "Association Rules", "Summary & Insights", "Settings"]
BOOTSTRAP_ICONS = ["house", "cloud-arrow-up", "magic", "bar-chart-line",
                   "search", "diagram-3", "robot", "graph-up-arrow", "share",
                   "clipboard-data", "gear"]
EMOJI = ["🏠", "📁", "🧹", "📊", "🔍", "🧠", "🤖", "📈", "🔗", "📝", "⚙️"]
ROUTES = {
    "Home": home, "Upload Dataset": upload, "Data Cleaning": cleaning,
    "Descriptive Analytics": descriptive, "Diagnostic Analytics": diagnostic,
    "Latent Analysis": latent, "Predictive Analytics": predictive,
    "Regression Lab": regression_reg, "Association Rules": assoc,
    "Summary & Insights": insights, "Settings": settings,
}


# --------------------------------------------------------------------------- #
# Sidebar navigation (premium option-menu, with a safe radio fallback)
# --------------------------------------------------------------------------- #
def sidebar_nav():
    # Honour a programmatic redirect (e.g. Home's "Upload" button)
    manual = None
    if "nav_to" in st.session_state:
        target = st.session_state.pop("nav_to")
        if target in PAGES:
            manual = PAGES.index(target)

    with st.sidebar:
        st.markdown(
            f'<div style="text-align:center;padding:8px 0 14px;">'
            f'<div style="font-family:Poppins;font-weight:800;font-size:26px;'
            f'color:{PALETTE["gold"]}">🪐 SATURN</div>'
            f'<div style="font-size:11.5px;letter-spacing:2px;opacity:.7">MARKET INTELLIGENCE</div></div>',
            unsafe_allow_html=True)

        try:
            from streamlit_option_menu import option_menu
            selected = option_menu(
                menu_title=None,
                options=PAGES,
                icons=BOOTSTRAP_ICONS,
                default_index=0,
                manual_select=manual,
                key="navmenu",
                styles={
                    "container": {"padding": "4px", "background-color": "rgba(0,0,0,0)"},
                    "icon": {"font-size": "16px"},
                    "nav-link": {
                        "font-size": "14px", "font-weight": "600",
                        "border-radius": "12px", "margin": "3px 0",
                        "--hover-color": "rgba(124,58,237,0.18)",
                    },
                    "nav-link-selected": {
                        "background": "linear-gradient(120deg,#1E3A8A,#7C3AED)",
                        "color": "white",
                    },
                },
            )
        except Exception:
            # Fallback: native radio (guarantees the app always runs)
            if manual is not None:
                st.session_state["nav_radio"] = PAGES[manual]
            labels = [f"{e}  {p}" for e, p in zip(EMOJI, PAGES)]
            default = PAGES.index(st.session_state.get("nav_radio", "Home"))
            choice = st.radio("Navigate", labels, index=default, label_visibility="collapsed")
            selected = PAGES[labels.index(choice)]
            st.session_state["nav_radio"] = selected

        st.markdown("<hr style='opacity:.15'>", unsafe_allow_html=True)
        df = st.session_state.get("df")
        if df is not None:
            st.caption(f"📦 {len(df):,} rows · {df.shape[1]} cols loaded")
        else:
            st.caption("⚪ No dataset loaded")
        st.caption("UAE Consumer Survey · Saturn BI")

    return selected


def main():
    selected = sidebar_nav()
    ROUTES.get(selected, home).render()


if __name__ == "__main__":
    main()
