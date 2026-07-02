import streamlit as st
from utils.theme import section, PALETTE
from components._common import set_flash

ACC = "#64748B"


def render():
    section("Settings", "Personalise the dashboard and manage your session.", ACC)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass"><b>🎨 Appearance</b></div>', unsafe_allow_html=True)
        st.write("")
        dark = st.toggle("Dark mode", value=st.session_state.get("dark", True))
        if dark != st.session_state.get("dark", True):
            st.session_state["dark"] = dark
            st.rerun()
        st.caption("Switches the entire app between the premium dark and light themes.")

    with c2:
        st.markdown('<div class="glass"><b>♻️ Session</b></div>', unsafe_allow_html=True)
        st.write("")
        if st.button("Reset to raw dataset", use_container_width=True):
            if st.session_state.get("df_raw") is not None:
                st.session_state["df"] = st.session_state["df_raw"].copy()
                st.session_state["log"] = []
                st.session_state.pop("pred_cache", None)
                set_flash("Working dataset reset to the original raw upload.")
                st.rerun()
            else:
                st.warning("No dataset loaded yet.")
        st.caption("Reverts all cleaning steps and clears trained models for this session.")

    st.write("")
    section("About", "", ACC)
    st.markdown(f"""
    <div class="glass">
      <b>🪐 Saturn — AI Market Understanding Dashboard</b><br>
      A self-service business-intelligence app for analysing UAE consumer-survey data for
      Saturn's premium men's accessories range (ties, pocket squares, dress socks, handkerchiefs).<br><br>
      <span class="pill">Streamlit</span><span class="pill">Plotly</span>
      <span class="pill">scikit-learn</span><span class="pill">pandas</span>
      <span class="pill">SciPy</span><span class="pill">openpyxl</span>
      <br><br>
      <span style="opacity:.8;font-size:13px">Built to deploy on GitHub + Streamlit Community Cloud.
      All analytics are computed live from the loaded dataset — no external data is used.</span>
    </div>
    """, unsafe_allow_html=True)
