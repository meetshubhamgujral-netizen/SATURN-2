"""Small shared helpers for page components."""
import streamlit as st
from utils.theme import plotly_layout, PAGE_ACCENT


def pdf_safe(s):
    """Make text safe for the core PDF font (latin-1), replacing common Unicode."""
    return (str(s)
            .replace("–", "-").replace("—", "-")
            .replace("’", "'").replace("‘", "'")
            .replace("“", '"').replace("”", '"')
            .replace("…", "...").replace("→", "->")
            .replace("•", "-").replace("≈", "~").replace("⚠️", "!")
            .encode("latin-1", "replace").decode("latin-1"))


def pdf_heading(pdf, text, size=13):
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", size)
    pdf.multi_cell(pdf.epw, 8, pdf_safe(text))


def pdf_line(pdf, text, size=11, bold=False):
    """Write a wrapped line at the left margin with an explicit width."""
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B" if bold else "", size)
    pdf.multi_cell(pdf.epw, 6, pdf_safe(text))


def dark():
    return st.session_state.get("dark", True)


def accent(page):
    return PAGE_ACCENT.get(page, "#1E3A8A")


def chart(fig, page="Home", height=420):
    """Apply the premium theme then render a plotly figure full-width."""
    st.plotly_chart(plotly_layout(fig, dark(), accent(page), height),
                    use_container_width=True)


def set_flash(msg, kind="success"):
    st.session_state["_flash"] = (kind, msg)


def show_flash():
    if "_flash" in st.session_state:
        kind, msg = st.session_state.pop("_flash")
        getattr(st, kind, st.info)(msg)


def require_data():
    """Stop the page with a friendly notice if no dataset is loaded."""
    if st.session_state.get("df") is None:
        st.warning("📁 No dataset loaded yet. Go to **Upload Dataset** "
                   "(or load the bundled Saturn sample there) to begin.")
        st.stop()
