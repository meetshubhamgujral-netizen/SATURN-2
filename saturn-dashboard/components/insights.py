import streamlit as st

from utils.theme import section, PALETTE
from utils import insights as INS
from utils import cleaning as CL
from utils.data_loader import to_excel_bytes
from components._common import require_data

ACC = PALETTE["gold"]

ICONS = {
    "Demographics": "👥", "Shopping Behaviour": "🛍️", "Spending Behaviour": "💳",
    "Purchase Intent": "🎯", "Key Purchase Drivers": "⭐", "Trial Triggers": "🎁",
    "Recommendations": "✅", "Business Risks": "⚠️",
}


def render():
    require_data()
    df = st.session_state["df"]
    score, _ = CL.quality_score(df)

    section("Summary & Business Insights", "Board-ready findings, generated entirely from the "
            "loaded dataset.", ACC)

    # Executive summary
    st.markdown(
        f'<div class="hero" style="background:linear-gradient(120deg,rgba(200,162,75,0.92),'
        f'rgba(30,58,138,0.85));"><div class="logo">🪐 Executive Summary</div>'
        f'<p style="font-size:16px">{INS.executive_summary(df, score)}</p></div>',
        unsafe_allow_html=True)

    st.write("")
    data = INS.generate(df)

    # Two-column card layout
    keys = list(data.keys())
    left, right = st.columns(2)
    for i, k in enumerate(keys):
        target = left if i % 2 == 0 else right
        with target:
            body = "".join(f'<li style="margin-bottom:6px">{pt}</li>' for pt in data[k])
            st.markdown(
                f'<div class="glass" style="margin-bottom:16px">'
                f'<div style="font-family:Poppins;font-weight:700;font-size:17px;margin-bottom:8px">'
                f'{ICONS.get(k,"📌")} {k}</div>'
                f'<ul style="margin:0;padding-left:18px;font-size:14px;line-height:1.5">{body}</ul></div>',
                unsafe_allow_html=True)

    st.write("")
    section("Export Business Insights", "Download the full findings.", ACC)

    # Build a flat report
    lines = [f"SATURN — BUSINESS INSIGHTS REPORT", "=" * 40,
             "", "EXECUTIVE SUMMARY", INS.executive_summary(df, score), ""]
    for k in keys:
        lines.append(k.upper())
        for pt in data[k]:
            clean = pt.replace("**", "")
            lines.append(f"  - {clean}")
        lines.append("")
    report = "\n".join(lines)

    import pandas as pd
    rows = []
    for k in keys:
        for pt in data[k]:
            rows.append({"Section": k, "Insight": pt.replace("**", "")})
    insights_df = pd.DataFrame(rows)

    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇️ Insights (TXT)", report.encode(),
                       "saturn_business_insights.txt", use_container_width=True)
    c2.download_button("⬇️ Insights (Excel)",
                       to_excel_bytes({"insights": insights_df}),
                       "saturn_business_insights.xlsx", use_container_width=True)
    c3.download_button("⬇️ Insights (PDF)", _pdf(df, score, keys, data),
                       "saturn_business_insights.pdf", "application/pdf",
                       use_container_width=True)


def _pdf(df, score, keys, data):
    try:
        from fpdf import FPDF
    except Exception:
        return b"PDF library unavailable."
    from components._common import pdf_heading, pdf_line
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf_heading(pdf, "Saturn - Business Insights", size=18)
    pdf_line(pdf, INS.executive_summary(df, score).replace("**", ""))
    pdf.ln(2)
    for k in keys:
        pdf_heading(pdf, k, size=13)
        for pt in data[k]:
            pdf_line(pdf, f"  - {pt.replace('**','')}", size=10)
        pdf.ln(1)
    out = pdf.output(dest="S")
    return bytes(out) if isinstance(out, (bytes, bytearray)) else out.encode("latin-1", "replace")
