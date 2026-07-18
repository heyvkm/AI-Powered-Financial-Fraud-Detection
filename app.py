import time
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.loader import load_model
from utils.predictor import predict, ALL_TYPES
from utils.helpers import validate_transaction, get_risk_factors
from utils.logger import log_prediction

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------
# Top bar: dark-mode toggle + About popover (top-right corner)
# --------------------------------------------------------------------------
top_spacer, top_toggle, top_about = st.columns([8, 1, 1.4])
with top_toggle:
    dark_mode = st.toggle("🌙", value=False, key="dark_mode", help="Toggle dark mode")
with top_about:
    with st.popover("ℹ️ About", use_container_width=True):
        st.markdown("### Vishal Kumar Maurya")
        st.caption("Builder of this AI Fraud Detection System")
        st.markdown(
            "🔗 [GitHub](https://github.com/heyvkm)  \n"
            "💼 [LinkedIn](https://www.linkedin.com/in/vishalkmaury/)"
        )

# --------------------------------------------------------------------------
# Theme: set CSS variables inline (dynamic, based on the toggle), then load
# the static rules from assets/style.css (which only ever references var(--x))
# --------------------------------------------------------------------------
LIGHT = dict(
    page_bg="#f9f9f7", surface_1="#fcfcfb", surface_2="#f9f9f7",
    text_primary="#0b0b0b", text_secondary="#52514e", text_muted="#898781",
    border="rgba(11,11,11,0.10)", gridline="#e1e0d9", series_1="#2a78d6",
    good="#0ca30c", warning="#b9820f", serious="#c2542c", critical="#d03b3b",
    input_bg="#ffffff",
)
DARK = dict(
    page_bg="#0d0d0d", surface_1="#1a1a19", surface_2="#222221",
    text_primary="#ffffff", text_secondary="#c3c2b7", text_muted="#898781",
    border="rgba(255,255,255,0.10)", gridline="#2c2c2a", series_1="#3987e5",
    good="#0ca30c", warning="#fab219", serious="#ec835a", critical="#e66767",
    input_bg="#262625",
)
T = DARK if dark_mode else LIGHT

root_vars = "\n".join(f"    --{key.replace('_', '-')}: {value};" for key, value in T.items())
st.markdown(f"<style>:root {{\n{root_vars}\n}}</style>", unsafe_allow_html=True)


def load_css(file_path: Path) -> None:
    if file_path.exists():
        st.markdown(f"<style>{file_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


load_css(BASE_DIR / "assets" / "style.css")


# --------------------------------------------------------------------------
# Load model artifacts (cached)
# --------------------------------------------------------------------------
model = load_model()


def severity_color(prob: float) -> str:
    if prob < 0.20:
        return "var(--good)"
    if prob < 0.50:
        return "var(--warning)"
    if prob < 0.80:
        return "var(--serious)"
    return "var(--critical)"


def risk_row_html(active: bool, active_text: str, inactive_text: str) -> str:
    icon = "⚠️" if active else "✅"
    text = active_text if active else inactive_text
    return f'<div class="risk-row"><span class="risk-icon">{icon}</span><span class="risk-text">{text}</span></div>'


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1>🛡️ AI-Powered Financial Fraud Detection</h1>
        <p>Enter a transaction's details to get a real-time fraud risk assessment from the trained Random Forest model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_form, col_result = st.columns([2, 3], gap="large")

# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------
with col_form:
    st.markdown('<div class="card"><h3>Transaction Details</h3>', unsafe_allow_html=True)

    # Outside the form so the Day/Hour preview updates live as you type
    step = st.number_input(
        "Time Step (hours since simulation start, 1-743)",
        min_value=1, max_value=743, value=1,
        help="PaySim's `step` counts hours continuously from the start of the simulated month "
             "(1-743 ≈ 31 days) — it does NOT reset to 0 every 24 hours. Hour-of-day and day "
             "number are derived from it automatically below.",
    )
    preview_hour = step % 24
    preview_day = (step // 24) + 1
    st.caption(f"→ Day {preview_day}, Hour {preview_hour} of the simulation")

    with st.form("txn_form"):
        txn_type = st.selectbox("Transaction Type", ALL_TYPES, index=4)
        amount = st.number_input("Amount (₹)", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")

        st.markdown("**Sender Account**")
        c1, c2 = st.columns(2)
        old_org = c1.number_input("Balance before", min_value=0.0, value=60000.0, step=1000.0, key="oldOrg", format="%.2f")
        new_org = c2.number_input("Balance after", min_value=0.0, value=10000.0, step=1000.0, key="newOrg", format="%.2f")

        st.markdown("**Receiver Account**")
        c3, c4 = st.columns(2)
        old_dest = c3.number_input("Balance before", min_value=0.0, value=1000.0, step=1000.0, key="oldDest", format="%.2f")
        new_dest = c4.number_input("Balance after", min_value=0.0, value=1000.0, step=1000.0, key="newDest", format="%.2f")

        submitted = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Result panel
# --------------------------------------------------------------------------
with col_result:
    if not submitted:
        st.markdown(
            """
            <div class="card">
                <div class="empty-state">
                    <div class="big">👈</div>
                    Enter a transaction on the left and click<br><b>Analyze Transaction</b> to see the fraud risk assessment.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        txn = {
            "step": step,
            "type": txn_type,
            "amount": amount,
            "oldbalanceOrg": old_org,
            "newbalanceOrig": new_org,
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
        }
        validation_issues = validate_transaction(txn)

        if validation_issues:
            # Input is internally inconsistent — skip the model entirely rather than
            # show a verdict/probability computed from data that doesn't add up.
            warn_html = '<div class="card" style="border-color: var(--warning);"><h3>⚠️ Input Validation Warning</h3>'
            for msg in validation_issues:
                warn_html += (
                    f'<div class="risk-row"><span class="risk-icon">⚠️</span>'
                    f'<span class="risk-text">{msg}</span></div>'
                )
            warn_html += (
                '<div class="verdict-sub" style="margin-top:10px;">'
                "No fraud assessment is shown for internally inconsistent data — "
                "correct the transaction details above and click <b>Analyze Transaction</b> again."
                "</div></div>"
            )
            st.markdown(warn_html, unsafe_allow_html=True)

        else:
            # ── Loading animation ────────────────────────────────
            # model.predict() itself finishes in milliseconds — this delay is
            # purely a UX device so the app visibly shows "the AI is working"
            # instead of a prediction appearing with no perceptible processing.
            # The stages mirror utils/predictor.predict()'s actual pipeline order.
            status_box = st.empty()
            progress_bar = st.progress(0)

            stages = [
                ("Engineering features (dest_not_credited, account_emptied, ...)...", 20),
                ("Encoding transaction type...", 40),
                ("Running Random Forest prediction...", 70),
                ("Computing risk factor breakdown...", 90),
                ("Logging & finalizing result...", 100),
            ]
            for stage_text, pct in stages:
                status_box.markdown(f"🤖 **{stage_text}**")
                progress_bar.progress(pct)
                time.sleep(0.4)

            status_box.empty()
            progress_bar.empty()

            result = predict(txn)  # ← the actual prediction happens here (fast)
            log_prediction(txn, result)

            engineered = result["engineered"]
            is_fraud = result["prediction"] == 1
            verdict_class = "fraud" if is_fraud else "legit"
            verdict_icon = "🚨" if is_fraud else "✅"
            verdict_text = "Fraud Detected" if is_fraud else "Legitimate Transaction"

            st.markdown(
                f"""
                <div class="verdict-card {verdict_class}">
                    <div class="verdict-label">Model Verdict</div>
                    <div class="verdict-value {verdict_class}">{verdict_icon} {verdict_text}</div>
                    <div class="verdict-sub">Based on the Balanced Random Forest model trained on {len(model.feature_names_in_)} features.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            color = severity_color(result["fraud_probability"])
            pct = round(result["fraud_probability"] * 100, 2)
            st.markdown(
                f"""
                <div class="card">
                    <h3>Fraud Probability</h3>
                    <div class="meter-row">
                        <span class="meter-label">Risk score ({result['risk_level']})</span>
                        <span class="meter-value">{pct:.2f}%</span>
                    </div>
                    <div class="meter-track">
                        <div class="meter-fill" style="width:{pct}%; background:{color};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            risk_html = '<div class="card"><h3>Risk Factor Breakdown</h3>'
            for active, active_text, inactive_text in get_risk_factors(txn, engineered):
                risk_html += risk_row_html(active, active_text, inactive_text)
            risk_html += (
                f'<div class="risk-row"><span class="risk-icon">📊</span>'
                f'<span class="risk-text">Amount is <b>{engineered["amount_balance_ratio"]*100:.1f}%</b> '
                f"of the sender's pre-transaction balance.</span></div>"
            )
            risk_html += "</div>"
            st.markdown(risk_html, unsafe_allow_html=True)

            with st.expander("View computed features (raw model input)"):
                st.table(result["input_df"].T.rename(columns={0: "value"}))

# --------------------------------------------------------------------------
# Global model explainability
# --------------------------------------------------------------------------
st.markdown('<div class="card"><h3>What the Model Weighs Most Heavily (Overall)</h3>', unsafe_allow_html=True)
importances = (
    pd.Series(model.feature_importances_, index=model.feature_names_in_)
    .sort_values(ascending=False)
    .head(8)
)
max_imp = importances.max()
bars_html = ""
for name, value in importances.items():
    width_pct = (value / max_imp) * 100
    bars_html += (
        f'<div class="imp-row">'
        f'<span class="imp-label">{name}</span>'
        f'<div class="imp-track"><div class="imp-fill" style="width:{width_pct:.1f}%;"></div></div>'
        f'<span class="imp-value">{value:.3f}</span>'
        f"</div>"
    )
st.markdown(bars_html + "</div>", unsafe_allow_html=True)

st.caption(
    "AI-Powered Financial Fraud Detection System · Balanced Random Forest · "
    "Built with Streamlit · Phase 14 of the project roadmap"
)
