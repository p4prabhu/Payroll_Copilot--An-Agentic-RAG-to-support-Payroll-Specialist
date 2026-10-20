"""Local Streamlit chat for payroll specialists."""
from pathlib import Path
from decimal import Decimal
import re
import sys
from urllib.parse import urlparse

import streamlit as st

SCRIPTS = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from langchain_core.messages import AIMessage, HumanMessage
from payroll_ui_runtime import run_ui_turn


def reset_chat():
    st.session_state.chat_messages = []
    st.session_state.employee_session = {}
    st.session_state.last_capabilities = None
    st.session_state.pop("queued_question", None)
    st.session_state.pop("ui_error", None)


def queue_question(question):
    st.session_state.queued_question = question


def apply_style():
    """Presentation only; native Streamlit controls retain their behavior."""
    st.markdown("""
    <style>
      .stApp {background: #0b1220; color: #e6edf7;}
      .block-container {max-width: 1040px; padding-top: 2.8rem; padding-bottom: 3rem;}
      section[data-testid="stSidebar"] {background: #0f192a; border-right: 1px solid #213047;}
      section[data-testid="stSidebar"] .block-container {padding-top: 2rem;}
      h1 {font-size: 2.5rem !important; font-weight: 650 !important; letter-spacing: -.07rem; color: #f1f6fc;}
      h2, h3 {letter-spacing: -.025rem; color: #e6edf7;}
      p, li {line-height: 1.75;}
      [data-testid="stCaptionContainer"] {color: #a6b4c9;}
      .payroll-brand {display: flex; align-items: center; gap: .8rem; margin-bottom: 1rem;}
      .payroll-mark {background: #123e40; border: 1px solid #257b75; color: #71e4ce; width: 40px; height: 40px;
        border-radius: 11px; display: grid; place-items: center; font: 700 21px sans-serif;}
      .payroll-brand-name {font-size: 1.12rem; font-weight: 650; letter-spacing: -.02rem;}
      .payroll-brand-sub {font-size: .7rem; letter-spacing: .09rem; opacity: .6; margin-top: .15rem;}
      .payroll-eyebrow {font-size: .7rem; font-weight: 650; letter-spacing: .17rem;
        color: #71e4ce; margin-bottom: .7rem;}
      .payroll-welcome {padding: 1.8rem 0 1.2rem;}
      .payroll-welcome h2 {font-size: 1.5rem; font-weight: 600; margin-bottom: .5rem;}
      .payroll-welcome p {opacity: .7; font-size: .96rem; line-height: 1.65;}
      [data-testid="stChatMessage"] {background: #111d30; border: 1px solid #24344c;
        border-radius: 18px; padding: 1.5rem; margin-bottom: 1.1rem; box-shadow: 0 8px 28px #00000014;}
      [data-testid="stChatMessageAvatarAssistant"], [data-testid="stChatMessageAvatarUser"] {
        background: #173d40; color: #71e4ce; border-radius: 10px;}
      [data-testid="stChatMessageContent"] {min-width: 0;}
      [data-testid="stChatMessageContent"] h3 {font-size: 1.24rem; padding-top: .35rem;}
      [data-testid="stChatInput"] {border: 1px solid #36576b; border-radius: 15px; background: #111d30;}
      [data-testid="stChatInput"] textarea {color: #e6edf7; background: transparent;}
      [data-testid="stBottom"] > div {background: #0b1220;}
      [data-testid="stExpander"] {border: 1px solid #293b53; border-radius: 12px; background: #101b2c;}
      [data-testid="stVerticalBlockBorderWrapper"] {border-color: #293b53; border-radius: 14px;}
      [data-testid="stMetric"] {padding: 1rem; background: #123137; border: 1px solid #235358; border-radius: 12px;}
      [data-testid="stMetricValue"] {color: #83ead5; font-size: 1.85rem;}
      .stButton > button, .stLinkButton > a {border-radius: 9px; font-weight: 500;}
      .payroll-note {border-left: 3px solid #d8b66c; background: #292719; color: #e7d5a7;
        border-radius: 0 8px 8px 0; padding: .65rem .9rem; font-size: .85rem; margin: .9rem 0 .2rem;}
      .payroll-status {display: inline-block; color: #80dccb; font-size: .67rem;
        font-weight: 650; letter-spacing: .13rem; margin-bottom: .1rem;}
      hr {border-color: #24344c;}
      @media (max-width: 700px) {
        .block-container {padding-top: 1.4rem;}
        h1 {font-size: 1.8rem !important;}
        .payroll-welcome {padding-top: 1rem;}
        [data-testid="stChatMessage"] {padding: 1rem;}
        [data-testid="stMetric"] {padding: .65rem;}
        [data-testid="stMetricValue"] {font-size: 1.25rem;}
      }
    </style>
    """, unsafe_allow_html=True)


REVIEW_NOTES = {
    "Source content review is pending internally within this project; this answer is provisional.",
    "Some cited editions need internal project update review; current applicability has not been verified.",
    "Some cited sources have unknown or incomplete effective dates; payment-date applicability remains unresolved.",
    "No payment date was supplied; date-specific applicability has not been assessed.",
}


def answer_sections(text):
    """Move only known trailing review notices; keep substantive answer text intact."""
    lines = text.rstrip().splitlines()
    notes = []
    while lines and (not lines[-1].strip() or lines[-1].strip() in REVIEW_NOTES):
        line = lines.pop().strip()
        if line:
            notes.insert(0, line)
    return "\n".join(lines).strip(), notes


def display_answer(message):
    outcome = message["outcome"]
    status = outcome["status"]
    st.markdown('<span class="payroll-status">PAYROLL COPILOT</span>', unsafe_allow_html=True)
    title = {"clarification": "A few details needed", "unsupported": "Request unavailable"}.get(status)
    if title is None:
        title = "Sample T4 breakdown" if outcome.get("t4_summary") else "Employee roster" if outcome.get("roster") else "Payroll estimate" if outcome.get("calculation") else "Workforce summary" if any(
            s.get("kind") == "workforce_summary" for s in outcome["sources"]) else "Employee information" if any(
            s.get("kind") == "employee_record" for s in outcome["sources"]) else "Payroll guidance"
    st.subheader(title)
    display_text = re.sub(r"\bfictional\s+", "", outcome["text"], flags=re.IGNORECASE)
    body, notes = answer_sections(display_text)
    calculation = outcome.get("calculation")
    annual = outcome.get("t4_summary")
    if annual and status == "answer":
        st.caption(f"Reporting year: {annual['reporting_year']} · All monetary amounts in CAD · Simulated year-end records")
        visible, separator, assumptions = body.partition("**Sample assumptions:**")
        st.markdown(visible.replace("$", r"\$"))
        if separator:
            with st.expander("Sample assumptions and calculation scope", expanded=False):
                st.markdown(assumptions.replace("$", r"\$"))
    elif calculation and status == "answer":
        amounts = calculation["amounts"]
        st.caption(f"Payment date: {calculation['inputs']['payment_date']} · All amounts in CAD")
        for column, label, key in zip(st.columns(3), ("Net pay", "Gross pay", "Total deductions"),
                                      ("net_pay", "gross_pay", "total_deductions")):
            column.metric(label, f"${Decimal(amounts[key]):,.2f}")
        st.markdown("**Deduction breakdown**")
        rows = [f"| {label} | \\${Decimal(amounts[key]):,.2f} |" for label, key in (
            ("Federal income tax", "federal_income_tax"), ("Ontario income tax", "ontario_income_tax"),
            ("CPP", "cpp"), ("CPP2", "cpp2"), ("EI", "ei"))]
        # Native Markdown keeps this small, static table independent of the
        # optional dataframe/Arrow stack on both Windows and macOS.
        st.markdown("| Deduction | CAD |\n| :--- | ---: |\n" + "\n".join(rows))
        st.caption("Estimate based on confirmed inputs · not CRA-certified. Review the full scope and assumptions below.")
        with st.expander("Calculation details", expanded=False):
            st.markdown(body.replace("$", r"\$"))
    else:
        st.markdown(body.replace("$", r"\$"))
    if notes:
        review_pending = any("review" in note for note in notes)
        label = ("Provisional guidance · source review or date applicability remains unresolved."
                 if review_pending else "Date-specific applicability has not been assessed.")
        st.markdown(f'<div class="payroll-note">{label}</div>', unsafe_allow_html=True)
        with st.expander("Review notes", expanded=False):
            for note in notes:
                st.caption(note)
    if outcome["sources"]:
        with st.expander(f"Sources · {len(outcome['sources'])}", expanded=False):
            for source in outcome["sources"]:
                with st.container(border=True):
                    if source.get("kind") in {"workforce_summary", "employee_roster"}:
                        label = "Employee database roster" if source["kind"] == "employee_roster" else "Employee database summary"
                        st.write(f"[{source['source_id']}] {label}")
                        st.caption(f"Database snapshot: {source['provenance']['dataset_snapshot_date']} · Read-only query")
                        st.caption("Counts reflect recorded employees; company-wide HR completeness is not established.")
                    elif source.get("kind") == "annual_payroll_sample":
                        st.write(f"[{source['source_id']}] Annual payroll sample · {source['reporting_year']}")
                        st.caption(f"Employee: {source['employee_id']} · Snapshot: {source['provenance']['dataset_snapshot_date']} · Read-only query")
                        st.caption("Reconciled simulated year-end records; sample preparation only, not an actual or filing-ready T4.")
                        st.caption(f"Calculation rules SHA-256: {source['rules_sha256']}")
                    elif source.get("kind") == "calculation":
                        provenance = source["provenance"]
                        st.write(f"[{source['source_id']}] Payroll calculation rules")
                        st.caption(provenance["validation_status"])
                        st.caption(f"Rules: {provenance['ruleset_id']} · Reviewed: {provenance['verified_on']}")
                        for index, item in enumerate(provenance["sources"], 1):
                            url = item["url"]
                            if urlparse(url).scheme == "https" and urlparse(url).netloc == "www.canada.ca":
                                st.link_button(f"Government calculation reference {index}", url)
                    elif source.get("kind") == "employee_record":
                        provenance = source["provenance"]
                        st.write(f"[{source['source_id']}] Employee record · {source['employee_id']}")
                        st.caption(f"Database snapshot: {provenance['dataset_snapshot_date']} · Stored amounts are not CRA-validated.")
                    else:
                        st.write(f"[{source['source_id']}] {source['title']}")
                        st.caption(f"PDF page {source['pdf_page_number']}")
                        url = source.get("source_url", "")
                        if urlparse(url).scheme == "https" and urlparse(url).netloc:
                            st.link_button(f"Open source {source['source_id']}", url)
    with st.expander("Tool activity", expanded=False):
        if message["trace"]:
            for event in message["trace"]:
                if event["event"] == "tool":
                    st.caption(f"Document search · {event.get('result_count', 0)} passages returned · {event.get('status', '')}")
                elif event["event"] == "employee_tool":
                    st.caption(f"Employee lookup · {event.get('tool', '')} · {event.get('status', '')}")
                elif event["event"] == "calculation_tool":
                    st.caption(f"Payroll calculator · {event.get('status', '')}")
                elif event["event"] == "workforce_tool":
                    st.caption(f"Workforce summary · {event.get('tool', '')} · {event.get('status', '')}")
                elif event["event"] == "t4_tool":
                    st.caption(f"Annual payroll sample · {event.get('tool', '')} · {event.get('status', '')}")
                elif event["event"] == "response_validation":
                    st.caption(f"Response check · {event.get('status', '')}")
            with st.expander("Technical details", expanded=False):
                st.json(message["trace"])
        else:
            st.caption("No tool calls were needed for this response.")


def main():
    st.set_page_config(page_title="Payroll Copilot", page_icon=":material/account_balance_wallet:", layout="wide")
    apply_style()
    if "chat_messages" not in st.session_state:
        reset_chat()
    with st.sidebar:
        st.markdown('''<div class="payroll-brand"><div class="payroll-mark">P</div>
          <div><div class="payroll-brand-name">Payroll Copilot</div>
          <div class="payroll-brand-sub">SPECIALIST WORKSPACE</div></div></div>''', unsafe_allow_html=True)
        st.button("Reset conversation", on_click=reset_chat, use_container_width=True)
        st.divider()
        st.caption("CURRENT EMPLOYEE")
        selected = st.session_state.employee_session.get("selected")
        with st.container(border=True):
            if selected:
                st.write(f"{selected['given_name']} {selected['family_name']}")
                st.caption(selected["employee_number"])
            else:
                st.write("No employee selected")
                st.caption("Search by name or employee number in chat.")
            st.divider()
            st.caption("LOOKUP DATE")
            st.write(st.session_state.employee_session.get("lookup_date", "Awaiting a date"))
        caps = st.session_state.last_capabilities
        st.caption("Employee connection: " + ("Not checked" if caps is None else "Available on last turn" if caps.get("mcp_transport") else "Unavailable on last turn"))
        st.divider()
        with st.expander("Available services"):
            st.write("Employee records · Scheduled paydays · Payment history · Payroll guidance")
            st.write("Employee counts · Pay-frequency and salary filters · Join/leave summaries")
            st.write("Active employee roster · Names, salaries and pay frequencies")
            st.write("Regular-salary calculations · Review inputs before calculating")
            st.caption("Calculations support 2026 Ontario regular salary: biweekly (26) or monthly (12). Compensation changes are not available.")
        st.caption("Guidance questions and natural-language calculation intake use the configured AI service and may incur API charges.")

    st.markdown('<div class="payroll-eyebrow">PAYROLL OPERATIONS</div>', unsafe_allow_html=True)
    st.title("Payroll Copilot")
    st.write("Your workspace for employee records, payroll questions and pay estimates.")
    st.divider()
    if not st.session_state.chat_messages:
        st.markdown('''<div class="payroll-welcome"><h2>Payroll clarity starts here.</h2>
          <p>Find the employee information you need, review recorded payments,
          or ask a question about payroll requirements.</p></div>''', unsafe_allow_html=True)
        for column, title, description in zip(st.columns(3),
                ("01  Employee records", "02  Payroll schedule", "03  Payroll guidance"),
                ("Review salary, employment details and pay frequency.",
                 "Check scheduled paydays and recorded payment history.",
                 "Explore government guidance with supporting sources.")):
            with column, st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(description)
        st.caption("Enter an employee name or number to begin. Include an explicit lookup date when requesting records.")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"], avatar=":material/person:" if message["role"] == "user" else ":material/account_balance_wallet:"):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                display_answer(message)

    candidates = st.session_state.employee_session.get("candidates", [])
    if candidates:
        st.subheader("Select an employee")
        st.caption("Multiple records match your search. Select the correct employee to continue.")
        for employee in candidates:
            number = employee["employee_number"]
            st.button(f"{employee['given_name']} {employee['family_name']} · {number}",
                      key=f"select_{number}", on_click=queue_question, args=(number,))
    if st.session_state.get("ui_error"):
        st.error(st.session_state.ui_error)

    roster = st.session_state.employee_session.get("_analytics", {})
    latest = st.session_state.chat_messages[-1] if st.session_state.chat_messages else {}
    if roster.get("kind") == "roster" and roster.get("page_ready") and latest.get("outcome", {}).get("roster"):
        previous, next_page = st.columns(2)
        previous.button("Previous employees", disabled=roster["arguments"]["offset"] == 0,
                        on_click=queue_question, args=("Previous employees",), use_container_width=True)
        next_page.button("Next employees", disabled=not roster.get("has_more"),
                         on_click=queue_question, args=("Next employees",), use_container_width=True)

    draft = st.session_state.employee_session.get("_calculation", {})
    if draft.get("awaiting_confirmation"):
        confirm, cancel = st.columns(2)
        confirm.button("Confirm calculation", on_click=queue_question, args=("Confirm calculation",), type="primary", use_container_width=True)
        cancel.button("Cancel calculation", on_click=queue_question, args=("Cancel calculation",), use_container_width=True)

    question = st.chat_input("Search for an employee or ask a payroll question…", max_chars=4000)
    queued = st.session_state.pop("queued_question", None)
    question = question or queued
    if question:
        history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                   for m in st.session_state.chat_messages]
        try:
            with st.spinner("Checking your request…"):
                result = run_ui_turn(question, history, st.session_state.employee_session)
            # Commit only a completed exchange. Failed requests are never retried
            # on reruns and never become incomplete model conversation history.
            st.session_state.employee_session = result["employee_session"]
            st.session_state.last_capabilities = result["capabilities"]
            st.session_state.chat_messages.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": result["outcome"]["text"],
                 "outcome": result["outcome"], "trace": result["trace"]},
            ])
            st.session_state.pop("ui_error", None)
        except Exception:
            st.session_state.ui_error = "The request could not be completed. Check the local database and, for guidance questions, your API configuration. Your previous conversation is preserved."
        st.rerun()


if __name__ == "__main__":
    main()
