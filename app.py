"""Local Streamlit chat for payroll specialists."""
from pathlib import Path
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
      .block-container {max-width: 1080px; padding-top: 2.5rem; padding-bottom: 3rem;}
      section[data-testid="stSidebar"] {border-right: 1px solid rgba(128,145,164,.18);}
      section[data-testid="stSidebar"] .block-container {padding-top: 2rem;}
      h1 {font-size: 2.15rem !important; font-weight: 650 !important; letter-spacing: -.055rem;}
      h2, h3 {letter-spacing: -.025rem;}
      .payroll-brand {display: flex; align-items: center; gap: .8rem; margin-bottom: 1rem;}
      .payroll-mark {background: #0f766e; color: white; width: 40px; height: 40px;
        border-radius: 11px; display: grid; place-items: center; font: 700 21px sans-serif;}
      .payroll-brand-name {font-size: 1.12rem; font-weight: 650; letter-spacing: -.02rem;}
      .payroll-brand-sub {font-size: .7rem; letter-spacing: .09rem; opacity: .6; margin-top: .15rem;}
      .payroll-eyebrow {font-size: .72rem; font-weight: 650; letter-spacing: .12rem;
        color: #0f9b8e; margin-bottom: .55rem;}
      .payroll-welcome {padding: 2rem 0 1.1rem;}
      .payroll-welcome h2 {font-size: 1.5rem; font-weight: 600; margin-bottom: .5rem;}
      .payroll-welcome p {opacity: .7; font-size: .96rem; line-height: 1.65;}
      [data-testid="stChatMessage"] {border: 1px solid rgba(128,145,164,.15);
        border-radius: 14px; padding: 1.2rem; margin-bottom: 1rem;}
      [data-testid="stChatInput"] {border-radius: 14px;}
      [data-testid="stExpander"] {border-radius: 10px;}
      .stButton > button {border-radius: 9px; font-weight: 500;}
      @media (max-width: 700px) {
        .block-container {padding-top: 1.4rem;}
        h1 {font-size: 1.8rem !important;}
        .payroll-welcome {padding-top: 1rem;}
      }
    </style>
    """, unsafe_allow_html=True)


def display_answer(message):
    outcome = message["outcome"]
    st.caption({"answer": "PAYROLL COPILOT", "clarification": "ADDITIONAL INFORMATION REQUIRED", "unsupported": "SERVICE LIMITATION"}.get(outcome["status"], outcome["status"]))
    display_text = re.sub(r"\bfictional\s+", "", outcome["text"], flags=re.IGNORECASE)
    st.markdown(display_text.replace("$", r"\$"))
    if outcome["sources"]:
        with st.expander("Sources", expanded=False):
            for source in outcome["sources"]:
                if source.get("kind") == "employee_record":
                    provenance = source["provenance"]
                    st.write(f"[{source['source_id']}] Employee record · {source['employee_id']}")
                    st.caption(f"Database snapshot: {provenance['dataset_snapshot_date']} · Stored amounts are not CRA-validated.")
                else:
                    st.write(f"[{source['source_id']}] {source['title']}")
                    st.caption(f"PDF page {source['pdf_page_number']}")
                    url = source.get("source_url", "")
                    if urlparse(url).scheme == "https" and urlparse(url).netloc:
                        st.link_button("Open source document", url)
    with st.expander("Tool activity", expanded=False):
        if message["trace"]:
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
            st.caption("Payroll calculations and compensation changes are not available.")
        st.caption("Guidance questions use the configured AI service and may incur API charges.")

    st.markdown('<div class="payroll-eyebrow">PAYROLL OPERATIONS</div>', unsafe_allow_html=True)
    st.title("Payroll Copilot")
    st.write("Employee information and payroll guidance, in one conversation.")
    st.divider()
    if not st.session_state.chat_messages:
        st.markdown('''<div class="payroll-welcome"><h2>How can I help with payroll today?</h2>
          <p>Find the employee information you need, review recorded payments,
          or ask a question about payroll requirements.</p></div>''', unsafe_allow_html=True)
        for column, title, description in zip(st.columns(3),
                ("Employee records", "Payroll schedule", "Payroll guidance"),
                ("Review salary, employment details and pay frequency.",
                 "Check scheduled paydays and recorded payment history.",
                 "Explore government guidance with supporting sources.")):
            with column, st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(description)
        st.caption("Enter an employee name or number to begin. Include an explicit lookup date when requesting records.")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
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
