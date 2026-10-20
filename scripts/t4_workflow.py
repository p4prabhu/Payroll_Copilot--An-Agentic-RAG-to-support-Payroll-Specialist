"""Bounded sample T4 breakdowns from reconciled annual MCP evidence; no model calls."""
from copy import deepcopy
from decimal import Decimal
import re

from payroll_mcp_client import PayrollMCPError


SESSION_KEY = "_t4"
GUIDANCE = r"\b(?:what information|what records|which records|what documents|checklist|requirements|needed|required|explain|how to|what is (?:a |the )?t4|which (?:t4 )?boxes)\b"
BOX_LABELS = {"14": "Employment income", "16": "Employee CPP contributions", "16A": "Employee CPP2 contributions",
              "18": "Employee EI premiums", "22": "Income tax deducted", "24": "EI insurable earnings",
              "26": "CPP pensionable earnings"}


def normal(question):
    return question.replace("\u2019", "'").replace("\u2018", "'").strip()


def name_query(value):
    """Accept a bounded name/identifier, never choose a database identity here."""
    value = value.strip(" \t.!?,:")
    if re.fullmatch(r"(?:EMP-\d+|emp-\d+)", value, re.I):
        return value
    if re.fullmatch(r"[^\W\d_]+(?:[-'][^\W\d_]+)*(?:\s+[^\W\d_]+(?:[-'][^\W\d_]+)*){0,3}", value):
        if not set(value.lower().split()) & set("show give generate prepare sample t4 breakdown for of the a me year today tomorrow what how information salary payroll yes no confirm find lookup search list employee employees records next pay reset explain checklist needed required details their his her".split()):
            return value
    return None


def target_query(question):
    text = normal(question)
    ids = re.findall(r"\bemp-\d+\b", text, re.I)
    if ids:
        return ids[0] if len({i.lower() for i in ids}) == 1 else None
    text = re.sub(r"\b(?:for\s+)?(?:(?:reporting|tax|calendar)\s+year\s+)?(?:19|20)\d{2}\b", "", text, flags=re.I).strip()
    text = re.sub(r"^(?:(?:please|show|give|me|a|the|generate|prepare|draft|sample)\s+)+", "", text, flags=re.I)
    for pattern in (r"^(.+?)'s\s+(?:sample\s+)?t4\b", r"\bt4(?:\s+(?:sample|annual|box.by.box|breakdown|summary|amounts|totals))*\s+(?:of|for)\s+(.+?)$",
                    r"^(.+?)\s+(?:sample\s+)?t4\b"):
        match = re.search(pattern, text, re.I)
        if match:
            return name_query(match.group(1))
    return None


def explicit_request(question):
    text = normal(question)
    if not re.search(r"\bt4\b", text, re.I) or re.search(GUIDANCE, text, re.I):
        return False
    return bool(target_query(text) or re.search(r"\b(?:generate|prepare|draft|breakdown|summary|amounts|totals)\b", text, re.I))


def should_route(question, session):
    if explicit_request(question):
        return True
    pending = session.get(SESSION_KEY, {}).get("pending")
    text = normal(question)
    return bool(pending and (re.fullmatch(r"(?:for\s+)?(?:19|20)\d{2}[.!?]?", text, re.I)
                             or name_query(text)))


def escape(text):
    return re.sub(r"([\\`*_{}\[\]<>|])", r"\\\1", str(text)).replace("\n", " ")


def cad(cents):
    return f"CAD ${Decimal(cents) / 100:,.2f}"


def render_breakdown(data):
    employee = data["employee"]
    name = escape(f"{employee['given_name']} {employee['family_name']} ({employee['employee_number']})")
    lines = [f"**{name} — {data['reporting_year']} sample T4 breakdown** [D1]",
             "Simulated year-end payroll records. This is a sample breakdown, not an actual T4, generated form or CRA filing.",
             "| T4 box | Description | CAD |", "| :--- | :--- | ---: |"]
    lines.extend(f"| {box} | {label} | {cad(data['box_candidates_cents'][box])} |" for box, label in BOX_LABELS.items())
    lines.extend([f"| 45 | Employer-offered dental benefits code | {data['dental_benefits_code']} |",
                  f"\nRecorded payments: **{data['totals']['payment_count']}**. Supplementary net pay: **{cad(data['totals']['net_pay_cents'])}** (not a T4 box).",
                  "\n**Required before real filing:** " + "; ".join(escape(v) for v in data["missing_filing_information"]) + ".",
                  "\n**Sample assumptions:** " + escape(data["assumptions"]),
                  f"\nAnnual sample snapshot: {data['provenance']['dataset_snapshot_date']}. Calculated sample deductions are not CRA-certified."])
    return "\n\n".join(lines[:2]) + "\n\n" + "\n".join(lines[2:])


async def t4_turn(client, question, employee_session):
    session = deepcopy(employee_session)
    # This context must never carry a previously approved paycheck calculation.
    session.pop("_calculation", None)
    session.pop("_analytics", None)
    previous = session.get(SESSION_KEY, {})
    fresh = explicit_request(question)
    plan = {} if fresh else deepcopy(previous)
    session[SESSION_KEY] = plan
    trace, sources = [], []

    def respond(status, text, annual=None):
        outcome = {"status": status, "text": text, "citations": [s["source_id"] for s in sources], "sources": sources}
        if annual is not None:
            outcome["t4_summary"] = annual
        return {"employee_session": session, "trace": trace, "outcome": outcome}

    if client is None or not client.available:
        return respond("unsupported", "The employee database connection is unavailable. No sample T4 amounts were retrieved.")
    text = normal(question)
    years = re.findall(r"\b(?:19|20)\d{2}\b", text)
    if len(set(years)) > 1 or re.search(r"\b\d{4}-\d{2}-\d{2}\b", text):
        plan.pop("year", None)
        plan["pending"] = "year"
        return respond("clarification", "Which single T4 reporting year should I use? Enter a year such as 2026, rather than a payment date.")
    if years:
        plan["year"] = int(years[0])
    target = target_query(text) if fresh else (name_query(text) if not years else None)
    if fresh and target is None and re.search(r"\b(?:their|his|her)\b", text, re.I):
        plan["identity"] = deepcopy(session.get("selected"))
    if fresh:
        session.pop("pending_action", None)
        session.pop("candidates", None)
    if fresh and len({v.lower() for v in re.findall(r"\bemp-\d+\b", text, re.I)}) > 1:
        plan["pending"] = "identity"
        return respond("clarification", "Which single employee name or employee number should I use for the sample T4 breakdown?")
    try:
        if target:
            session.pop("selected", None)
            plan.pop("identity", None)
            found = await client.call("find_employees", {"query": target, "limit": 10})
            trace.append({"event": "t4_tool", "tool": "find_employees", "status": found["status"]})
            matches = found["matches"]
            if found["requires_selection"]:
                plan["pending"] = "selection"
                options = "; ".join(escape(f"{e['given_name']} {e['family_name']} ({e['employee_number']})") for e in matches)
                more = " More matches exist; narrow the name." if found.get("has_more") else ""
                return respond("clarification", f"I found multiple employees: {options}.{more} Which employee number should I use?")
            if not matches:
                plan["pending"] = "identity"
                return respond("clarification", "No employee matched that name or number. Which employee should I search for?")
            if len(matches) != 1:
                raise PayrollMCPError("Employee selection could not be verified")
            plan["identity"] = matches[0]
        identity = plan.get("identity")
        if not identity:
            plan["pending"] = "identity"
            return respond("clarification", "Which employee name or employee number should I use for the sample T4 breakdown?")
        session["selected"] = deepcopy(identity)
        if "year" not in plan:
            plan["pending"] = "year"
            return respond("clarification", "Which T4 reporting year should I use? Enter a year such as 2026. A payment date is not needed for this annual sample breakdown.")
        arguments = {"employee_id": identity["employee_id"], "reporting_year": plan["year"]}
        data = await client.call("get_annual_payroll_summary", arguments)
        trace.append({"event": "t4_tool", "tool": "get_annual_payroll_summary", "status": data["status"]})
        plan.pop("pending", None)
        if data["status"] != "ok":
            return respond("unsupported", f"Complete reconciled annual sample records are unavailable for {escape(identity['given_name'] + ' ' + identity['family_name'])} for {plan['year']}. No T4 amounts were produced. The annual sample database must be available and cover that employee and year.")
        if data["employee"] != identity:
            raise PayrollMCPError("Annual identity does not match the selected employee")
        sources.append({"source_id": "D1", "kind": "annual_payroll_sample", "tool": "get_annual_payroll_summary",
                        "arguments": arguments, "employee_id": identity["employee_id"], "reporting_year": plan["year"],
                        "provenance": data["provenance"], "rules_sha256": data["rules_sha256"]})
        return respond("answer", render_breakdown(data), data)
    except (PayrollMCPError, KeyError, TypeError, ValueError):
        return respond("unsupported", "The annual sample payroll records could not be verified. No T4 amounts were produced. Check the employee selection and annual database connection.")
