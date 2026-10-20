"""Controlled natural-language workforce summaries; no generated SQL or model calls."""
from copy import deepcopy
from decimal import Decimal
import re

from employee_repository import iso_date
from payroll_mcp_client import PayrollMCPError

SESSION_KEY = "_analytics"
DATE = r"\b\d{4}-\d{2}-\d{2}\b"
NUMBER = r"\d[\d,]*(?:\.\d+)?(?:[kK]\b)?"
WORDS = set("""how many employee employees people staff workers active inactive current currently working work employed
    total count number workforce headcount head show me please paid payment frequency pay monthly biweekly weekly semi
    semimonthly versus vs and by breakdown earn earning earns more than less above below over under at least most
    salary salaries annual annually year yearly per between from to dollars cad joined join started hired left
    leave leavers ended end employment endings who with a an is are there the in company on as of for our us have
    do does they them that date given specific want i would like know working company what about those these
    exactly now today tomorrow all records recorded can you tell filter filtered greater equal inclusive both
    start stop calculate summary range during through until hires please get list group grouped or bi
    give details detail display roster names name numbers their""".split())
FREQUENCIES = {"biweekly": r"\bbi[ -]?weekly\b", "semimonthly": r"\bsemi[ -]?monthly\b",
               "weekly": r"\bweekly\b", "monthly": r"\bmonthly\b"}


def is_summary_question(question):
    lower = question.lower()
    if re.search(r"\bhow many\s+(?:(?:vacation|sick|working|business|calendar)\s+)?(?:days|hours|weeks|months|years|pay periods|payments|paychecks|paycheques)\b", lower):
        return False
    return bool(re.search(r"\b(?:head\s*count|workforce summary|employee count|number of employees|total employees)\b", lower)
                or re.search(r"\bhow many\b", lower) and re.search(r"\b(?:employees?|people|staff|workers|paid|earn|joined|left|active|them)\b", lower)
                or re.search(r"\b(?:breakdown|grouped|group)\b.*\b(?:pay|employees?|frequency)\b", lower))


PAGE_COMMANDS = {"next page": "next", "next employees": "next", "previous page": "previous", "previous employees": "previous"}


def is_roster_question(question):
    lower = question.lower()
    return bool(re.search(r"\b(?:employee roster|staff roster|roster of employees)\b", lower)
                or re.search(r"\b(?:list|display)\b.*\b(?:employees|staff|workers|people)\b", lower)
                or re.search(r"\b(?:show|give|details)\b", lower) and re.search(r"\ball\b.*\b(?:employees?|staff|workers|people)\b", lower))


def should_route(question, session):
    if is_summary_question(question) or is_roster_question(question) or question.strip().lower() in PAGE_COMMANDS:
        return True
    pending = session.get(SESSION_KEY, {}).get("pending", False)
    return bool(pending and (re.fullmatch(r"\s*(?:as of |on |from )?\d{4}-\d{2}-\d{2}(?:\s+(?:to|through|and)\s+\d{4}-\d{2}-\d{2})?[.!]?\s*", question, re.I)
                            or re.fullmatch(r"\s*\$?" + NUMBER + r"\s*(?:CAD|dollars)?[.!]?\s*", question, re.I)))


def cents(text):
    number = Decimal(text.rstrip("kK").replace(",", "")) * (1000 if text[-1:] in {"k", "K"} else 1)
    if not number.is_finite() or not 0 <= number <= Decimal("1000000000") or number != number.quantize(Decimal(".01")):
        raise ValueError("Salary must be a nonnegative CAD amount with cent precision")
    return int(number * 100)


def plan_request(question, previous):
    """Recognize bounded supported forms; unknown filters never disappear silently."""
    lower = question.lower().strip()
    if re.search(r"\b(?:department|location|gender|age|born|nationality|manager|contractors?|full.time|part.time|bonus|salary changes?|inactive|all records|all employees ever)\b", lower):
        return None, "That summary dimension is not supported. I can count active employees, filter pay frequency or annual salary, and count recorded joiners and employment endings."
    cleaned = re.sub(DATE, " ", lower)
    if re.search(r"[<>!=]", cleaned):
        return None, "Please state salary comparisons in words, such as 'above' or 'at least'."
    if re.search(r"(?:-\s*\$?\s*\d|\$\s*-\s*\d)", cleaned):
        return None, "Please supply nonnegative annual salary bounds."
    cleaned = re.sub(NUMBER, " ", cleaned)
    unknown = set(re.findall(r"[a-z]+", cleaned)) - WORDS
    if unknown:
        return None, "I could not identify a supported summary filter. Please ask for active headcount, pay-frequency counts, annual-salary filters, or join/leave counts with explicit dates."
    if re.search(r"\b(?:today|tomorrow|now|currently)\b", lower):
        return None, "Please replace the relative date with an explicit YYYY-MM-DD date. I will not assume today's date."
    new_question = is_summary_question(question) or is_roster_question(question)
    if new_question:
        kind = "roster" if is_roster_question(question) else "movements" if re.search(r"\b(?:joined|join|hired|hires|started|left|leavers|ended|endings)\b", lower) else "summary"
        plan = {"kind": kind, "arguments": {"limit": 20, "offset": 0} if kind == "roster" else {}}
        if kind in {"summary", "roster"} and previous.get("kind") in {"summary", "roster"} and "as_of_date" in previous.get("arguments", {}):
            plan["arguments"]["as_of_date"] = previous["arguments"]["as_of_date"]
    else:
        plan = deepcopy(previous)
    args = plan.setdefault("arguments", {})
    dates = re.findall(DATE, question)
    for value in dates:
        iso_date(value, "Date")
    if plan.get("kind") == "movements":
        if re.search(r"\b(?:salary|salaries|paid|pay|frequency|weekly|monthly|biweekly|semimonthly|earn|earning)\b", lower):
            return None, "Join/leave counts do not support salary or pay-frequency filters. Please ask those as an active-workforce question."
        if len(dates) == 2:
            args.update(start_date=dates[0], end_date=dates[1])
        elif len(dates) == 1 and re.search(r"\bon\b", lower):
            args.update(start_date=dates[0], end_date=dates[0])
        elif dates:
            args.clear()
        return plan, None
    if len(dates) > 1:
        return None, "For active headcount, provide one as-of date. Use a join/leave question for a date range."
    if dates:
        args["as_of_date"] = dates[0]
    if plan.get("kind") == "roster":
        if re.search(r"\b(?:weekly|monthly|biweekly|semimonthly|above|below|over|under|than|between|least|most|earn|earning|earns|joined|left|ended|hired|leavers)\b", lower) or re.search(NUMBER, re.sub(DATE, " ", lower)):
            return None, "The roster currently lists all recorded active employees on one date. Salary and pay-frequency filters are available for summary counts; they are not roster filters yet."
        return plan, None
    compact = re.sub(r"\bbi[ -]?weekly\b", " biweekly ", lower)
    compact = re.sub(r"\bsemi[ -]?monthly\b", " semimonthly ", compact)
    frequencies = {name for name, pattern in FREQUENCIES.items() if re.search(pattern, compact)}
    if re.search(r"\bor\b", re.sub(r"(?:greater|less) than or equal to", "", lower)):
        return None, "Combined OR filters are not supported. Please ask each filtered count separately."
    if len(frequencies) == 1:
        args["pay_frequency"] = next(iter(frequencies))
    elif frequencies:
        if not re.search(r"\b(?:versus|vs|breakdown|by|grouped)\b", lower):
            return None, "Please request a pay-frequency breakdown, or filter one pay frequency at a time."
        args.pop("pay_frequency", None)  # Return the explicit full frequency breakdown.
    comparisons = [
        ("gte", r"\b(?:at least|greater than or equal to)\s*\$?\s*(" + NUMBER + r")?"),
        ("lte", r"\b(?:at most|less than or equal to)\s*\$?\s*(" + NUMBER + r")?"),
        ("gt", r"\b(?:above|over|more than|greater than)\s*\$?\s*(" + NUMBER + r")?"),
        ("lt", r"\b(?:below|under|less than)\s*\$?\s*(" + NUMBER + r")?"),
    ]
    between = re.search(r"\bbetween\s*\$?\s*(" + NUMBER + r")\s*(?:and|to)\s*\$?\s*(" + NUMBER + r")", re.sub(DATE, " ", lower))
    numbers = re.findall(NUMBER, re.sub(DATE, " ", lower))
    if len(numbers) > (2 if between else 1):
        return None, "Please provide one salary bound, or two bounds using 'between X and Y'."
    if between:
        args.update(salary_comparison="between", salary_min_cents=cents(between[1]), salary_max_cents=cents(between[2]))
    else:
        matches = [(kind, match) for kind, pattern in comparisons if (match := re.search(pattern, lower))]
        # Inclusive forms also contain the simpler words; prefer the complete form.
        if any(kind == "gte" for kind, _ in matches):
            matches = [(kind, m) for kind, m in matches if kind != "gt"]
        if any(kind == "lte" for kind, _ in matches):
            matches = [(kind, m) for kind, m in matches if kind != "lt"]
        if len(matches) > 1:
            return None, "Please use one salary comparison, or 'between X and Y' for inclusive salary bounds."
        if matches:
            kind, match = matches[0]
            args["salary_comparison"] = kind
            if match[1]:
                args["salary_min_cents" if kind in {"gt", "gte"} else "salary_max_cents"] = cents(match[1])
        elif not new_question and re.fullmatch(r"\s*\$?" + NUMBER + r"\s*(?:CAD|dollars)?[.!]?\s*", question, re.I):
            number = re.search(NUMBER, question)[0]
            key = "salary_min_cents" if args.get("salary_comparison") in {"gt", "gte"} else "salary_max_cents"
            if args.get("salary_comparison") in {"gt", "gte", "lt", "lte"}:
                args[key] = cents(number)
            else:
                return None, "Please state whether the annual salary should be above or below that amount."
        elif re.search(r"\b(?:earn|earning|earns|salary|salaries|annually)\b", lower):
            return None, "What annual salary comparison should I use? For example, 'above $70,000' or 'between $50,000 and $80,000' (inclusive)."
        elif numbers:
            return None, "Please label the salary comparison, or provide a full YYYY-MM-DD as-of date."
    return plan, None


def render_summary(result):
    args = result["filters"]
    description = []
    if args["pay_frequency"]:
        description.append("pay frequency: " + args["pay_frequency"])
    kind = args["salary_comparison"]
    labels = {"gt": "above", "gte": "at least", "lt": "below", "lte": "at most"}
    if kind == "between":
        description.append(f"annual salary between CAD {Decimal(args['salary_min_cents'])/100:,.2f} and CAD {Decimal(args['salary_max_cents'])/100:,.2f}, inclusive")
    elif kind != "any":
        value = args["salary_min_cents"] if kind in {"gt", "gte"} else args["salary_max_cents"]
        description.append(f"annual salary {labels[kind]} CAD {Decimal(value)/100:,.2f}")
    count = result["employee_count"]
    text = f"**{count} active {'employee' if count == 1 else 'employees'}** match{'es' if count == 1 else ''} as of **{result['as_of_date']}**. [D1]"
    text += "\n\nFilters: " + ("; ".join(description) if description else "all recorded active employees") + "."
    if result["pay_frequency_counts"]:
        text += "\n\n**Pay-frequency breakdown of matching employees**\n\n" + "\n".join(
            f"- {g['frequency'].capitalize()}: **{g['employee_count']}**" for g in result["pay_frequency_counts"])
    if result["missing_salary_count"] or result["missing_pay_group_count"]:
        text += f"\n\nData gaps in the active workforce: {result['missing_salary_count']} missing salary records; {result['missing_pay_group_count']} missing pay-group records. Unknown pay groups remain visible."
    return text


def render_roster(result):
    def cell(value):
        return re.sub(r"([\\`*_{}\[\]<>()#!|])", r"\\\1", " ".join(str(value).split()))
    count, offset = result["returned_count"], result["offset"]
    text = f"**Active employee roster as of {result['as_of_date']}**. [D1]\n\n"
    text += (f"Showing {offset + 1}–{offset + count} of **{result['match_count']}** recorded active employees."
             if count else f"No employees on this page. Total recorded active employees: **{result['match_count']}**.")
    if result["employees"]:
        text += "\n\n| Employee | Employee number | Annual salary (CAD) | Pay frequency |\n| :--- | :--- | ---: | :--- |\n"
        for row in result["employees"]:
            salary = "Not recorded" if row["annual_salary_cents"] is None else f"${Decimal(row['annual_salary_cents'])/100:,.2f}"
            text += f"| {cell(row['given_name'] + ' ' + row['family_name'])} | {cell(row['employee_number'])} | {salary} | {cell(row['frequency'] or 'Not recorded')} |\n"
    if any(row["annual_salary_cents"] is None or row["frequency"] is None for row in result["employees"]):
        text += "\nSome salary or pay-frequency fields are not recorded; missing values are not zero."
    text += "\n\nAnnual salaries are stored compensation effective on the requested date, not new payroll calculations. This list excludes employees inactive on that date."
    return text


async def analytics_turn(client, question, session):
    session = deepcopy(session)
    session.pop("_calculation", None)
    previous = session.get(SESSION_KEY, {})
    trace = []

    def respond(status, text, result=None, tool=None, arguments=None):
        sources = [] if result is None else [{"source_id": "D1", "kind": "employee_roster" if tool == "list_employees" else "workforce_summary", "tool": tool,
            "arguments": arguments, "provenance": result["provenance"]}]
        outcome = {"status": status, "text": text, "citations": ["D1"] if sources else [], "sources": sources}
        if result is not None and tool == "list_employees" and result["status"] == "ok":
            outcome["roster"] = result
        return {"employee_session": session, "trace": trace, "outcome": outcome}

    page = PAGE_COMMANDS.get(question.strip().lower())
    if page:
        if previous.get("kind") != "roster" or not previous.get("page_ready"):
            return respond("clarification", "Which employee roster should I page through? Please request a roster with an explicit as-of date first.")
        plan = deepcopy(previous)
        if page == "next":
            if not plan.get("has_more"):
                return respond("clarification", "You are already on the last roster page. Would you like the previous page or a roster for another date?")
            plan["arguments"]["offset"] = plan["next_offset"]
        else:
            if plan["arguments"]["offset"] == 0:
                return respond("clarification", "You are already on the first roster page. Would you like a roster for another date?")
            plan["arguments"]["offset"] = max(0, plan["arguments"]["offset"] - plan["arguments"]["limit"])
    try:
        if not page:
            plan, error = plan_request(question, previous)
        else:
            error = None
    except (ValueError, ArithmeticError):
        session.pop(SESSION_KEY, None)
        return respond("clarification", "Please provide valid YYYY-MM-DD dates and annual salary amounts with cent precision. What corrected values should I use?")
    if error:
        session.pop(SESSION_KEY, None)
        return respond("clarification", error + " What supported query would you like to run?")
    if not client or not client.available:
        return respond("unsupported", "The employee database connection is unavailable. No employee count could be retrieved.")
    args = plan["arguments"]
    required = ["start_date", "end_date"] if plan["kind"] == "movements" else ["as_of_date"]
    if plan["kind"] == "summary":
        comp = args.get("salary_comparison", "any")
        required += ["salary_min_cents"] if comp in {"gt", "gte"} else ["salary_max_cents"] if comp in {"lt", "lte"} else []
    missing = [key for key in required if key not in args]
    plan["pending"] = bool(missing)
    session[SESSION_KEY] = plan
    if missing:
        if missing == ["as_of_date"]:
            return respond("clarification", "As of what date? Please use YYYY-MM-DD.")
        return respond("clarification", "What values should I use for " + ", ".join(key.replace("_", " ") for key in missing) + "? Please use YYYY-MM-DD dates and annual salary amounts in CAD.")
    tool = {"movements": "get_employment_movements", "roster": "list_employees", "summary": "get_workforce_summary"}[plan["kind"]]
    try:
        result = await client.call(tool, args)
        trace.append({"event": "workforce_tool", "tool": tool, "status": result["status"]})
    except PayrollMCPError:
        return respond("unsupported", "The employee summary could not be retrieved. Check the explicit dates, filters and local MCP connection.")
    if result["status"] != "ok":
        limitation = ("Some active employees lack data required by the filter; I cannot establish the full matching count."
                      if result["status"] == "data_gap" else "The requested dates are after the database snapshot; I cannot project employee counts.")
        return respond("unsupported", limitation, result, tool, args)
    if plan["kind"] == "roster":
        plan.update(page_ready=True, has_more=result["has_more"], next_offset=result["next_offset"])
        text = render_roster(result)
    elif plan["kind"] == "summary":
        text = render_summary(result)
    else:
        text = (f"**Recorded employment movements: {result['start_date']} to {result['end_date']}**, inclusive. [D1]\n\n"
                f"- Employees joining: **{result['joined_employee_count']}** · Recorded start events: {result['join_event_count']}.\n"
                f"- Employees with employment ending: **{result['ended_employee_count']}** · Recorded end events: {result['end_event_count']}.\n\n"
                "An ending is counted on the first inactive date (ends_before), not the last day worked. Rehires can produce multiple events per person; these counts are not net headcount change.")
    text += f"\n\nDatabase snapshot: {result['provenance']['dataset_snapshot_date']}. These are counts of recorded employees, not a complete company-wide HR roster."
    return respond("answer", text, result, tool, args)
