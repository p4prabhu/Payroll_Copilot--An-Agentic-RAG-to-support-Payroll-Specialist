"""Read-only, offline 2026 Ontario regular-salary calculator; not wired to chat.

Input/output monetary values are CAD decimal strings, never binary floats.
Implements T4127 Option 1 with the optional YTD projection for K2/K2P credits.
"""
import argparse
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_UP, localcontext
import hashlib
import json
from pathlib import Path
import sys
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, StrictBool, StrictInt, StrictStr, ValidationError


ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "data/payroll_rules_2026.json"
CENT = Decimal("0.01")
ZERO = Decimal("0")


def monetary_input(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise ValueError("Use a CAD decimal string or integer dollars, not a float or boolean")
    if isinstance(value, str) and (len(value) > 32 or not value.strip()):
        raise ValueError("Invalid monetary value")
    try:
        amount = Decimal(value)
        if not amount.is_finite() or not ZERO <= amount <= Decimal("1000000000"):
            raise ValueError("Amount must be finite and between 0 and 1,000,000,000 CAD")
        if amount != amount.quantize(CENT):
            raise ValueError("Monetary inputs must have cent precision; inputs are not silently rounded")
        return abs(amount) if amount == 0 else amount
    except InvalidOperation:
        raise ValueError("Invalid monetary value") from None


Money = Annotated[Decimal, BeforeValidator(monetary_input)]


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Claim(StrictRecord):
    mode: Literal["basic_personal", "amount"]
    amount: Money | None = None


class YearToDate(StrictRecord):
    pensionable_earnings: Money
    cpp: Money
    cpp2: Money
    ei: Money
    is_complete: StrictBool
    employer_scope: StrictStr


class PayrollInput(StrictRecord):
    annual_salary: Money
    pay_frequency: StrictStr
    pay_periods_per_year: StrictInt
    pay_period_number: StrictInt
    payment_date: StrictStr
    province: StrictStr
    federal_claim: Claim
    ontario_claim: Claim
    cpp_status: StrictStr
    ei_status: StrictStr
    ytd: YearToDate
    regular_salary_only: StrictBool
    no_other_adjustments: StrictBool
    ontario_only_employment: StrictBool


def cents(amount: Decimal) -> Decimal:
    return amount.quantize(CENT, rounding=ROUND_HALF_UP)


def money(amount: Decimal) -> str:
    return format(cents(amount), ".2f")


def load_rules(path: Path) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    rules = json.loads(raw)
    if (rules["schema_version"] != 1 or rules["ruleset_id"] != "cra-2026-on-option1-regular-v2"
            or rules["province"] != "ON" or rules["currency"] != "CAD"
            or rules["effective_from"] != "2026-01-01" or rules["effective_to"] != "2026-12-31"
            or rules["supported_frequencies"] != {"biweekly": 26, "monthly": 12}):
        raise ValueError("Unsupported ruleset")
    for group in ("cpp", "ei", "federal", "ontario"):
        def check_numbers(value):
            if isinstance(value, dict):
                for item in value.values():
                    check_numbers(item)
            elif isinstance(value, list):
                for item in value:
                    check_numbers(item)
            elif (not isinstance(value, str) or not Decimal(value).is_finite()
                  or not ZERO <= Decimal(value) <= Decimal("1000000000")):
                raise ValueError("Invalid rule constant")
        check_numbers(rules[group])
    for group in ("federal", "ontario"):
        brackets = rules[group]["brackets"]
        thresholds = [Decimal(row["over"]) for row in brackets]
        if thresholds[0] != 0 or thresholds != sorted(set(thresholds)):
            raise ValueError("Invalid tax brackets")
    cpp, ei = rules["cpp"], rules["ei"]
    d = Decimal
    if ((d(cpp["ympe"]) - d(cpp["annual_exemption"])) * d(cpp["rate"]) != d(cpp["maximum"])
            or (d(cpp["ympe"]) - d(cpp["annual_exemption"])) * d(cpp["base_rate"]) != d(cpp["base_maximum"])
            or (d(cpp["yampe"]) - d(cpp["ympe"])) * d(cpp["cpp2_rate"]) != d(cpp["cpp2_maximum"])
            or d(ei["maximum_insurable_earnings"]) * d(ei["rate"]) != d(ei["maximum"])):
        raise ValueError("Inconsistent contribution limits")
    if not rules["sources"] or any(not s["url"].startswith("https://www.canada.ca/") for s in rules["sources"]):
        raise ValueError("Missing official rule provenance")
    return rules, hashlib.sha256(raw).hexdigest()


def issue(status: str, field: str, message: str) -> dict:
    return {"status": status, "issues": [{"field": field, "message": message}]}


def validate_scope(request: PayrollInput, rules: dict) -> dict | None:
    if request.province != "ON":
        return issue("unsupported", "province", "Only Ontario province of employment is supported")
    try:
        payment = date.fromisoformat(request.payment_date)
        if payment.isoformat() != request.payment_date:
            raise ValueError()
    except ValueError:
        return issue("invalid_input", "payment_date", "Supply a real YYYY-MM-DD payment date")
    if not rules["effective_from"] <= request.payment_date <= rules["effective_to"]:
        return issue("unsupported", "payment_date", "No reviewed rules cover this payment date; only 2026 is supported")
    periods = rules["supported_frequencies"].get(request.pay_frequency)
    if periods is None or request.pay_periods_per_year != periods:
        return issue("unsupported", "pay_periods_per_year", "Support is limited to biweekly with exactly 26 or monthly with exactly 12 periods")
    if not 1 <= request.pay_period_number <= periods:
        return issue("invalid_input", "pay_period_number", "Supply the current calendar-year pay-period index, including this payment")
    for field in ("regular_salary_only", "no_other_adjustments", "ontario_only_employment"):
        if not getattr(request, field):
            return issue("unsupported", field, "This calculation is outside the documented regular-salary scope")
    if request.cpp_status not in ("eligible_all_year", "exempt_all_year"):
        return issue("unsupported", "cpp_status", "Confirm CPP eligibility for all 12 months or exemption for the whole year; partial-year changes are unsupported")
    if request.ei_status not in ("insurable", "exempt_all_year"):
        return issue("unsupported", "ei_status", "Confirm insurable salary or whole-year EI exemption")
    if not request.ytd.is_complete or request.ytd.employer_scope != "current_employer":
        return issue("unsupported", "ytd", "Supply complete current-employer YTD values before this payment; partial fixture history cannot be used")
    for field in ("federal_claim", "ontario_claim"):
        claim = getattr(request, field)
        if (claim.mode == "amount") != (claim.amount is not None):
            return issue("invalid_input", field, "Supply amount only with mode=amount; basic_personal must be an explicit choice")
    limits = {"cpp": rules["cpp"]["maximum"], "cpp2": rules["cpp"]["cpp2_maximum"], "ei": rules["ei"]["maximum"]}
    for field, limit in limits.items():
        if getattr(request.ytd, field) > Decimal(limit):
            return issue("invalid_input", "ytd." + field, "YTD contributions exceed the supported annual limit; resolve the payroll record first")
    if request.pay_period_number == 1 and any(getattr(request.ytd, f) != 0 for f in ("pensionable_earnings", "cpp", "cpp2", "ei")):
        return issue("invalid_input", "ytd", "The first regular payment requires explicit zero prior YTD values in this regular-salary-only workflow")
    if request.cpp_status == "exempt_all_year" and (request.ytd.cpp or request.ytd.cpp2):
        return issue("invalid_input", "ytd.cpp", "Whole-year CPP exemption conflicts with existing contributions")
    if request.ei_status == "exempt_all_year" and request.ytd.ei:
        return issue("invalid_input", "ytd.ei", "Whole-year EI exemption conflicts with existing premiums")
    if request.ytd.cpp2 and request.ytd.pensionable_earnings <= Decimal(rules["cpp"]["ympe"]):
        return issue("invalid_input", "ytd.cpp2", "CPP2 contributions conflict with the supplied pensionable earnings")
    # These are upper-bound consistency checks, not a reconstruction of prior
    # payroll: prior salaries can differ and cents round once per pay period.
    tolerance = Decimal(request.pay_period_number - 1) * CENT
    if request.ytd.cpp > request.ytd.pensionable_earnings * Decimal(rules["cpp"]["rate"]) + tolerance:
        return issue("invalid_input", "ytd.cpp", "CPP contributions conflict with the supplied pensionable earnings")
    cpp2_upper = max(ZERO, request.ytd.pensionable_earnings - Decimal(rules["cpp"]["ympe"])) * Decimal(rules["cpp"]["cpp2_rate"])
    if request.ytd.cpp2 > cpp2_upper + tolerance:
        return issue("invalid_input", "ytd.cpp2", "CPP2 contributions conflict with the supplied pensionable earnings")
    return None


def bracket_tax(income: Decimal, brackets: list[dict]) -> Decimal:
    selected = brackets[0]
    for bracket in brackets[1:]:
        if income > Decimal(bracket["over"]):
            selected = bracket
    return max(ZERO, income * Decimal(selected["rate"]) - Decimal(selected["constant"]))


def federal_basic_personal(income: Decimal, federal: dict) -> Decimal:
    high, low = Decimal(federal["basic_personal_maximum"]), Decimal(federal["basic_personal_minimum"])
    start, end = Decimal(federal["basic_personal_phaseout_from"]), Decimal(federal["basic_personal_phaseout_to"])
    if income <= start:
        return high
    if income >= end:
        return low
    return cents(high - (income - start) * (high - low) / (end - start))


def health_premium(income: Decimal, ontario: dict) -> Decimal:
    premium = ZERO
    for band in ontario["health_premium"]:
        if income > Decimal(band["over"]):
            premium = min(Decimal(band["cap"]), Decimal(band["base"]) + (income - Decimal(band["over"])) * Decimal(band["rate"]))
    return premium


def _calculate(request: PayrollInput, rules: dict, rules_hash: str) -> dict:
    p = Decimal(request.pay_periods_per_year)
    remaining = Decimal(request.pay_periods_per_year - request.pay_period_number + 1)
    gross = cents(request.annual_salary / p)
    cpp_rules, ei_rules, federal, ontario = [rules[name] for name in ("cpp", "ei", "federal", "ontario")]
    d = Decimal
    exemption = (d(cpp_rules["annual_exemption"]) / p).quantize(CENT, rounding=ROUND_DOWN)
    cpp = cpp2 = ZERO
    if request.cpp_status == "eligible_all_year":
        cpp = cents(max(ZERO, min(d(cpp_rules["maximum"]) - request.ytd.cpp, d(cpp_rules["rate"]) * (gross - exemption))))
        w = max(request.ytd.pensionable_earnings, d(cpp_rules["ympe"]))
        cpp2 = cents(max(ZERO, min(d(cpp_rules["cpp2_maximum"]) - request.ytd.cpp2,
                                  (request.ytd.pensionable_earnings + gross - w) * d(cpp_rules["cpp2_rate"]))))
    ei = ZERO if request.ei_status == "exempt_all_year" else cents(max(ZERO, min(d(ei_rules["maximum"]) - request.ytd.ei, gross * d(ei_rules["rate"]))))
    enhanced_cpp = cents(cpp * d(cpp_rules["first_additional_rate"]) / d(cpp_rules["rate"]) + cpp2)
    annual_gross = p * gross
    taxable = max(ZERO, p * (gross - enhanced_cpp))
    # PR includes the payment being calculated. Rate ratios retain precision.
    base_credit = min(d(cpp_rules["base_maximum"]), (request.ytd.cpp + remaining * cpp) * d(cpp_rules["base_rate"]) / d(cpp_rules["rate"]))
    ei_credit = min(d(ei_rules["maximum"]), request.ytd.ei + remaining * ei)
    federal_claim = federal_basic_personal(taxable, federal) if request.federal_claim.mode == "basic_personal" else request.federal_claim.amount
    ontario_claim = d(ontario["basic_personal_amount"]) if request.ontario_claim.mode == "basic_personal" else request.ontario_claim.amount
    federal_tax = max(ZERO, bracket_tax(taxable, federal["brackets"]) - d(federal["lowest_rate"]) * (federal_claim + base_credit + ei_credit + min(annual_gross, d(federal["canada_employment_amount"]))))
    ontario_basic = max(ZERO, bracket_tax(taxable, ontario["brackets"]) - d(ontario["lowest_rate"]) * (ontario_claim + base_credit + ei_credit))
    surtax = sum((max(ZERO, ontario_basic - d(b["over"])) * d(b["rate"]) for b in ontario["surtaxes"]), ZERO)
    reduction = max(ZERO, min(ontario_basic + surtax, 2 * d(ontario["tax_reduction_basic"]) - (ontario_basic + surtax)))
    premium = health_premium(taxable, ontario)
    ontario_tax = max(ZERO, ontario_basic + surtax - reduction) + premium
    federal_period = cents(federal_tax / p)
    ontario_period = cents(ontario_tax / p)
    total_tax = federal_period + ontario_period
    rounding_difference = total_tax - cents((federal_tax + ontario_tax) / p)
    deductions = cpp + cpp2 + ei + total_tax
    amounts = {"gross_pay": gross, "cpp": cpp, "cpp2": cpp2, "ei": ei,
               "federal_income_tax": federal_period, "ontario_income_tax": ontario_period,
               "total_income_tax": total_tax, "total_deductions": deductions, "net_pay": gross - deductions}
    if amounts["net_pay"] < 0:
        return issue("unsupported", "annual_salary", "Calculated deductions exceed gross; this case needs payroll review")
    sources = [s for s in rules["sources"] if s["effective_from"] <= request.payment_date]
    return {
        "status": "calculated", "is_estimate": True, "currency": rules["currency"],
        "inputs": request.model_dump(mode="json"), "amounts": {key: money(value) for key, value in amounts.items()},
        "calculation_details": {
            "pay_periods_remaining_including_current": int(remaining), "cpp_period_exemption": money(exemption),
            "enhanced_cpp_deduction_f5": money(enhanced_cpp), "annualized_gross_pay": money(annual_gross),
            "annual_taxable_income": money(taxable), "federal_claim_used": money(federal_claim),
            "ontario_claim_used": money(ontario_claim), "projected_annual_base_cpp_credit_amount": str(base_credit),
            "projected_annual_ei_credit_amount": str(ei_credit), "annual_federal_tax": str(federal_tax),
            "annual_ontario_basic_tax": str(ontario_basic), "annual_ontario_surtax": str(surtax),
            "annual_ontario_tax_reduction": str(reduction), "annual_ontario_health_premium": money(premium),
            "annual_ontario_tax": str(ontario_tax), "income_tax_rounding_difference": money(rounding_difference),
        },
        "provenance": {"ruleset_id": rules["ruleset_id"], "rules_sha256": rules_hash,
                       "verified_on": rules["verified_on"], "effective_from": rules["effective_from"],
                       "effective_to": rules["effective_to"], "method": rules["method"],
                       "validation_status": rules["validation_status"], "sources": sources,
                       "income_tax_rounding": rules["rounding"]["income_tax"],
                       "validation_evidence": rules["validation_evidence"]},
        "limitations": [
            "Regular cash salary only, within the explicitly confirmed scope; no employee database was read or modified.",
            "YTD amounts were explicitly supplied, not inferred from partial payment history.",
            "This is a withholding estimate, not an annual tax return or a posted payroll payment.",
            "Gross pay is rounded per period; this engine does not reconcile annual salary rounding on the final payment.",
            "Federal and Ontario income tax are each rounded to cents, then summed; income_tax_rounding_difference compares that total with rounding combined tax once.",
            rules["validation_status"],
        ],
    }


def calculate_payroll(payload: dict, *, rules_path: Path = RULES_PATH) -> dict:
    """Return calculated, clarification, invalid_input, unsupported or unavailable."""
    with localcontext() as context:
        context.prec = 38
        try:
            request = PayrollInput.model_validate(payload)
        except ValidationError as error:
            errors = error.errors(include_url=False, include_input=False)
            status = "clarification" if all(e["type"] == "missing" for e in errors) else "invalid_input"
            return {"status": status, "issues": [{"field": ".".join(str(part) for part in e["loc"]), "message": e["msg"]} for e in errors]}
        try:
            rules, rules_hash = load_rules(rules_path)
            problem = validate_scope(request, rules)
            return problem if problem else _calculate(request, rules, rules_hash)
        except (OSError, UnicodeError, ValueError, KeyError, TypeError, InvalidOperation, ZeroDivisionError):
            return issue("unavailable", "rules", "Calculation rules are unavailable or invalid; no amounts were produced")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="JSON request file; omit to read JSON from stdin")
    args = parser.parse_args()
    try:
        text = args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read()
        result = calculate_payroll(json.loads(text))
    except (OSError, UnicodeError, ValueError):
        result = issue("invalid_input", "request", "Supply a readable JSON request using decimal strings for money")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "calculated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
