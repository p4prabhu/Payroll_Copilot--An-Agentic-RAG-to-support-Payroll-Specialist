"""Build reproducible 2026 simulated year-end payroll; never modifies a database.

This creates sample accounting facts from withholding estimates, not real payroll
or a filing-ready T4. It deliberately replaces the old illustrative deductions
only in a separate fixture; the original fixture and database remain unchanged.
"""

import argparse
import calendar
import copy
from datetime import date, timedelta
from decimal import Decimal
import json
from pathlib import Path

from init_employee_db import FIXTURE_PATH, ROOT, load_fixture
from payroll_calculator import RULES_PATH, calculate_payroll, load_rules


OUTPUT = ROOT / "data/fixtures_t4_2026.json"
ASSUMPTIONS = (
    "Simulated closed 2026 calendar year, not actual or forecast company payroll. "
    "One Ontario employer, CAD, regular salary only, basic federal and Ontario TD1 claims, "
    "CPP eligibility all 12 months and EI-insurable employment. No bonuses, taxable benefits, "
    "pensions, union dues, additional withholding, other employment or off-cycle payments. "
    "First biweekly payment is a guaranteed full regular payment despite the short onboarding window; "
    "no first-pay proration. Salary changes apply to the rate effective at period end. "
    "Sam's final regular payment is October 16 for a period ending October 15; all final entitlements "
    "are assumed included and no further settlement is due in this sample. "
    "Per-period gross rounding is retained without annual salary true-up. Dental code 1: "
    "no employer-offered dental coverage. Addresses are sample placeholders; SIN and employer "
    "payroll account are intentionally absent. Calculations inherit the existing engine's limited "
    "PDOC validation; a closed sample dataset does not certify tax correctness or external completeness."
)


def money(value: int) -> str:
    return format(Decimal(value) / 100, ".2f")


def build_fixture() -> dict:
    data = copy.deepcopy(load_fixture(FIXTURE_PATH))
    rules, rules_hash = load_rules(RULES_PATH)
    data["fixture_version"] = 2
    data["notes"] = ASSUMPTIONS
    data["dataset_info"][0].update(snapshot_date="2026-12-31", history_complete_from="2026-01-01",
                                   description=ASSUMPTIONS)
    data["pay_runs"], data["payroll_payments"], data["payroll_lines"] = [], [], []
    data["employer_tax_profiles"] = [{"reporting_year": 2026,
        "legal_name": "Maple Harbour Sample Employer", "mailing_address": "SAMPLE ADDRESS — NOT FOR MAILING, Ontario, Canada",
        "payroll_account": None, "dataset_kind": "simulated_year_end"}]
    data["employee_tax_profiles"] = [{"employee_id": employee["employee_id"], "reporting_year": 2026,
        "mailing_address": "SAMPLE ADDRESS — NOT FOR MAILING, Ontario, Canada", "social_insurance_number": None,
        "dental_benefits_code": 1} for employee in data["employees"]]
    data["annual_payroll_coverage"], data["payroll_tax_bases"] = [], []
    schedules = {}
    for group in data["pay_groups"]:
        runs = []
        for index in range(1, group["periods_per_year"] + 1):
            if group["frequency"] == "biweekly":
                paid = date(2026, 1, 9) + timedelta(days=14 * (index - 1))
                end = paid - timedelta(days=1)
                start = max(date(2026, 1, 1), end - timedelta(days=13))
            else:
                end = date(2026, index, calendar.monthrange(2026, index)[1])
                paid, start = end, date(2026, index, 1)
            run = {"pay_run_id": f"sample-{group['frequency']}-{index:02d}", "pay_group_id": group["pay_group_id"],
                   "period_start": start.isoformat(), "period_end": end.isoformat(), "payment_date": paid.isoformat(),
                   "cutoff_at_utc": start.isoformat() + "T12:00:00Z", "status": "posted"}
            runs.append(run)
            data["pay_runs"].append(run)
        schedules[group["pay_group_id"]] = runs
    for job in data["employments"]:
        employee_id = job["employee_id"]
        ytd = {key: 0 for key in ("pensionable_earnings", "cpp", "cpp2", "ei")}
        manifest = []
        assignments = [a for a in data["pay_group_assignments"] if a["employment_id"] == job["employment_id"]]
        if len(assignments) != 1:
            raise ValueError("Sample builder supports one pay group per employment")
        group = next(g for g in data["pay_groups"] if g["pay_group_id"] == assignments[0]["pay_group_id"])
        for index, run in enumerate(schedules[group["pay_group_id"]], 1):
            if run["period_end"] < job["starts_on"] or (job["ends_before"] and run["period_end"] >= job["ends_before"]):
                continue
            rate = next(c for c in data["compensation_rates"] if c["employment_id"] == job["employment_id"]
                        and c["valid_from"] <= run["period_end"] and (c["valid_to"] is None or run["period_end"] < c["valid_to"]))
            payload = {"annual_salary": money(rate["annual_salary_cents"]), "pay_frequency": group["frequency"],
                       "pay_periods_per_year": group["periods_per_year"], "pay_period_number": index,
                       "payment_date": run["payment_date"], "province": "ON",
                       "federal_claim": {"mode": "basic_personal"}, "ontario_claim": {"mode": "basic_personal"},
                       "cpp_status": "eligible_all_year", "ei_status": "insurable",
                       "ytd": {**{key: money(value) for key, value in ytd.items()}, "is_complete": True, "employer_scope": "current_employer"},
                       "regular_salary_only": True, "no_other_adjustments": True, "ontario_only_employment": True}
            result = calculate_payroll(payload)
            if result["status"] != "calculated":
                raise ValueError(f"Sample calculation failed for {employee_id}: {result['status']}")
            amounts = {key: int(Decimal(value) * 100) for key, value in result["amounts"].items()}
            payment_id = f"sample-{employee_id}-{index:02d}"
            data["payroll_payments"].append({"payment_id": payment_id, "employment_id": job["employment_id"],
                "pay_run_id": run["pay_run_id"], "compensation_id": rate["compensation_id"], "payment_type": "regular",
                "province_code": "ON", "currency": "CAD", "gross_cents": amounts["gross_pay"],
                "deduction_cents": amounts["total_deductions"], "net_cents": amounts["net_pay"], "status": "posted",
                "recorded_at": run["payment_date"] + "T12:00:00Z", "recorded_by": "simulated-year-end-builder"})
            for number, (code, key) in enumerate((("SALARY", "gross_pay"), ("FED_TAX", "federal_income_tax"),
                    ("ON_TAX", "ontario_income_tax"), ("CPP", "cpp"), ("CPP2", "cpp2"), ("EI", "ei")), 1):
                data["payroll_lines"].append({"payment_id": payment_id, "line_number": number,
                    "category": "earning" if code == "SALARY" else "deduction", "code": code, "amount_cents": amounts[key]})
            data["payroll_tax_bases"].append({"payment_id": payment_id,
                "pensionable_earnings_cents": amounts["gross_pay"], "insurable_earnings_cents": amounts["gross_pay"],
                "taxable_benefits_cents": 0, "calculation_input_json": json.dumps(payload, sort_keys=True)})
            manifest.append({"payment_id": payment_id, "pay_run_id": run["pay_run_id"], "payment_date": run["payment_date"]})
            ytd["pensionable_earnings"] += amounts["gross_pay"]
            for key in ("cpp", "cpp2", "ei"):
                ytd[key] += amounts[key]
        data["annual_payroll_coverage"].append({"employee_id": employee_id, "reporting_year": 2026,
            "coverage_start": "2026-01-01", "coverage_end": "2026-12-31", "expected_payments_json": json.dumps(manifest, sort_keys=True),
            "rules_sha256": rules_hash, "pensionable_ceiling_cents": int(Decimal(rules["cpp"]["yampe"]) * 100),
            "insurable_ceiling_cents": int(Decimal(rules["ei"]["maximum_insurable_earnings"]) * 100), "assumptions": ASSUMPTIONS})
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    # Rebuilding this generated JSON is intentional; live databases are never opened.
    if args.output.resolve() == FIXTURE_PATH.resolve():
        parser.error("Choose a separate output; the original fixture is preserved")
    args.output.write_text(json.dumps(build_fixture(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Created simulated year-end fixture: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
