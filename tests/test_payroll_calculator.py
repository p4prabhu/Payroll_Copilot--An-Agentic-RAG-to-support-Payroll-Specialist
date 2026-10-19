"""Financial reference, boundary and failure tests; no network, LLM or database."""
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from payroll_calculator import RULES_PATH, bracket_tax, calculate_payroll, federal_basic_personal, health_premium, load_rules


REFERENCES = json.loads((ROOT / "tests/fixtures/payroll_reference_cases.json").read_text(encoding="utf-8"))


class PayrollCalculatorTests(unittest.TestCase):
    def request(self, case=0):
        return deepcopy(REFERENCES["cases"][case]["input"])

    def assert_no_amounts(self, result, status):
        self.assertEqual(result["status"], status, result)
        self.assertNotIn("amounts", result)
        self.assertTrue(result["issues"])

    def test_all_independently_computed_reference_cases_and_reconciliation(self):
        for case in REFERENCES["cases"]:
            with self.subTest(case=case["id"]):
                result = calculate_payroll(case["input"])
                self.assertEqual(result["status"], "calculated", result)
                self.assertEqual(result["amounts"], case["expected_amounts"])
                for key, value in case["expected_details"].items():
                    self.assertEqual(result["calculation_details"][key], value)
                a = {key: Decimal(value) for key, value in result["amounts"].items()}
                self.assertEqual(a["federal_income_tax"] + a["ontario_income_tax"], a["total_income_tax"])
                self.assertEqual(a["cpp"] + a["cpp2"] + a["ei"] + a["total_income_tax"], a["total_deductions"])
                self.assertEqual(a["net_pay"] + a["total_deductions"], a["gross_pay"])
                self.assertTrue(all(value >= 0 for value in a.values()))

    def test_published_cra_component_examples(self):
        for check in REFERENCES["published_component_checks"]:
            case = next(c for c in REFERENCES["cases"] if c["id"] == check["case_id"])
            result = calculate_payroll(case["input"])
            if "expected_cpp2" in check:
                self.assertEqual(result["amounts"]["cpp2"], check["expected_cpp2"])
            if "expected_cpp_period_exemption" in check:
                self.assertEqual(result["calculation_details"]["cpp_period_exemption"], check["expected_cpp_period_exemption"])

    def test_user_supplied_pdoc_screenshot_matches_each_displayed_amount(self):
        for observation in REFERENCES["pdoc_observations"]:
            case = next(c for c in REFERENCES["cases"] if c["id"] == observation["case_id"])
            result = calculate_payroll(case["input"])
            inputs = observation["observed_inputs"]
            for field in ("payment_date", "province", "pay_periods_per_year"):
                self.assertEqual(case["input"][field], inputs[field])
            self.assertEqual(result["amounts"], observation["observed_amounts"])
            details = result["calculation_details"]
            self.assertEqual(details["federal_claim_used"], inputs["federal_td1_claim"])
            self.assertEqual(details["ontario_claim_used"], inputs["ontario_td1_claim"])
            for field in ("pensionable_earnings", "cpp", "cpp2", "ei"):
                self.assertEqual(case["input"]["ytd"][field], inputs["prior_" + field])
            self.assertEqual(details["enhanced_cpp_deduction_f5"], observation["observed_details"]["enhanced_cpp_deduction_f5"])
            self.assertEqual(Decimal(details["annual_taxable_income"]) / case["input"]["pay_periods_per_year"],
                             Decimal(observation["observed_details"]["taxable_income_per_period"]))

    def test_separate_tax_rounding_preserves_both_shares_and_reconciles_totals(self):
        for index, federal, ontario, total, difference in (
                (0, "153.28", "88.01", "241.29", "0.01"),
                (3, "51.75", "48.09", "99.84", "-0.01")):
            result = calculate_payroll(self.request(index))
            amounts, details = result["amounts"], result["calculation_details"]
            self.assertEqual(amounts["federal_income_tax"], federal)
            self.assertEqual(amounts["ontario_income_tax"], ontario)
            self.assertEqual(amounts["total_income_tax"], total)
            self.assertEqual(details["income_tax_rounding_difference"], difference)
            self.assertNotIn("ontario_tax_allocation_adjustment", details)

    def test_missing_required_inputs_request_clarification_instead_of_defaults(self):
        payload = self.request()
        for field in payload:
            with self.subTest(field=field):
                incomplete = deepcopy(payload)
                incomplete.pop(field)
                result = calculate_payroll(incomplete)
                self.assert_no_amounts(result, "clarification")
                self.assertIn(field, [e["field"] for e in result["issues"]])
        for field in payload["ytd"]:
            incomplete = deepcopy(payload)
            incomplete["ytd"].pop(field)
            self.assert_no_amounts(calculate_payroll(incomplete), "clarification")

    def test_monetary_inputs_are_never_coerced_from_floats_or_rounded(self):
        for value in (50000.0, True, "NaN", "Infinity", "-1.00", "50000.001", "1000000001", None, [], ""):
            with self.subTest(value=value):
                payload = self.request()
                payload["annual_salary"] = value
                self.assert_no_amounts(calculate_payroll(payload), "invalid_input")
        payload = self.request()
        payload["ytd"]["cpp"] = 0.0
        self.assert_no_amounts(calculate_payroll(payload), "invalid_input")

    def test_integer_and_decimal_money_match_string_input(self):
        expected = calculate_payroll(self.request())["amounts"]
        for value in (50000, Decimal("50000.00")):
            payload = self.request()
            payload["annual_salary"] = value
            self.assertEqual(calculate_payroll(payload)["amounts"], expected)

    def test_boolean_confirmation_fields_require_actual_booleans(self):
        for field in ("no_other_adjustments", "regular_salary_only", "ontario_only_employment"):
            for value in (1, "true"):
                payload = self.request()
                payload[field] = value
                self.assert_no_amounts(calculate_payroll(payload), "invalid_input")
        payload = self.request()
        payload["ytd"]["is_complete"] = 1
        self.assert_no_amounts(calculate_payroll(payload), "invalid_input")

    def test_unknown_fields_and_non_object_requests_are_rejected(self):
        payload = self.request()
        payload["bonus"] = "1000.00"
        self.assert_no_amounts(calculate_payroll(payload), "invalid_input")
        for value in (None, [], "50000", 50000):
            self.assert_no_amounts(calculate_payroll(value), "invalid_input")

    def test_only_explicit_supported_payment_dates_are_accepted(self):
        for value, status in (("today", "invalid_input"), ("2026-02-30", "invalid_input"),
                              ("20261113", "invalid_input"), ("2026-1-9", "invalid_input"),
                              ("2025-12-31", "unsupported"), ("2027-01-01", "unsupported")):
            payload = self.request()
            payload["payment_date"] = value
            self.assert_no_amounts(calculate_payroll(payload), status)

    def test_effective_dates_select_source_versions_and_do_not_extend_into_2027(self):
        outputs = []
        for payment in ("2026-01-01", "2026-06-30", "2026-07-01", "2026-12-31"):
            payload = self.request()
            payload["payment_date"] = payment
            result = calculate_payroll(payload)
            outputs.append(result["amounts"])
            self.assertEqual(len(result["provenance"]["sources"]), 1 if payment < "2026-07-01" else 2)
        self.assertTrue(all(output == outputs[0] for output in outputs))

    def test_period_count_and_current_period_are_explicit_and_bounded(self):
        for frequency, count in (("weekly", 52), ("biweekly", 27), ("monthly", 26)):
            payload = self.request()
            payload.update(pay_frequency=frequency, pay_periods_per_year=count)
            self.assert_no_amounts(calculate_payroll(payload), "unsupported")
        for number in (0, 27):
            payload = self.request()
            payload["pay_period_number"] = number
            self.assert_no_amounts(calculate_payroll(payload), "invalid_input")
        for field in ("pay_period_number", "pay_periods_per_year"):
            payload = self.request()
            payload[field] = True
            self.assert_no_amounts(calculate_payroll(payload), "invalid_input")

    def test_outside_scope_cases_never_produce_partial_payroll_amounts(self):
        for field, value in (("province", "QC"), ("province", "BC"),
                             ("regular_salary_only", False), ("no_other_adjustments", False),
                             ("ontario_only_employment", False), ("cpp_status", "eligible_part_year"),
                             ("ei_status", "unknown")):
            payload = self.request()
            payload[field] = value
            self.assert_no_amounts(calculate_payroll(payload), "unsupported")

    def test_partial_fixture_history_and_other_employer_ytd_are_unsupported(self):
        for field, value in (("is_complete", False), ("employer_scope", "all_employers")):
            payload = self.request(5)
            payload["ytd"][field] = value
            self.assert_no_amounts(calculate_payroll(payload), "unsupported")

    def test_td1_claim_choices_are_explicit_and_dynamic_bpa_is_not_always_maximum(self):
        for field in ("federal_claim", "ontario_claim"):
            for claim in ({"mode": "amount"}, {"mode": "basic_personal", "amount": "16452.00"}, {"mode": "claim_code", "amount": "1.00"}):
                payload = self.request()
                payload[field] = claim
                self.assert_no_amounts(calculate_payroll(payload), "invalid_input")
        high = calculate_payroll(self.request(10))
        self.assertEqual(high["calculation_details"]["federal_claim_used"], "14829.00")
        phaseout = calculate_payroll(self.request(9))
        claim = Decimal(phaseout["calculation_details"]["federal_claim_used"])
        self.assertTrue(Decimal("14829") < claim < Decimal("16452"))

    def test_ytd_limits_and_inconsistent_records_are_rejected(self):
        for field, amount in (("cpp", "4230.46"), ("cpp2", "416.01"), ("ei", "1123.08")):
            payload = self.request(8)
            payload["ytd"][field] = amount
            self.assert_no_amounts(calculate_payroll(payload), "invalid_input")
        payload = self.request()
        payload["ytd"]["cpp"] = "1.00"
        self.assert_no_amounts(calculate_payroll(payload), "invalid_input")
        payload = self.request(5)
        payload["ytd"]["pensionable_earnings"] = "0.00"
        self.assert_no_amounts(calculate_payroll(payload), "invalid_input")
        payload = self.request(5)
        payload["ytd"]["cpp2"] = "1.00"
        self.assert_no_amounts(calculate_payroll(payload), "invalid_input")

    def test_exemption_status_cannot_conflict_with_prior_contributions(self):
        for field, value in (("cpp_status", "exempt_all_year"), ("ei_status", "exempt_all_year")):
            payload = self.request(5)
            payload[field] = value
            self.assert_no_amounts(calculate_payroll(payload), "invalid_input")

    def test_caps_remain_reached_for_later_periods_and_credits_do_not_disappear(self):
        result = calculate_payroll(self.request(8))
        self.assertEqual([result["amounts"][key] for key in ("cpp", "cpp2", "ei")], ["0.00"] * 3)
        self.assertEqual(Decimal(result["calculation_details"]["projected_annual_base_cpp_credit_amount"]), Decimal("3519.45"))
        self.assertEqual(Decimal(result["calculation_details"]["projected_annual_ei_credit_amount"]), Decimal("1123.07"))

    def test_contribution_half_cent_rounding_and_exemption_truncation(self):
        payload = self.request()
        # $50 per period gives EI=$0.815: an exact half-cent rounds up.
        payload["annual_salary"] = "1300.00"
        result = calculate_payroll(payload)
        self.assertEqual(result["amounts"]["ei"], "0.82")
        self.assertEqual(result["amounts"]["cpp"], "0.00")
        self.assertEqual(result["calculation_details"]["cpp_period_exemption"], "134.61")

    def test_ontario_health_premium_at_all_piecewise_boundaries(self):
        rules, _ = load_rules(RULES_PATH)
        expected = {"20000": "0", "20000.01": "0.0006", "25000": "300", "36000": "300", "38500": "450", "48000": "450", "48600": "600", "72000": "600", "72600": "750", "200000": "750", "200600": "900"}
        for income, premium in expected.items():
            self.assertEqual(health_premium(Decimal(income), rules["ontario"]), Decimal(premium))

    def test_federal_basic_personal_formula_at_phaseout_boundaries(self):
        rules, _ = load_rules(RULES_PATH)
        expected = {"181440": "16452", "219961": "15640.50", "258482": "14829", "1000000": "14829"}
        for income, claim in expected.items():
            self.assertEqual(federal_basic_personal(Decimal(income), rules["federal"]), Decimal(claim))

    def test_tax_brackets_use_published_constants_at_each_threshold(self):
        rules, _ = load_rules(RULES_PATH)
        # The published constants are rounded integers; they are not replaced
        # by constants derived from a mathematically continuous bracket table.
        for group, threshold, exact, above in (("federal", "58523", "8193.22", "8193.21705"),
                                               ("ontario", "53891", "2721.4955", "2721.027415")):
            value = Decimal(threshold)
            self.assertEqual(bracket_tax(value, rules[group]["brackets"]), Decimal(exact))
            self.assertEqual(bracket_tax(value + Decimal("0.01"), rules[group]["brackets"]), Decimal(above))

    def test_full_year_caps_and_payroll_balance_across_twenty_six_payments(self):
        payload = self.request()
        payload["annual_salary"] = "91000.00"
        totals = {key: Decimal("0") for key in ("cpp", "cpp2", "ei")}
        pensionable = Decimal("0")
        for number in range(1, 27):
            payload["pay_period_number"] = number
            payload["payment_date"] = (date(2026, 1, 9) + timedelta(days=14 * (number - 1))).isoformat()
            payload["ytd"].update({key: str(value) for key, value in totals.items()})
            payload["ytd"]["pensionable_earnings"] = str(pensionable)
            result = calculate_payroll(payload)
            self.assertEqual(result["status"], "calculated", result)
            for key in totals:
                totals[key] += Decimal(result["amounts"][key])
            pensionable += Decimal(result["amounts"]["gross_pay"])
        self.assertEqual(totals, {"cpp": Decimal("4230.45"), "cpp2": Decimal("416.00"), "ei": Decimal("1123.07")})

    def test_rules_hash_provenance_and_inputs_are_stable_and_unmodified(self):
        payload = self.request()
        original = deepcopy(payload)
        result = calculate_payroll(payload)
        self.assertEqual(payload, original)
        self.assertEqual(result, calculate_payroll(payload))
        self.assertEqual(result["provenance"]["rules_sha256"], hashlib.sha256(RULES_PATH.read_bytes()).hexdigest())
        self.assertTrue(result["is_estimate"])
        self.assertIn("broader PDOC validation pending", result["provenance"]["validation_status"])
        self.assertIn("not CRA-certified", result["provenance"]["validation_status"])
        self.assertEqual(result["provenance"]["ruleset_id"], "cra-2026-on-option1-regular-v2")
        self.assertEqual(result["provenance"]["validation_evidence"]["case_id"], "salary_50000_biweekly_first")

    def test_caller_decimal_precision_does_not_change_payroll_results(self):
        expected = calculate_payroll(self.request())["amounts"]
        with localcontext() as context:
            context.prec = 4
            self.assertEqual(calculate_payroll(self.request())["amounts"], expected)

    def test_missing_corrupt_and_unknown_rule_versions_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.json"
            self.assert_no_amounts(calculate_payroll(self.request(), rules_path=path), "unavailable")
            for content in ("not JSON", "{}", '{"schema_version":2}'):
                path.write_text(content, encoding="utf-8")
                self.assert_no_amounts(calculate_payroll(self.request(), rules_path=path), "unavailable")
            rules = json.loads(RULES_PATH.read_text())
            rules["ruleset_id"] = "cra-2026-on-option1-regular-v1"
            path.write_text(json.dumps(rules), encoding="utf-8")
            self.assert_no_amounts(calculate_payroll(self.request(), rules_path=path), "unavailable")
            rules = json.loads(RULES_PATH.read_text())
            rules["cpp"]["maximum"] = "4000"
            path.write_text(json.dumps(rules), encoding="utf-8")
            self.assert_no_amounts(calculate_payroll(self.request(), rules_path=path), "unavailable")

    def test_cli_accepts_json_stdin_and_returns_structured_failure(self):
        command = [sys.executable, str(ROOT / "scripts/payroll_calculator.py")]
        successful = subprocess.run(command, input=json.dumps(self.request()), text=True, capture_output=True, timeout=30)
        self.assertEqual(successful.returncode, 0, successful.stderr)
        self.assertEqual(json.loads(successful.stdout)["amounts"], REFERENCES["cases"][0]["expected_amounts"])
        failed = subprocess.run(command, input="not JSON", text=True, capture_output=True, timeout=30)
        self.assertEqual(failed.returncode, 2)
        self.assertEqual(json.loads(failed.stdout)["status"], "invalid_input")


if __name__ == "__main__":
    unittest.main()
