"""Offline calculation intake, confirmation and rule-provenance checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from langchain_core.messages import AIMessage
from payroll_calculation_workflow import calculation_turn, SESSION_KEY, EXTRACT_TOOL, flatten, should_route
from payroll_calculator import calculate_payroll
from test_payroll_agent import FakeModel

REFERENCE = json.loads((Path(__file__).parent / "fixtures/payroll_reference_cases.json").read_text(encoding="utf-8"))


def extraction(updates):
    return AIMessage(content="", tool_calls=[{"name": EXTRACT_TOOL, "id": "inputs", "type": "tool_call",
                                             "args": {"updates": updates}}])


class CalculationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.payload = deepcopy(REFERENCE["cases"][0]["input"])
        self.no_model = Mock()
        self.no_model.bind_tools.side_effect = AssertionError("JSON and confirmation must be offline")

    def prepare(self, payload=None):
        return calculation_turn(self.no_model, "Calculate payroll: " + json.dumps(payload or self.payload), {})

    def test_three_pdoc_cases_require_confirmation_and_match_all_amounts(self):
        observed = {item["case_id"] for item in REFERENCE["pdoc_observations"]}
        for case in REFERENCE["cases"]:
            if case["id"] not in observed:
                continue
            with self.subTest(case=case["id"]):
                with patch("payroll_calculation_workflow.calculate_payroll", wraps=calculate_payroll) as calculator:
                    ready = self.prepare(case["input"])
                    self.assertEqual(ready["outcome"]["status"], "clarification")
                    self.assertIn("Confirm calculation", ready["outcome"]["text"])
                    calculator.assert_not_called()
                    result = calculation_turn(self.no_model, "Confirm calculation", ready["employee_session"])
                    calculator.assert_called_once()
                self.assertEqual(result["outcome"]["calculation"]["amounts"], case["expected_amounts"])
                self.assertEqual(result["outcome"]["sources"][0]["kind"], "calculation")
                self.assertIn("not CRA-certified", result["outcome"]["text"])
                self.assertEqual(result["model_calls"], 0)
        self.no_model.bind_tools.assert_not_called()

    def test_missing_fields_never_default_zero_claims_or_eligibility(self):
        result = self.prepare({"annual_salary": "50000.00"})
        draft = result["employee_session"][SESSION_KEY]
        self.assertEqual(draft["values"], {"annual_salary": "50000.00"})
        self.assertNotIn("awaiting_confirmation", draft)
        for text in ("Payment date", "TD1", "CPP status", "complete", "zero must be explicitly"):
            self.assertIn(text, result["outcome"]["text"])

    def test_natural_language_updates_are_quoted_and_structured_once(self):
        question = "Calculate deductions: annual salary $50,000.00."
        model = FakeModel([extraction([{"field": "annual_salary", "value": "50000.00", "evidence": "$50,000.00"}])])
        result = calculation_turn(model, question, {})
        self.assertEqual(result["employee_session"][SESSION_KEY]["values"]["annual_salary"], "50000.00")
        self.assertEqual(len(model.seen), 1)
        self.assertEqual(result["model_calls"], 1)
        self.assertNotIn("calculation", result["outcome"])

    def test_complete_natural_language_intake_then_confirmation_uses_only_python(self):
        question = ("Calculate deductions for annual salary $50,000.00, biweekly, 26 periods, period 1, "
                    "payment date 2026-01-09, province Ontario. Basic personal federal and Ontario TD1. "
                    "CPP eligible all year, EI insurable. Complete current-employer YTD before this payment: "
                    "pensionable earnings zero, CPP zero, CPP2 zero, EI zero. Regular salary only, "
                    "no other adjustments, Ontario-only employment.")
        updates = [{"field": name, "value": str(value).lower() if isinstance(value, bool) else str(value),
                    "evidence": question} for name, value in flatten(self.payload).items()]
        model = FakeModel([extraction(updates)])
        ready = calculation_turn(model, question, {})
        self.assertTrue(ready["employee_session"][SESSION_KEY]["awaiting_confirmation"])
        calculated = calculation_turn(model, "Confirm calculation", ready["employee_session"])
        self.assertEqual(calculated["outcome"]["calculation"]["amounts"]["net_pay"], "1544.03")
        self.assertEqual(len(model.seen), 1)

    def test_multiple_calls_or_extra_extraction_fields_never_execute(self):
        invalid = extraction([])
        invalid.tool_calls.append(deepcopy(invalid.tool_calls[0]))
        extra = extraction([])
        extra.tool_calls[0]["args"]["net_pay"] = "1544.03"
        for response in (invalid, extra):
            result = calculation_turn(FakeModel([response]), "Calculate net pay", {})
            self.assertEqual(result["trace"][-1]["status"], "invalid_extraction")
            self.assertNotIn("calculation", result["outcome"])

    def test_unknown_json_objects_and_tampered_confirmation_are_rejected(self):
        payload = deepcopy(self.payload)
        payload["unknown"] = {}
        rejected = self.prepare(payload)
        self.assertNotIn("calculation", rejected["outcome"])
        self.assertNotIn("awaiting_confirmation", rejected["employee_session"][SESSION_KEY])
        ready = self.prepare()
        ready["employee_session"][SESSION_KEY]["values"]["annual_salary"] = "60000.00"
        with patch("payroll_calculation_workflow.calculate_payroll") as calculator:
            result = calculation_turn(self.no_model, "Confirm calculation", ready["employee_session"])
        calculator.assert_not_called()
        self.assertNotIn("calculation", result["outcome"])

    def test_unquoted_or_derived_money_and_unknown_fields_are_rejected(self):
        for update in ({"field": "annual_salary", "value": "60000", "evidence": "$50,000"},
                       {"field": "annual_salary", "value": "50000", "evidence": "not in message"},
                       {"field": "net_pay", "value": "1544.03", "evidence": "$50,000"}):
            result = calculation_turn(FakeModel([extraction([update])]), "Calculate salary $50,000", {})
            self.assertNotIn("calculation", result["outcome"])
            self.assertEqual(result["trace"][-1]["status"], "invalid_extraction")
            self.assertEqual(result["employee_session"][SESSION_KEY]["values"], {})

    def test_correction_requires_new_confirmation_and_preserves_other_inputs(self):
        ready = self.prepare()
        original = deepcopy(ready["employee_session"])
        model = FakeModel([extraction([{"field": "annual_salary", "value": "60000.00", "evidence": "$60,000"}])])
        with patch("payroll_calculation_workflow.calculate_payroll", wraps=calculate_payroll) as calculator:
            corrected = calculation_turn(model, "Change annual salary to $60,000", original)
            calculator.assert_not_called()
            self.assertIn("60000.00", corrected["outcome"]["text"])
            self.assertEqual(original[SESSION_KEY]["payload"]["annual_salary"], "50000.00")
            result = calculation_turn(self.no_model, "Confirm calculation", corrected["employee_session"])
        self.assertEqual(result["outcome"]["calculation"]["inputs"]["annual_salary"], "60000.00")

    def test_bad_correction_invalidates_previous_confirmation(self):
        ready = self.prepare()
        broken = calculation_turn(FakeModel([AIMessage(content="secret-api-key")]), "Change the salary", ready["employee_session"])
        final = calculation_turn(self.no_model, "Confirm calculation", broken["employee_session"])
        self.assertNotIn("calculation", final["outcome"])
        self.assertNotIn("secret-api-key", broken["outcome"]["text"])

    def test_rules_change_requires_new_summary(self):
        ready = self.prepare()
        with patch("payroll_calculation_workflow.load_rules", return_value=({}, "changed")), patch("payroll_calculation_workflow.validate_scope", return_value=None), patch("payroll_calculation_workflow.calculate_payroll") as calculator:
            result = calculation_turn(self.no_model, "Confirm calculation", ready["employee_session"])
        calculator.assert_not_called()
        self.assertIn("rules changed", result["outcome"]["text"])

    def test_rules_missing_never_produces_amounts(self):
        with patch("payroll_calculation_workflow.load_rules", side_effect=OSError):
            result = self.prepare()
        self.assertEqual(result["outcome"]["status"], "unsupported")
        self.assertNotIn("calculation", result["outcome"])

    def test_unsupported_scope_partial_ytd_and_invalid_inputs(self):
        changes = [("province", "BC"), ("payment_date", "2027-01-01"), ("pay_frequency", "weekly"),
                   ("regular_salary_only", False), ("no_other_adjustments", False), ("cpp_status", "partial_year")]
        for field, value in changes:
            payload = deepcopy(self.payload)
            payload[field] = value
            result = self.prepare(payload)
            self.assertEqual(result["outcome"]["status"], "unsupported", field)
            self.assertNotIn("calculation", result["outcome"])
        payload = deepcopy(self.payload)
        payload["ytd"]["is_complete"] = False
        result = self.prepare(payload)
        self.assertIn("partial fixture history cannot be used", result["outcome"]["text"])
        for invalid in (True, 50000.25, "NaN", "-1"):
            payload = deepcopy(self.payload)
            payload["annual_salary"] = invalid
            result = self.prepare(payload)
            self.assertNotIn("awaiting_confirmation", result["employee_session"][SESSION_KEY])

    def test_cancel_new_reset_and_repeat_confirmation_never_reuse_result(self):
        ready = self.prepare()
        completed = calculation_turn(self.no_model, "Confirm calculation", ready["employee_session"])
        for session in ({}, completed["employee_session"],
                        calculation_turn(self.no_model, "Cancel calculation", ready["employee_session"])["employee_session"],
                        calculation_turn(self.no_model, "New calculation", ready["employee_session"])["employee_session"]):
            with patch("payroll_calculation_workflow.calculate_payroll") as calculator:
                result = calculation_turn(self.no_model, "Confirm calculation", session)
            calculator.assert_not_called()
            self.assertNotIn("calculation", result["outcome"])

    def test_policy_and_employee_questions_can_interrupt_intake(self):
        ready = self.prepare()["employee_session"]
        for question in ("What is CPP?", "Explain how deductions are calculated", "Find Alex", "Show their history", "Salary of Priya"):
            self.assertFalse(should_route(question, ready), question)
        self.assertTrue(should_route("YTD CPP is $0.00", ready))

    def test_basic_claim_switch_removes_old_explicit_amount(self):
        payload = deepcopy(self.payload)
        payload["federal_claim"] = {"mode": "amount", "amount": "15000.00"}
        ready = self.prepare(payload)
        model = FakeModel([extraction([{"field": "federal_claim.mode", "value": "basic_personal", "evidence": "basic federal TD1"}])])
        updated = calculation_turn(model, "Use basic federal TD1", ready["employee_session"])
        self.assertEqual(updated["employee_session"][SESSION_KEY]["payload"]["federal_claim"], {"mode": "basic_personal"})


if __name__ == "__main__":
    unittest.main()
