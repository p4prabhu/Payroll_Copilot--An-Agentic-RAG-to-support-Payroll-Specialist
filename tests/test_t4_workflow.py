"""Annual sample conversation checks; local records only, no model API."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from init_employee_db import FIXTURE_PATH, init_database
from employee_repository import EmployeeRepository
from t4_workflow import explicit_request, should_route, target_query, t4_turn


class LocalClient:
    available = True

    def __init__(self, records, annual):
        self.records, self.annual = records, annual
        self.calls = []

    async def call(self, name, args):
        self.calls.append((name, args))
        return getattr(self.annual if name == "get_annual_payroll_summary" else self.records, name)(**args)


class T4WorkflowTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.partial = Path(cls.directory.name) / "partial.db"
        cls.annual = Path(cls.directory.name) / "annual.db"
        init_database(cls.partial)
        init_database(cls.annual, fixture_path=FIXTURE_PATH.with_name("fixtures_t4_2026.json"))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.client = LocalClient(EmployeeRepository(self.partial), EmployeeRepository(self.annual))

    async def turn(self, text, session=None):
        return await t4_turn(self.client, text, session or {})

    def test_exact_question_and_supported_name_forms(self):
        for text in ("Show Alex Morgan’s sample T4 breakdown for 2026", "Give a T4 breakdown of Alex Morgan for 2026",
                     "Generate T4 for Alex Morgan for 2026", "Alex Morgan T4 for 2026"):
            self.assertTrue(explicit_request(text), text)
            self.assertEqual(target_query(text), "Alex Morgan", text)
        for text in ("What is T4?", "What information would be needed to prepare Alex Morgan's T4 for 2026?",
                     "Which T4 boxes report income?", "Explain Alex Morgan's T4 boxes"):
            self.assertFalse(should_route(text, {}), text)

    async def test_exact_question_returns_verified_box_table_and_source_without_writes(self):
        before = self.annual.read_bytes(), self.partial.read_bytes()
        result = await self.turn("Show Alex Morgan’s sample T4 breakdown for 2026")
        outcome = result["outcome"]
        self.assertEqual(outcome["status"], "answer")
        self.assertIn("| 14 | Employment income | CAD $50,000.08 |", outcome["text"])
        self.assertIn("CAD $6,273.54", outcome["text"])
        self.assertIn("not an actual T4", outcome["text"])
        self.assertFalse(outcome["t4_summary"]["ready_for_filing"])
        self.assertEqual(outcome["sources"][0]["kind"], "annual_payroll_sample")
        self.assertEqual([name for name, _ in self.client.calls], ["find_employees", "get_annual_payroll_summary"])
        self.assertEqual(before, (self.annual.read_bytes(), self.partial.read_bytes()))

    async def test_missing_year_followup_keeps_identity_without_payment_date(self):
        first = await self.turn("Give a T4 breakdown of Alex Morgan")
        self.assertEqual(first["outcome"]["status"], "clarification")
        self.assertIn("Which T4 reporting year", first["outcome"]["text"])
        self.assertTrue(should_route("2026", first["employee_session"]))
        second = await self.turn("2026", first["employee_session"])
        self.assertEqual(second["outcome"]["status"], "answer")
        self.assertNotIn("lookup_date", second["employee_session"])

    async def test_ambiguous_name_requires_selection_and_retains_year(self):
        first = await self.turn("Generate T4 for Alex for 2026")
        self.assertEqual(first["outcome"]["status"], "clarification")
        self.assertIn("EMP-1001", first["outcome"]["text"])
        self.assertIn("EMP-1004", first["outcome"]["text"])
        self.assertEqual(len(self.client.calls), 1)
        second = await self.turn("EMP-1004", first["employee_session"])
        self.assertEqual(second["outcome"]["t4_summary"]["employee"]["given_name"], "Alex")
        self.assertEqual(second["outcome"]["t4_summary"]["employee"]["family_name"], "Chen")

    async def test_employee_prompt_and_unknown_name_do_not_choose_first_record(self):
        first = await self.turn("Generate a sample T4 for 2026")
        self.assertIn("Which employee", first["outcome"]["text"])
        second = await self.turn("Sam Reed", first["employee_session"])
        self.assertEqual(second["outcome"]["t4_summary"]["totals"]["payment_count"], 21)
        unknown = await self.turn("Generate T4 for Nobody Known for 2026")
        self.assertIn("No employee matched", unknown["outcome"]["text"])

    async def test_missing_annual_database_and_unsupported_year_produce_no_amounts(self):
        result = await self.turn("Show Alex Morgan's sample T4 breakdown for 2025")
        self.assertEqual(result["outcome"]["status"], "unsupported")
        self.assertNotIn("t4_summary", result["outcome"])
        self.client.annual = EmployeeRepository(self.partial)
        partial = await self.turn("Show Alex Morgan's sample T4 breakdown for 2026")
        self.assertEqual(partial["outcome"]["status"], "unsupported")
        self.assertNotIn("$", partial["outcome"]["text"])

    async def test_conflicting_years_and_multiple_employees_need_clarification(self):
        for text in ("Generate T4 for Alex Morgan for 2025 and 2026", "Generate T4 for emp-001 and emp-004 for 2026",
                     "Generate T4 for Alex Morgan for 2026-10-20"):
            result = await self.turn(text)
            self.assertEqual(result["outcome"]["status"], "clarification")
            self.assertNotIn("t4_summary", result["outcome"])

    async def test_mismatched_annual_identity_is_rejected(self):
        original = self.client.call
        async def wrong_identity(name, args):
            result = await original(name, args)
            if name == "get_annual_payroll_summary":
                result["employee"]["family_name"] = "Other"
            return result
        self.client.call = wrong_identity
        result = await self.turn("Generate T4 for Alex Morgan for 2026")
        self.assertEqual(result["outcome"]["status"], "unsupported")
        self.assertNotIn("$", result["outcome"]["text"])

    async def test_context_is_copied_and_pending_paycheck_confirmation_cleared(self):
        session = {"_calculation": {"awaiting_confirmation": True}, "_analytics": {"kind": "roster"}}
        original = deepcopy(session)
        result = await self.turn("Generate T4 for Alex Morgan for 2026", session)
        self.assertEqual(original, session)
        self.assertNotIn("_calculation", result["employee_session"])
        self.assertNotIn("_analytics", result["employee_session"])
        followup = await self.turn("Show their T4 breakdown for 2026", result["employee_session"])
        self.assertEqual(followup["outcome"]["status"], "answer")
        self.assertEqual(followup["outcome"]["t4_summary"]["employee"]["family_name"], "Morgan")

    async def test_unavailable_mcp_does_not_fall_back_to_invented_amounts(self):
        self.client.available = False
        result = await self.turn("Generate T4 for Alex Morgan for 2026")
        self.assertEqual(result["outcome"]["status"], "unsupported")
        self.assertEqual(self.client.calls, [])


if __name__ == "__main__":
    unittest.main()
