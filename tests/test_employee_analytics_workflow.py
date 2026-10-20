"""Read-only workforce questions against real local MCP, without model calls."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from init_employee_db import init_database
from payroll_mcp_client import PayrollMCPClient
from payroll_mcp_server import create_server
from employee_analytics_workflow import analytics_turn, should_route, SESSION_KEY


class AnalyticsWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_roster_followup_reuses_summary_date_and_stored_salary(self):
        summary = await self.turn("How many employee are there in the company as of 2026-10-19?")
        question = "give me the details of all the employee"
        self.assertTrue(should_route(question, summary["employee_session"]))
        result = await self.turn(question, summary["employee_session"])
        self.assertEqual(result["outcome"]["status"], "answer", result["outcome"]["text"])
        roster = result["outcome"]["roster"]
        self.assertEqual(roster["as_of_date"], "2026-10-19")
        self.assertEqual(roster["match_count"], 4)
        self.assertIn("Priya Shah", result["outcome"]["text"])
        self.assertIn("$75,000.00", result["outcome"]["text"])
        self.assertNotIn("Sam Reed", result["outcome"]["text"])
        self.assertEqual(result["outcome"]["sources"][0]["kind"], "employee_roster")

    async def test_roster_collects_date_and_supports_forward_backward_pages(self):
        first = await self.turn("List all employees")
        self.assertEqual(first["outcome"]["text"], "As of what date? Please use YYYY-MM-DD.")
        first["employee_session"][SESSION_KEY]["arguments"]["limit"] = 2
        page1 = await self.turn("2026-10-19", first["employee_session"])
        original = deepcopy(page1["employee_session"])
        page2 = await self.turn("Next employees", original)
        self.assertEqual(page2["outcome"]["roster"]["offset"], 2)
        self.assertIn("Jordan Lee", page2["outcome"]["text"])
        self.assertNotIn("Priya Shah", page2["outcome"]["text"])
        self.assertEqual(original[SESSION_KEY]["arguments"]["offset"], 0)
        back = await self.turn("Previous page", page2["employee_session"])
        self.assertEqual(back["outcome"]["roster"]["offset"], 0)
        end = await self.turn("Next page", page2["employee_session"])
        self.assertEqual(end["trace"], [])
        self.assertIn("last roster page", end["outcome"]["text"])
        no_context = await self.turn("Next employees")
        self.assertEqual(no_context["trace"], [])

    async def test_roster_resets_filters_but_does_not_ignore_new_unsupported_filters(self):
        summary = await self.turn("How many employees earn above $70,000 as of 2026-10-19?")
        all_people = await self.turn("List all employees", summary["employee_session"])
        self.assertEqual(all_people["outcome"]["roster"]["match_count"], 4)
        for question in ("List all monthly employees", "List all employees who left", "List all employees by department"):
            result = await self.turn(question, summary["employee_session"])
            self.assertEqual(result["outcome"]["status"], "clarification")
            self.assertEqual(result["trace"], [])
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "workforce.db"
        init_database(self.path)
        self.client = await self.enterAsyncContext(PayrollMCPClient(transport=create_server(self.path)))

    async def turn(self, question, session=None):
        return await analytics_turn(self.client, question, session or {})

    async def test_headcount_missing_date_and_followup_preserve_state_without_writes(self):
        before = self.path.read_bytes()
        first = await self.turn("How many employees are working in the company?")
        self.assertEqual(first["outcome"]["status"], "clarification")
        self.assertEqual(first["trace"], [])
        original = deepcopy(first["employee_session"])
        self.assertTrue(should_route("2026-11-01", original))
        second = await self.turn("2026-11-01", original)
        self.assertIn("**4 active employees**", second["outcome"]["text"])
        self.assertEqual(second["outcome"]["sources"][0]["kind"], "workforce_summary")
        self.assertEqual(original, first["employee_session"])
        self.assertEqual(before, self.path.read_bytes())

    async def test_frequency_salary_filters_and_boundaries(self):
        cases = [("How many employees are paid monthly as of 2026-11-01?", 2),
                 ("How many employees earn more than $70,000 as of 2026-11-01?", 1),
                 ("How many employees earn at least $60,000 as of 2026-11-01?", 3),
                 ("How many employees earn less than $60,000 as of 2026-11-01?", 1),
                 ("How many employees earn between $50,000 and $65,000 as of 2026-11-01?", 3),
                 ("How many employees are paid monthly and earn above $60k as of 2026-11-01?", 1)]
        for question, expected in cases:
            with self.subTest(question=question):
                result = await self.turn(question)
                self.assertEqual(result["outcome"]["status"], "answer", result["outcome"]["text"])
                self.assertIn(f"**{expected} active {'employee' if expected == 1 else 'employees'}**", result["outcome"]["text"])

    async def test_frequency_breakdown_and_explicit_date_corrections(self):
        first = await self.turn("How many employees are paid monthly versus biweekly as of 2026-11-01?")
        self.assertIn("Monthly: **2**", first["outcome"]["text"])
        self.assertIn("Biweekly: **2**", first["outcome"]["text"])
        second = await self.turn("How many employees are there as of 2026-10-15?", first["employee_session"])
        self.assertIn("**5 active employees**", second["outcome"]["text"])
        invalid = await self.turn("How many employees as of 2026-02-30?", second["employee_session"])
        self.assertNotIn(SESSION_KEY, invalid["employee_session"])
        self.assertEqual(invalid["trace"], [])

    async def test_salary_missing_threshold_then_date(self):
        first = await self.turn("How many employees earn above as of 2026-11-01?")
        self.assertEqual(first["outcome"]["status"], "clarification")
        second = await self.turn("$70,000", first["employee_session"])
        self.assertIn("**1 active employee**", second["outcome"]["text"])

    async def test_movements_collect_range_and_explain_first_inactive_date(self):
        first = await self.turn("How many employees joined or left?")
        self.assertEqual(first["outcome"]["status"], "clarification")
        second = await self.turn("from 2026-10-01 to 2026-11-01", first["employee_session"])
        self.assertEqual(second["outcome"]["status"], "answer", second["outcome"]["text"])
        self.assertIn("Employees with employment ending: **1**", second["outcome"]["text"])
        self.assertIn("first inactive date", second["outcome"]["text"])

    async def test_unsupported_unknown_and_or_filters_never_return_unfiltered_counts(self):
        for question in ("How many employees in Finance as of 2026-11-01?", "How many employees by department?",
                         "How many employees earn above $50k and below $80k as of 2026-11-01?",
                         "How many employees are paid monthly or biweekly?", "How many employees as of today?",
                         "How many employees earn above $0.001 as of 2026-11-01?",
                         "How many employees != monthly as of 2026-11-01?",
                         "How many employees earn above $50k and above $70k as of 2026-11-01?",
                         "How many weekly employees joined from 2026-01-01 to 2026-11-01?"):
            with self.subTest(question=question):
                result = await self.turn(question)
                self.assertNotEqual(result["outcome"]["status"], "answer")
                self.assertEqual(result["trace"], [])

    async def test_future_snapshot_and_disconnected_database(self):
        future = await self.turn("How many employees as of 2027-01-01?")
        self.assertEqual(future["outcome"]["status"], "unsupported")
        self.assertIn("snapshot", future["outcome"]["text"])
        unavailable = await analytics_turn(None, "How many employees as of 2026-11-01?", {})
        self.assertEqual(unavailable["outcome"]["status"], "unsupported")

    async def test_unrelated_policy_and_named_employee_questions_do_not_route(self):
        for question in ("What is T4?", "Find Alex as of 2026-11-01", "How many vacation days does an employee get?"):
            self.assertFalse(should_route(question, {}))


if __name__ == "__main__":
    unittest.main()
