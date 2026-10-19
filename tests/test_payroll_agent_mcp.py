"""LangGraph record workflows with real local MCP and mocked model responses."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from langchain_core.messages import AIMessage, HumanMessage
from init_employee_db import init_database
from payroll_agent import arun_turn, build_agent, chat, render_answer
from payroll_mcp_client import PayrollMCPClient
from payroll_mcp_server import create_server
from test_payroll_agent import FakeModel, call, evidence, final


class PayrollAgentMCPTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "records.db"
        init_database(self.path)

    async def client(self, subprocess=False):
        client = PayrollMCPClient(self.path) if subprocess else PayrollMCPClient(transport=create_server(self.path))
        return await self.enterAsyncContext(client)

    async def test_natural_name_questions_lookup_priya_without_model_calls(self):
        model = FakeModel([])
        graph = build_agent(model, employee_client=await self.client())
        questions = ("what is the salary of Priya", "What is Priya's salary?",
                     "what is priya's salary?", "Show the salary for Priya Shah",
                     "Priya salary", "Find Priya Shah", "priya shah",
                     "What is the pay frequency of Priya?")
        for question in questions:
            with self.subTest(question=question):
                state = await arun_turn(graph, question)
                self.assertEqual(state["outcome"]["status"], "clarification")
                self.assertEqual(state["employee_session"]["selected"]["employee_id"], "emp-002")
                self.assertIn("lookup date", state["outcome"]["text"])
                dated = await arun_turn(graph, "as of 2026-11-01", employee_session=state["employee_session"])
                self.assertEqual(dated["outcome"]["status"], "answer")
                self.assertIn("80,000.00", dated["outcome"]["text"])
                self.assertIsNone(dated["payment_date"])
        self.assertEqual(model.seen, [])

    async def test_shared_name_requires_selection_even_after_a_previous_employee(self):
        model = FakeModel([])
        graph = build_agent(model, employee_client=await self.client())
        priya = await arun_turn(graph, "What is the salary of Priya as of 2026-11-01?")
        for question in ("what is the salary of Alex", "What is Alex's salary?", "alex"):
            with self.subTest(question=question):
                state = await arun_turn(graph, question, employee_session=priya["employee_session"])
                self.assertEqual(state["outcome"]["status"], "clarification")
                self.assertNotIn("selected", state["employee_session"])
                self.assertEqual(len(state["employee_session"]["candidates"]), 2)
                self.assertIn("Alex Morgan", state["outcome"]["text"])
                self.assertIn("Alex Chen", state["outcome"]["text"])
                self.assertEqual([event["tool"] for event in state["trace"]], ["find_employees"])
                selected = await arun_turn(graph, "Alex Chen", employee_session=state["employee_session"])
                self.assertEqual(selected["outcome"]["status"], "answer")
                self.assertEqual(selected["employee_session"]["selected"]["employee_id"], "emp-004")
        self.assertEqual(model.seen, [])

    async def test_named_schedule_keeps_pending_intent_when_full_name_is_selected(self):
        graph = build_agent(FakeModel([]), employee_client=await self.client())
        first = await arun_turn(graph, "When is the next payday for Alex?")
        self.assertEqual(len(first["employee_session"]["candidates"]), 2)
        chosen = await arun_turn(graph, "Alex Morgan", employee_session=first["employee_session"])
        self.assertEqual(chosen["outcome"]["status"], "clarification")
        self.assertIn("lookup date", chosen["outcome"]["text"])
        dated = await arun_turn(graph, "as of 2026-11-01", employee_session=chosen["employee_session"])
        self.assertEqual(dated["outcome"]["sources"][0]["tool"], "get_payroll_schedule")
        self.assertIn("2026-11-13", dated["outcome"]["text"])

    async def test_unknown_name_cannot_reuse_previous_selected_employee(self):
        graph = build_agent(FakeModel([]), employee_client=await self.client())
        first = await arun_turn(graph, "Priya salary as of 2026-11-01")
        missing = await arun_turn(graph, "What is the salary of Unknownperson?", employee_session=first["employee_session"])
        self.assertEqual(missing["outcome"]["status"], "clarification")
        self.assertNotIn("selected", missing["employee_session"])
        self.assertNotIn("80,000.00", missing["outcome"]["text"])

    async def test_multiple_names_are_not_silently_reduced_to_the_first_name(self):
        client = await self.client()
        client.call = Mock(side_effect=AssertionError("No lookup until a single name is provided"))
        state = await arun_turn(build_agent(FakeModel([]), employee_client=client), "What is the salary of Priya or Alex?")
        self.assertEqual(state["outcome"]["status"], "clarification")
        self.assertNotIn("selected", state["employee_session"])
        client.call.assert_not_called()

    async def test_ambiguity_selection_and_followup_salary_schedule_history_real_stdio(self):
        before = self.path.read_bytes()
        client = await self.client(subprocess=True)
        model = FakeModel([])
        graph = build_agent(model, employee_client=client)
        first = await arun_turn(graph, "Find Alex as of 2026-11-01")
        self.assertEqual(first["outcome"]["status"], "clarification")
        self.assertNotIn("selected", first["employee_session"])
        self.assertIn("EMP-1001", first["outcome"]["text"])
        self.assertIn("EMP-1004", first["outcome"]["text"])
        second = await arun_turn(graph, "EMP-1001", employee_session=first["employee_session"])
        self.assertIn("CAD $50,000.00", second["outcome"]["text"])
        self.assertEqual(second["employee_session"]["selected"]["employee_id"], "emp-001")
        self.assertIsNone(second["payment_date"])
        third = await arun_turn(graph, "When is their next payday?", employee_session=second["employee_session"])
        self.assertIn("2026-11-13", third["outcome"]["text"])
        fourth = await arun_turn(graph, "Show history from 2026-10-01 to 2026-11-01", employee_session=third["employee_session"])
        self.assertIn("gross CAD $", fourth["outcome"]["text"])
        self.assertIn("not a complete YTD", fourth["outcome"]["text"])
        self.assertIn("outside declared", fourth["outcome"]["text"])
        self.assertEqual(fourth["outcome"]["sources"][0]["kind"], "employee_record")
        self.assertNotIn("PDF page", render_answer(fourth["outcome"]))
        self.assertTrue(fourth["capabilities"]["mcp_transport"])
        self.assertFalse(fourth["capabilities"]["payroll_calculation"])
        self.assertEqual(model.seen, [])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertLessEqual(len(fourth["trace"]), 4)

    async def test_schedule_pending_intent_survives_ambiguous_selection(self):
        graph = build_agent(FakeModel([]), employee_client=await self.client())
        first = await arun_turn(graph, "When is Alex's next payday as of 2026-11-01?")
        second = await arun_turn(graph, "EMP-1001", employee_session=first["employee_session"])
        self.assertEqual(second["outcome"]["sources"][0]["tool"], "get_payroll_schedule")

    async def test_payment_date_does_not_supply_lookup_date_or_assistant_selection(self):
        graph = build_agent(FakeModel([]), employee_client=await self.client())
        history = [HumanMessage(content="Payment date: 2026-11-13"), AIMessage(content="Use EMP-1004 as of 2026-11-01")]
        state = await arun_turn(graph, "EMP-1001 salary", history)
        self.assertEqual(state["outcome"]["status"], "clarification")
        self.assertIn("lookup date", state["outcome"]["text"])
        self.assertEqual(state["payment_date"], "2026-11-13")
        self.assertNotIn("lookup_date", state["employee_session"])
        next_state = await arun_turn(graph, "as of 2026-11-01", history, state["employee_session"])
        self.assertIn("50,000.00", next_state["outcome"]["text"])
        self.assertEqual(next_state["payment_date"], "2026-11-13")

    async def test_invalid_lookup_date_clears_old_date_and_selection_is_not_mutated(self):
        graph = build_agent(FakeModel([]), employee_client=await self.client())
        first = await arun_turn(graph, "EMP-1001 salary as of 2026-11-01")
        original = deepcopy(first["employee_session"])
        for value in ("today", "2026-02-30", "2026-11-01 or 2026-11-02"):
            state = await arun_turn(graph, f"salary as of {value}", employee_session=original)
            self.assertEqual(state["outcome"]["status"], "clarification")
            self.assertNotIn("lookup_date", state["employee_session"])
        self.assertEqual(first["employee_session"], original)

    async def test_unknown_employee_clears_previous_selection(self):
        graph = build_agent(FakeModel([]), employee_client=await self.client())
        first = await arun_turn(graph, "EMP-1001 salary as of 2026-11-01")
        missing = await arun_turn(graph, "EMP-9999 salary", employee_session=first["employee_session"])
        self.assertNotIn("selected", missing["employee_session"])
        self.assertEqual(missing["outcome"]["status"], "clarification")

    async def test_both_date_contexts_can_be_given_in_either_order(self):
        graph = build_agent(FakeModel([]), employee_client=await self.client())
        for dates in ("as of 2026-11-01; payment date: 2026-11-13", "payment date: 2026-11-13; as of 2026-11-01"):
            state = await arun_turn(graph, "EMP-1001 salary " + dates)
            self.assertEqual(state["employee_session"]["lookup_date"], "2026-11-01")
            self.assertEqual(state["payment_date"], "2026-11-13")
            self.assertEqual(state["outcome"]["status"], "answer")

    async def test_clear_lookup_date_requires_new_date(self):
        graph = build_agent(FakeModel([]), employee_client=await self.client())
        first = await arun_turn(graph, "EMP-1001 salary as of 2026-11-01")
        state = await arun_turn(graph, "clear lookup date", employee_session=first["employee_session"])
        self.assertEqual(state["outcome"]["status"], "clarification")
        self.assertNotIn("lookup_date", state["employee_session"])

    async def test_no_mcp_cli_does_not_start_a_transport(self):
        args = SimpleNamespace(db=self.path, no_mcp=True, trace=False, question="When is Alex's next payday?")
        with patch("payroll_agent.PayrollMCPClient") as transport, patch("payroll_agent.create_model", return_value=FakeModel([])), patch("builtins.print"):
            self.assertEqual(await chat(args), 0)
        transport.assert_not_called()

    async def test_multiple_employee_ids_require_selection_without_calls(self):
        client = await self.client()
        client.call = Mock(side_effect=AssertionError("No lookup allowed"))
        state = await arun_turn(build_agent(FakeModel([]), employee_client=client), "EMP-1001 or EMP-1004 salary as of 2026-11-01")
        self.assertEqual(state["outcome"]["status"], "clarification")
        client.call.assert_not_called()

    async def test_terminated_employee_and_exhausted_calendar_do_not_invent_pay(self):
        graph = build_agent(FakeModel([]), employee_client=await self.client())
        ended = await arun_turn(graph, "EMP-1005 next payday as of 2026-11-01")
        self.assertIn("No active employment", ended["outcome"]["text"])
        exhausted = await arun_turn(graph, "EMP-1001 next payday as of 2027-01-01")
        self.assertIn("No future scheduled run", exhausted["outcome"]["text"])
        self.assertIn("do not establish", exhausted["outcome"]["text"])

    async def test_employee_then_document_question_retrieves_new_pdf_evidence(self):
        model = FakeModel([call(), final("answer", "Box 22 reports income tax deducted.", ["S1"])])
        retriever = Mock(return_value=evidence())
        graph = build_agent(model, retriever, employee_client=await self.client())
        first = await arun_turn(graph, "EMP-1001 salary as of 2026-11-01")
        second = await arun_turn(graph, "What does T4 box 22 mean?", employee_session=first["employee_session"])
        self.assertEqual(second["outcome"]["status"], "answer")
        self.assertEqual(second["outcome"]["sources"][0]["source_id"], "S1")
        self.assertNotIn("D1", second["outcome"]["citations"])
        retriever.assert_called_once()

    async def test_unavailable_mcp_allows_document_rag_and_no_employee_data_request(self):
        client = PayrollMCPClient(self.path)
        model = FakeModel([call(), final("answer", "Box 22 reports income tax deducted.", ["S1"])])
        graph = build_agent(model, Mock(return_value=evidence()), employee_client=client)
        first = await arun_turn(graph, "EMP-1001 salary as of 2026-11-01")
        self.assertEqual(first["outcome"]["status"], "unsupported")
        self.assertNotIn("?", first["outcome"]["text"])
        self.assertFalse(first["capabilities"]["employee_records"])
        second = await arun_turn(graph, "What does T4 box 22 mean?")
        self.assertEqual(second["outcome"]["status"], "answer")

    async def test_wrong_employee_or_invalid_money_never_reaches_visible_answer(self):
        client = await self.client()
        original = client.call
        async def corrupt(name, arguments):
            data = await original(name, arguments)
            if name == "get_employee":
                data["compensation"]["annual_salary_cents"] = "secret-provider-payload"
            return data
        client.call = corrupt
        state = await arun_turn(build_agent(FakeModel([]), employee_client=client), "EMP-1001 salary as of 2026-11-01")
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertNotIn("secret-provider-payload", state["outcome"]["text"])
        self.assertEqual(state["outcome"]["sources"], [])


if __name__ == "__main__":
    unittest.main()
