"""UI async bridge checks using local MCP and mocked models; no API calls."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from init_employee_db import init_database
from payroll_mcp_client import PayrollMCPClient
from payroll_mcp_server import create_server
from payroll_ui_runtime import run_ui_turn, ui_search
from test_payroll_agent import FakeModel, call, evidence, final


class UIRuntimeTests(unittest.TestCase):
    def test_t4_uses_separate_annual_database_through_real_stdio_without_model_calls(self):
        from init_employee_db import FIXTURE_PATH
        annual = self.path.with_name("annual.db")
        init_database(annual, fixture_path=FIXTURE_PATH.with_name("fixtures_t4_2026.json"))
        before = self.path.read_bytes(), annual.read_bytes()
        result = self.turn("Show Alex Morgan’s sample T4 breakdown for 2026", annual_db_path=annual)
        self.assertEqual(result["outcome"]["status"], "answer")
        self.assertEqual(result["outcome"]["t4_summary"]["box_candidates_cents"]["14"], 5000008)
        self.assertFalse(result["outcome"]["t4_summary"]["ready_for_filing"])
        self.assertEqual(before, (self.path.read_bytes(), annual.read_bytes()))
        self.no_model.assert_not_called()

    def test_calculation_context_survives_turns_and_reset_requires_fresh_inputs(self):
        import json
        payload = json.loads((Path(__file__).parent / "fixtures/payroll_reference_cases.json").read_text(encoding="utf-8"))["cases"][0]["input"]
        before = self.path.read_bytes()
        ready = self.turn("Calculate payroll: " + json.dumps(payload))
        original = deepcopy(ready["employee_session"])
        completed = self.turn("Confirm calculation", original)
        self.assertEqual(completed["outcome"]["calculation"]["amounts"]["net_pay"], "1544.03")
        self.assertTrue(original["_calculation"]["awaiting_confirmation"])
        reset = self.turn("Confirm calculation")
        self.assertNotIn("calculation", reset["outcome"])
        self.assertEqual(before, self.path.read_bytes())
        self.no_model.assert_not_called()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "ui records.db"
        init_database(self.path)
        self.no_model = Mock(side_effect=AssertionError("Employee turn must not initialize the model"))

    def turn(self, question, session=None, **kwargs):
        return run_ui_turn(question, [], session or {}, db_path=self.path, model_factory=self.no_model, **kwargs)

    def test_consecutive_real_stdio_turns_retain_selection_without_model_calls(self):
        before = self.path.read_bytes()
        first = self.turn("Find Alex as of 2026-11-01")
        self.assertEqual(first["outcome"]["status"], "clarification")
        second = self.turn("EMP-1001", first["employee_session"])
        self.assertIn("50,000.00", second["outcome"]["text"])
        third = self.turn("When is their next payday?", second["employee_session"])
        self.assertIn("2026-11-13", third["outcome"]["text"])
        self.no_model.assert_not_called()
        self.assertEqual(before, self.path.read_bytes())

    def test_client_closes_in_same_task_and_loop_on_success_and_failure(self):
        owners, closed = [], []

        class OwnedClient:
            async def __aenter__(inner):
                owners.append((asyncio.current_task(), asyncio.get_running_loop()))
                inner.client = PayrollMCPClient(transport=create_server(self.path))
                return await inner.client.__aenter__()

            async def __aexit__(inner, *args):
                self.assertEqual(owners[-1], (asyncio.current_task(), asyncio.get_running_loop()))
                await inner.client.__aexit__(*args)
                closed.append(not inner.client.available)

        self.turn("EMP-1001 salary as of 2026-11-01", client_factory=lambda _: OwnedClient())
        with self.assertRaises(ValueError):
            self.turn("", client_factory=lambda _: OwnedClient())
        self.assertEqual(closed, [True, True])

    def test_document_route_uses_mock_model_and_current_turn_evidence(self):
        model = FakeModel([call(), final("answer", "Box 22 reports income tax deducted.", ["S1"])])
        factory, retriever = Mock(return_value=model), Mock(return_value=evidence())
        state = run_ui_turn("What does T4 box 22 mean?", [], {}, db_path=self.path,
                            model_factory=factory, retriever=retriever)
        self.assertEqual(state["outcome"]["status"], "answer")
        self.assertEqual(state["outcome"]["sources"][0]["source_id"], "S1")
        factory.assert_called_once()
        retriever.assert_called_once()

    def test_missing_database_falls_back_to_document_model(self):
        model = FakeModel([call(), final("answer", "Box 22 reports income tax deducted.", ["S1"])])
        missing = self.path.parent / "missing.db"
        state = run_ui_turn("What does T4 box 22 mean?", [], {}, db_path=missing,
                            model_factory=lambda: model, retriever=Mock(return_value=evidence()))
        self.assertEqual(state["outcome"]["status"], "answer")
        self.assertFalse(state["capabilities"]["mcp_transport"])
        self.assertFalse(missing.exists())

    def test_input_state_is_not_mutated_and_other_session_has_no_selection(self):
        first = self.turn("EMP-1001 salary as of 2026-11-01")
        original = deepcopy(first["employee_session"])
        self.turn("EMP-1004 salary", original)
        self.assertEqual(original["selected"]["employee_id"], "emp-001")
        other = self.turn("When is their next payday?")
        self.assertNotIn("selected", other["employee_session"])
        self.assertEqual(other["outcome"]["status"], "clarification")

    def test_ui_searches_are_serialized_for_local_qdrant(self):
        active, maximum = 0, 0

        def search(*args, **kwargs):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            time.sleep(.02)
            active -= 1
            return {"status": "no_matches", "results": []}

        with patch("payroll_ui_runtime.search_documents", side_effect=search), ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(ui_search, ["first", "second"]))
        self.assertEqual(maximum, 1)


if __name__ == "__main__":
    unittest.main()
