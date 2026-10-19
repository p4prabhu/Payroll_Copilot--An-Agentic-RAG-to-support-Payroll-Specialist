"""Streamlit interactions through AppTest; all model responses are mocked."""
from copy import deepcopy
from functools import partial
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from streamlit.testing.v1 import AppTest
from init_employee_db import init_database
from payroll_ui_runtime import run_ui_turn


def result(text="Stored salary: CAD $50,000.00", *, session=None, sources=None, status="answer"):
    return {"employee_session": deepcopy(session or {}), "capabilities": {"mcp_transport": True},
            "outcome": {"status": status, "text": text, "sources": sources or [], "citations": []},
            "trace": [{"event": "employee_tool", "tool": "get_employee", "status": "ok"}]}


class PayrollUITests(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()

    def test_startup_does_not_run_agent_and_submit_is_not_repeated_on_rerun(self):
        with patch("payroll_ui_runtime.run_ui_turn", return_value=result()) as run:
            app = self.app()
            run.assert_not_called()
            self.assertFalse(app.exception)
            app.chat_input[0].set_value("EMP-1001 salary as of 2026-11-01").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state["chat_messages"]), 2)
            app.run()
            run.assert_called_once()

    def test_reset_clears_messages_employee_context_and_connection_status(self):
        selected = {"employee_id": "emp-001", "employee_number": "EMP-1001", "given_name": "Alex", "family_name": "Morgan"}
        with patch("payroll_ui_runtime.run_ui_turn", return_value=result(session={"selected": selected, "lookup_date": "2026-11-01"})):
            app = self.app()
            app.chat_input[0].set_value("EMP-1001 salary as of 2026-11-01").run()
            next(b for b in app.button if b.label == "Reset conversation").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["chat_messages"], [])
        self.assertEqual(app.session_state["employee_session"], {})
        self.assertIsNone(app.session_state["last_capabilities"])

    def test_document_and_database_sources_have_separate_displays(self):
        sources = [{"source_id": "S1", "title": "Government guide", "pdf_page_number": 18, "source_url": "https://www.canada.ca/test.pdf"},
                   {"source_id": "D1", "kind": "employee_record", "employee_id": "emp-001", "provenance": {"dataset_snapshot_date": "2026-11-01"}}]
        with patch("payroll_ui_runtime.run_ui_turn", return_value=result("Fictional database record", sources=sources)):
            app = self.app()
            app.chat_input[0].set_value("Show records").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Government guide" in m.value for m in app.markdown))
        self.assertTrue(any("Employee record" in m.value for m in app.markdown))
        self.assertFalse(any("fictional" in m.value.lower() for m in app.markdown))
        self.assertEqual(app.session_state["chat_messages"][-1]["outcome"]["text"], "Fictional database record")
        self.assertTrue(any("PDF page 18" in c.value for c in app.caption))
        self.assertTrue(any(e.label == "Tool activity" for e in app.expander))

    def test_failure_preserves_conversation_and_never_displays_exception_secrets(self):
        with patch("payroll_ui_runtime.run_ui_turn", side_effect=[result(), RuntimeError("secret-api-key")]) as run:
            app = self.app()
            app.chat_input[0].set_value("EMP-1001 salary as of 2026-11-01").run()
            original = deepcopy(app.session_state["chat_messages"])
            app.chat_input[0].set_value("A follow-up").run()
            self.assertEqual(app.session_state["chat_messages"], original)
            self.assertTrue(app.error)
            self.assertNotIn("secret-api-key", app.error[0].value)
            app.run()
            self.assertEqual(run.call_count, 2)
        self.assertFalse(app.exception)

    def test_separate_browser_sessions_do_not_share_history(self):
        with patch("payroll_ui_runtime.run_ui_turn", return_value=result()):
            first, second = self.app(), self.app()
            first.chat_input[0].set_value("EMP-1001 salary as of 2026-11-01").run()
        self.assertEqual(len(first.session_state["chat_messages"]), 2)
        self.assertEqual(second.session_state["chat_messages"], [])

    def test_real_mcp_ambiguity_buttons_selection_and_followup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ui records.db"
            init_database(path)
            no_model = Mock(side_effect=AssertionError("No paid model calls"))
            with patch("payroll_ui_runtime.run_ui_turn", side_effect=partial(run_ui_turn, db_path=path, model_factory=no_model)):
                app = self.app()
                app.chat_input[0].set_value("Find Alex as of 2026-11-01").run()
                self.assertEqual(len(app.session_state["employee_session"]["candidates"]), 2)
                next(b for b in app.button if "EMP-1001" in b.label).click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state["employee_session"]["selected"]["employee_id"], "emp-001")
                self.assertIn("50,000.00", app.session_state["chat_messages"][-1]["content"])
                app.chat_input[0].set_value("When is their next payday?").run()
                self.assertIn("2026-11-13", app.session_state["chat_messages"][-1]["content"])
                self.assertNotIn("candidates", app.session_state["employee_session"])
            no_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
