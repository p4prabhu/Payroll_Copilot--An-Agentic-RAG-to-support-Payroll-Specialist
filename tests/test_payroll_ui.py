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
    def test_fresh_deployment_initializes_both_databases_then_answers_employee_and_t4_questions(self):
        from init_employee_db import ensure_sample_databases
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            no_model = Mock(side_effect=AssertionError("Sample employee queries must not call a paid model"))
            runtime = partial(run_ui_turn, db_path=data / "payroll.db", annual_db_path=data / "payroll_t4_2026.db", model_factory=no_model)
            with patch("init_employee_db.ensure_sample_databases", side_effect=lambda: ensure_sample_databases(data)), \
                    patch("payroll_ui_runtime.run_ui_turn", side_effect=runtime):
                app = self.app()
                self.assertFalse(app.exception)
                self.assertTrue((data / "payroll.db").exists())
                self.assertTrue((data / "payroll_t4_2026.db").exists())
                app.chat_input[0].set_value("Show all active employees as of 2026-10-20").run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state["chat_messages"][-1]["outcome"]["roster"]["match_count"], 4)
                app.chat_input[0].set_value("How many employees are active as of 2026-10-20?").run()
                self.assertIn("**4 active employees**", app.session_state["chat_messages"][-1]["outcome"]["text"])
                app.chat_input[0].set_value("Show Alex Morgan’s sample T4 breakdown for 2026").run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state["chat_messages"][-1]["outcome"]["t4_summary"]["box_candidates_cents"]["14"], 5000008)
            no_model.assert_not_called()

    def test_database_startup_failure_shows_clear_error_and_stops_chat(self):
        with patch("init_employee_db.ensure_sample_databases", side_effect=PermissionError("private-internal-path")), \
                patch("payroll_ui_runtime.run_ui_turn") as run:
            app = self.app()
            self.assertFalse(app.exception)
            self.assertTrue(any("could not be initialized or validated" in e.value for e in app.error))
            self.assertFalse(any("private-internal-path" in e.value for e in app.error))
            self.assertEqual(len(app.chat_input), 0)
            run.assert_not_called()

    def test_exact_t4_question_shows_box_table_sample_source_and_missing_filing_data(self):
        from init_employee_db import FIXTURE_PATH
        with tempfile.TemporaryDirectory() as directory:
            path, annual = Path(directory) / "records.db", Path(directory) / "annual.db"
            init_database(path)
            init_database(annual, fixture_path=FIXTURE_PATH.with_name("fixtures_t4_2026.json"))
            no_model = Mock(side_effect=AssertionError("T4 breakdown must not initialize a paid model"))
            with patch("payroll_ui_runtime.run_ui_turn", side_effect=partial(run_ui_turn, db_path=path, annual_db_path=annual, model_factory=no_model)):
                app = self.app()
                app.chat_input[0].set_value("Show Alex Morgan’s sample T4 breakdown for 2026").run()
                self.assertFalse(app.exception)
                latest = app.session_state["chat_messages"][-1]["outcome"]
                self.assertEqual(latest["status"], "answer")
                self.assertEqual(latest["t4_summary"]["box_candidates_cents"]["14"], 5000008)
                self.assertTrue(any(s.value == "Sample T4 breakdown" for s in app.subheader))
                self.assertTrue(any("| 14 | Employment income" in m.value and "50,000.08" in m.value for m in app.markdown))
                self.assertTrue(any("Annual payroll sample" in m.value for m in app.markdown))
                self.assertIn("employee SIN", latest["text"])
                self.assertIn("not an actual T4", latest["text"])
                next(b for b in app.button if b.label == "Reset conversation").click().run()
                self.assertEqual(app.session_state["employee_session"], {})
            no_model.assert_not_called()

    def test_exact_roster_followup_shows_table_and_database_source_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ui roster.db"
            init_database(path)
            no_model = Mock(side_effect=AssertionError("Roster must not initialize paid models"))
            with patch("payroll_ui_runtime.run_ui_turn", side_effect=partial(run_ui_turn, db_path=path, model_factory=no_model)):
                app = self.app()
                app.chat_input[0].set_value("How many employees as of 2026-10-19?").run()
                app.chat_input[0].set_value("give me the details of all the employee").run()
                self.assertFalse(app.exception)
                latest = app.session_state["chat_messages"][-1]["outcome"]
                self.assertEqual(latest["roster"]["match_count"], 4)
                self.assertIn("Priya Shah", latest["text"])
                self.assertIn("$75,000.00", latest["text"])
                self.assertTrue(any("Employee database roster" in m.value for m in app.markdown))
                self.assertTrue(next(b for b in app.button if b.label == "Next employees").disabled)
                self.assertTrue(next(b for b in app.button if b.label == "Previous employees").disabled)
            no_model.assert_not_called()

    def test_roster_pagination_buttons_send_commands_and_reset_clears_controls(self):
        from employee_repository import EmployeeRepository
        from employee_analytics_workflow import render_roster
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pages.db"
            init_database(path)
            repository = EmployeeRepository(path)
            outputs = []
            for offset in (0, 2):
                page = repository.list_employees("2026-10-19", limit=2, offset=offset)
                args = {"as_of_date": "2026-10-19", "limit": 2, "offset": offset}
                session = {"_analytics": {"kind": "roster", "arguments": args, "page_ready": True,
                                          "has_more": page["has_more"], "next_offset": page["next_offset"]}}
                sources = [{"source_id": "D1", "kind": "employee_roster", "tool": "list_employees", "arguments": args, "provenance": page["provenance"]}]
                output = result(render_roster(page), session=session, sources=sources)
                output["outcome"]["roster"] = page
                outputs.append(output)
            with patch("payroll_ui_runtime.run_ui_turn", side_effect=[outputs[0], outputs[1], outputs[0]]) as run:
                app = self.app()
                app.chat_input[0].set_value("List all employees as of 2026-10-19").run()
                next(b for b in app.button if b.label == "Next employees").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(run.call_args.args[0], "Next employees")
                self.assertEqual(app.session_state["chat_messages"][-1]["outcome"]["roster"]["offset"], 2)
                next(b for b in app.button if b.label == "Previous employees").click().run()
                self.assertEqual(run.call_args.args[0], "Previous employees")
                next(b for b in app.button if b.label == "Reset conversation").click().run()
                self.assertFalse(any(b.label == "Next employees" for b in app.button))
                self.assertEqual(app.session_state["employee_session"], {})
    def test_real_workforce_count_and_followup_render_sources_without_paid_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ui workforce.db"
            init_database(path)
            no_model = Mock(side_effect=AssertionError("Workforce queries must not initialize a paid model"))
            with patch("payroll_ui_runtime.run_ui_turn", side_effect=partial(run_ui_turn, db_path=path, model_factory=no_model)):
                app = self.app()
                app.chat_input[0].set_value("How many employees are active?").run()
                self.assertFalse(app.exception)
                app.chat_input[0].set_value("2026-11-01").run()
                self.assertFalse(app.exception)
                self.assertTrue(any(s.value == "Workforce summary" for s in app.subheader))
                self.assertIn("**4 active employees**", app.session_state["chat_messages"][-1]["content"])
                self.assertTrue(any("Employee database summary" in m.value for m in app.markdown))
                next(b for b in app.button if b.label == "Reset conversation").click().run()
                self.assertEqual(app.session_state["employee_session"], {})
            no_model.assert_not_called()
    def test_calculation_confirmation_buttons_result_sources_and_reset(self):
        import json
        from payroll_calculation_workflow import calculation_turn
        payload = json.loads((ROOT / "tests/fixtures/payroll_reference_cases.json").read_text(encoding="utf-8"))["cases"][0]["input"]
        ready = calculation_turn(Mock(), "Calculate payroll: " + json.dumps(payload), {})
        ready["capabilities"] = {"payroll_calculation": True, "mcp_transport": False}
        calculated = calculation_turn(Mock(), "Confirm calculation", ready["employee_session"])
        calculated["capabilities"] = ready["capabilities"]
        with patch("payroll_ui_runtime.run_ui_turn", side_effect=[ready, calculated]) as run:
            app = self.app()
            app.chat_input[0].set_value("Calculate payroll").run()
            self.assertFalse(app.exception)
            next(b for b in app.button if b.label == "Confirm calculation").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(run.call_args.args[0], "Confirm calculation")
            self.assertIn("1,544.03", app.session_state["chat_messages"][-1]["content"])
            self.assertEqual([(m.label, m.value) for m in app.metric],
                             [("Net pay", "$1,544.03"), ("Gross pay", "$1,923.08"), ("Total deductions", "$379.05")])
            table = next(m.value for m in app.markdown if "| Deduction | CAD |" in m.value)
            for row in ("Federal income tax | \\$153.28", "Ontario income tax | \\$88.01", "CPP | \\$106.41", "CPP2 | \\$0.00", "EI | \\$31.35"):
                self.assertIn(row, table)
            self.assertTrue(any(e.label == "Calculation details" for e in app.expander))
            self.assertTrue(any("Payroll calculation rules" in m.value for m in app.markdown))
            self.assertFalse(any(b.label == "Confirm calculation" for b in app.button))
            next(b for b in app.button if b.label == "Reset conversation").click().run()
            self.assertEqual(app.session_state["employee_session"], {})

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

    def test_review_notices_are_compact_but_full_text_and_history_are_preserved(self):
        notes = ["Source content review is pending internally within this project; this answer is provisional.",
                 "Some cited editions need internal project update review; current applicability has not been verified.",
                 "Some cited sources have unknown or incomplete effective dates; payment-date applicability remains unresolved."]
        body = "A T4 summarizes employment income and deductions.\n\n**What it includes**\n\n- Salary\n- Income tax, CPP and EI"
        text = body + "\n\n" + "\n".join(notes)
        with patch("payroll_ui_runtime.run_ui_turn", return_value=result(text)):
            app = self.app()
            app.chat_input[0].set_value("What is T4?").run()
        self.assertFalse(app.exception)
        self.assertTrue(any(m.value == body for m in app.markdown))
        self.assertTrue(any("Provisional guidance" in m.value for m in app.markdown))
        self.assertTrue(any(e.label == "Review notes" for e in app.expander))
        for note in notes:
            self.assertTrue(any(c.value == note for c in app.caption))
        self.assertEqual(app.session_state["chat_messages"][-1]["content"], text)
        self.assertEqual(app.session_state["chat_messages"][-1]["outcome"]["text"], text)

    def test_substantive_limitations_stay_in_answer_and_dynamic_text_is_not_html(self):
        text = "Employee history is incomplete. Do not use it as a complete YTD total.\n\n<script>UI_SENTINEL</script>"
        with patch("payroll_ui_runtime.run_ui_turn", return_value=result(text)):
            app = self.app()
            app.chat_input[0].set_value("Show history").run()
        self.assertFalse(app.exception)
        body = next(m for m in app.markdown if "UI_SENTINEL" in m.value)
        self.assertIn("Do not use it as a complete YTD total", body.value)
        self.assertFalse(body.proto.allow_html)

    def test_repeated_pdf_sources_keep_distinct_citations_and_links(self):
        sources = [{"source_id": "S1", "title": "T4 guide", "pdf_page_number": 2, "source_url": "https://www.canada.ca/test.pdf"},
                   {"source_id": "S2", "title": "T4 guide", "pdf_page_number": 7, "source_url": "https://www.canada.ca/test.pdf"}]
        with patch("payroll_ui_runtime.run_ui_turn", return_value=result("T4 guidance [S1] [S2]", sources=sources)):
            app = self.app()
            app.chat_input[0].set_value("What is T4?").run()
        self.assertFalse(app.exception)
        for source_id in ("S1", "S2"):
            self.assertTrue(any(f"[{source_id}] T4 guide" in m.value for m in app.markdown))
        self.assertTrue(any(e.label == "Sources · 2" for e in app.expander))

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
