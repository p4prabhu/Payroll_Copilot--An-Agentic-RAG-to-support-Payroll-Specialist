"""Offline workflow tests and explicitly enabled paid acceptance scenarios.

Normal runs skip live scenarios. After approving API charges, set
RUN_LIVE_AGENT_TESTS=1 to enable them. Four scenarios execute seven turns total,
each subject to the agent's model-call and search limits. Keyword assertions
are acceptance screens, not independent verification of payroll facts.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from payroll_agent import MAX_MODEL_CALLS, MAX_SEARCHES, SUBMIT_NAME, TOOL_NAME, applicability_violation, build_agent, create_model, document_violation, payment_context, payroll_amount_violation, render_answer, requested_documents, source_applicability, submission_tool, run_turn


def final(status, text, citations=None):
    return AIMessage(content="", tool_calls=[{
        "name": SUBMIT_NAME, "id": "submit_final", "type": "tool_call",
        "args": {"status": status, "text": text, "citations": citations or []},
    }])


def call(number=1, **args):
    return AIMessage(content="", tool_calls=[{
        "name": TOOL_NAME, "args": {"question": "T4 box 22", **args},
        "id": f"call_{number}", "type": "tool_call",
    }])


def evidence():
    return {"status": "evidence_found", "results": [{
        "chunk_id": "test-chunk", "content": "Box 22 reports income tax deducted.",
        "source_url": "https://www.canada.ca/test-guide.pdf", "title": "Test government guide",
        "pdf_page_number": 18, "review_status": "pending", "needs_update_review": True,
    }]}


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.seen = []

    def bind_tools(self, tools, **kwargs):
        self.tools = tools
        self.bound_options = kwargs
        return self

    def invoke(self, messages):
        self.seen.append(messages)
        # Each API response is a new message, even when text/arguments repeat.
        return next(self.responses).model_copy(deep=True, update={"id": None})


class PayrollAgentTests(unittest.TestCase):
    def test_submission_tool_is_bound_without_rejected_response_format(self):
        model = FakeModel([])
        build_agent(model, Mock())
        self.assertNotIn("response_format", model.bound_options)
        self.assertEqual(model.tools[0].name, TOOL_NAME)
        config = model.tools[1]["function"]
        self.assertEqual(config["name"], SUBMIT_NAME)
        schema = config["parameters"]
        self.assertEqual(set(schema["required"]), {"status", "text", "citations"})
        self.assertFalse(schema["additionalProperties"])

    def test_installed_integration_converts_submission_tool_for_responses_api(self):
        from langchain_perplexity import ChatPerplexity
        model = ChatPerplexity(api_key="offline-placeholder", use_responses_api=True)
        payload = model._to_responses_payload([], {"tools": [submission_tool()], "max_tokens": 1600})
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["tools"][0]["name"], SUBMIT_NAME)
        self.assertEqual(set(payload["tools"][0]["parameters"]["required"]), {"status", "text", "citations"})
        self.assertEqual(payload["max_output_tokens"], 1600)

    def test_schema_rejects_missing_wrong_type_and_extra_fields(self):
        for draft in ({"status": "clarification", "text": "Missing field"},
                      {"status": "clarification", "text": 23, "citations": []},
                      {"status": "unsupported", "text": "Gap", "citations": [], "extra": "private-text"}):
            with self.subTest(draft=draft):
                response = final("unsupported", "Placeholder")
                response.tool_calls[0]["args"] = draft
                model = FakeModel([response, response])
                state = run_turn(build_agent(model, Mock()), "Question")
                self.assertEqual(state["outcome"]["status"], "unsupported")
                self.assertEqual(state["trace"][-1]["reason"], "Final response schema validation failed")
                self.assertNotIn("private-text", str(state["trace"]))

    def test_incomplete_response_does_not_execute_partial_tool_call(self):
        response = call()
        response.response_metadata["status"] = "incomplete"
        retriever = Mock()
        state = run_turn(build_agent(FakeModel([response]), retriever), "T4 box 22?")
        retriever.assert_not_called()
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["trace"][-1]["reason"], "Model response was incomplete or unsuccessful")

    def test_incomplete_response_rejected_even_with_valid_submission(self):
        response = final("clarification", "What payment date?")
        response.response_metadata["status"] = "incomplete"
        state = run_turn(build_agent(FakeModel([response]), Mock()), "Question")
        self.assertEqual(state["trace"][-1]["status"], "rejected")

    def test_search_then_cited_answer_and_review_flags(self):
        retriever = Mock(return_value=evidence())
        model = FakeModel([call(edition_year=2025, document_type="guide"),
                           final("answer", "Box 22 reports income tax deducted [S1].", ["S1"])])
        state = run_turn(build_agent(model, retriever), "What is box 22?")
        self.assertEqual(state["model_calls"], 2)
        self.assertEqual(state["searches"], 1)
        self.assertIsInstance(model.seen[1][-1], ToolMessage)
        self.assertIsInstance(model.seen[0][0], SystemMessage)
        self.assertIn("Ontario Payroll", model.seen[0][0].content)
        self.assertEqual(retriever.call_args.kwargs["edition_year"], 2025)
        self.assertEqual(state["outcome"]["status"], "answer")
        answer = render_answer(state["outcome"])
        self.assertIn("PDF page 18", answer)
        self.assertIn("https://www.canada.ca/test-guide.pdf", answer)
        self.assertIn("provisional", answer)
        self.assertIn("update review", answer)

    def test_refines_empty_search(self):
        retriever = Mock(side_effect=[{"status": "no_matches", "results": []}, evidence()])
        model = FakeModel([call(1), call(2, question="Income tax deducted on T4 box 22"),
                           final("answer", "Income tax deducted.", ["S1"])])
        state = run_turn(build_agent(model, retriever), "T4 box 22?")
        self.assertEqual(state["searches"], 2)
        self.assertEqual(state["outcome"]["status"], "answer")

    def test_two_searches_preserve_distinct_evidence_and_citation_mapping(self):
        box14 = {**evidence()["results"][0], "chunk_id": "box14-chunk",
                 "content": "Box 14 reports employment income.", "pdf_page_number": 14}
        box22 = {**evidence()["results"][0], "chunk_id": "box22-chunk"}
        retriever = Mock(side_effect=[
            {"status": "evidence_found", "results": [box14]},
            {"status": "evidence_found", "results": [box22, box14]},
        ])
        model = FakeModel([
            call(1, question="T4 box 14"), call(2, question="T4 box 22"),
            final("answer", "Employment income [S1]; income tax deducted [S2].", ["S1", "S2"]),
        ])
        state = run_turn(build_agent(model, retriever), "What do T4 boxes 14 and 22 mean?")
        self.assertEqual(state["searches"], 2)
        self.assertEqual(state["outcome"]["status"], "answer")
        self.assertEqual(set(state["evidence"]), {"S1", "S2"})
        for source_id, expected in (("S1", box14), ("S2", box22)):
            self.assertEqual(state["evidence"][source_id], expected)
        second_output = json.loads(model.seen[2][-1].content)
        self.assertEqual([item["source_id"] for item in second_output["results"]], ["S2", "S1"])
        sources = {source["source_id"]: source for source in state["outcome"]["sources"]}
        self.assertEqual(sources["S1"]["pdf_page_number"], 14)
        self.assertEqual(sources["S2"]["pdf_page_number"], 18)
        rendered = render_answer(state["outcome"])
        self.assertIn("[S1] Test government guide — PDF page 14", rendered)
        self.assertIn("[S2] Test government guide — PDF page 18", rendered)

    def test_search_and_model_limits_stop_repeating_agent(self):
        retriever = Mock(return_value=evidence())
        model = FakeModel([call(i) for i in range(MAX_MODEL_CALLS)])
        state = run_turn(build_agent(model, retriever), "Keep searching")
        self.assertEqual(retriever.call_count, MAX_SEARCHES)
        self.assertEqual(state["model_calls"], MAX_MODEL_CALLS)
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["trace"][-2]["status"], "limit_reached")

    def test_parallel_requested_searches_are_executed_within_budget(self):
        calls = [call(i).tool_calls[0] for i in range(7)]
        model = FakeModel([AIMessage(content="", tool_calls=calls), final("answer", "Tax deducted.", ["S1"])])
        retriever = Mock(return_value=evidence())
        state = run_turn(build_agent(model, retriever), "Several searches")
        self.assertEqual(retriever.call_count, MAX_SEARCHES)
        self.assertEqual(len(state["evidence"]), 1)

    def test_invalid_tool_arguments_never_reach_retrieval(self):
        retriever = Mock()
        model = FakeModel([call(limit=21), final("unsupported", "Search arguments were invalid.")])
        state = run_turn(build_agent(model, retriever), "T4 box 22?")
        retriever.assert_not_called()
        self.assertEqual(state["trace"][1]["status"], "tool_error")

    def test_tool_error_is_sanitized(self):
        retriever = Mock(side_effect=RuntimeError("secret-request-detail"))
        model = FakeModel([call(), final("unsupported", "Search unavailable.")])
        state = run_turn(build_agent(model, retriever), "T4 box 22?")
        self.assertNotIn("secret-request-detail", str(state))
        self.assertEqual(state["outcome"]["status"], "unsupported")

    def test_unknown_citation_is_rejected(self):
        rejected = final("answer", "Income tax deducted.", ["S999"])
        model = FakeModel([call(), rejected, rejected])
        state = run_turn(build_agent(model, Mock(return_value=evidence())), "T4 box 22?")
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["outcome"]["sources"], [])

    def test_followup_context_and_per_turn_budget_reset(self):
        model = FakeModel([final("clarification", "How often is payroll paid?"),
                           final("clarification", "What is the payment date?")])
        retriever = Mock()
        graph = build_agent(model, retriever)
        first = run_turn(graph, "What are deductions on $50,000?")
        history = [HumanMessage(content="What are deductions on $50,000?"),
                   AIMessage(content=render_answer(first["outcome"]))]
        second = run_turn(graph, "Biweekly", history)
        self.assertIn("$50,000", model.seen[1][1].content)
        self.assertEqual(second["model_calls"], 1)
        self.assertEqual(second["searches"], 0)
        retriever.assert_not_called()

    def test_previous_turn_sources_cannot_support_new_answer_without_search(self):
        model = FakeModel([call(), final("answer", "Tax deducted.", ["S1"]),
                           final("answer", "Still tax deducted.", ["S1"]),
                           final("answer", "Still tax deducted.", ["S1"])])
        graph = build_agent(model, Mock(return_value=evidence()))
        first = run_turn(graph, "Box 22?")
        history = [HumanMessage(content="Box 22?"), AIMessage(content=render_answer(first["outcome"]))]
        second = run_turn(graph, "And this year?", history)
        self.assertEqual(second["evidence"], {})
        self.assertEqual(second["outcome"]["status"], "unsupported")

    def test_malformed_final_and_uncited_answer_are_rejected(self):
        for response in (AIMessage(content="not json"), final("answer", "An unsupported claim.")):
            with self.subTest(response=response.content):
                state = run_turn(build_agent(FakeModel([response, response]), Mock()), "Policy question")
                self.assertEqual(state["outcome"]["status"], "unsupported")

    def test_plain_json_is_not_accepted_in_place_of_submission_tool(self):
        reply = AIMessage(content=json.dumps({"status": "clarification", "text": "What date?", "citations": []}))
        state = run_turn(build_agent(FakeModel([reply, reply]), Mock()), "Question")
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["trace"][-1]["reason"], "A single submit_response tool call is required")

    def test_submission_cannot_be_combined_with_search(self):
        combined = AIMessage(content="", tool_calls=[call().tool_calls[0], final("answer", "Claim.", ["S1"]).tool_calls[0]])
        retriever = Mock()
        state = run_turn(build_agent(FakeModel([combined, combined]), retriever), "Question")
        retriever.assert_not_called()
        self.assertEqual(state["outcome"]["status"], "unsupported")

    def test_submission_ends_turn_without_extra_model_or_search_call(self):
        model = FakeModel([final("clarification", "What payment date?")])
        retriever = Mock()
        state = run_turn(build_agent(model, retriever), "Calculate deductions")
        self.assertEqual(len(model.seen), 1)
        retriever.assert_not_called()
        self.assertEqual(state["outcome"]["status"], "clarification")
        self.assertEqual(state["trace"][-1]["status"], "accepted")

    def test_urls_must_match_retrieved_cited_sources(self):
        for url, expected in [("https://www.canada.ca/test-guide.pdf", "answer"),
                              ("https://example.com/invented", "unsupported")]:
            with self.subTest(url=url):
                reply = final("answer", f"Tax deducted. See {url}", ["S1"])
                model = FakeModel([call(), reply] + ([reply] if expected == "unsupported" else []))
                state = run_turn(build_agent(model, Mock(return_value=evidence())), "Box 22?")
                self.assertEqual(state["outcome"]["status"], expected)

    def test_plain_clarification_can_be_corrected_without_retrieval(self):
        retriever = Mock()
        model = FakeModel([AIMessage(content="How often are you paid?"),
                           final("clarification", "How often are you paid?")])
        state = run_turn(build_agent(model, retriever), "Deductions on $50,000?")
        self.assertEqual(state["outcome"]["status"], "clarification")
        self.assertEqual(state["correction_attempts"], 1)
        self.assertEqual(state["model_calls"], 2)
        retriever.assert_not_called()
        self.assertIsInstance(model.seen[1][-1], HumanMessage)
        self.assertIn("Application validation feedback", model.seen[1][-1].content)

    def test_invalid_submission_gets_paired_feedback_and_can_explain_gap(self):
        invalid = final("unsupported", "Employee data is unavailable.")
        del invalid.tool_calls[0]["args"]["citations"]
        model = FakeModel([invalid, final("unsupported", "Employee data is unavailable.")])
        state = run_turn(build_agent(model, Mock()), "Alex's next payday?")
        self.assertEqual(state["trace"][-1]["status"], "accepted")
        self.assertEqual(state["outcome"]["text"], "Employee data is unavailable.")
        feedback = model.seen[1][-1]
        self.assertIsInstance(feedback, ToolMessage)
        self.assertEqual(feedback.tool_call_id, invalid.tool_calls[0]["id"])
        self.assertEqual(json.loads(feedback.content)["status"], "validation_error")

    def test_correction_preserves_evidence_and_requires_valid_citation(self):
        retriever = Mock(return_value=evidence())
        model = FakeModel([call(), final("answer", "Income tax deducted.", ["S999"]),
                           final("answer", "Income tax deducted [S1].", ["S1"])])
        state = run_turn(build_agent(model, retriever), "Box 22?")
        self.assertEqual(state["outcome"]["status"], "answer")
        self.assertEqual(state["outcome"]["sources"][0]["chunk_id"], "test-chunk")
        self.assertEqual(state["correction_attempts"], 1)
        self.assertEqual(retriever.call_count, 1)
        self.assertIn('Available source IDs: ["S1"]', json.loads(model.seen[2][-1].content)["feedback"])

    def test_repeated_invalid_responses_stop_after_one_correction(self):
        model = FakeModel([AIMessage(content="plain response"), AIMessage(content="still plain")])
        state = run_turn(build_agent(model, Mock()), "Question")
        self.assertEqual(state["model_calls"], 2)
        self.assertEqual(state["correction_attempts"], 1)
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["trace"][-1]["status"], "rejected")

    def test_correction_cannot_exceed_model_call_budget(self):
        model = FakeModel([*[call(i) for i in range(MAX_MODEL_CALLS - 1)], AIMessage(content="invalid final")])
        state = run_turn(build_agent(model, Mock(return_value=evidence())), "Question")
        self.assertEqual(state["model_calls"], MAX_MODEL_CALLS)
        self.assertEqual(state["correction_attempts"], 0)
        self.assertEqual(state["outcome"]["status"], "unsupported")

    def test_correction_budget_resets_for_followup_turn(self):
        model = FakeModel([AIMessage(content="plain"), final("clarification", "What frequency?"),
                           AIMessage(content="plain again"), final("clarification", "What payment date?")])
        graph = build_agent(model, Mock())
        first = run_turn(graph, "Deductions on $50,000?")
        history = [HumanMessage(content="Deductions on $50,000?"),
                   AIMessage(content=render_answer(first["outcome"]))]
        second = run_turn(graph, "Biweekly", history)
        self.assertEqual(first["correction_attempts"], 1)
        self.assertEqual(second["correction_attempts"], 1)
        self.assertEqual(second["model_calls"], 2)
        self.assertEqual(second["outcome"]["status"], "clarification")

    def test_employee_access_gap_requires_unsupported_status(self):
        invalid = final("clarification", "I do not have access to employee records or the company payroll schedule. What is Alex's employee ID?")
        corrected = final("unsupported", "I cannot determine Alex's next payday because employee records and the company payroll calendar are not connected.")
        retriever = Mock()
        state = run_turn(build_agent(FakeModel([invalid, corrected]), retriever), "Alex's next payday?")
        retriever.assert_not_called()
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["trace"][-1]["status"], "accepted")
        self.assertEqual(state["correction_attempts"], 1)
        self.assertIn("requires unsupported status", state["trace"][1]["reason"])

    def test_calculation_promises_are_rejected_and_can_be_corrected(self):
        for promise in (
            "I can calculate your payroll deductions.",
            "Once you provide TD1 details, I will calculate your net pay.",
            "I can search the applicable 2026 tables and calculate specific deductions.",
        ):
            with self.subTest(promise=promise):
                invalid = final("clarification", promise + " What is your frequency?")
                corrected = final("clarification", "Payroll calculation is not implemented. I can collect details for intake: what is your pay frequency and intended payment date?")
                state = run_turn(build_agent(FakeModel([invalid, corrected]), Mock()), "Deductions on $50,000?")
                self.assertEqual(state["outcome"]["status"], "clarification")
                self.assertEqual(state["correction_attempts"], 1)
                self.assertIn("do not promise to calculate", state["trace"][1]["reason"])
                self.assertEqual(state["trace"][-1]["status"], "accepted")

    def test_honest_limits_allow_intake_and_document_explanations(self):
        text = "I cannot calculate payroll deductions; the calculation tool is not implemented. I can explain how deductions are calculated from official guidance. What is the intended payment date?"
        state = run_turn(build_agent(FakeModel([final("clarification", text)]), Mock()), "Salary intake")
        self.assertEqual(state["outcome"]["status"], "clarification")
        self.assertEqual(state["correction_attempts"], 0)

    def test_cited_document_cannot_enable_net_pay_or_employee_access(self):
        for claim in ("Your net pay is $1,500.", "I can retrieve the employee records."):
            with self.subTest(claim=claim):
                invalid = final("answer", claim, ["S1"])
                state = run_turn(build_agent(FakeModel([call(), invalid, invalid]), Mock(return_value=evidence())), "Payroll question")
                self.assertEqual(state["outcome"]["status"], "unsupported")
                self.assertEqual(state["trace"][-1]["status"], "rejected")

    def test_payment_context_uses_user_dates_and_explicit_changes_only(self):
        history = [HumanMessage(content="The payment date is 2026-11-13."),
                   AIMessage(content="Assuming the payment date is 2026-10-16."),
                   HumanMessage(content="Biweekly.")]
        self.assertEqual(payment_context(history), {"date": "2026-11-13", "status": "provided"})
        changed = payment_context([*history, HumanMessage(content="Change payment date to 2026-12-04.")])
        self.assertEqual(changed["date"], "2026-12-04")
        self.assertEqual(payment_context([*history, HumanMessage(content="Using that same payment date, explain the T4001 guide")])["date"], "2026-11-13")
        self.assertEqual(payment_context([AIMessage(content="The payment date is 2026-11-13")])["status"], "missing")
        self.assertEqual(payment_context([HumanMessage(content="Compare the 2026-07-01 formula edition")])["date"], None)

    def test_invalid_or_ambiguous_payment_date_clears_previous_date(self):
        previous = [HumanMessage(content="Payment date: 2026-11-13")]
        for message, expected in (("Payment date is 2026-02-30", "invalid"),
                                  ("Payment date is 2026-11-13 or 2026-11-20", "ambiguous"),
                                  ("Payment date is 03/04/2026", "needs_iso_date"),
                                  ("Payment date is tomorrow", "needs_iso_date"),
                                  ("Payment date is next Friday", "needs_iso_date"),
                                  ("Clear the payment date", "missing")):
            with self.subTest(message=message):
                self.assertEqual(payment_context([*previous, HumanMessage(content=message)]),
                                 {"date": None, "status": expected})

    def test_date_retained_when_old_exchange_is_trimmed_from_model_history(self):
        history = [HumanMessage(content="Payment date: 2026-11-13"), AIMessage(content="Noted.")]
        for number in range(7):
            history.extend([HumanMessage(content=f"General follow-up {number}"), AIMessage(content="Noted.")])
        model = FakeModel([final("clarification", "Using payment date 2026-11-13. What TD1 information is available?")])
        state = run_turn(build_agent(model, Mock()), "Keep the same date", history)
        self.assertEqual(state["payment_date"], "2026-11-13")
        self.assertEqual(state["outcome"]["payment_date"], "2026-11-13")
        self.assertNotIn("Payment date:", str([message.content for message in model.seen[0][1:]]))
        self.assertIn('"payment_date": "2026-11-13"', model.seen[0][0].content)

    def test_assumed_payment_date_is_rejected_and_corrected(self):
        for text in ("As of October 16, 2026, what are your TD1 amounts?",
                     "Your payment date is 2026-10-16. What frequency?",
                     "Using payment date today, what is your frequency?"):
            with self.subTest(text=text):
                corrected = final("clarification", "What is the intended payment date in YYYY-MM-DD format? Payroll calculation is not implemented.")
                state = run_turn(build_agent(FakeModel([final("clarification", text), corrected]), Mock()), "Salary intake")
                self.assertEqual(state["correction_attempts"], 1)
                self.assertEqual(state["outcome"]["status"], "clarification")
                self.assertIsNone(state["payment_date"])

    def test_frequency_table_selection_requires_user_payment_date(self):
        model = FakeModel([call(document_type="deduction_table", pay_periods_per_year=26),
                           final("clarification", "What is the payment date in YYYY-MM-DD format?")])
        retriever = Mock()
        state = run_turn(build_agent(model, retriever), "Ontario biweekly deduction table")
        retriever.assert_not_called()
        self.assertEqual(state["trace"][1]["status"], "clarification_required")
        self.assertEqual(state["outcome"]["status"], "clarification")

    def test_source_applicability_checks_bounds_without_inferring_missing_dates(self):
        cases = [
            ({"effective_from": "2026-01-01", "effective_to": "2026-06-30"}, "2026-06-30", "within_known_effective_dates"),
            ({"effective_from": "2026-07-01"}, "2026-06-30", "before_effective_from"),
            ({"effective_to": "2026-06-30"}, "2026-07-01", "after_effective_to"),
            ({"effective_from": "2026-01-01"}, "2026-11-13", "effective_dates_incomplete"),
            ({"edition_year": 2026}, "2026-11-13", "effective_dates_incomplete"),
            ({"effective_from": "2026-01-01", "effective_to": "2026-12-31"}, None, "payment_date_missing"),
            ({"effective_from": "bad-date"}, "2026-11-13", "invalid_effective_dates"),
            ({"effective_from": "2027-01-01", "effective_to": "2026-01-01"}, "2026-11-13", "invalid_effective_dates"),
        ]
        for source, payment, expected in cases:
            with self.subTest(source=source, payment=payment):
                self.assertEqual(source_applicability(source, payment), expected)

    def test_unknown_dates_are_visible_in_tool_output_and_final_answer(self):
        model = FakeModel([call(), final("answer", "Box 22 reports tax deducted [S1].", ["S1"])])
        state = run_turn(build_agent(model, Mock(return_value=evidence())), "Payment date: 2026-11-13. What is box 22?")
        self.assertEqual(state["outcome"]["sources"][0]["payment_applicability"], "effective_dates_incomplete")
        self.assertIn("applicability remains unresolved", state["outcome"]["text"])
        tool_data = json.loads(model.seen[1][-1].content)
        self.assertEqual(tool_data["results"][0]["payment_applicability"], "effective_dates_incomplete")

    def test_answer_cannot_rely_on_source_outside_known_dates(self):
        saved = evidence()
        saved["results"][0].update(effective_from="2025-01-01", effective_to="2025-12-31")
        model = FakeModel([call(), final("answer", "Use this guidance [S1].", ["S1"]),
                           final("unsupported", "The retrieved guide has expired for your intended payment date.")])
        state = run_turn(build_agent(model, Mock(return_value=saved)), "Payment date: 2026-11-13. Which guidance applies?")
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["correction_attempts"], 1)
        self.assertIn("conflicts with known effective-date bounds", state["trace"][3]["reason"])

    def test_unknown_effective_dates_cannot_support_definitive_application_claim(self):
        invalid = final("answer", "Both deductions apply to your payment date of 2026-11-13 under 2026 guidance.", ["S1"])
        corrected = final("answer", "The guide describes deductions. I cannot confirm whether these rules apply to your payment date because effective dates are incomplete [S1].", ["S1"])
        state = run_turn(build_agent(FakeModel([call(), invalid, corrected]), Mock(return_value=evidence())),
                         "Payment date: 2026-11-13. Explain this guide.")
        self.assertEqual(state["correction_attempts"], 1)
        self.assertIn("Effective dates are incomplete", state["trace"][3]["reason"])
        self.assertEqual(state["trace"][-1]["status"], "accepted")
        self.assertEqual(state["outcome"]["status"], "answer")
        self.assertIn("applicability remains unresolved", state["outcome"]["text"])

    def test_unknown_date_overclaim_cannot_bypass_correction_limit(self):
        invalid = final("answer", "These rules apply to your payment date of 2026-11-13.", ["S1"])
        state = run_turn(build_agent(FakeModel([call(), invalid, invalid]), Mock(return_value=evidence())),
                         "Payment date: 2026-11-13. Explain this guide.")
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["trace"][-1]["status"], "rejected")

    def test_explicit_publication_is_resolved_without_treating_t4_boxes_as_guides(self):
        self.assertEqual(requested_documents("Use the T4001 guide."), {"t4001": ["t4001-25e"]})
        self.assertEqual(requested_documents("Use T4001-25e, not an assumed edition."), {"t4001-25e": ["t4001-25e"]})
        self.assertEqual(requested_documents("What are T4 boxes 14 and 22?"), {})
        self.assertEqual(requested_documents("Use T9999."), {"t9999": []})

    def test_requested_guide_is_enforced_before_retrieval(self):
        saved = evidence()
        saved["results"][0]["document_id"] = "t4001-25e"
        retriever = Mock(return_value=saved)
        model = FakeModel([call(document_id="t4127-01-26e"), call(2),
                           final("answer", "The requested guide describes deductions [S1].", ["S1"])])
        state = run_turn(build_agent(model, retriever), "What does T4001 say?")
        self.assertEqual(retriever.call_count, 1)
        self.assertEqual(retriever.call_args.kwargs["document_id"], "t4001-25e")
        self.assertEqual(state["trace"][1]["status"], "tool_error")
        self.assertEqual(state["outcome"]["status"], "answer")

    def test_unavailable_requested_guide_never_calls_retriever(self):
        retriever = Mock()
        model = FakeModel([call(), final("unsupported", "The requested T9999 publication is not available in the catalog.")])
        state = run_turn(build_agent(model, retriever), "Explain T9999.")
        retriever.assert_not_called()
        self.assertEqual(state["outcome"]["status"], "unsupported")

    def test_retriever_cannot_leak_substitute_publication_into_evidence(self):
        saved = evidence()
        saved["results"][0]["document_id"] = "t4127-01-26e"
        invalid = final("answer", "Use another guide [S1].", ["S1"])
        state = run_turn(build_agent(FakeModel([call(), invalid,
            final("unsupported", "No suitable evidence from T4001 was retrieved.")]), Mock(return_value=saved)), "Explain T4001.")
        self.assertEqual(state["evidence"], {})
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["trace"][1]["status"], "no_evidence")

    def test_multiple_requested_publications_require_evidence_from_each(self):
        requested = requested_documents("Compare T4001 and T4127")
        first = {"document_id": "t4001-25e"}
        second = {"document_id": "t4127-jul-26e"}
        self.assertIsNotNone(document_violation(requested, [first]))
        self.assertIsNotNone(document_violation(requested, [{"document_id": "rc4120-25e"}]))
        self.assertIsNone(document_violation(requested, [first, second]))

    def test_request_for_specific_edition_does_not_allow_another_edition(self):
        retriever = Mock()
        model = FakeModel([call(document_id="t4127-jul-26e"),
                           final("unsupported", "I could not retrieve the requested January edition.")])
        state = run_turn(build_agent(model, retriever), "Use T4127-01-26e.")
        retriever.assert_not_called()
        self.assertEqual(state["outcome"]["status"], "unsupported")

    def test_payroll_estimates_are_rejected_even_with_a_disclaimer(self):
        for estimate in ("Gross pay per period: approximately $1,923.08.",
                         "You receive $1,923.08 per pay period.",
                         "Your CPP deduction is approximately $100.00.",
                         "Your tax deductions: 200 CAD.",
                         "$50,000 / 26 = $1,923.08."):
            with self.subTest(estimate=estimate):
                invalid = final("clarification", estimate + " Calculation tools are unavailable. What TD1 details do you have?")
                corrected = final("clarification", "Your annual salary is $50,000. Payroll calculation is unavailable. What TD1 details do you have?")
                state = run_turn(build_agent(FakeModel([invalid, corrected]), Mock()), "Annual salary $50,000, biweekly.")
                self.assertEqual(state["correction_attempts"], 1)
                self.assertIn("Payroll amounts and estimates", state["trace"][1]["reason"])
                self.assertEqual(state["trace"][-1]["status"], "accepted")

    def test_salary_confirmation_and_general_explanation_are_allowed(self):
        for text in ("Annual salary: $50,000. Frequency: biweekly (26 periods).",
                     "The user supplied a $50,000 annual salary. Calculation is unavailable.",
                     "Gross pay is the amount before deductions. No estimate is available."):
            self.assertIsNone(payroll_amount_violation(text))

    def test_payroll_estimate_cannot_bypass_correction_limit(self):
        invalid = final("clarification", "Gross pay per period: $1,923.08. What TD1 details?")
        state = run_turn(build_agent(FakeModel([invalid, invalid]), Mock()), "Biweekly $50,000 salary.")
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["trace"][-1]["status"], "rejected")

    def test_ordinarily_govern_claim_is_rejected_despite_appended_uncertainty(self):
        for claim in ("For your payment date of 2026-11-13, the January edition would ordinarily govern, but the end date is unknown.",
                      "This edition should apply, although applicability remains unresolved.",
                      "The January edition governs your payment.",
                      "This guidance is valid for 2026-11-13."):
            with self.subTest(claim=claim):
                invalid = final("answer", claim, ["S1"])
                corrected = final("answer", "The edition describes deductions; applicability cannot be confirmed because dates are incomplete [S1].", ["S1"])
                state = run_turn(build_agent(FakeModel([call(), invalid, corrected]), Mock(return_value=evidence())),
                                 "Payment date: 2026-11-13. Explain this guide.")
                self.assertEqual(state["correction_attempts"], 1)
                self.assertEqual(state["trace"][-1]["status"], "accepted")

    def test_qualified_or_confirmed_date_explanations_remain_allowed(self):
        for text in ("I cannot confirm whether this edition would ordinarily govern.",
                     "Applicability cannot be confirmed; the guide describes deductions."):
            self.assertIsNone(applicability_violation(text, evidence()["results"], "2026-11-13"))
        known = [{"effective_from": "2026-01-01", "effective_to": "2026-12-31"}]
        self.assertIsNone(applicability_violation("This edition applies to your payment date.", known, "2026-11-13"))


@unittest.skipUnless(os.getenv("RUN_LIVE_AGENT_TESTS") == "1",
                     "Paid API scenario disabled; requires explicit RUN_LIVE_AGENT_TESTS=1")
class LivePayrollAgentTests(unittest.TestCase):
    """Run only with approved API usage; each scenario builds a fresh agent."""

    def setUp(self):
        self.graph = build_agent(create_model())

    def checked_turn(self, question, history=None):
        state = run_turn(self.graph, question, history)
        outcome = state["outcome"]
        print(json.dumps({
            "input": question, "status": outcome["status"],
            "output": render_answer(outcome),
            "model_calls": state["model_calls"], "searches": state["searches"],
            "correction_attempts": state["correction_attempts"], "trace": state["trace"],
            "payment_date": state["payment_date"], "payment_date_status": state["payment_date_status"],
        }, ensure_ascii=True, indent=2), flush=True)
        self.assertEqual(state["trace"][-1]["status"], "accepted",
                         "A fallback is not a successful model response")
        self.assertLessEqual(state["model_calls"], MAX_MODEL_CALLS)
        self.assertLessEqual(state["searches"], MAX_SEARCHES)
        self.assertIsNone(payroll_amount_violation(outcome["text"]),
                          "A disclaimer cannot excuse a payroll estimate without a calculation tool")
        return state

    def test_salary_clarification_and_followup_context(self):
        question = "What are deductions on a $50,000 annual salary in Ontario?"
        first = self.checked_turn(question)
        self.assertEqual(first["outcome"]["status"], "clarification")
        self.assertIn("?", first["outcome"]["text"])
        self.assert_intake_limits(first["outcome"]["text"])
        history = [HumanMessage(content=question),
                   AIMessage(content=render_answer(first["outcome"]))]
        second = self.checked_turn(
            "Biweekly. Please confirm the annual salary I gave you and tell me what else you need.", history)
        self.assertIn(second["outcome"]["status"], ("clarification", "unsupported"))
        self.assertRegex(second["outcome"]["text"].lower(), r"50[ ,]?000|50\s*k")
        self.assert_intake_limits(second["outcome"]["text"])

    def assert_intake_limits(self, text):
        text = text.lower()
        self.assertRegex(text, r"calculat(?:e|ion|or|ions)")
        self.assertRegex(text, r"not (?:yet )?(?:implemented|available|supported|connected|enabled)|unavailable|cannot (?:calculate|compute)|can't (?:calculate|compute)|do not have|does not (?:yet )?(?:have|include|support)")
        self.assertNotRegex(text, r"\b(?:i|we)\s+(?:can|will)\s+(?:then\s+)?(?:calculate|compute)\b")

    def test_t4_answer_has_evidence_for_both_boxes(self):
        state = self.checked_turn("What do T4 boxes 14 and 22 mean? Cite evidence for each and state edition limitations.")
        outcome = state["outcome"]
        self.assertEqual(outcome["status"], "answer")
        self.assertGreaterEqual(state["searches"], 1)
        # Coverage can come from one chunk or multiple; do not force extra calls.
        text = outcome["text"].lower()
        self.assertRegex(text, r"box(?:es)?\s*14")
        self.assertRegex(text, r"box(?:es)?\s*22|\b22\b")
        self.assertIn("employment income", text)
        self.assertIn("income tax", text)
        cited_content = "\n".join(source["content"] for source in outcome["sources"])
        for number in (14, 22):
            self.assertRegex(cited_content.lower(), rf"box\s+{number}\b")

    def test_employee_payday_is_reported_as_unavailable(self):
        state = self.checked_turn("When is employee Alex's next payday?")
        outcome = state["outcome"]
        self.assertEqual(outcome["status"], "unsupported")
        text = outcome["text"].lower()
        self.assertRegex(text, r"(?:employee|company|payroll|schedule|calendar|record|database)")
        self.assertRegex(text, r"(?:not|cannot|can't|unavailable|don't|do not|no access|unable)")
        self.assertNotRegex(text, r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
        self.assertNotRegex(text, r"(?:what is|provide|tell me)[^.!?\n]{0,80}employee id")

    def test_payment_date_conversation(self):
        question = "What are deductions on a $50,000 annual salary in Ontario?"
        first = self.checked_turn(question)
        self.assertEqual(first["outcome"]["status"], "clarification")
        self.assertIsNone(first["payment_date"])
        self.assertRegex(first["outcome"]["text"].lower(), r"payment date|pay date|payroll date")
        # A labeled format example is allowed; it must never become user context.
        self.assertEqual(first["payment_date_status"], "missing")
        self.assert_intake_limits(first["outcome"]["text"])
        history = [HumanMessage(content=question), AIMessage(content=render_answer(first["outcome"]))]
        second_question = "The payment date is 2026-11-13. Biweekly. Confirm the salary and date we are using."
        second = self.checked_turn(second_question, history)
        self.assertEqual(second["payment_date"], "2026-11-13")
        self.assertIn("2026-11-13", second["outcome"]["text"])
        self.assertRegex(second["outcome"]["text"].lower(), r"50[ ,]?000|50\s*k")
        self.assert_intake_limits(second["outcome"]["text"])
        history.extend([HumanMessage(content=second_question), AIMessage(content=render_answer(second["outcome"]))])
        third = self.checked_turn(
            "Using that same payment date, what does the indexed T4001 guide say about deducting CPP and EI from bonuses? Explain any effective-date uncertainty.", history)
        self.assertEqual(third["payment_date"], "2026-11-13")
        self.assertEqual(third["outcome"]["status"], "answer")
        self.assertTrue(third["outcome"]["sources"])
        self.assertIsNone(document_violation(requested_documents("T4001"), third["outcome"]["sources"]),
                          "The requested T4001 guide cannot be replaced by T4127")
        self.assertTrue(any(source["payment_applicability"] == "effective_dates_incomplete" for source in third["outcome"]["sources"]))
        self.assertIn("applicability remains unresolved", third["outcome"]["text"])
        self.assertIsNone(applicability_violation(third["outcome"]["text"], third["outcome"]["sources"], third["payment_date"]),
                          "An appended warning cannot excuse a definitive applicability claim in the answer")


if __name__ == "__main__":
    unittest.main()
