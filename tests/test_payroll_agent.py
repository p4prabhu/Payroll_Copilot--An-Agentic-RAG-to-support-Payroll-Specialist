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
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from payroll_agent import MAX_MODEL_CALLS, MAX_SEARCHES, SUBMIT_NAME, TOOL_NAME, applicability_violation, build_agent, calculation_limit_disclosed, capability_violation, confirmation_violation, create_model, document_violation, intake_route, payment_context, payment_date_violation, payroll_amount_violation, render_answer, reporting_year_violation, requested_confirmations, requested_documents, review_metadata_violation, source_applicability, submission_tool, run_turn, year_selection_violation


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


def assert_turn_date_safety(test_case, state):
    """Apply date checks to every accepted live turn, not just the final follow-up."""
    outcome = state["outcome"]
    text, payment = outcome["text"], state["payment_date"]
    test_case.assertIsNone(payment_date_violation(text, payment))
    test_case.assertIsNone(year_selection_violation(text, payment))
    test_case.assertIsNone(applicability_violation(text, outcome["sources"], payment))
    # Independent checks of the exact observed failures, in addition to runtime validators.
    if payment is None:
        test_case.assertNotRegex(text.lower(), r"applicable\s+2026\s+ontario|withheld\s+so\s+far\s+in\s+2026")
    if any(not source.get("effective_from") or not source.get("effective_to") for source in outcome["sources"]):
        test_case.assertNotIn("these tables apply to your november 13, 2026 payment date", text.lower())


def assert_intake_disclosure(test_case, text, history=None):
    """Require a limitation in the visible exchange, and no new capability overclaim."""
    test_case.assertTrue(calculation_limit_disclosed([*(history or []), AIMessage(content=text)]),
                         "Payroll calculation limitations must be disclosed in assistant history or this response")
    test_case.assertIsNone(capability_violation("clarification", text))
    test_case.assertIsNone(payroll_amount_violation(text))


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.seen = []
        self.bindings = []

    def bind_tools(self, tools, **kwargs):
        self.tools = tools
        self.bound_options = kwargs
        self.bindings.append((tools, kwargs))
        return self

    def invoke(self, messages):
        self.seen.append(messages)
        # Each API response is a new message, even when text/arguments repeat.
        return next(self.responses).model_copy(deep=True, update={"id": None})


def build_model_agent(model, retriever, **kwargs):
    """Isolate model-loop unit tests; production routing is tested separately below."""
    return build_agent(model, retriever, route_intake=False, **kwargs)


class PayrollAgentTests(unittest.TestCase):
    def test_document_only_graph_keeps_employee_capabilities_disabled(self):
        model = FakeModel([])
        state = run_turn(build_agent(model), "When is Alex's next payday?")
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertFalse(state["capabilities"]["employee_records"])
        self.assertFalse(state["capabilities"]["mcp_transport"])
        self.assertEqual(model.seen, [])

    def test_submission_tool_is_bound_without_rejected_response_format(self):
        model = FakeModel([])
        build_model_agent(model, Mock())
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
                state = run_turn(build_model_agent(model, Mock()), "Question")
                self.assertEqual(state["outcome"]["status"], "unsupported")
                self.assertEqual(state["trace"][-1]["reason"], "Final response schema validation failed")
                self.assertNotIn("private-text", str(state["trace"]))

    def test_incomplete_response_does_not_execute_partial_tool_call(self):
        response = call()
        response.response_metadata["status"] = "incomplete"
        retriever = Mock()
        state = run_turn(build_model_agent(FakeModel([response]), retriever), "T4 box 22?")
        retriever.assert_not_called()
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["trace"][-1]["reason"], "Model response was incomplete or unsuccessful")

    def test_incomplete_response_rejected_even_with_valid_submission(self):
        response = final("clarification", "What payment date?")
        response.response_metadata["status"] = "incomplete"
        state = run_turn(build_model_agent(FakeModel([response]), Mock()), "Question")
        self.assertEqual(state["trace"][-1]["status"], "rejected")

    def test_search_then_cited_answer_and_review_flags(self):
        retriever = Mock(return_value=evidence())
        model = FakeModel([call(edition_year=2025, document_type="guide"),
                           final("answer", "Box 22 reports income tax deducted [S1].", ["S1"])])
        state = run_turn(build_model_agent(model, retriever), "What is box 22?")
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
        state = run_turn(build_model_agent(model, retriever), "T4 box 22?")
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
        state = run_turn(build_model_agent(model, retriever), "What do T4 boxes 14 and 22 mean?")
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
        state = run_turn(build_model_agent(model, retriever), "Keep searching")
        self.assertEqual(retriever.call_count, MAX_SEARCHES)
        self.assertEqual(state["model_calls"], MAX_MODEL_CALLS)
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["trace"][-2]["status"], "limit_reached")

    def test_parallel_requested_searches_are_executed_within_budget(self):
        calls = [call(i).tool_calls[0] for i in range(7)]
        model = FakeModel([AIMessage(content="", tool_calls=calls), final("answer", "Tax deducted.", ["S1"])])
        retriever = Mock(return_value=evidence())
        state = run_turn(build_model_agent(model, retriever), "Several searches")
        self.assertEqual(retriever.call_count, MAX_SEARCHES)
        self.assertEqual(len(state["evidence"]), 1)

    def test_invalid_tool_arguments_never_reach_retrieval(self):
        retriever = Mock()
        model = FakeModel([call(limit=21), final("unsupported", "Search arguments were invalid.")])
        state = run_turn(build_model_agent(model, retriever), "T4 box 22?")
        retriever.assert_not_called()
        self.assertEqual(state["trace"][1]["status"], "tool_error")

    def test_tool_error_is_sanitized(self):
        retriever = Mock(side_effect=RuntimeError("secret-request-detail"))
        model = FakeModel([call(), final("unsupported", "Search unavailable.")])
        state = run_turn(build_model_agent(model, retriever), "T4 box 22?")
        self.assertNotIn("secret-request-detail", str(state))
        self.assertEqual(state["outcome"]["status"], "unsupported")

    def test_unknown_citation_is_rejected(self):
        rejected = final("answer", "Income tax deducted.", ["S999"])
        model = FakeModel([call(), rejected, rejected])
        state = run_turn(build_model_agent(model, Mock(return_value=evidence())), "T4 box 22?")
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["outcome"]["sources"], [])

    def test_followup_context_and_per_turn_budget_reset(self):
        model = FakeModel([final("clarification", "How often is payroll paid?"),
                           final("clarification", "What is the payment date?")])
        retriever = Mock()
        graph = build_model_agent(model, retriever)
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
        graph = build_model_agent(model, Mock(return_value=evidence()))
        first = run_turn(graph, "Box 22?")
        history = [HumanMessage(content="Box 22?"), AIMessage(content=render_answer(first["outcome"]))]
        second = run_turn(graph, "And this year?", history)
        self.assertEqual(second["evidence"], {})
        self.assertEqual(second["outcome"]["status"], "unsupported")

    def test_malformed_final_and_uncited_answer_are_rejected(self):
        for response in (AIMessage(content="not json"), final("answer", "An unsupported claim.")):
            with self.subTest(response=response.content):
                state = run_turn(build_model_agent(FakeModel([response, response]), Mock()), "Policy question")
                self.assertEqual(state["outcome"]["status"], "unsupported")

    def test_strict_json_is_accepted_only_during_bounded_correction(self):
        reply = AIMessage(content=json.dumps({"status": "clarification", "text": "What date?", "citations": []}))
        state = run_turn(build_model_agent(FakeModel([reply, reply]), Mock()), "Question")
        self.assertEqual(state["trace"][1]["reason"], "A single submit_response tool call is required")
        self.assertEqual(state["outcome"]["status"], "clarification")
        self.assertEqual(state["correction_attempts"], 1)
        self.assertEqual(state["model_calls"], 2)
        self.assertEqual(state["trace"][-1]["format"], "validated_json_repair")

    def test_correction_exposes_submission_tool_only_without_unsupported_tool_choice(self):
        model = FakeModel([AIMessage(content="Plain final"), final("clarification", "What date?")])
        state = run_turn(build_model_agent(model, Mock()), "Question")
        self.assertEqual(len(model.bindings), 2)
        self.assertEqual([tool["function"]["name"] for tool in model.bindings[1][0]], [SUBMIT_NAME])
        self.assertNotIn("tool_choice", model.bindings[1][1])
        self.assertEqual(state["trace"][-1]["format"], "submit_response")

    def test_correction_search_call_is_never_executed(self):
        retriever = Mock()
        state = run_turn(build_model_agent(FakeModel([AIMessage(content="Plain"), call()]), retriever), "Question")
        retriever.assert_not_called()
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["searches"], 0)

    def test_json_repair_rejects_prose_fences_wrong_types_and_extra_fields(self):
        valid = {"status": "clarification", "text": "What date?", "citations": []}
        for content in ("```json\n" + json.dumps(valid) + "\n```",
                        "Here is the result: " + json.dumps(valid),
                        json.dumps({**valid, "extra": "private-value"}),
                        json.dumps({**valid, "text": 123}),
                        json.dumps({"status": "clarification", "text": "What date?"})):
            with self.subTest(content=content):
                state = run_turn(build_model_agent(FakeModel([AIMessage(content="Plain"), AIMessage(content=content)]), Mock()), "Question")
                self.assertEqual(state["outcome"]["status"], "unsupported")
                self.assertEqual(state["trace"][-1]["status"], "rejected")
                self.assertEqual(state["model_calls"], 2)

    def test_json_repair_retains_citation_and_calculation_checks(self):
        for text, citations in (("A claim [S999].", ["S999"]),
                                ("Gross pay per period: $1,923.08 [S1].", ["S1"])):
            with self.subTest(text=text):
                repaired = AIMessage(content=json.dumps({"status": "answer", "text": text, "citations": citations}))
                state = run_turn(build_model_agent(FakeModel([call(), AIMessage(content="Plain"), repaired]),
                                             Mock(return_value=evidence())), "Question")
                self.assertEqual(state["outcome"]["status"], "unsupported")
                self.assertEqual(state["trace"][-1]["status"], "rejected")

    def test_clarification_can_explain_retrieved_guidance_and_ask_a_question(self):
        text = "Box 22 reports income tax deducted [S1]. Which reporting year are you asking about?"
        state = run_turn(build_model_agent(FakeModel([call(), final("clarification", text, ["S1"])]),
                                     Mock(return_value=evidence())), "Explain box 22 for my reporting year.")
        self.assertEqual(state["outcome"]["status"], "clarification")
        self.assertEqual(state["outcome"]["citations"], ["S1"])
        self.assertEqual(state["outcome"]["sources"][0]["pdf_page_number"], 18)
        self.assertIn("Source content review is pending", render_answer(state["outcome"]))
        self.assertEqual(state["correction_attempts"], 0)

    def test_observed_cited_clarification_can_be_repaired_as_json(self):
        text = "Annual salary: $50,000. Payment date: 2026-11-13. Biweekly. The retrieved guide describes deductions [S1]. Payroll calculation is unavailable. What TD1 information is available?"
        repaired = AIMessage(content=json.dumps({"status": "clarification", "text": text, "citations": ["S1"]}))
        state = run_turn(build_model_agent(FakeModel([call(), AIMessage(content="Plain"), repaired]),
                                     Mock(return_value=evidence())), "Payment date: 2026-11-13. Confirm $50,000 salary and biweekly frequency.")
        self.assertEqual(state["outcome"]["status"], "clarification")
        self.assertEqual(state["payment_date"], "2026-11-13")
        self.assertEqual(state["trace"][-1]["format"], "validated_json_repair")

    def test_cited_clarification_cannot_bypass_date_or_document_checks(self):
        for kind in ("expired", "wrong_document", "amount", "unknown_citation", "uncertain_dates"):
            with self.subTest(kind=kind):
                saved = evidence()
                saved["results"][0]["document_id"] = "t4001-25e"
                text = "This guide describes deductions [S1]. What TD1 details are available?"
                citations = ["S1"]
                question = "Payment date: 2026-11-13. Explain T4001."
                if kind == "expired":
                    saved["results"][0].update(effective_from="2025-01-01", effective_to="2025-12-31")
                elif kind == "wrong_document":
                    saved["results"][0]["document_id"] = "t4127-01-26e"
                elif kind == "amount":
                    text = "Gross pay per period: $1,923.08. What TD1 details?"
                elif kind == "unknown_citation":
                    citations = ["S999"]
                else:
                    text = "The edition would ordinarily govern. What TD1 details?"
                invalid = final("clarification", text, citations)
                state = run_turn(build_model_agent(FakeModel([call(), invalid, invalid]), Mock(return_value=saved)), question)
                self.assertEqual(state["outcome"]["status"], "unsupported")
                self.assertEqual(state["trace"][-1]["status"], "rejected")

    def test_clarification_must_still_ask_a_question(self):
        invalid = final("clarification", "The salary was confirmed.")
        state = run_turn(build_model_agent(FakeModel([invalid, invalid]), Mock()), "Confirm salary")
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["trace"][-1]["reason"], "Clarification must ask a specific question")

    def test_submission_cannot_be_combined_with_search(self):
        combined = AIMessage(content="", tool_calls=[call().tool_calls[0], final("answer", "Claim.", ["S1"]).tool_calls[0]])
        retriever = Mock()
        state = run_turn(build_model_agent(FakeModel([combined, combined]), retriever), "Question")
        retriever.assert_not_called()
        self.assertEqual(state["outcome"]["status"], "unsupported")

    def test_submission_ends_turn_without_extra_model_or_search_call(self):
        model = FakeModel([final("clarification", "What payment date?")])
        retriever = Mock()
        state = run_turn(build_model_agent(model, retriever), "Calculate deductions")
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
                state = run_turn(build_model_agent(model, Mock(return_value=evidence())), "Box 22?")
                self.assertEqual(state["outcome"]["status"], expected)

    def test_plain_clarification_can_be_corrected_without_retrieval(self):
        retriever = Mock()
        model = FakeModel([AIMessage(content="How often are you paid?"),
                           final("clarification", "How often are you paid?")])
        state = run_turn(build_model_agent(model, retriever), "Deductions on $50,000?")
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
        state = run_turn(build_model_agent(model, Mock()), "Alex's next payday?")
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
        state = run_turn(build_model_agent(model, retriever), "Box 22?")
        self.assertEqual(state["outcome"]["status"], "answer")
        self.assertEqual(state["outcome"]["sources"][0]["chunk_id"], "test-chunk")
        self.assertEqual(state["correction_attempts"], 1)
        self.assertEqual(retriever.call_count, 1)
        self.assertIn('Available source IDs: ["S1"]', json.loads(model.seen[2][-1].content)["feedback"])

    def test_repeated_invalid_responses_stop_after_one_correction(self):
        model = FakeModel([AIMessage(content="plain response"), AIMessage(content="still plain")])
        state = run_turn(build_model_agent(model, Mock()), "Question")
        self.assertEqual(state["model_calls"], 2)
        self.assertEqual(state["correction_attempts"], 1)
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["trace"][-1]["status"], "rejected")

    def test_correction_cannot_exceed_model_call_budget(self):
        model = FakeModel([*[call(i) for i in range(MAX_MODEL_CALLS - 1)], AIMessage(content="invalid final")])
        state = run_turn(build_model_agent(model, Mock(return_value=evidence())), "Question")
        self.assertEqual(state["model_calls"], MAX_MODEL_CALLS)
        self.assertEqual(state["correction_attempts"], 0)
        self.assertEqual(state["outcome"]["status"], "unsupported")

    def test_correction_budget_resets_for_followup_turn(self):
        model = FakeModel([AIMessage(content="plain"), final("clarification", "What frequency?"),
                           AIMessage(content="plain again"), final("clarification", "What payment date?")])
        graph = build_model_agent(model, Mock())
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
        state = run_turn(build_model_agent(FakeModel([invalid, corrected]), retriever), "Alex's next payday?")
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
                state = run_turn(build_model_agent(FakeModel([invalid, corrected]), Mock()), "Deductions on $50,000?")
                self.assertEqual(state["outcome"]["status"], "clarification")
                self.assertEqual(state["correction_attempts"], 1)
                self.assertIn("do not promise to calculate", state["trace"][1]["reason"])
                self.assertEqual(state["trace"][-1]["status"], "accepted")

    def test_honest_limits_allow_intake_and_document_explanations(self):
        text = "I cannot calculate payroll deductions; the calculation tool is not implemented. I can explain how deductions are calculated from official guidance. What is the intended payment date?"
        state = run_turn(build_model_agent(FakeModel([final("clarification", text)]), Mock()), "Salary intake")
        self.assertEqual(state["outcome"]["status"], "clarification")
        self.assertEqual(state["correction_attempts"], 0)

    def test_cited_document_cannot_enable_net_pay_or_employee_access(self):
        for claim in ("Your net pay is $1,500.", "I can retrieve the employee records."):
            with self.subTest(claim=claim):
                invalid = final("answer", claim, ["S1"])
                state = run_turn(build_model_agent(FakeModel([call(), invalid, invalid]), Mock(return_value=evidence())), "Payroll question")
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
        state = run_turn(build_model_agent(model, Mock()), "Keep the same date", history)
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
                state = run_turn(build_model_agent(FakeModel([final("clarification", text), corrected]), Mock()), "Salary intake")
                self.assertEqual(state["correction_attempts"], 1)
                self.assertEqual(state["outcome"]["status"], "clarification")
                self.assertIsNone(state["payment_date"])

    def test_frequency_table_selection_requires_user_payment_date(self):
        model = FakeModel([call(document_type="deduction_table", pay_periods_per_year=26),
                           final("clarification", "What is the payment date in YYYY-MM-DD format?")])
        retriever = Mock()
        state = run_turn(build_model_agent(model, retriever), "Ontario biweekly deduction table")
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
        state = run_turn(build_model_agent(model, Mock(return_value=evidence())), "Payment date: 2026-11-13. What is box 22?")
        self.assertEqual(state["outcome"]["sources"][0]["payment_applicability"], "effective_dates_incomplete")
        self.assertIn("applicability remains unresolved", state["outcome"]["text"])
        tool_data = json.loads(model.seen[1][-1].content)
        self.assertEqual(tool_data["results"][0]["payment_applicability"], "effective_dates_incomplete")

    def test_answer_cannot_rely_on_source_outside_known_dates(self):
        saved = evidence()
        saved["results"][0].update(effective_from="2025-01-01", effective_to="2025-12-31")
        model = FakeModel([call(), final("answer", "Use this guidance [S1].", ["S1"]),
                           final("unsupported", "The retrieved guide has expired for your intended payment date.")])
        state = run_turn(build_model_agent(model, Mock(return_value=saved)), "Payment date: 2026-11-13. Which guidance applies?")
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["correction_attempts"], 1)
        self.assertIn("conflicts with known effective-date bounds", state["trace"][3]["reason"])

    def test_unknown_effective_dates_cannot_support_definitive_application_claim(self):
        invalid = final("answer", "Both deductions apply to your payment date of 2026-11-13 under 2026 guidance.", ["S1"])
        corrected = final("answer", "The guide describes deductions. I cannot confirm whether these rules apply to your payment date because effective dates are incomplete [S1].", ["S1"])
        state = run_turn(build_model_agent(FakeModel([call(), invalid, corrected]), Mock(return_value=evidence())),
                         "Payment date: 2026-11-13. Explain this guide.")
        self.assertEqual(state["correction_attempts"], 1)
        self.assertIn("Effective dates are incomplete", state["trace"][3]["reason"])
        self.assertEqual(state["trace"][-1]["status"], "accepted")
        self.assertEqual(state["outcome"]["status"], "answer")
        self.assertIn("applicability remains unresolved", state["outcome"]["text"])

    def test_unknown_date_overclaim_cannot_bypass_correction_limit(self):
        invalid = final("answer", "These rules apply to your payment date of 2026-11-13.", ["S1"])
        state = run_turn(build_model_agent(FakeModel([call(), invalid, invalid]), Mock(return_value=evidence())),
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
        state = run_turn(build_model_agent(model, retriever), "What does T4001 say?")
        self.assertEqual(retriever.call_count, 1)
        self.assertEqual(retriever.call_args.kwargs["document_id"], "t4001-25e")
        self.assertEqual(state["trace"][1]["status"], "tool_error")
        self.assertEqual(state["outcome"]["status"], "answer")

    def test_unavailable_requested_guide_never_calls_retriever(self):
        retriever = Mock()
        model = FakeModel([call(), final("unsupported", "The requested T9999 publication is not available in the catalog.")])
        state = run_turn(build_model_agent(model, retriever), "Explain T9999.")
        retriever.assert_not_called()
        self.assertEqual(state["outcome"]["status"], "unsupported")

    def test_retriever_cannot_leak_substitute_publication_into_evidence(self):
        saved = evidence()
        saved["results"][0]["document_id"] = "t4127-01-26e"
        invalid = final("answer", "Use another guide [S1].", ["S1"])
        state = run_turn(build_model_agent(FakeModel([call(), invalid,
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
        state = run_turn(build_model_agent(model, retriever), "Use T4127-01-26e.")
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
                state = run_turn(build_model_agent(FakeModel([invalid, corrected]), Mock()), "Annual salary $50,000, biweekly.")
                self.assertEqual(state["correction_attempts"], 1)
                self.assertIn("Payroll amounts and estimates", state["trace"][1]["reason"])
                self.assertEqual(state["trace"][-1]["status"], "accepted")

    def test_salary_confirmation_and_general_explanation_are_allowed(self):
        for text in ("Annual salary: $50,000. Frequency: biweekly (26 periods).",
                     "The user supplied a $50,000 annual salary. Calculation is unavailable.",
                     "Gross pay is the amount before deductions. No estimate is available."):
            self.assertIsNone(payroll_amount_violation(text))

    def test_annual_salary_near_deduction_words_is_not_a_payroll_estimate(self):
        for text in ("I can explain payroll deductions for a $50,000 annual salary. What TD1 details are available?",
                     "Gross pay is based on the $50,000 annual salary you supplied. What other details are available?",
                     "Tax deductions depend on the $50,000 annual salary and TD1 information. What details are available?"):
            with self.subTest(text=text):
                state = run_turn(build_model_agent(FakeModel([final("clarification", text)]), Mock()),
                                 "Confirm my annual salary of $50,000.")
                self.assertEqual(state["trace"][-1]["status"], "accepted")
                self.assertEqual(state["correction_attempts"], 0)
                self.assertEqual(state["outcome"]["text"], text)

    def test_formatted_amount_assignments_remain_blocked(self):
        for text in ("**Gross pay per period:** approximately **$1,923.08**.",
                     "Gross pay (biweekly) is about $1,923.08.",
                     "EI deductions total 100 dollars.",
                     "Net pay would be $1,500.00."):
            with self.subTest(text=text):
                self.assertIsNotNone(payroll_amount_violation(text))

    def test_rejected_draft_diagnostics_are_opt_in_and_never_become_the_answer(self):
        text = "Gross pay per period: $1,923.08. What TD1 details?"
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                model = FakeModel([final("clarification", text), final("clarification", text)])
                state = run_turn(build_model_agent(model, Mock(), diagnostics=enabled), "Salary intake")
                rejected = [event for event in state["trace"] if event["event"] == "response_validation"]
                self.assertEqual(state["outcome"]["status"], "unsupported")
                self.assertNotIn("1,923.08", state["outcome"]["text"])
                for event in rejected:
                    self.assertEqual("rejected_draft" in event, enabled)
                    if enabled:
                        self.assertEqual(event["rejected_draft"]["text"], text)
                        self.assertEqual(event["rejected_draft"]["kind"], "submit_response")

    def test_diagnostics_redact_secrets_and_exclude_provider_metadata(self):
        secret = "test-secret-value-do-not-log"
        response = AIMessage(content="Plain final response: " + secret,
                             additional_kwargs={"reasoning": "PRIVATE_REASONING"},
                             response_metadata={"status": "completed", "internal": "PRIVATE_METADATA"})
        corrected = final("clarification", "What is the payment date?")
        with patch.dict(os.environ, {"PAYROLL_TEST_API_KEY": secret}):
            model = FakeModel([response, corrected])
            state = run_turn(build_model_agent(model, Mock(), diagnostics=True), "Salary intake")
        encoded = json.dumps(state["trace"])
        self.assertIn("[REDACTED]", encoded)
        self.assertNotIn(secret, encoded)
        self.assertNotIn("PRIVATE_REASONING", encoded)
        self.assertNotIn("PRIVATE_METADATA", encoded)

    def test_diagnostics_do_not_dump_invalid_tool_arguments_or_content_blocks(self):
        malformed = final("unsupported", "Gap")
        malformed.tool_calls[0]["args"]["extra"] = "PRIVATE_ARGUMENT"
        blocks = AIMessage(content=[{"type": "text", "text": "PRIVATE_BLOCK"}])
        incomplete = AIMessage(content="PRIVATE_INCOMPLETE", response_metadata={"status": "incomplete"})
        for response in (malformed, blocks, incomplete):
            with self.subTest(response=response):
                model = FakeModel([response, final("unsupported", "Unable to answer.")])
                state = run_turn(build_model_agent(model, Mock(), diagnostics=True), "Question")
                encoded = json.dumps(state["trace"])
                for private in ("PRIVATE_ARGUMENT", "PRIVATE_BLOCK", "PRIVATE_INCOMPLETE"):
                    self.assertNotIn(private, encoded)

    def test_format_correction_also_explains_amount_constraint(self):
        model = FakeModel([AIMessage(content="Plain salary confirmation"),
                           final("clarification", "Your salary is $50,000 annually. Calculation is unavailable. What TD1 details?")])
        state = run_turn(build_model_agent(model, Mock()), "Annual salary $50,000 intake")
        feedback = model.seen[1][-1].content
        self.assertIn("Do not include any derived gross pay per period", feedback)
        self.assertEqual(state["trace"][-1]["status"], "accepted")

    def test_payroll_estimate_cannot_bypass_correction_limit(self):
        invalid = final("clarification", "Gross pay per period: $1,923.08. What TD1 details?")
        state = run_turn(build_model_agent(FakeModel([invalid, invalid]), Mock()), "Biweekly $50,000 salary.")
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
                state = run_turn(build_model_agent(FakeModel([call(), invalid, corrected]), Mock(return_value=evidence())),
                                 "Payment date: 2026-11-13. Explain this guide.")
                self.assertEqual(state["correction_attempts"], 1)
                self.assertEqual(state["trace"][-1]["status"], "accepted")

    def test_qualified_or_confirmed_date_explanations_remain_allowed(self):
        for text in ("I cannot confirm whether this edition would ordinarily govern.",
                     "Applicability cannot be confirmed; the guide describes deductions."):
            self.assertIsNone(applicability_violation(text, evidence()["results"], "2026-11-13"))
        known = [{"effective_from": "2026-01-01", "effective_to": "2026-12-31"}]
        self.assertIsNone(applicability_violation("This edition applies to your payment date.", known, "2026-11-13"))

    def test_written_date_applicability_claims_are_rejected_in_each_response_status(self):
        for status in ("answer", "clarification", "unsupported"):
            for claim in ("These tables apply to your November 13, 2026 payment date.",
                          "The January edition covers your November 2026 payment date.",
                          "The effective start date aligns with your payment date of 2026-11-13.",
                          "These tables cover November 2026."):
                with self.subTest(status=status, claim=claim):
                    invalid = final(status, claim + " What other details are available?", ["S1"])
                    corrected = final("answer", "The edition describes deductions, but applicability cannot be confirmed [S1].", ["S1"])
                    state = run_turn(build_model_agent(FakeModel([call(), invalid, corrected]), Mock(return_value=evidence())),
                                     "Payment date: 2026-11-13. Explain this guide.")
                    self.assertEqual(state["correction_attempts"], 1)
                    self.assertEqual(state["trace"][-1]["status"], "accepted")

    def test_written_date_claim_cannot_bypass_correction_limit(self):
        invalid = final("answer", "These tables apply to your November 13, 2026 payment date.", ["S1"])
        state = run_turn(build_model_agent(FakeModel([call(), invalid, invalid]), Mock(return_value=evidence())),
                         "Payment date: 2026-11-13. Explain this guide.")
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["trace"][-1]["status"], "rejected")

    def test_missing_date_prevents_year_selection_but_retains_salary(self):
        for claim in ("I can retrieve and explain the applicable 2026 Ontario and federal deduction tables.",
                      "How much gross pay and income tax have been withheld so far in 2026?",
                      "Once you provide the payment date, I can search for the correct 2026 biweekly deduction tables."):
            with self.subTest(claim=claim):
                invalid = final("clarification", "Annual salary: $50,000, biweekly. " + claim + " What is the payment date?")
                corrected = final("clarification", "Your annual salary is $50,000, paid biweekly. What is the payment date in YYYY-MM-DD? Payroll calculation is unavailable.")
                state = run_turn(build_model_agent(FakeModel([invalid, corrected]), Mock()), "Confirm $50,000 annual salary and biweekly frequency.")
                self.assertEqual(state["correction_attempts"], 1)
                self.assertIsNone(state["payment_date"])
                self.assertIn("do not select a tax year", state["trace"][1]["reason"])
                self.assertEqual(state["trace"][-1]["status"], "accepted")

    def test_date_examples_and_edition_descriptions_do_not_choose_a_tax_year(self):
        for text in ("What is your payment date? For example: 2026-10-31. I can retrieve applicable deduction tables after you provide it.",
                     "The 2025 edition describes bonuses. Applicability to a payment date is unconfirmed.",
                     "The source is effective from January 1, 2026; its end date is unknown.",
                     "What are your TD1 amounts and year-to-date contributions? Please provide a payment date first."):
            with self.subTest(text=text):
                self.assertIsNone(year_selection_violation(text, None))
                self.assertIsNone(applicability_violation(text, evidence()["results"], None))
        self.assertIsNone(year_selection_violation("I can retrieve the 2026 Ontario deduction tables.", "2026-11-13"))

    def test_date_qualifications_and_known_bounds_remain_accepted(self):
        for text in ("I cannot confirm whether these tables apply to your November 13, 2026 payment date.",
                     "Edition year does not prove these tables apply to your November 13, 2026 payment date.",
                     "The source describes deductions; applicability cannot be confirmed for November 2026."):
            with self.subTest(text=text):
                self.assertIsNone(applicability_violation(text, evidence()["results"], "2026-11-13"))
        known = [{"effective_from": "2026-01-01", "effective_to": "2026-12-31"}]
        self.assertIsNone(applicability_violation("These tables apply to your November 13, 2026 payment date.", known, "2026-11-13"))

    def test_live_turn_date_checks_catch_both_recorded_failures(self):
        cases = [("These tables apply to your November 13, 2026 payment date.", "2026-11-13", evidence()["results"]),
                 ("I can retrieve applicable 2026 Ontario deduction tables.", None, []),
                 ("How much income tax was withheld so far in 2026?", None, [])]
        for text, payment, sources in cases:
            with self.subTest(text=text):
                state = {"payment_date": payment, "outcome": {"text": text, "sources": sources}}
                with self.assertRaises(AssertionError):
                    assert_turn_date_safety(self, state)

    def test_json_repair_cannot_bypass_year_selection_check(self):
        draft = AIMessage(content=json.dumps({"status": "clarification", "text": "I can retrieve applicable 2026 Ontario deduction tables. What payment date?", "citations": []}))
        state = run_turn(build_model_agent(FakeModel([AIMessage(content="Plain"), draft]), Mock()), "Salary guidance")
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertIn("do not select a tax year", state["trace"][-1]["reason"])

    def test_general_deduction_followup_is_not_a_date_applicability_claim(self):
        text = "Confirmed annual salary: $50,000. Payment date: 2026-11-13. Biweekly. I retrieved the 2026 Ontario tables [S1], effective January 1, 2026. Once you provide the CC code, I can explain which deductions would apply, but I cannot compute exact deduction amounts. What is the employee's TD1 claim code?"
        state = run_turn(build_model_agent(FakeModel([call(), final("clarification", text, ["S1"])]),
                                     Mock(return_value=evidence())), "Annual salary $50,000. Payment date: 2026-11-13. Confirm salary and date.")
        self.assertEqual(state["outcome"]["status"], "clarification")
        self.assertEqual(state["correction_attempts"], 0)
        self.assertIn("applicability remains unresolved", state["outcome"]["text"])

    def test_general_apply_language_is_allowed_but_publication_validity_claims_are_not(self):
        for text in ("I can explain which deductions would apply.",
                     "Which deductions should apply to this salary?",
                     "I can describe how withholding rules would apply to a hypothetical bonus."):
            with self.subTest(text=text):
                self.assertIsNone(applicability_violation(text, evidence()["results"], "2026-11-13"))
        for text in ("These tables would apply.", "The edition should apply.",
                     "The 2026 guidance will normally apply.",
                     "These deductions would apply to your November 13, 2026 payment date."):
            with self.subTest(text=text):
                self.assertIsNotNone(applicability_violation(text, evidence()["results"], "2026-11-13"))

    def test_disclosed_limitation_allows_short_salary_followup(self):
        first_question = "Deductions on a $50,000 annual salary?"
        first_text = "Payroll calculation is not yet implemented. What is your payment date and pay frequency?"
        followup = "I confirm the annual salary: $50,000, and pay frequency: biweekly. What is the payment date in YYYY-MM-DD?"
        model = FakeModel([final("clarification", first_text), final("clarification", followup)])
        graph = build_model_agent(model, Mock())
        first = run_turn(graph, first_question)
        assert_intake_disclosure(self, first["outcome"]["text"])
        history = [HumanMessage(content=first_question), AIMessage(content=render_answer(first["outcome"]))]
        second = run_turn(graph, "Biweekly. Confirm the salary.", history)
        assert_intake_disclosure(self, second["outcome"]["text"], history)
        self.assertTrue(second["calculation_limit_disclosed"])
        self.assertIn('CALCULATION LIMIT DISCLOSED IN COMPLETED HISTORY: true', model.seen[1][0].content)
        self.assertEqual(second["trace"][-1]["status"], "accepted")

    def test_undisclosed_intake_still_fails_the_disclosure_check(self):
        text = "Your salary is $50,000. What is the payment date?"
        for history in ([], [HumanMessage(content="Payroll calculation is unavailable."), AIMessage(content="Noted.")],
                        [HumanMessage(content="Question"), AIMessage(content="I can calculate your payroll deductions.")]):
            with self.subTest(history=history):
                with self.assertRaises(AssertionError):
                    assert_intake_disclosure(self, text, history)

    def test_previous_disclosure_never_enables_promises_or_estimates(self):
        history = [HumanMessage(content="Question"), AIMessage(content="Payroll calculation is not implemented.")]
        for text in ("I can calculate your deductions. What TD1 details?",
                     "Gross pay per period is $1,923.08. What TD1 details?"):
            with self.subTest(text=text):
                with self.assertRaises(AssertionError):
                    assert_intake_disclosure(self, text, history)

    def test_disclosure_uses_retained_completed_history_only(self):
        history = [HumanMessage(content="Salary intake"), AIMessage(content="Payroll calculation is unavailable.")]
        for number in range(7):
            history.extend([HumanMessage(content=f"Followup {number}"), AIMessage(content="Noted.")])
        model = FakeModel([final("clarification", "Payroll calculation is unavailable. What is your payment date?")])
        state = run_turn(build_model_agent(model, Mock()), "Salary intake", history)
        self.assertFalse(state["calculation_limit_disclosed"])
        self.assertFalse(calculation_limit_disclosed([HumanMessage(content="Payroll calculation is unavailable.")]))
        rejected = final("clarification", "Payroll calculation is unavailable. What date?")
        self.assertFalse(calculation_limit_disclosed([rejected]))

    def test_correction_cannot_drop_requested_salary_date_or_frequency(self):
        question = "Payment date: 2026-11-13. Biweekly. Confirm salary, date, and pay frequency."
        history = [HumanMessage(content="What are deductions on a $50,000 annual salary?"),
                   AIMessage(content="Payroll calculation is unavailable. What date and frequency?")]
        initial = AIMessage(content="Confirmed $50,000 annual salary, payment date 2026-11-13, biweekly.")
        corrected = final("clarification", "Confirmed $50,000 annual salary, payment date 2026-11-13, biweekly. What TD1 information do you have?")
        model = FakeModel([initial, corrected])
        state = run_turn(build_model_agent(model, Mock()), question, history)
        self.assertEqual(state["requested_confirmations"], {"annual_salary": "50000", "payment_date": "2026-11-13", "pay_frequency": "biweekly"})
        self.assertEqual(state["trace"][-1]["status"], "accepted")
        feedback = model.seen[1][-1].content
        for value in ("50000", "2026-11-13", "biweekly"):
            self.assertIn(value, feedback)

    def test_recorded_missing_salary_response_is_rejected(self):
        history = [HumanMessage(content="What are deductions on a $50,000 annual salary in Ontario?"), AIMessage(content="What is the date?")]
        missing = final("clarification", "To retrieve applicable tables for the 2026-11-13 payment date, I need TD1 details. Do you have the form?")
        corrected = final("clarification", "Confirmed annual salary $50,000 and payment date 2026-11-13. What TD1 details do you have?")
        state = run_turn(build_model_agent(FakeModel([missing, corrected]), Mock()),
                         "The payment date is 2026-11-13. Biweekly. Confirm the salary and date we are using.", history)
        self.assertEqual(state["correction_attempts"], 1)
        self.assertIn("annual_salary=50000", state["trace"][1]["reason"])
        self.assertEqual(state["trace"][-1]["status"], "accepted")

    def test_confirmation_omission_after_correction_fails_closed(self):
        history = [HumanMessage(content="Annual salary $50,000"), AIMessage(content="Noted.")]
        missing = final("clarification", "Payment date: 2026-11-13. What TD1 information?")
        state = run_turn(build_model_agent(FakeModel([AIMessage(content="Plain response"), missing]), Mock()),
                         "Payment date: 2026-11-13. Confirm salary and date.", history)
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertIn("annual_salary=50000", state["trace"][-1]["reason"])

    def test_confirmation_context_ignores_assistant_values_and_ambiguous_user_amounts(self):
        question = "Confirm salary, date and frequency."
        messages = [HumanMessage(content="Annual salary $50k, biweekly."),
                    AIMessage(content="Annual salary $60,000, monthly."),
                    HumanMessage(content=question)]
        required = requested_confirmations(messages, question, "2026-11-13")
        self.assertEqual(required["annual_salary"], "50000")
        self.assertEqual(required["pay_frequency"], "biweekly")
        updated = [*messages, HumanMessage(content="Annual salary is now $60,000, paid monthly.")]
        self.assertEqual(requested_confirmations(updated, question, None), {"annual_salary": "60000", "pay_frequency": "monthly"})
        ambiguous = [*messages, HumanMessage(content="Annual salary is $50,000 or $60,000.")]
        self.assertNotIn("annual_salary", requested_confirmations(ambiguous, question, None))
        self.assertEqual(requested_confirmations(messages, "Explain T4001", None), {})

    def test_confirmation_checks_equivalent_formats_and_missing_fields(self):
        required = {"annual_salary": "50000", "payment_date": "2026-11-13", "pay_frequency": "biweekly"}
        self.assertIsNone(confirmation_violation("Annual salary $50k; payment date 2026-11-13; bi-weekly.", required))
        for text in ("Annual salary $60,000; payment date 2026-11-13; biweekly.",
                     "Annual salary $50,000; payment date 2026-11-20; biweekly.",
                     "Annual salary $50,000; payment date 2026-11-13; monthly."):
            self.assertIsNotNone(confirmation_violation(text, required))

    def test_internal_review_metadata_cannot_be_attributed_to_cra(self):
        for claim in ("This guidance is pending CRA review and may require updates.",
                      "The guide is awaiting Canada Revenue Agency review.",
                      "The CRA has flagged this edition for potential update review.",
                      "This review status comes from the CRA."):
            with self.subTest(claim=claim):
                invalid = final("answer", claim + " [S1]", ["S1"])
                corrected = final("answer", "The guide describes payroll rules [S1]. Our project's internal content review is pending; that is not CRA review.", ["S1"])
                state = run_turn(build_model_agent(FakeModel([call(), invalid, corrected]), Mock(return_value=evidence())), "Explain this guide.")
                self.assertEqual(state["correction_attempts"], 1)
                self.assertEqual(state["trace"][-1]["status"], "accepted")
                self.assertIn("Review flags belong", state["trace"][3]["reason"])

    def test_internal_review_explanations_and_tool_provenance_are_allowed(self):
        for text in ("Internal content review is pending in our project.",
                     "This flag does not mean pending CRA review.",
                     "CRA publishes the guide; our project marks it for internal review."):
            self.assertIsNone(review_metadata_violation(text))
        model = FakeModel([call(), final("answer", "Box 22 reports tax deducted [S1].", ["S1"])])
        state = run_turn(build_model_agent(model, Mock(return_value=evidence())), "Explain box 22.")
        output = json.loads(model.seen[1][-1].content)
        self.assertEqual(output["review_metadata_origin"], "internal_project")
        self.assertIn("within this project", state["outcome"]["text"])

    def test_review_metadata_and_confirmations_are_checked_in_json_repair(self):
        for draft in ({"status": "answer", "text": "Annual salary: $50,000. Pending CRA review [S1].", "citations": ["S1"]},
                      {"status": "clarification", "text": "What payment date?", "citations": []}):
            with self.subTest(draft=draft):
                history = [HumanMessage(content="Annual salary $50,000."), AIMessage(content="Noted.")]
                model = FakeModel([call(), AIMessage(content="Plain"), AIMessage(content=json.dumps(draft))])
                state = run_turn(build_model_agent(model, Mock(return_value=evidence())), "Confirm salary.", history)
                self.assertEqual(state["outcome"]["status"], "unsupported")
                self.assertEqual(state["trace"][-1]["status"], "rejected")


class DeterministicWorkflowTests(unittest.TestCase):
    def test_new_money_amounts_are_rejected_in_document_answers_and_json_repairs(self):
        for money in ("your gross per pay period is approximately $1,923",
                      "your paycheck would be 1,923 CAD", "the amount is 1923 dollars"):
            with self.subTest(money=money):
                text = money + " [S1]."
                invalid = final("answer", text, ["S1"])
                repaired = AIMessage(content=json.dumps({"status": "answer", "text": text, "citations": ["S1"]}))
                state = run_turn(build_agent(FakeModel([call(), invalid, repaired]), Mock(return_value=evidence())),
                                 "Explain general deduction guidance for an annual salary of $50,000.")
                self.assertEqual(state["outcome"]["status"], "unsupported")
                self.assertIn("New currency amounts require", state["trace"][-1]["reason"])

    def test_salary_intake_and_confirmation_do_not_invoke_model_or_search(self):
        model, retriever = FakeModel([]), Mock()
        graph = build_agent(model, retriever)
        question = "What are deductions on a $50,000 annual salary in Ontario?"
        first = run_turn(graph, question)
        self.assertEqual(first["outcome"]["status"], "clarification")
        self.assertIn("YYYY-MM-DD", first["outcome"]["text"])
        self.assertNotIn("2026", first["outcome"]["text"])
        history = [HumanMessage(content=question), AIMessage(content=render_answer(first["outcome"]))]
        second = run_turn(graph, "The payment date is 2026-11-13. Biweekly. Confirm salary and date.", history)
        text = second["outcome"]["text"]
        for value in ("$50,000", "2026-11-13", "biweekly"):
            self.assertIn(value, text)
        self.assertNotIn("1,923", text)
        self.assertEqual(second["model_calls"], 0)
        self.assertEqual(second["searches"], 0)
        self.assertEqual(second["correction_attempts"], 0)
        self.assertEqual(second["trace"][-1]["status"], "accepted")
        self.assertEqual(model.seen, [])
        retriever.assert_not_called()

    def test_short_frequency_followup_never_suggests_a_payday(self):
        history = [HumanMessage(content="Annual salary $50,000, what are deductions?"),
                   AIMessage(content="Payroll calculation is unavailable. What date and frequency?")]
        state = run_turn(build_agent(FakeModel([]), Mock()), "Biweekly. Confirm annual salary.", history)
        self.assertIn("$50,000", state["outcome"]["text"])
        self.assertIn("YYYY-MM-DD", state["outcome"]["text"])
        self.assertNotRegex(state["outcome"]["text"], r"\b\d{4}-\d{2}-\d{2}\b|October|next payday")
        self.assertEqual(state["model_calls"], 0)

    def test_explicit_calculation_requests_explain_missing_tool_without_arithmetic(self):
        for question in ("Calculate deductions on my $50,000 annual salary.",
                         "Estimate my net pay.", "What is my take-home pay?"):
            with self.subTest(question=question):
                model, retriever = FakeModel([]), Mock()
                state = run_turn(build_agent(model, retriever), question)
                self.assertEqual(state["outcome"]["status"], "unsupported")
                self.assertIn("not implemented", state["outcome"]["text"])
                self.assertEqual(state["model_calls"], 0)
                self.assertEqual(model.seen, [])
                retriever.assert_not_called()

    def test_next_payday_requires_calendar_data(self):
        for question in ("When is employee Alex's next payday?", "When is my next pay date?"):
            with self.subTest(question=question):
                state = run_turn(build_agent(FakeModel([]), Mock()), question)
                self.assertEqual(state["outcome"]["status"], "unsupported")
                self.assertNotRegex(state["outcome"]["text"], r"\d{4}|\b(?:Monday|Friday|October)\b")
                self.assertIn("calendar", state["outcome"]["text"])
                self.assertEqual(state["model_calls"], 0)

    def test_document_followup_after_intake_still_uses_retrieval(self):
        saved = evidence()
        saved["results"][0]["document_id"] = "t4001-25e"
        model = FakeModel([call(), final("answer", "The guide describes bonus deductions [S1].", ["S1"])])
        retriever = Mock(return_value=saved)
        graph = build_agent(model, retriever)
        question = "What are deductions on a $50,000 annual salary?"
        first = run_turn(graph, question)
        history = [HumanMessage(content=question), AIMessage(content=render_answer(first["outcome"]))]
        second_question = "Payment date: 2026-11-13. Biweekly. Confirm salary and date."
        second = run_turn(graph, second_question, history)
        history.extend([HumanMessage(content=second_question), AIMessage(content=render_answer(second["outcome"]))])
        third = run_turn(graph, "Using that same payment date, what does T4001 say about bonuses?", history)
        self.assertEqual(third["outcome"]["status"], "answer")
        self.assertEqual(third["model_calls"], 2)
        self.assertEqual(third["searches"], 1)
        self.assertEqual(third["payment_date"], "2026-11-13")
        self.assertEqual(retriever.call_args.kwargs["document_id"], "t4001-25e")

    def test_how_to_calculate_is_a_document_explanation_not_a_personal_calculation(self):
        for question in ("Explain how CPP is calculated.", "How do I calculate tax using the T4001 guide?",
                         "What do T4 boxes 14 and 22 mean?"):
            self.assertIsNone(intake_route(question, []))
        model = FakeModel([call(), final("answer", "The guide describes tax deductions [S1].", ["S1"])])
        state = run_turn(build_agent(model, Mock(return_value=evidence())), "Explain how payroll deductions are calculated.")
        self.assertEqual(state["searches"], 1)
        self.assertEqual(state["outcome"]["status"], "answer")

    def test_invalid_or_ambiguous_dates_ask_for_clarification_without_assuming(self):
        history = [HumanMessage(content="Annual salary $50,000, biweekly. Payment date: 2026-11-13."), AIMessage(content="Noted.")]
        for question in ("Payment date is 2026-02-30.", "Payment date is 2026-11-13 or 2026-11-20.", "Payment date is tomorrow."):
            with self.subTest(question=question):
                state = run_turn(build_agent(FakeModel([]), Mock()), question, history)
                self.assertIsNone(state["payment_date"])
                self.assertIn("YYYY-MM-DD", state["outcome"]["text"])
                self.assertNotIn("Payment date: 2026-11-13", state["outcome"]["text"])

    def test_ambiguous_salary_does_not_choose_one_amount(self):
        state = run_turn(build_agent(FakeModel([]), Mock()), "Annual salary is $50,000 or $60,000, paid biweekly.")
        self.assertNotIn("annual_salary", state["intake_context"])
        self.assertNotIn("$50,000", state["outcome"]["text"])
        self.assertIn("one annual salary amount", state["outcome"]["text"])

    def test_reporting_year_needs_explicit_evidence_not_an_edition_label(self):
        claim = "The 2025 edition applies to T4 reporting for the 2024 calendar year."
        self.assertIsNotNone(reporting_year_violation(claim, evidence()["results"]))
        source = {"content": "Use these instructions for the 2024 calendar year."}
        self.assertIsNone(reporting_year_violation(claim, [source]))
        self.assertIsNone(reporting_year_violation("This is the 2025 edition; reporting-year applicability is unconfirmed.", evidence()["results"]))

    def test_unsupported_reporting_year_is_rejected_in_document_flow(self):
        invalid = final("answer", "This guide applies to T4 reporting for the 2024 calendar year [S1].", ["S1"])
        corrected = final("answer", "The indexed edition describes box 22. Reporting-year applicability is unconfirmed [S1].", ["S1"])
        state = run_turn(build_agent(FakeModel([call(), invalid, corrected]), Mock(return_value=evidence())), "Explain T4 box 22.")
        self.assertEqual(state["correction_attempts"], 1)
        self.assertIn("reporting-year coverage requires", state["trace"][3]["reason"])
        self.assertEqual(state["trace"][-1]["status"], "accepted")


@unittest.skipUnless(os.getenv("RUN_LIVE_AGENT_TESTS") == "1",
                     "Paid API scenario disabled; requires explicit RUN_LIVE_AGENT_TESTS=1")
class LivePayrollAgentTests(unittest.TestCase):
    """Run only with approved API usage; each scenario builds a fresh agent."""

    def setUp(self):
        # Explicit test-only diagnostics retain rejected final drafts in the log.
        self.graph = build_agent(create_model(), diagnostics=True)

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
        assert_turn_date_safety(self, state)
        self.assertIsNone(confirmation_violation(outcome["text"], state["requested_confirmations"]))
        self.assertIsNone(review_metadata_violation(outcome["text"]))
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
        self.assert_intake_limits(second["outcome"]["text"], history)

    def assert_intake_limits(self, text, history=None):
        assert_intake_disclosure(self, text, history)

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
        self.assert_intake_limits(second["outcome"]["text"], history)
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
