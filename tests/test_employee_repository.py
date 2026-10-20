"""Read-only employee tool semantics and effective-date boundaries."""

from pathlib import Path
from contextlib import closing
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from employee_repository import EmployeeRepository, RepositoryError
from init_employee_db import init_database


class AnnualPayrollRepositoryTests(unittest.TestCase):
    def setUp(self):
        from init_employee_db import FIXTURE_PATH
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "annual.db"
        init_database(self.path, fixture_path=FIXTURE_PATH.with_name("fixtures_t4_2026.json"))
        self.repository = EmployeeRepository(self.path)

    def test_all_employees_reconcile_and_reads_preserve_database(self):
        before = self.path.read_bytes()
        for number, expected_count in enumerate((26, 26, 12, 12, 21), 1):
            with self.subTest(employee=number):
                result = self.repository.get_annual_payroll_summary(f"emp-{number:03d}", 2026)
                self.assertEqual(result["status"], "ok")
                self.assertTrue(result["ready_for_sample_breakdown"])
                self.assertTrue(result["is_complete_calendar_year"])
                self.assertFalse(result["ready_for_filing"])
                self.assertEqual(result["dataset_kind"], "simulated_year_end")
                self.assertEqual(result["totals"]["payment_count"], expected_count)
                boxes, totals = result["box_candidates_cents"], result["totals"]
                self.assertEqual(boxes["14"], totals["employment_income_cents"])
                self.assertEqual(sum(boxes[key] for key in ("16", "16A", "18", "22")), totals["total_deductions_cents"])
                self.assertEqual(boxes["14"] - totals["total_deductions_cents"], totals["net_pay_cents"])
                self.assertIn("employee SIN", result["missing_filing_information"])
        self.assertEqual(self.path.read_bytes(), before)

    def test_priya_salary_change_contribution_caps_and_departed_sam(self):
        priya = self.repository.get_annual_payroll_summary("emp-002", 2026)
        self.assertEqual(priya["box_candidates_cents"]["14"], 22 * 288462 + 4 * 307692)
        self.assertEqual(priya["box_candidates_cents"]["24"], 6890000)
        self.assertGreater(priya["box_candidates_cents"]["16A"], 0)
        self.assertEqual(priya["box_candidates_cents"]["16"], 423045)
        self.assertEqual(priya["box_candidates_cents"]["18"], 112307)
        sam = self.repository.get_annual_payroll_summary("emp-005", 2026)
        self.assertEqual(sam["box_candidates_cents"]["14"], 21 * 153846)
        self.assertEqual(sam["totals"]["payment_count"], 21)
        with closing(sqlite3.connect(self.path)) as connection:
            last = connection.execute("SELECT MAX(payment_date) FROM payroll_history WHERE employee_id = 'emp-005'").fetchone()[0]
            self.assertEqual(last, "2026-10-16")

    def test_known_first_pay_and_per_period_rounding_are_preserved(self):
        with closing(sqlite3.connect(self.path)) as connection:
            rows = dict(connection.execute("SELECT code, amount_cents FROM payroll_lines WHERE payment_id = 'sample-emp-001-01'"))
        self.assertEqual(rows, {"SALARY": 192308, "FED_TAX": 15328, "ON_TAX": 8801, "CPP": 10641, "CPP2": 0, "EI": 3135})
        result = self.repository.get_annual_payroll_summary("emp-001", 2026)
        self.assertEqual(result["box_candidates_cents"]["14"], 5000008)
        self.assertIn("without annual salary true-up", result["assumptions"])

    def test_partial_database_unsupported_year_and_invalid_ids(self):
        partial = self.path.with_name("partial.db")
        init_database(partial)
        result = EmployeeRepository(partial).get_annual_payroll_summary("emp-001", 2026)
        self.assertEqual(result["status"], "incomplete_annual_records")
        self.assertNotIn("box_candidates_cents", result)
        self.assertEqual(self.repository.get_annual_payroll_summary("emp-001", 2025)["status"], "incomplete_annual_records")
        self.assertEqual(self.repository.get_annual_payroll_summary("missing", 2026)["status"], "not_found")
        for bad_year in (True, "2026", 0):
            with self.assertRaises(ValueError):
                self.repository.get_annual_payroll_summary("emp-001", bad_year)

    def test_missing_or_tampered_coverage_returns_no_amounts(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("UPDATE annual_payroll_coverage SET expected_payments_json = '[]' WHERE employee_id = 'emp-001'")
            connection.commit()
        result = self.repository.get_annual_payroll_summary("emp-001", 2026)
        self.assertEqual(result["status"], "incomplete_annual_records")
        self.assertNotIn("box_candidates_cents", result)


class EmployeeRepositoryTests(unittest.TestCase):
    def test_roster_pages_effective_salary_and_active_boundaries_without_writes(self):
        before = self.path.read_bytes()
        first = self.repository.list_employees("2026-10-19", limit=2)
        self.assertEqual(first["match_count"], 4)
        self.assertEqual([r["employee_number"] for r in first["employees"]], ["EMP-1001", "EMP-1002"])
        self.assertEqual(first["employees"][1]["annual_salary_cents"], 7500000)
        self.assertEqual(first["next_offset"], 2)
        second = self.repository.list_employees("2026-10-19", limit=2, offset=2)
        self.assertEqual([r["employee_number"] for r in second["employees"]], ["EMP-1003", "EMP-1004"])
        self.assertFalse(second["has_more"])
        self.assertIsNone(second["next_offset"])
        self.assertEqual(self.repository.list_employees("2026-11-01")["employees"][1]["annual_salary_cents"], 8000000)
        self.assertEqual(self.repository.list_employees("2026-10-15")["match_count"], 5)
        self.assertEqual(self.repository.list_employees("2026-10-16")["match_count"], 4)
        empty = self.repository.list_employees("2026-10-19", offset=100)
        self.assertEqual(empty["employees"], [])
        self.assertEqual(empty["match_count"], 4)
        self.assertEqual(self.repository.list_employees("2027-01-01")["status"], "outside_snapshot")
        for kwargs in ({"limit": True}, {"limit": 101}, {"offset": -1}):
            with self.assertRaises(ValueError):
                self.repository.list_employees("2026-10-19", **kwargs)
        self.assertEqual(before, self.path.read_bytes())
    def test_workforce_effective_dates_filters_and_distinct_counts(self):
        before = self.path.read_bytes()
        summary = self.repository.get_workforce_summary("2026-11-01")
        self.assertEqual(summary["employee_count"], 4)
        self.assertEqual(summary["total_employee_records"], 5)
        self.assertEqual({g["frequency"]: g["employee_count"] for g in summary["pay_frequency_counts"]}, {"biweekly": 2, "monthly": 2})
        self.assertEqual(self.repository.get_workforce_summary("2026-10-15")["employee_count"], 5)
        self.assertEqual(self.repository.get_workforce_summary("2026-10-16")["employee_count"], 4)
        for comparison, minimum, maximum, count in (("gt", 6000000, None, 2), ("gte", 6000000, None, 3),
                                                     ("lt", None, 6000000, 1), ("lte", None, 6000000, 2),
                                                     ("between", 5000000, 6500000, 3)):
            result = self.repository.get_workforce_summary("2026-11-01", salary_comparison=comparison,
                                                         salary_min_cents=minimum, salary_max_cents=maximum)
            self.assertEqual(result["employee_count"], count)
        self.assertEqual(self.repository.get_workforce_summary("2026-10-31", salary_comparison="gt", salary_min_cents=7500000)["employee_count"], 0)
        self.assertEqual(self.repository.get_workforce_summary("2026-11-01", "monthly", "gt", 6000000)["employee_count"], 1)
        self.assertEqual(self.repository.get_workforce_summary("2026-11-02")["status"], "outside_snapshot")
        self.assertEqual(before, self.path.read_bytes())

    def test_workforce_invalid_filters_do_not_become_unfiltered_queries(self):
        for args in ({"salary_min_cents": 1}, {"salary_comparison": "gt"}, {"pay_frequency": "x' OR 1=1--"},
                     {"salary_comparison": "gt", "salary_min_cents": True}, {"salary_comparison": "lt", "salary_max_cents": 1.5},
                     {"salary_comparison": "between", "salary_min_cents": 200, "salary_max_cents": 100}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.repository.get_workforce_summary("2026-11-01", **args)
        with self.assertRaises(ValueError):
            self.repository.get_workforce_summary("2026-02-30")

    def test_movements_use_inclusive_first_inactive_dates(self):
        self.assertEqual(self.repository.get_employment_movements("2026-01-01", "2026-01-01")["joined_employee_count"], 5)
        self.assertEqual(self.repository.get_employment_movements("2026-10-15", "2026-10-15")["ended_employee_count"], 0)
        self.assertEqual(self.repository.get_employment_movements("2026-10-16", "2026-10-16")["ended_employee_count"], 1)
        self.assertEqual(self.repository.get_employment_movements("2026-11-01", "2026-11-02")["status"], "outside_snapshot")
        with self.assertRaises(ValueError):
            self.repository.get_employment_movements("2026-11-01", "2026-01-01")

    def test_rehires_are_distinct_people_and_missing_filter_data_is_not_zero(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("INSERT INTO employments VALUES (?, ?, ?, ?, ?, ?)",
                               ("rehire-sam", "emp-005", "2026-11-01", None, "ON", "CAD"))
            connection.commit()
        movements = self.repository.get_employment_movements("2026-01-01", "2026-11-01")
        self.assertEqual(movements["joined_employee_count"], 5)
        self.assertEqual(movements["join_event_count"], 6)
        result = self.repository.get_workforce_summary("2026-11-01")
        self.assertEqual(result["employee_count"], 5)
        self.assertEqual(result["missing_salary_count"], 1)
        roster = self.repository.list_employees("2026-11-01")
        self.assertEqual(roster["match_count"], 5)
        sam = next(row for row in roster["employees"] if row["employee_id"] == "emp-005")
        self.assertIsNone(sam["annual_salary_cents"])
        self.assertIsNone(sam["frequency"])
        self.assertIn({"frequency": "unknown", "employee_count": 1}, result["pay_frequency_counts"])
        for args in ({"salary_comparison": "gt", "salary_min_cents": 0}, {"pay_frequency": "monthly"}):
            filtered = self.repository.get_workforce_summary("2026-11-01", **args)
            self.assertEqual(filtered["status"], "data_gap")
            self.assertIsNone(filtered["employee_count"])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "records with spaces.db"
        init_database(self.path)
        self.repository = EmployeeRepository(self.path)

    def test_ambiguous_name_is_not_resolved_even_when_page_has_one_match(self):
        result = self.repository.find_employees("Alex", limit=1)
        self.assertEqual(result["status"], "ambiguous")
        self.assertTrue(result["requires_selection"])
        self.assertEqual(result["match_count"], 2)
        self.assertEqual(result["next_offset"], 1)
        second = self.repository.find_employees("Alex", limit=1, offset=1)
        self.assertNotEqual(result["matches"][0]["employee_id"], second["matches"][0]["employee_id"])
        self.assertFalse(second["has_more"])
        self.assertNotIn("compensation", result["matches"][0])

    def test_exact_employee_number_id_and_full_name(self):
        for query in ("emp-001", "EMP-1001", "alex morgan"):
            with self.subTest(query=query):
                result = self.repository.find_employees(query)
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["matches"][0]["employee_id"], "emp-001")

    def test_parameterized_queries_and_literal_wildcards(self):
        for query in ("%", "_", "Alex' OR 1=1 --"):
            self.assertEqual(self.repository.find_employees(query)["match_count"], 0)
        self.assertEqual(self.repository.find_employees("Alex")["match_count"], 2)

    def test_compensation_changes_at_exclusive_end_boundary(self):
        before = self.repository.get_employee("emp-002", "2026-10-31")
        after = self.repository.get_employee("emp-002", "2026-11-01")
        self.assertEqual(before["compensation"]["annual_salary_cents"], 7500000)
        self.assertEqual(after["compensation"]["annual_salary_cents"], 8000000)
        self.assertEqual(after["pay_group"]["frequency"], "biweekly")
        self.assertFalse(after["as_of_after_snapshot"])
        self.assertTrue(self.repository.get_employee("emp-002", "2026-11-02")["as_of_after_snapshot"])

    def test_employment_boundaries_and_unknown_ids(self):
        self.assertEqual(self.repository.get_employee("emp-005", "2026-10-15")["status"], "ok")
        self.assertEqual(self.repository.get_employee("emp-005", "2026-10-16")["status"], "not_employed_on_date")
        self.assertEqual(self.repository.get_employee("emp-001", "2025-12-31")["status"], "not_employed_on_date")
        self.assertEqual(self.repository.get_employee("missing", "2026-11-01")["status"], "not_found")

    def test_schedule_uses_strictly_future_recorded_dates(self):
        result = self.repository.get_payroll_schedule("emp-001", "2026-11-01", limit=1)
        self.assertEqual(result["runs"][0]["payment_date"], "2026-11-13")
        self.assertTrue(result["has_more"])
        same_day = self.repository.get_payroll_schedule("emp-001", "2026-11-13")
        self.assertEqual([r["payment_date"] for r in same_day["runs"]], ["2026-11-27"])
        self.assertEqual(self.repository.get_payroll_schedule("emp-001", "2026-12-01")["status"], "no_scheduled_run")
        self.assertEqual(self.repository.get_payroll_schedule("emp-005", "2026-11-01")["status"], "not_employed_on_date")

    def test_schedule_observes_future_assignment_change(self):
        connection = sqlite3.connect(self.path)
        try:
            with connection:
                connection.execute("UPDATE pay_group_assignments SET valid_to = '2026-11-15' WHERE assignment_id = 'assign-001'")
                connection.execute("INSERT INTO pay_group_assignments VALUES ('transfer', 'job-001', 'group-monthly', '2026-11-15', NULL)")
        finally:
            connection.close()
        result = self.repository.get_payroll_schedule("emp-001", "2026-11-01")
        self.assertEqual([r["payment_date"] for r in result["runs"]], ["2026-11-13", "2026-11-30"])

    def test_schedule_does_not_extend_past_recorded_employment_end(self):
        connection = sqlite3.connect(self.path)
        try:
            with connection:
                connection.execute("UPDATE employments SET ends_before = '2026-11-20' WHERE employment_id = 'job-001'")
        finally:
            connection.close()
        result = self.repository.get_payroll_schedule("emp-001", "2026-11-01")
        self.assertEqual([r["payment_date"] for r in result["runs"]], ["2026-11-13"])

    def test_history_inclusive_dates_and_partial_year_coverage(self):
        result = self.repository.get_payroll_history("emp-001", "2026-10-16", "2026-10-16")
        self.assertEqual(result["match_count"], 1)
        self.assertEqual(result["payments"][0]["payment_id"], "pay-001")
        self.assertTrue(result["request_within_declared_coverage"])
        self.assertFalse(result["is_complete_ytd"])
        all_year = self.repository.get_payroll_history("emp-001", "2026-01-01", "2026-12-31")
        self.assertEqual(all_year["match_count"], 2)
        self.assertFalse(all_year["request_within_declared_coverage"])
        self.assertFalse(all_year["is_complete_ytd"])
        self.assertEqual(all_year["provenance"]["history_coverage_from"], "2026-10-16")

    def test_history_pagination_preserves_stored_amounts_and_lines(self):
        first = self.repository.get_payroll_history("emp-001", "2026-10-01", "2026-11-01", limit=1)
        second = self.repository.get_payroll_history("emp-001", "2026-10-01", "2026-11-01", limit=1, offset=first["next_offset"])
        self.assertEqual(first["payments"][0]["payment_id"], "pay-001")
        self.assertEqual(second["payments"][0]["payment_id"], "pay-004")
        self.assertEqual(first["payments"][0]["gross_cents"], 192308)
        self.assertEqual(len(first["payments"][0]["lines"]), 5)
        self.assertFalse(second["has_more"])

    def test_former_employees_have_history_but_no_future_schedule(self):
        history = self.repository.get_payroll_history("emp-005", "2026-10-01", "2026-11-01")
        self.assertEqual(history["match_count"], 1)
        self.assertEqual(self.repository.get_payroll_history("emp-005", "2026-11-01", "2026-11-30")["status"], "no_recorded_payments")

    def test_invalid_inputs_are_rejected_without_date_inference(self):
        for invalid in ("today", "2026-02-30", "20261101", "2026-11-01T00:00:00Z", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.repository.get_employee("emp-001", invalid)
        for kwargs in ({"limit": True}, {"limit": 0}, {"limit": 101}, {"offset": -1}, {"offset": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.repository.find_employees("Alex", **kwargs)
        with self.assertRaises(ValueError):
            self.repository.get_payroll_history("emp-001", "2026-11-01", "2026-01-01")
        with self.assertRaises(ValueError):
            self.repository.find_employees(" ")

    def test_missing_database_is_not_created_and_error_does_not_expose_path(self):
        path = self.path.parent / "private-missing.db"
        with self.assertRaises(RepositoryError) as error:
            EmployeeRepository(path).find_employees("Alex")
        self.assertNotIn(str(path), str(error.exception))
        self.assertFalse(path.exists())

    def test_unsupported_or_nonfictional_database_is_rejected(self):
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA user_version = 2")
        finally:
            connection.close()
        with self.assertRaisesRegex(RepositoryError, "schema version"):
            self.repository.check_database()
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA user_version = 1")
            connection.execute("PRAGMA ignore_check_constraints = ON")
            connection.execute("UPDATE dataset_info SET is_fictional = 0")
            connection.commit()
        finally:
            connection.close()
        with self.assertRaisesRegex(RepositoryError, "fictional"):
            self.repository.check_database()

    def test_all_reads_preserve_database_bytes(self):
        before = self.path.read_bytes()
        self.repository.find_employees("Alex")
        employee = self.repository.get_employee("emp-001", "2026-11-01")
        self.repository.get_payroll_schedule("emp-001", "2026-11-01")
        self.repository.get_payroll_history("emp-001", "2026-01-01", "2026-11-01")
        self.assertTrue(employee["provenance"]["is_fictional"])
        self.assertFalse(employee["provenance"]["amounts_are_cra_validated"])
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
