"""Read-only employee tool semantics and effective-date boundaries."""

from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from employee_repository import EmployeeRepository, RepositoryError
from init_employee_db import init_database


class EmployeeRepositoryTests(unittest.TestCase):
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
