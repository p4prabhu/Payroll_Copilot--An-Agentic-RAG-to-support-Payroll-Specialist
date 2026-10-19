"""Database integrity and historical-record tests; no network or model required."""

import copy
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from init_employee_db import FIXTURE_PATH, connect_readonly, init_database, load_fixture


class EmployeeDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.path = self.root / "payroll.db"
        self.summary = init_database(self.path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.addCleanup(self.connection.close)

    def changed_fixture(self, change):
        fixture = copy.deepcopy(load_fixture())
        change(fixture)
        path = self.root / "changed.json"
        path.write_text(json.dumps(fixture), encoding="utf-8")
        return path

    def draft_bonus(self, **changes):
        values = dict(self.connection.execute("SELECT * FROM payroll_payments WHERE payment_id = 'pay-001'").fetchone())
        values.update(payment_id="draft-bonus", payment_type="bonus", status="draft", gross_cents=10000, deduction_cents=2000, net_cents=8000)
        values.update(changes)
        columns = list(values)
        self.connection.execute(f"INSERT INTO payroll_payments ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})", list(values.values()))

    def test_seed_counts_and_integrity(self):
        self.assertEqual(self.summary["counts"]["employees"], 5)
        self.assertEqual(self.summary["counts"]["compensation_rates"], 6)
        self.assertEqual(self.summary["counts"]["payroll_payments"], 7)
        self.assertEqual(self.summary["counts"]["payroll_lines"], 35)
        self.assertEqual(self.connection.execute("PRAGMA foreign_key_check").fetchall(), [])
        self.assertEqual(self.connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        self.assertEqual(self.summary["is_fictional"], 1)

    def test_salary_as_of_date_preserves_old_and_new_rates(self):
        query = """SELECT annual_salary_cents FROM compensation_rates
                   WHERE employment_id = ? AND valid_from <= ? AND (valid_to IS NULL OR ? < valid_to)"""
        for as_of, expected in (("2026-10-31", 7500000), ("2026-11-01", 8000000)):
            self.assertEqual(self.connection.execute(query, ("job-002", as_of, as_of)).fetchone()[0], expected)
        historical = self.connection.execute("""SELECT p.gross_cents, c.annual_salary_cents
            FROM payroll_payments p JOIN compensation_rates c USING (compensation_id)
            WHERE p.payment_id = 'pay-005'""").fetchone()
        self.assertEqual(tuple(historical), (288462, 7500000))

    def test_same_first_name_does_not_identify_one_employee(self):
        rows = self.connection.execute("SELECT employee_number FROM employees WHERE given_name = ? ORDER BY employee_number", ("Alex",)).fetchall()
        self.assertEqual([row[0] for row in rows], ["EMP-1001", "EMP-1004"])

    def test_employee_number_is_unique_and_foreign_keys_are_enforced(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.connection.execute("INSERT INTO employees VALUES ('different', 'EMP-1001', 'A', 'B', '2026-01-01T00:00:00Z')")
        with self.assertRaises(sqlite3.IntegrityError):
            self.connection.execute("UPDATE employments SET employee_id = 'missing' WHERE employment_id = 'job-001'")

    def test_money_is_integer_and_negative_amounts_are_rejected(self):
        for amount in (-1, 5000000.25):
            with self.subTest(amount=amount), self.assertRaises(sqlite3.IntegrityError):
                self.connection.execute("UPDATE compensation_rates SET annual_salary_cents = ? WHERE compensation_id = 'comp-002b'", (amount,))
        self.assertEqual(self.connection.execute("SELECT typeof(annual_salary_cents) FROM compensation_rates LIMIT 1").fetchone()[0], "integer")

    def test_overlapping_compensation_and_assignments_are_rejected(self):
        with self.assertRaisesRegex(sqlite3.IntegrityError, "overlapping compensation"):
            self.connection.execute("UPDATE compensation_rates SET valid_from = '2026-10-31' WHERE compensation_id = 'comp-002b'")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "overlapping pay-group"):
            self.connection.execute("INSERT INTO pay_group_assignments VALUES ('overlap', 'job-001', 'group-monthly', '2026-11-01', NULL)")

    def test_posted_payment_and_lines_cannot_be_rewritten_or_deleted(self):
        statements = [
            "UPDATE payroll_payments SET net_cents = 1 WHERE payment_id = 'pay-001'",
            "DELETE FROM payroll_payments WHERE payment_id = 'pay-001'",
            "UPDATE payroll_lines SET amount_cents = 1 WHERE payment_id = 'pay-001'",
            "DELETE FROM payroll_lines WHERE payment_id = 'pay-001'",
            "INSERT INTO payroll_lines VALUES ('pay-001', 9, 'deduction', 'BENEFITS', 1)",
            "UPDATE pay_runs SET payment_date = '2026-10-20' WHERE pay_run_id = 'bw-2026-10-16'",
        ]
        for statement in statements:
            with self.subTest(statement=statement), self.assertRaisesRegex(sqlite3.IntegrityError, "immutable"):
                self.connection.execute(statement)

    def test_posted_compensation_cannot_be_rewritten(self):
        with self.assertRaisesRegex(sqlite3.IntegrityError, "cannot be rewritten"):
            self.connection.execute("UPDATE compensation_rates SET annual_salary_cents = 1 WHERE compensation_id = 'comp-001'")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "cannot invalidate"):
            self.connection.execute("UPDATE compensation_rates SET valid_to = '2026-10-01' WHERE compensation_id = 'comp-001'")

    def test_posted_employment_cannot_be_reassigned_or_invalidated(self):
        with self.assertRaisesRegex(sqlite3.IntegrityError, "cannot be reassigned"):
            self.connection.execute("UPDATE employments SET employee_id = 'emp-005' WHERE employment_id = 'job-001'")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "cannot invalidate"):
            self.connection.execute("UPDATE employments SET ends_before = '2026-10-01' WHERE employment_id = 'job-001'")

    def test_posting_requires_balanced_header_and_lines(self):
        self.draft_bonus()
        self.connection.execute("INSERT INTO payroll_lines VALUES ('draft-bonus', 1, 'earning', 'BONUS', 10000)")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "totals do not match"):
            self.connection.execute("UPDATE payroll_payments SET status = 'posted' WHERE payment_id = 'draft-bonus'")
        self.connection.execute("INSERT INTO payroll_lines VALUES ('draft-bonus', 2, 'deduction', 'BENEFITS', 2000)")
        self.connection.execute("UPDATE payroll_payments SET status = 'posted' WHERE payment_id = 'draft-bonus'")
        self.assertEqual(self.connection.execute("SELECT status FROM payroll_payments WHERE payment_id = 'draft-bonus'").fetchone()[0], "posted")

    def test_posting_rejects_wrong_employee_compensation(self):
        self.draft_bonus(compensation_id="comp-002a")
        self.connection.execute("INSERT INTO payroll_lines VALUES ('draft-bonus', 1, 'earning', 'BONUS', 10000)")
        self.connection.execute("INSERT INTO payroll_lines VALUES ('draft-bonus', 2, 'deduction', 'BENEFITS', 2000)")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "invalid for run"):
            self.connection.execute("UPDATE payroll_payments SET status = 'posted' WHERE payment_id = 'draft-bonus'")

    def test_payment_identity_cannot_be_changed_during_posting(self):
        self.draft_bonus()
        with self.assertRaisesRegex(sqlite3.IntegrityError, "identity"):
            self.connection.execute("UPDATE payroll_payments SET payment_id = 'renamed', status = 'posted' WHERE payment_id = 'draft-bonus'")

    def test_history_is_labeled_partial_not_full_ytd(self):
        coverage = self.connection.execute("SELECT DISTINCT history_complete_from, snapshot_date, is_fictional FROM payroll_history").fetchall()
        self.assertEqual([tuple(row) for row in coverage], [("2026-10-16", "2026-11-01", 1)])
        for row in self.connection.execute("SELECT gross_cents, deduction_cents, net_cents FROM payroll_history"):
            self.assertEqual(row[0] - row[1], row[2])

    def test_payday_query_reads_explicit_calendar_for_given_as_of_date(self):
        # An example query, not yet an MCP tool. Never extrapolate after the last row.
        query = """SELECT MIN(r.payment_date) FROM pay_runs r
          JOIN pay_group_assignments a USING (pay_group_id) JOIN employments e USING (employment_id)
          WHERE e.employee_id = ? AND r.status = 'scheduled' AND r.payment_date > ?
          AND e.starts_on <= r.payment_date AND (e.ends_before IS NULL OR r.payment_date < e.ends_before)
          AND a.valid_from <= r.payment_date AND (a.valid_to IS NULL OR r.payment_date < a.valid_to)"""
        self.assertEqual(self.connection.execute(query, ("emp-001", "2026-11-01")).fetchone()[0], "2026-11-13")
        self.assertIsNone(self.connection.execute(query, ("emp-005", "2026-11-01")).fetchone()[0])
        self.assertIsNone(self.connection.execute(query, ("emp-001", "2026-12-01")).fetchone()[0])

    def test_readonly_connection_rejects_writes_and_missing_database(self):
        reader = connect_readonly(self.path)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                reader.execute("DELETE FROM employees")
        finally:
            reader.close()
        absent = self.root / "absent.db"
        with self.assertRaises(sqlite3.OperationalError):
            connect_readonly(absent)
        self.assertFalse(absent.exists())

    def test_existing_database_is_never_overwritten(self):
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            init_database(self.path)
        self.assertEqual(self.path.read_bytes(), before)

    def test_dry_run_does_not_create_output_or_parent_directory(self):
        target = self.root / "not-created" / "payroll.db"
        result = init_database(target, dry_run=True)
        self.assertEqual(result["mode"], "dry_run")
        self.assertFalse(target.parent.exists())

    def test_bad_fixture_leaves_no_database_or_temporary_files(self):
        fixture = self.changed_fixture(lambda data: data["payroll_lines"][0].update(amount_cents=1))
        target = self.root / "bad.db"
        with self.assertRaisesRegex(sqlite3.IntegrityError, "totals do not match"):
            init_database(target, fixture_path=fixture)
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.glob(".payroll-seed-*")), [])

    def test_fixture_rejects_invalid_dates_float_money_and_unknown_fields(self):
        for change in (lambda data: data["pay_runs"][0].update(payment_date="2026-02-30"),
                       lambda data: data["payroll_lines"][0].update(amount_cents=1.5),
                       lambda data: data["employees"][0].update(sin="not-supported")):
            with self.subTest(change=change):
                fixture = self.changed_fixture(change)
                with self.assertRaises(ValueError):
                    init_database(self.root / "invalid.db", fixture_path=fixture)
                self.assertFalse((self.root / "invalid.db").exists())

    def test_concurrent_target_creation_is_preserved(self):
        target = self.root / "race.db"
        def competing_writer(source, destination):
            Path(destination).write_bytes(b"another creator owns this file")
            raise FileExistsError("target appeared")
        with patch("init_employee_db.os.link", side_effect=competing_writer):
            with self.assertRaises(FileExistsError):
                init_database(target)
        self.assertEqual(target.read_bytes(), b"another creator owns this file")
        self.assertEqual(list(self.root.glob(".payroll-seed-*")), [])


if __name__ == "__main__":
    unittest.main()
