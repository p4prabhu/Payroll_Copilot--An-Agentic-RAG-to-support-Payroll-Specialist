"""Bounded, parameterized reads of the fictional employee dataset."""

from contextlib import contextmanager
from datetime import date
from pathlib import Path
import sqlite3

from init_employee_db import DEFAULT_DB, connect_readonly


class RepositoryError(RuntimeError):
    """A safe operational message, without SQL, credentials, or filesystem paths."""


def iso_date(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an explicit YYYY-MM-DD date")
    try:
        if date.fromisoformat(value).isoformat() == value:
            return value
    except ValueError:
        pass
    raise ValueError(f"{field} must be an explicit YYYY-MM-DD date")


def text_input(value: str, field: str, maximum: int = 80) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or any(ord(c) < 32 for c in value):
        raise ValueError(f"{field} must contain 1 to {maximum} printable characters")
    return value.strip()


def paging(limit: int, offset: int) -> None:
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer from 1 to 100")
    if type(offset) is not int or not 0 <= offset <= 100000:
        raise ValueError("offset must be an integer from 0 to 100000")


class EmployeeRepository:
    def __init__(self, db_path: Path = DEFAULT_DB):
        self.db_path = Path(db_path)

    @contextmanager
    def connection(self):
        connection = None
        try:
            connection = connect_readonly(self.db_path)
            connection.execute("BEGIN")  # One consistent read snapshot per tool call.
            if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise RepositoryError("Unsupported employee database schema version")
            rows = connection.execute("SELECT * FROM dataset_info").fetchall()
            if len(rows) != 1 or rows[0]["is_fictional"] != 1 or rows[0]["schema_version"] != 1:
                raise RepositoryError("Only the versioned fictional employee dataset is supported")
            metadata = dict(rows[0])
            iso_date(metadata["snapshot_date"], "dataset snapshot")
            iso_date(metadata["history_complete_from"], "history coverage")
            if metadata["history_complete_from"] > metadata["snapshot_date"]:
                raise RepositoryError("Invalid employee database coverage metadata")
            yield connection, metadata
        except sqlite3.Error:
            raise RepositoryError("Employee database unavailable or invalid; initialize the fictional dataset first") from None
        finally:
            if connection is not None:
                connection.close()

    def check_database(self) -> None:
        with self.connection():
            pass

    @staticmethod
    def result(metadata: dict, status: str, **values) -> dict:
        return {"status": status, "provenance": {
            "source": "local_fictional_payroll_database", "is_fictional": True,
            "employer_name": metadata["employer_name"], "dataset_snapshot_date": metadata["snapshot_date"],
            "history_coverage_from": metadata["history_complete_from"],
            "amounts_are_cra_validated": False,
        }, **values}

    @staticmethod
    def identity(connection, employee_id):
        row = connection.execute("SELECT employee_id, employee_number, given_name, family_name FROM employees WHERE employee_id = ?", (employee_id,)).fetchone()
        return dict(row) if row else None

    @staticmethod
    def employment(connection, employee_id, as_of_date):
        rows = connection.execute("""SELECT * FROM employments WHERE employee_id = ?
            AND starts_on <= ? AND (ends_before IS NULL OR ? < ends_before)""",
            (employee_id, as_of_date, as_of_date)).fetchall()
        if len(rows) > 1:
            raise RepositoryError("Conflicting employment records; no employee details were selected")
        return dict(rows[0]) if rows else None

    def find_employees(self, query: str, limit: int = 10, offset: int = 0) -> dict:
        query = text_input(query, "query")
        paging(limit, offset)
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        where = """employee_id = ? COLLATE NOCASE OR employee_number = ? COLLATE NOCASE
            OR given_name LIKE ? ESCAPE '\\' OR family_name LIKE ? ESCAPE '\\'
            OR (given_name || ' ' || family_name) LIKE ? ESCAPE '\\'"""
        params = (query, query, escaped, escaped, escaped)
        with self.connection() as (connection, metadata):
            total = connection.execute(f"SELECT COUNT(*) FROM employees WHERE {where}", params).fetchone()[0]
            rows = connection.execute(f"""SELECT employee_id, employee_number, given_name, family_name
                FROM employees WHERE {where} ORDER BY employee_number LIMIT ? OFFSET ?""", (*params, limit, offset)).fetchall()
            more = offset + len(rows) < total
            return self.result(metadata, "not_found" if total == 0 else "ambiguous" if total > 1 else "ok",
                               matches=[dict(row) for row in rows], match_count=total, requires_selection=total > 1,
                               has_more=more, next_offset=offset + len(rows) if more else None)

    def get_employee(self, employee_id: str, as_of_date: str) -> dict:
        employee_id = text_input(employee_id, "employee_id", 64)
        as_of_date = iso_date(as_of_date, "as_of_date")
        with self.connection() as (connection, metadata):
            identity = self.identity(connection, employee_id)
            if identity is None:
                return self.result(metadata, "not_found", employee_id=employee_id)
            employment = self.employment(connection, employee_id, as_of_date)
            if employment is None:
                return self.result(metadata, "not_employed_on_date", employee=identity, as_of_date=as_of_date)
            rates = connection.execute("""SELECT compensation_id, pay_basis, annual_salary_cents, currency, valid_from, valid_to
                FROM compensation_rates WHERE employment_id = ? AND valid_from <= ? AND (valid_to IS NULL OR ? < valid_to)""",
                (employment["employment_id"], as_of_date, as_of_date)).fetchall()
            groups = connection.execute("""SELECT a.assignment_id, a.valid_from, a.valid_to, g.*
                FROM pay_group_assignments a JOIN pay_groups g USING (pay_group_id)
                WHERE a.employment_id = ? AND a.valid_from <= ? AND (a.valid_to IS NULL OR ? < a.valid_to)""",
                (employment["employment_id"], as_of_date, as_of_date)).fetchall()
            if len(rates) > 1 or len(groups) > 1:
                raise RepositoryError("Conflicting effective-dated employee records")
            return self.result(metadata, "ok" if rates and groups else "data_gap", employee=identity,
                               employment=employment, compensation=dict(rates[0]) if rates else None,
                               pay_group=dict(groups[0]) if groups else None, as_of_date=as_of_date,
                               as_of_after_snapshot=as_of_date > metadata["snapshot_date"])

    def get_payroll_schedule(self, employee_id: str, as_of_date: str, limit: int = 5, offset: int = 0) -> dict:
        employee_id = text_input(employee_id, "employee_id", 64)
        as_of_date = iso_date(as_of_date, "as_of_date")
        paging(limit, offset)
        with self.connection() as (connection, metadata):
            identity = self.identity(connection, employee_id)
            if identity is None:
                return self.result(metadata, "not_found", employee_id=employee_id)
            employment = self.employment(connection, employee_id, as_of_date)
            if employment is None:
                return self.result(metadata, "not_employed_on_date", employee=identity, as_of_date=as_of_date)
            where = """a.employment_id = ? AND r.status = 'scheduled' AND r.payment_date > ?
                AND a.valid_from <= r.payment_date AND (a.valid_to IS NULL OR r.payment_date < a.valid_to)
                AND (? IS NULL OR r.payment_date < ?)"""
            params = (employment["employment_id"], as_of_date, employment["ends_before"], employment["ends_before"])
            joins = "pay_runs r JOIN pay_group_assignments a USING (pay_group_id) JOIN pay_groups g USING (pay_group_id)"
            total = connection.execute(f"SELECT COUNT(*) FROM {joins} WHERE {where}", params).fetchone()[0]
            rows = connection.execute(f"""SELECT r.*, g.timezone FROM {joins} WHERE {where}
                ORDER BY r.payment_date, r.pay_run_id LIMIT ? OFFSET ?""", (*params, limit, offset)).fetchall()
            more = offset + len(rows) < total
            return self.result(metadata, "ok" if total else "no_scheduled_run", employee=identity,
                               as_of_date=as_of_date, date_comparison="payment_date strictly after as_of_date",
                               runs=[dict(row) for row in rows], match_count=total, has_more=more,
                               next_offset=offset + len(rows) if more else None,
                               limitation="Only scheduled rows recorded in this fixture snapshot are returned. No dates are extrapolated; absence is not proof that no pay is owed.")

    def get_payroll_history(self, employee_id: str, start_date: str, end_date: str,
                            limit: int = 20, offset: int = 0) -> dict:
        employee_id = text_input(employee_id, "employee_id", 64)
        start_date, end_date = iso_date(start_date, "start_date"), iso_date(end_date, "end_date")
        if start_date > end_date:
            raise ValueError("start_date must not be after end_date")
        paging(limit, offset)
        with self.connection() as (connection, metadata):
            identity = self.identity(connection, employee_id)
            if identity is None:
                return self.result(metadata, "not_found", employee_id=employee_id)
            params = (employee_id, start_date, end_date)
            where = "employee_id = ? AND payment_date >= ? AND payment_date <= ?"
            total = connection.execute(f"SELECT COUNT(*) FROM payroll_history WHERE {where}", params).fetchone()[0]
            rows = connection.execute(f"""SELECT payment_id, employment_id, pay_run_id, compensation_id,
                payment_type, province_code, currency, gross_cents, deduction_cents, net_cents,
                payment_date, period_start, period_end, status FROM payroll_history WHERE {where}
                ORDER BY payment_date, payment_id LIMIT ? OFFSET ?""", (*params, limit, offset)).fetchall()
            payments = []
            for row in rows:
                lines = connection.execute("SELECT line_number, category, code, amount_cents FROM payroll_lines WHERE payment_id = ? ORDER BY line_number", (row["payment_id"],)).fetchall()
                payments.append({**dict(row), "lines": [dict(line) for line in lines]})
            more = offset + len(rows) < total
            return self.result(metadata, "ok" if total else "no_recorded_payments", employee=identity,
                               start_date=start_date, end_date=end_date, date_comparison="inclusive payment-date range",
                               payments=payments, match_count=total, has_more=more,
                               next_offset=offset + len(rows) if more else None, is_complete_ytd=False,
                               request_within_declared_coverage=metadata["history_complete_from"] <= start_date <= end_date <= metadata["snapshot_date"],
                               limitation="Partial-year fictional stored payments, not calculated or CRA-validated amounts. Do not present these records or page totals as a complete YTD total.")
