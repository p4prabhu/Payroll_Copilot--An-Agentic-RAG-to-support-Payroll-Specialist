"""Bounded, parameterized reads of the fictional employee dataset."""

from contextlib import contextmanager
from datetime import date
from pathlib import Path
import sqlite3

from init_employee_db import DEFAULT_DB, connect_readonly, validate_annual_reporting


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

    def get_annual_payroll_summary(self, employee_id: str, reporting_year: int) -> dict:
        """Read reconciled sample T4 candidates, never estimate from annual salary.

        This repository operation is deliberately not yet exposed to chat/MCP.
        Inactive employees remain reportable when they were paid during the year.
        """
        employee_id = text_input(employee_id, "employee_id", 64)
        if type(reporting_year) is not int or not 1900 <= reporting_year <= 2100:
            raise ValueError("reporting_year must be an integer year")
        with self.connection() as (connection, metadata):
            identity = self.identity(connection, employee_id)
            if identity is None:
                return self.result(metadata, "not_found", employee_id=employee_id)
            present = connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'annual_payroll_coverage'").fetchone()
            coverage = connection.execute("SELECT * FROM annual_payroll_coverage WHERE employee_id = ? AND reporting_year = ?",
                                          (employee_id, reporting_year)).fetchone() if present else None
            if coverage is None:
                return self.result(metadata, "incomplete_annual_records", employee=identity,
                                   reporting_year=reporting_year, is_complete_calendar_year=False,
                                   ready_for_sample_breakdown=False, ready_for_filing=False,
                                   limitation="No reconciled complete calendar-year payroll coverage; annual salary and partial history are insufficient")
            try:
                validate_annual_reporting(connection, employee_id, reporting_year)
            except (ValueError, TypeError, KeyError):
                return self.result(metadata, "incomplete_annual_records", employee=identity, reporting_year=reporting_year,
                                   is_complete_calendar_year=False, ready_for_sample_breakdown=False, ready_for_filing=False,
                                   limitation="Annual sample records failed completeness or reconciliation checks; no T4 amounts were produced")
            totals = dict(connection.execute("""SELECT COUNT(*) AS payment_count, SUM(p.gross_cents) AS employment_income_cents,
                SUM(p.deduction_cents) AS total_deductions_cents, SUM(p.net_cents) AS net_pay_cents,
                SUM(b.pensionable_earnings_cents) AS pensionable_earnings_cents,
                SUM(b.insurable_earnings_cents) AS insurable_earnings_cents
                FROM payroll_payments p JOIN employments e USING (employment_id)
                JOIN pay_runs r USING (pay_run_id) JOIN payroll_tax_bases b USING (payment_id)
                WHERE e.employee_id = ? AND r.payment_date BETWEEN ? AND ? AND p.status = 'posted'""",
                (employee_id, coverage["coverage_start"], coverage["coverage_end"])).fetchone())
            lines = dict(connection.execute("""SELECT l.code, SUM(l.amount_cents) FROM payroll_lines l
                JOIN payroll_payments p USING (payment_id) JOIN employments e USING (employment_id)
                JOIN pay_runs r USING (pay_run_id) WHERE e.employee_id = ? AND r.payment_date BETWEEN ? AND ?
                AND p.status = 'posted' GROUP BY l.code""", (employee_id, coverage["coverage_start"], coverage["coverage_end"])))
            profile = dict(connection.execute("SELECT dental_benefits_code FROM employee_tax_profiles WHERE employee_id = ? AND reporting_year = ?",
                                              (employee_id, reporting_year)).fetchone())
            return self.result(metadata, "ok", employee=identity, reporting_year=reporting_year, currency="CAD",
                dataset_kind="simulated_year_end", is_complete_calendar_year=True, ready_for_sample_breakdown=True,
                ready_for_filing=False, missing_filing_information=["employee SIN", "employer payroll account", "verified mailing addresses", "payroll review and approval"],
                totals=totals, box_candidates_cents={"14": totals["employment_income_cents"], "16": lines["CPP"],
                    "16A": lines["CPP2"], "18": lines["EI"], "22": lines["FED_TAX"] + lines["ON_TAX"],
                    "24": min(totals["insurable_earnings_cents"], coverage["insurable_ceiling_cents"]),
                    "26": min(totals["pensionable_earnings_cents"], coverage["pensionable_ceiling_cents"])},
                dental_benefits_code=profile["dental_benefits_code"], rules_sha256=coverage["rules_sha256"],
                assumptions=coverage["assumptions"],
                limitation="Reconciled simulated year-end data for sample preparation only; not an actual or filing-ready T4. Net pay is supplementary and is not a T4 box.")

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

    def get_workforce_summary(self, as_of_date: str, pay_frequency: str | None = None,
                              salary_comparison: str = "any", salary_min_cents: int | None = None,
                              salary_max_cents: int | None = None) -> dict:
        """Count distinct active employees using effective salary/pay-group records."""
        as_of_date = iso_date(as_of_date, "as_of_date")
        if pay_frequency not in (None, "weekly", "biweekly", "semimonthly", "monthly"):
            raise ValueError("Unsupported pay frequency")
        if salary_comparison not in ("any", "gt", "gte", "lt", "lte", "between"):
            raise ValueError("Unsupported salary comparison")
        for value in (salary_min_cents, salary_max_cents):
            if value is not None and (type(value) is not int or not 0 <= value <= 100000000000):
                raise ValueError("Salary filters must be nonnegative integer CAD cents")
        required = {"any": (False, False), "gt": (True, False), "gte": (True, False),
                    "lt": (False, True), "lte": (False, True), "between": (True, True)}[salary_comparison]
        if required != (salary_min_cents is not None, salary_max_cents is not None):
            raise ValueError("Salary bounds do not match the comparison")
        if salary_comparison == "between" and salary_min_cents > salary_max_cents:
            raise ValueError("Salary lower bound must not exceed upper bound")
        params = {"date": as_of_date, "frequency": pay_frequency, "minimum": salary_min_cents, "maximum": salary_max_cents}
        predicates = ["1 = 1"]
        if pay_frequency is not None:
            predicates.append("frequency = :frequency")
        comparisons = {"any": "1 = 1", "gt": "annual_salary_cents > :minimum", "gte": "annual_salary_cents >= :minimum",
                       "lt": "annual_salary_cents < :maximum", "lte": "annual_salary_cents <= :maximum",
                       "between": "annual_salary_cents >= :minimum AND annual_salary_cents <= :maximum"}
        predicates.append(comparisons[salary_comparison])
        where = " AND ".join(predicates)  # Only fixed, allowlisted SQL fragments.
        cte = """WITH active AS (
            SELECT employment_id, employee_id FROM employments
            WHERE starts_on <= :date AND (ends_before IS NULL OR :date < ends_before)
        ), effective AS (
            SELECT e.employee_id, c.annual_salary_cents, g.frequency
            FROM active e LEFT JOIN compensation_rates c ON c.employment_id = e.employment_id
              AND c.valid_from <= :date AND (c.valid_to IS NULL OR :date < c.valid_to)
            LEFT JOIN pay_group_assignments a ON a.employment_id = e.employment_id
              AND a.valid_from <= :date AND (a.valid_to IS NULL OR :date < a.valid_to)
            LEFT JOIN pay_groups g ON g.pay_group_id = a.pay_group_id
        ) """
        with self.connection() as (connection, metadata):
            if as_of_date > metadata["snapshot_date"]:
                return self.result(metadata, "outside_snapshot", as_of_date=as_of_date,
                                   limitation="Headcounts after the dataset snapshot are not projected")
            conflict = connection.execute("""SELECT 1 FROM employments e
                LEFT JOIN compensation_rates c ON c.employment_id=e.employment_id
                  AND c.valid_from <= :date AND (c.valid_to IS NULL OR :date < c.valid_to)
                LEFT JOIN pay_group_assignments a ON a.employment_id=e.employment_id
                  AND a.valid_from <= :date AND (a.valid_to IS NULL OR :date < a.valid_to)
                WHERE e.starts_on <= :date AND (e.ends_before IS NULL OR :date < e.ends_before)
                GROUP BY e.employee_id HAVING COUNT(DISTINCT e.employment_id)>1
                  OR COUNT(c.compensation_id)>1 OR COUNT(a.assignment_id)>1 LIMIT 1""", params).fetchone()
            if conflict:
                raise RepositoryError("Conflicting effective-dated workforce records; no summary produced")
            coverage = connection.execute(cte + """SELECT COUNT(*) AS active_count,
                COUNT(*)-COUNT(annual_salary_cents) AS missing_salary_count,
                COUNT(*)-COUNT(frequency) AS missing_pay_group_count FROM effective""", params).fetchone()
            incomplete = (salary_comparison != "any" and coverage["missing_salary_count"] > 0
                          or pay_frequency is not None and coverage["missing_pay_group_count"] > 0)
            count = connection.execute(cte + f"SELECT COUNT(DISTINCT employee_id) FROM effective WHERE {where}", params).fetchone()[0]
            groups = connection.execute(cte + f"""SELECT COALESCE(frequency, 'unknown') AS frequency,
                COUNT(DISTINCT employee_id) AS employee_count FROM effective WHERE {where}
                GROUP BY frequency ORDER BY frequency""", params).fetchall()
            return self.result(metadata, "data_gap" if incomplete else "ok", as_of_date=as_of_date,
                employee_count=None if incomplete else count, unfiltered_active_count=coverage["active_count"],
                total_employee_records=connection.execute("SELECT COUNT(*) FROM employees").fetchone()[0],
                pay_frequency_counts=[dict(row) for row in groups],
                missing_salary_count=coverage["missing_salary_count"], missing_pay_group_count=coverage["missing_pay_group_count"],
                filters={"pay_frequency": pay_frequency, "salary_comparison": salary_comparison,
                         "salary_min_cents": salary_min_cents, "salary_max_cents": salary_max_cents},
                limitation="Counts reflect recorded distinct employees, not complete company-wide HR coverage. Active intervals include starts_on and exclude ends_before. Missing filter data cannot be treated as zero matches.")

    def list_employees(self, as_of_date: str, limit: int = 20, offset: int = 0) -> dict:
        """Page through recorded active employees with salary/pay group effective on date."""
        as_of_date = iso_date(as_of_date, "as_of_date")
        paging(limit, offset)
        with self.connection() as (connection, metadata):
            if as_of_date > metadata["snapshot_date"]:
                return self.result(metadata, "outside_snapshot", as_of_date=as_of_date,
                                   limitation="Employee rosters after the dataset snapshot are not projected")
            params = {"date": as_of_date, "limit": limit, "offset": offset}
            joins = """FROM employments e JOIN employees p ON p.employee_id = e.employee_id
                LEFT JOIN compensation_rates c ON c.employment_id = e.employment_id
                  AND c.valid_from <= :date AND (c.valid_to IS NULL OR :date < c.valid_to)
                LEFT JOIN pay_group_assignments a ON a.employment_id = e.employment_id
                  AND a.valid_from <= :date AND (a.valid_to IS NULL OR :date < a.valid_to)
                LEFT JOIN pay_groups g ON g.pay_group_id = a.pay_group_id
                WHERE e.starts_on <= :date AND (e.ends_before IS NULL OR :date < e.ends_before)"""
            conflict = connection.execute("SELECT 1 " + joins + """ GROUP BY e.employee_id
                HAVING COUNT(DISTINCT e.employment_id)>1 OR COUNT(c.compensation_id)>1
                  OR COUNT(a.assignment_id)>1 LIMIT 1""", params).fetchone()
            if conflict:
                raise RepositoryError("Conflicting effective-dated workforce records; no roster produced")
            total = connection.execute("SELECT COUNT(DISTINCT e.employee_id) " + joins, params).fetchone()[0]
            rows = connection.execute("""SELECT p.employee_id, p.employee_number, p.given_name, p.family_name,
                e.employment_id, e.starts_on, e.ends_before, e.province_code, e.currency,
                c.annual_salary_cents, g.frequency """ + joins +
                " ORDER BY p.employee_number, p.employee_id LIMIT :limit OFFSET :offset", params).fetchall()
            employees = [dict(row) for row in rows]
            more = offset + len(employees) < total
            return self.result(metadata, "ok", as_of_date=as_of_date, employees=employees,
                match_count=total, returned_count=len(employees), limit=limit, offset=offset,
                has_more=more, next_offset=offset + len(employees) if more else None,
                limitation="Recorded active employees only; starts_on is included and ends_before is excluded. Missing salary/pay-group fields remain null, not zero. Salary is stored annual CAD compensation, not a payroll calculation. This roster does not establish complete company-wide HR coverage.")

    def get_employment_movements(self, start_date: str, end_date: str) -> dict:
        """Count distinct recorded joiners/leavers and employment events, inclusively."""
        start_date, end_date = iso_date(start_date, "start_date"), iso_date(end_date, "end_date")
        if start_date > end_date:
            raise ValueError("start_date must not be after end_date")
        with self.connection() as (connection, metadata):
            if end_date > metadata["snapshot_date"]:
                return self.result(metadata, "outside_snapshot", start_date=start_date, end_date=end_date,
                                   limitation="Employment movements after the dataset snapshot are not projected")
            counts = connection.execute("""SELECT
                COUNT(DISTINCT CASE WHEN starts_on BETWEEN :start AND :end THEN employee_id END) AS joined_employee_count,
                COUNT(DISTINCT CASE WHEN ends_before BETWEEN :start AND :end THEN employee_id END) AS ended_employee_count,
                COUNT(CASE WHEN starts_on BETWEEN :start AND :end THEN 1 END) AS join_event_count,
                COUNT(CASE WHEN ends_before BETWEEN :start AND :end THEN 1 END) AS end_event_count
                FROM employments""", {"start": start_date, "end": end_date}).fetchone()
            return self.result(metadata, "ok", start_date=start_date, end_date=end_date, **dict(counts),
                limitation="Inclusive recorded employment-event dates. An ending uses ends_before (the first inactive date), not the last day worked. Rehires may create multiple events per person; these counts are not a net-headcount change or a complete HR roster.")

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
