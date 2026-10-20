"""Create a validated fictional payroll database without replacing existing files."""

import argparse
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
import sqlite3
import tempfile
from decimal import Decimal
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "data/schema.sql"
FIXTURE_PATH = ROOT / "data/fixtures.json"
DEFAULT_DB = ROOT / "data/payroll.db"
TABLES = (
    "dataset_info", "employees", "employments", "pay_groups", "pay_group_assignments",
    "compensation_rates", "pay_runs", "payroll_payments", "payroll_lines",
)
ANNUAL_TABLES = ("employer_tax_profiles", "employee_tax_profiles", "annual_payroll_coverage", "payroll_tax_bases")
DATE_FIELDS = {
    "snapshot_date", "history_complete_from", "starts_on", "ends_before",
    "valid_from", "valid_to", "period_start", "period_end", "payment_date",
}
UTC_FIELDS = {"created_at", "recorded_at", "cutoff_at_utc"}


def connect_readonly(path: Path = DEFAULT_DB) -> sqlite3.Connection:
    """Open only an existing database; no creation or write fallback."""
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA query_only = ON")
    return connection


def load_fixture(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(data) is not dict or data.get("fixture_version") not in (1, 2):
        raise ValueError("Unsupported fixture version")
    tables = TABLES + (ANNUAL_TABLES if data["fixture_version"] == 2 else ())
    if set(data) != {*tables, "fixture_version", "notes"}:
        raise ValueError("Fixture must contain exactly the documented tables and metadata")
    for table in tables:
        if type(data[table]) is not list or not data[table]:
            raise ValueError(f"Fixture table {table} must contain records")
        for row in data[table]:
            if type(row) is not dict:
                raise ValueError(f"Invalid fixture record in {table}")
            for field, value in row.items():
                if value is None:
                    continue
                if field in DATE_FIELDS:
                    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
                        raise ValueError(f"Invalid ISO date in {table}.{field}")
                elif field in UTC_FIELDS:
                    if not isinstance(value, str):
                        raise ValueError(f"Invalid UTC timestamp in {table}.{field}")
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    if parsed.utcoffset() != timezone.utc.utcoffset(parsed) or parsed.isoformat(timespec="seconds").replace("+00:00", "Z") != value:
                        raise ValueError(f"Expected canonical UTC timestamp in {table}.{field}")
                elif field.endswith("_cents") and (type(value) is not int or value < 0):
                    raise ValueError(f"Expected nonnegative integer cents in {table}.{field}")
    if len(data["dataset_info"]) != 1 or data["dataset_info"][0].get("is_fictional") != 1:
        raise ValueError("Only the explicitly fictional dataset is supported")
    if any(row.get("status") != "posted" for row in data["payroll_payments"]):
        raise ValueError("Seed history must contain posted payments")
    return data


def insert_rows(connection: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    """Insert known schema columns with parameter values, never string-built values."""
    if table not in TABLES + ANNUAL_TABLES:
        raise ValueError("Unknown fixture table")
    columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
    sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})"
    for row in rows:
        if set(row) != set(columns):
            raise ValueError(f"Fixture columns do not match {table}")
        values = dict(row)
        if table == "payroll_payments":
            values["status"] = "draft"
        connection.execute(sql, [values[column] for column in columns])


def validate_database(connection: sqlite3.Connection) -> None:
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise ValueError("Foreign-key validation failed")
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise ValueError("SQLite integrity validation failed")
    # All effective intervals must fit their employment, including ended employment.
    for table in ("compensation_rates", "pay_group_assignments"):
        invalid = connection.execute(f"""
            SELECT 1 FROM {table} t JOIN employments e USING (employment_id)
            WHERE t.valid_from < e.starts_on
               OR (e.ends_before IS NOT NULL AND (t.valid_to IS NULL OR t.valid_to > e.ends_before))
            LIMIT 1
        """).fetchone()
        if invalid:
            raise ValueError(f"Effective dates fall outside employment in {table}")
    invalid = connection.execute("""
        SELECT 1 FROM pay_runs r CROSS JOIN dataset_info d
        WHERE (r.status = 'posted' AND r.payment_date > d.snapshot_date)
           OR (r.status = 'scheduled' AND r.payment_date <= d.snapshot_date)
        LIMIT 1
    """).fetchone()
    if invalid:
        raise ValueError("Run status conflicts with the fixture snapshot date")
    invalid = connection.execute("""
        SELECT 1 FROM payroll_history
        WHERE payment_date < history_complete_from OR payment_date > snapshot_date LIMIT 1
    """).fetchone()
    if invalid:
        raise ValueError("Payment history falls outside declared fixture coverage")
    if connection.execute("SELECT COUNT(*) FROM annual_payroll_coverage").fetchone()[0]:
        employee_ids = {row[0] for row in connection.execute("SELECT employee_id FROM employees")}
        covered = {row[0] for row in connection.execute("SELECT employee_id FROM annual_payroll_coverage")}
        if employee_ids != covered:
            raise ValueError("Annual sample must cover every stored employee")
        for employee_id in sorted(employee_ids):
            validate_annual_reporting(connection, employee_id, 2026)


def validate_annual_reporting(connection: sqlite3.Connection, employee_id: str, year: int) -> None:
    """Reconcile a closed sample year against every posted payment and calculator input.

    This establishes internal consistency of a simulated dataset, not external HR
    completeness or filing eligibility. Never use coverage metadata alone as proof.
    """
    from payroll_calculator import RULES_PATH, calculate_payroll, load_rules

    coverage = connection.execute("SELECT * FROM annual_payroll_coverage WHERE employee_id = ? AND reporting_year = ?",
                                  (employee_id, year)).fetchone()
    if coverage is None:
        raise ValueError("Complete annual coverage is unavailable")
    columns = [row[1] for row in connection.execute("PRAGMA table_info(annual_payroll_coverage)")]
    coverage = dict(zip(columns, coverage))
    rules, rules_hash = load_rules(RULES_PATH)
    as_cents = lambda value: int(Decimal(value) * 100)
    if (coverage["rules_sha256"] != rules_hash
            or coverage["pensionable_ceiling_cents"] != as_cents(rules["cpp"]["yampe"])
            or coverage["insurable_ceiling_cents"] != as_cents(rules["ei"]["maximum_insurable_earnings"])):
        raise ValueError("Annual sample calculation rules or ceilings do not match")
    metadata = connection.execute("SELECT snapshot_date, history_complete_from FROM dataset_info").fetchone()
    if metadata[0] < coverage["coverage_end"] or metadata[1] > coverage["coverage_start"]:
        raise ValueError("Snapshot does not cover the complete reporting year")
    if not coverage["assumptions"].strip():
        raise ValueError("Annual sample assumptions are missing")
    profile = connection.execute("""SELECT p.mailing_address, p.dental_benefits_code, e.legal_name, e.mailing_address,
        e.dataset_kind FROM employee_tax_profiles p JOIN employer_tax_profiles e USING (reporting_year)
        WHERE p.employee_id = ? AND p.reporting_year = ?""", (employee_id, year)).fetchone()
    if profile is None or not all(str(v).strip() for v in profile) or profile[4] != "simulated_year_end":
        raise ValueError("Annual sample tax profile is missing")
    cursor = connection.execute("""SELECT p.*, r.payment_date, r.period_end, r.status AS run_status,
        c.annual_salary_cents, g.frequency, g.periods_per_year, b.pensionable_earnings_cents,
        b.insurable_earnings_cents, b.taxable_benefits_cents, b.calculation_input_json
        FROM payroll_payments p JOIN employments e USING (employment_id)
        JOIN pay_runs r USING (pay_run_id) JOIN pay_groups g USING (pay_group_id)
        JOIN compensation_rates c USING (compensation_id)
        LEFT JOIN payroll_tax_bases b USING (payment_id)
        WHERE e.employee_id = ? AND r.payment_date BETWEEN ? AND ? ORDER BY r.payment_date, p.payment_id""",
        (employee_id, coverage["coverage_start"], coverage["coverage_end"]))
    rows = [dict(zip([c[0] for c in cursor.description], row)) for row in cursor.fetchall()]
    manifest = json.loads(coverage["expected_payments_json"])
    actual = [{k: row[k] for k in ("payment_id", "pay_run_id", "payment_date")} for row in rows]
    if not rows or manifest != actual:
        raise ValueError("Annual payment manifest does not reconcile; payments are missing or unexpected")
    ytd = {key: 0 for key in ("pensionable_earnings", "cpp", "cpp2", "ei")}
    for index, row in enumerate(rows, 1):
        if (row["status"] != "posted" or row["run_status"] != "posted" or row["payment_type"] != "regular"
                or row["calculation_input_json"] is None
                or row["pensionable_earnings_cents"] != row["gross_cents"]
                or row["insurable_earnings_cents"] != row["gross_cents"] or row["taxable_benefits_cents"] != 0):
            raise ValueError("Annual sample requires posted regular payments with explicit tax bases")
        payload = json.loads(row["calculation_input_json"])
        expected_ytd = {key: format(Decimal(value) / 100, '.2f') for key, value in ytd.items()}
        expected_ytd.update(is_complete=True, employer_scope="current_employer")
        if (payload.get("ytd") != expected_ytd or payload.get("payment_date") != row["payment_date"]
                or payload.get("annual_salary") != format(Decimal(row["annual_salary_cents"]) / 100, '.2f')
                or payload.get("pay_frequency") != row["frequency"]
                or payload.get("pay_periods_per_year") != row["periods_per_year"]
                or payload.get("pay_period_number") != index
                or payload.get("cpp_status") != "eligible_all_year" or payload.get("ei_status") != "insurable"
                or payload.get("federal_claim") != {"mode": "basic_personal"}
                or payload.get("ontario_claim") != {"mode": "basic_personal"}):
            raise ValueError("Annual calculation input does not match salary, dates or sequential YTD")
        result = calculate_payroll(payload)
        if result["status"] != "calculated":
            raise ValueError("Annual sample calculation failed validation")
        amounts = {key: as_cents(value) for key, value in result["amounts"].items()}
        lines = dict(connection.execute("SELECT code, amount_cents FROM payroll_lines WHERE payment_id = ?", (row["payment_id"],)))
        expected_lines = {code: amounts[key] for code, key in
                          (("SALARY", "gross_pay"), ("FED_TAX", "federal_income_tax"), ("ON_TAX", "ontario_income_tax"),
                           ("CPP", "cpp"), ("CPP2", "cpp2"), ("EI", "ei"))}
        if (lines != expected_lines or row["gross_cents"] != amounts["gross_pay"]
                or row["deduction_cents"] != amounts["total_deductions"] or row["net_cents"] != amounts["net_pay"]):
            raise ValueError("Annual payment amounts do not reconcile with lines and calculation")
        ytd["pensionable_earnings"] += amounts["gross_pay"]
        for key in ("cpp", "cpp2", "ei"):
            ytd[key] += amounts[key]


def database_summary(connection: sqlite3.Connection) -> dict[str, Any]:
    metadata = dict(zip(("employer_name", "is_fictional", "snapshot_date", "history_complete_from"),
                        connection.execute("SELECT employer_name, is_fictional, snapshot_date, history_complete_from FROM dataset_info").fetchone()))
    tables = TABLES + (ANNUAL_TABLES if connection.execute("SELECT COUNT(*) FROM annual_payroll_coverage").fetchone()[0] else ())
    return {"schema_version": connection.execute("PRAGMA user_version").fetchone()[0],
            **metadata, "counts": {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables}}


def populate(connection: sqlite3.Connection, data: dict) -> dict[str, Any]:
    if sqlite3.sqlite_version_info < (3, 37, 0):
        raise RuntimeError("SQLite 3.37 or later is required for STRICT tables")
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    with connection:
        for table in TABLES:
            insert_rows(connection, table, data[table])
        if data["fixture_version"] == 2:
            for table in ANNUAL_TABLES:
                insert_rows(connection, table, data[table])
        # Posting validates the fixed fixture totals against the fixed fixture lines.
        # It does not derive tax deductions or calculate a live paycheck.
        connection.execute("UPDATE payroll_payments SET status = 'posted' WHERE status = 'draft'")
        validate_database(connection)
        summary = database_summary(connection)
    return summary


def init_database(path: Path = DEFAULT_DB, *, fixture_path: Path = FIXTURE_PATH,
                  dry_run: bool = False) -> dict[str, Any]:
    data = load_fixture(fixture_path)
    if dry_run:
        connection = sqlite3.connect(":memory:")
        try:
            return {"mode": "dry_run", **populate(connection, data)}
        finally:
            connection.close()
    # Do not resolve an existing symlink into another user's database.
    requested = Path(path).expanduser().absolute()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError("Database already exists; choose a new --db path. Existing data was not changed")
    requested.parent.mkdir(parents=True, exist_ok=True)
    target = requested.parent.resolve() / requested.name
    descriptor, name = tempfile.mkstemp(prefix=".payroll-seed-", suffix=".db", dir=target.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        connection = sqlite3.connect(temporary)
        try:
            summary = populate(connection, data)
        finally:
            connection.close()
        # Same-directory hard link publishes a complete DB and fails if a concurrent
        # creator won the target name. No destructive overwrite or partial target.
        os.link(temporary, target)
        return {"mode": "created", "database": str(target), **summary}
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="New database path; existing files are never replaced")
    parser.add_argument("--dry-run", action="store_true", help="Validate schema and fixtures in memory without creating files")
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH, help="Versioned fixture JSON; defaults to partial-history data")
    args = parser.parse_args()
    try:
        print(json.dumps(init_database(args.db, fixture_path=args.fixture, dry_run=args.dry_run), indent=2))
    except (ValueError, OSError, sqlite3.Error, RuntimeError) as error:
        print(f"Database initialization failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
