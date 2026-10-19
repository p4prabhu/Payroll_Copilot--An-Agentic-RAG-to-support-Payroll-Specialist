"""Create a validated fictional payroll database without replacing existing files."""

import argparse
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
import sqlite3
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "data/schema.sql"
FIXTURE_PATH = ROOT / "data/fixtures.json"
DEFAULT_DB = ROOT / "data/payroll.db"
TABLES = (
    "dataset_info", "employees", "employments", "pay_groups", "pay_group_assignments",
    "compensation_rates", "pay_runs", "payroll_payments", "payroll_lines",
)
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
    if type(data) is not dict or data.get("fixture_version") != 1:
        raise ValueError("Unsupported fixture version")
    if set(data) != {*TABLES, "fixture_version", "notes"}:
        raise ValueError("Fixture must contain exactly the documented tables and metadata")
    for table in TABLES:
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
    if table not in TABLES:
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


def database_summary(connection: sqlite3.Connection) -> dict[str, Any]:
    metadata = dict(zip(("employer_name", "is_fictional", "snapshot_date", "history_complete_from"),
                        connection.execute("SELECT employer_name, is_fictional, snapshot_date, history_complete_from FROM dataset_info").fetchone()))
    return {"schema_version": connection.execute("PRAGMA user_version").fetchone()[0],
            **metadata, "counts": {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in TABLES}}


def populate(connection: sqlite3.Connection, data: dict) -> dict[str, Any]:
    if sqlite3.sqlite_version_info < (3, 37, 0):
        raise RuntimeError("SQLite 3.37 or later is required for STRICT tables")
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    with connection:
        for table in TABLES:
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
    args = parser.parse_args()
    try:
        print(json.dumps(init_database(args.db, dry_run=args.dry_run), indent=2))
    except (ValueError, OSError, sqlite3.Error, RuntimeError) as error:
        print(f"Database initialization failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
