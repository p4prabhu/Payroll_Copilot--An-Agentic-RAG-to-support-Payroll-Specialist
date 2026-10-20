"""Local stdio MCP server exposing only fictional employee database reads."""

import argparse
from pathlib import Path
import sys
from typing import Annotated, Any, Callable, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from employee_repository import EmployeeRepository, RepositoryError
from init_employee_db import DEFAULT_DB


Query = Annotated[str, Field(strict=True, min_length=1, max_length=80)]
EmployeeID = Annotated[str, Field(strict=True, min_length=1, max_length=64)]
ISODate = Annotated[str, Field(strict=True, pattern=r"^\d{4}-\d{2}-\d{2}$")]
Limit = Annotated[int, Field(strict=True, ge=1, le=100)]
Offset = Annotated[int, Field(strict=True, ge=0, le=100000)]
SalaryCents = Annotated[int, Field(strict=True, ge=0, le=100000000000)]


def create_server(db_path: Path = DEFAULT_DB, annual_db_path: Path | None = None) -> MCPServer:
    repository = EmployeeRepository(db_path)
    annual_repository = EmployeeRepository(annual_db_path) if annual_db_path is not None else repository
    repository.check_database()
    server = MCPServer("Fictional Ontario Payroll Records", version="1.0.0",
                       instructions="Read-only fictional records. Resolve ambiguous names before using an employee_id. Dates must be explicit; do not infer today. Money is integer CAD cents. Regular history is not complete YTD. The annual-summary tool can separately return reconciled simulated year-end T4 candidates for an explicit reporting year, without a payment date. It cannot generate a form or file a T4. This server cannot calculate live payroll or modify records.",
                       log_level="WARNING")
    annotations = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)

    def call(method: Callable, *args):
        try:
            return method(*args)
        except (ValueError, RepositoryError) as error:
            raise ToolError(str(error)) from None

    @server.tool(annotations=annotations, structured_output=True)
    def find_employees(query: Query, limit: Limit = 10, offset: Offset = 0) -> dict[str, Any]:
        """Find fictional employees by name prefix or exact employee number/ID. Multiple matches require user selection; names are not identifiers. Follow next_offset if has_more."""
        return call(repository.find_employees, query, limit, offset)

    @server.tool(annotations=annotations, structured_output=True)
    def get_employee(employee_id: EmployeeID, as_of_date: ISODate) -> dict[str, Any]:
        """Read employment, annual salary cents and pay frequency effective on explicit YYYY-MM-DD. Use an ID returned by find_employees. This is a stored record, not a payroll calculation."""
        return call(repository.get_employee, employee_id, as_of_date)

    @server.tool(annotations=annotations, structured_output=True)
    def get_payroll_schedule(employee_id: EmployeeID, as_of_date: ISODate,
                             limit: Limit = 5, offset: Offset = 0) -> dict[str, Any]:
        """Read scheduled company payment dates strictly AFTER explicit as_of_date for this employee's recorded employment/assignments. Never extrapolate a payday. Status reflects the fixture snapshot, not a reconstructed past schedule."""
        return call(repository.get_payroll_schedule, employee_id, as_of_date, limit, offset)

    @server.tool(annotations=annotations, structured_output=True)
    def get_payroll_history(employee_id: EmployeeID, start_date: ISODate, end_date: ISODate,
                            limit: Limit = 20, offset: Offset = 0) -> dict[str, Any]:
        """Read posted fictional payments and lines in an inclusive payment-date range. Check has_more/next_offset and coverage. Amounts are stored examples, not verified deductions or complete YTD totals."""
        return call(repository.get_payroll_history, employee_id, start_date, end_date, limit, offset)

    @server.tool(annotations=annotations, structured_output=True)
    def get_workforce_summary(as_of_date: ISODate,
                              pay_frequency: Literal["weekly", "biweekly", "semimonthly", "monthly"] | None = None,
                              salary_comparison: Literal["any", "gt", "gte", "lt", "lte", "between"] = "any",
                              salary_min_cents: SalaryCents | None = None,
                              salary_max_cents: SalaryCents | None = None) -> dict[str, Any]:
        """Count recorded active employees on an explicit date, optionally filtering effective pay frequency and annual salary in integer CAD cents. gt/gte use minimum; lt/lte use maximum; between uses inclusive bounds. Dates after snapshot and missing filter data cannot establish a count. No arbitrary SQL or payroll arithmetic."""
        return call(repository.get_workforce_summary, as_of_date, pay_frequency, salary_comparison, salary_min_cents, salary_max_cents)

    @server.tool(annotations=annotations, structured_output=True)
    def list_employees(as_of_date: ISODate, limit: Limit = 20, offset: Offset = 0) -> dict[str, Any]:
        """List recorded ACTIVE employees on an explicit date, including names/numbers, effective annual salary CAD cents and pay frequency. Read one bounded page ordered by employee number; follow next_offset while has_more. Missing fields are null. Dates after snapshot are unsupported. This is not a complete HR roster or a payroll calculation."""
        return call(repository.list_employees, as_of_date, limit, offset)

    @server.tool(annotations=annotations, structured_output=True)
    def get_employment_movements(start_date: ISODate, end_date: ISODate) -> dict[str, Any]:
        """Count distinct recorded joiners and employment endings in an inclusive explicit date range. Endings use ends_before, the first inactive date, not the last workday. Separate event counts account for rehires. Do not infer complete company-wide coverage or net headcount change."""
        return call(repository.get_employment_movements, start_date, end_date)

    @server.tool(annotations=annotations, structured_output=True)
    def get_annual_payroll_summary(employee_id: EmployeeID,
                                   reporting_year: Annotated[int, Field(strict=True, ge=1900, le=2100)]) -> dict[str, Any]:
        """Read reconciled SAMPLE annual T4 box candidates for a resolved employee ID and explicit calendar year. Uses the separately configured annual database. No payment date required. Not an actual T4, filing or document generation. Incomplete/unavailable annual records return no amounts; never derive totals from salary or partial history."""
        with repository.connection() as (connection, metadata):
            identity = repository.identity(connection, employee_id)
            unavailable = repository.result(metadata, "incomplete_annual_records", employee=identity,
                reporting_year=reporting_year, is_complete_calendar_year=False,
                ready_for_sample_breakdown=False, ready_for_filing=False,
                limitation="Reconciled annual sample records are unavailable; no T4 amounts were produced")
        if identity is None:
            return {**unavailable, "status": "not_found", "employee_id": employee_id}
        try:
            result = annual_repository.get_annual_payroll_summary(employee_id, reporting_year)
        except RepositoryError:
            return unavailable
        if result.get("employee") != identity:
            raise ToolError("Annual sample identity does not match the resolved employee record")
        result["reporting_year"] = reporting_year
        return result

    return server


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Existing fictional database; never created or written by the server")
    parser.add_argument("--annual-db", type=Path, help="Separate existing annual sample database; never created or written")
    args = parser.parse_args()
    try:
        server = create_server(args.db, args.annual_db)
    except (RepositoryError, ValueError) as error:
        print(f"Payroll MCP startup failed: {error}", file=sys.stderr)
        return 1
    server.run(transport="stdio")  # stdout belongs exclusively to MCP protocol messages.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
