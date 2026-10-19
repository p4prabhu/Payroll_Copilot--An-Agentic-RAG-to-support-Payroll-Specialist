"""Local stdio MCP server exposing only fictional employee database reads."""

import argparse
from pathlib import Path
import sys
from typing import Annotated, Any, Callable

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


def create_server(db_path: Path = DEFAULT_DB) -> MCPServer:
    repository = EmployeeRepository(db_path)
    repository.check_database()
    server = MCPServer("Fictional Ontario Payroll Records", version="1.0.0",
                       instructions="Read-only fictional records. Resolve ambiguous names before using an employee_id. Dates must be explicit; do not infer today. Money is integer CAD cents. Stored history is partial-year and not CRA-validated. This server cannot calculate payroll or modify records.",
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

    return server


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Existing fictional database; never created or written by the server")
    args = parser.parse_args()
    try:
        server = create_server(args.db)
    except (RepositoryError, ValueError) as error:
        print(f"Payroll MCP startup failed: {error}", file=sys.stderr)
        return 1
    server.run(transport="stdio")  # stdout belongs exclusively to MCP protocol messages.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
