"""Discover and call all four employee tools through a real MCP subprocess."""

import argparse
import asyncio
import json
from pathlib import Path
import sys

from mcp import Client
from mcp.client.stdio import StdioServerParameters

from init_employee_db import DEFAULT_DB, ROOT


TOOL_NAMES = {"find_employees", "get_employee", "get_payroll_schedule", "get_payroll_history"}


def server_parameters(db_path: Path = DEFAULT_DB) -> StdioServerParameters:
    return StdioServerParameters(command=sys.executable,
        args=[str(ROOT / "scripts/payroll_mcp_server.py"), "--db", str(Path(db_path).resolve())],
        cwd=str(ROOT), env={"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})


async def run_smoke(db_path: Path = DEFAULT_DB) -> dict:
    async with asyncio.timeout(60):
        async with Client(server_parameters(db_path), read_timeout_seconds=20) as client:
            listed = await client.list_tools()
            names = {tool.name for tool in listed.tools}
            if names != TOOL_NAMES:
                raise RuntimeError("MCP discovery did not return the four approved read-only tools")
            calls = [
                ("find_employees", {"query": "Alex"}),
                ("get_employee", {"employee_id": "emp-001", "as_of_date": "2026-11-01"}),
                ("get_payroll_schedule", {"employee_id": "emp-001", "as_of_date": "2026-11-01"}),
                ("get_payroll_history", {"employee_id": "emp-001", "start_date": "2026-01-01", "end_date": "2026-11-01"}),
            ]
            outputs = []
            for name, arguments in calls:
                result = await client.call_tool(name, arguments)
                if result.is_error or not isinstance(result.structured_content, dict):
                    raise RuntimeError(f"MCP smoke call failed: {name}")
                outputs.append({"tool": name, "arguments": arguments, "result": result.structured_content})
            records = {item["tool"]: item["result"] for item in outputs}
            checks = {
                "discovered_exactly_four_tools": names == TOOL_NAMES,
                "alex_requires_selection": records["find_employees"]["status"] == "ambiguous",
                "salary_is_stored_cents": records["get_employee"]["compensation"]["annual_salary_cents"] == 5000000,
                "next_date_is_recorded": records["get_payroll_schedule"]["runs"][0]["payment_date"] == "2026-11-13",
                "history_not_complete_ytd": records["get_payroll_history"]["is_complete_ytd"] is False,
                "every_result_is_fictional": all(item["result"]["provenance"]["is_fictional"] for item in outputs),
            }
            if not all(checks.values()):
                raise RuntimeError("The database did not match the documented fictional smoke scenario")
            return {"status": "passed", "transport": "stdio", "tools": sorted(names), "checks": checks, "calls": outputs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    try:
        result = asyncio.run(run_smoke(args.db))
    except Exception:
        print("MCP smoke test failed. Check the initialized fictional database and server stderr.", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
