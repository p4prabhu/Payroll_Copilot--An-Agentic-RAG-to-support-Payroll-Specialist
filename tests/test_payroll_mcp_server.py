"""MCP contracts in-process and across a real stdio subprocess; no model API."""

import asyncio
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mcp import Client
from init_employee_db import init_database
from payroll_mcp_server import create_server
from smoke_payroll_mcp import TOOL_NAMES, run_smoke, server_parameters


class PayrollMCPTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="payroll mcp ")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "fictional records.db"
        init_database(self.path)

    async def test_discovery_contracts_and_readonly_annotations(self):
        async with Client(create_server(self.path)) as client:
            result = await client.list_tools()
        self.assertEqual({tool.name for tool in result.tools}, TOOL_NAMES)
        for tool in result.tools:
            self.assertTrue(tool.annotations.read_only_hint)
            self.assertTrue(tool.annotations.idempotent_hint)
            self.assertFalse(tool.annotations.destructive_hint)
            self.assertFalse(tool.annotations.open_world_hint)
            self.assertIsNotNone(tool.output_schema)
            self.assertNotIn("db_path", tool.input_schema["properties"])
            if tool.name == "get_employee":
                self.assertEqual(set(tool.input_schema["required"]), {"employee_id", "as_of_date"})

    async def test_structured_tool_outputs_not_found_and_ambiguous_states(self):
        async with Client(create_server(self.path)) as client:
            ambiguous = await client.call_tool("find_employees", {"query": "Alex", "limit": 1})
            missing = await client.call_tool("get_employee", {"employee_id": "unknown", "as_of_date": "2026-11-01"})
            employee = await client.call_tool("get_employee", {"employee_id": "emp-001", "as_of_date": "2026-11-01"})
        self.assertFalse(ambiguous.is_error)
        self.assertEqual(ambiguous.structured_content["status"], "ambiguous")
        self.assertTrue(ambiguous.structured_content["requires_selection"])
        self.assertEqual(missing.structured_content["status"], "not_found")
        self.assertIs(type(employee.structured_content["compensation"]["annual_salary_cents"]), int)

    async def test_missing_invalid_dates_and_wrong_types_are_tool_errors(self):
        async with Client(create_server(self.path)) as client:
            errors = []
            for args in ({"employee_id": "emp-001"},
                         {"employee_id": "emp-001", "as_of_date": "today"},
                         {"employee_id": "emp-001", "as_of_date": "2026-02-30"},
                         {"employee_id": 123, "as_of_date": "2026-11-01"}):
                errors.append(await client.call_tool("get_employee", args))
            errors.append(await client.call_tool("find_employees", {"query": "Alex", "limit": True}))
            errors.append(await client.call_tool("get_payroll_history", {"employee_id": "emp-001", "start_date": "2026-11-01", "end_date": "2026-01-01"}))
        for error in errors:
            self.assertTrue(error.is_error)

    async def test_no_write_tool_and_calls_leave_database_unchanged(self):
        before = self.path.read_bytes()
        async with Client(create_server(self.path)) as client:
            rejected = await client.call_tool("update_salary", {"employee_id": "emp-001", "amount": 1})
            history = await client.call_tool("get_payroll_history", {"employee_id": "emp-001", "start_date": "2026-01-01", "end_date": "2026-12-31", "limit": 1})
        self.assertTrue(rejected.is_error)
        self.assertTrue(history.structured_content["has_more"])
        self.assertFalse(history.structured_content["is_complete_ytd"])
        self.assertEqual(self.path.read_bytes(), before)

    async def test_real_subprocess_discovery_and_all_four_tool_calls(self):
        before = self.path.read_bytes()
        result = await run_smoke(self.path)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(len(result["calls"]), 4)
        self.assertEqual(self.path.read_bytes(), before)

    async def test_subprocess_validation_errors_stay_on_protocol_and_connection_remains_usable(self):
        async with asyncio.timeout(45):
            async with Client(server_parameters(self.path), read_timeout_seconds=20) as client:
                error = await client.call_tool("get_employee", {"employee_id": "emp-001", "as_of_date": "2026-02-30"})
                valid = await client.call_tool("get_employee", {"employee_id": "emp-001", "as_of_date": "2026-11-01"})
        self.assertTrue(error.is_error)
        self.assertNotIn(str(self.path), str(error.content))
        self.assertFalse(valid.is_error)
        self.assertEqual(valid.structured_content["status"], "ok")


if __name__ == "__main__":
    unittest.main()
