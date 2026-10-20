"""Client discovery, validation, lifecycle and transport failure checks; no AI API."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from init_employee_db import init_database
from payroll_mcp_client import PayrollMCPClient, PayrollMCPError
from payroll_mcp_server import create_server


class PayrollMCPClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_annual_summary_validates_identity_year_amounts_and_filing_status(self):
        from init_employee_db import FIXTURE_PATH
        from copy import deepcopy
        annual = self.path.with_name("annual.db")
        init_database(annual, fixture_path=FIXTURE_PATH.with_name("fixtures_t4_2026.json"))
        args = {"employee_id": "emp-001", "reporting_year": 2026}
        async with PayrollMCPClient(transport=create_server(self.path, annual)) as client:
            result = await client.call("get_annual_payroll_summary", args)
            self.assertEqual(result["box_candidates_cents"]["14"], 5000008)
            unknown = await client.call("get_annual_payroll_summary", {"employee_id": "missing", "reporting_year": 2026})
            self.assertEqual(unknown["status"], "not_found")
        changes = [lambda d: d.update(reporting_year=2025), lambda d: d["employee"].update(employee_id="emp-004"),
                   lambda d: d["box_candidates_cents"].update({"14": True}), lambda d: d.update(ready_for_filing=True),
                   lambda d: d["totals"].update(net_pay_cents=1), lambda d: d.update(is_complete_calendar_year=False)]
        for change in changes:
            bad = deepcopy(result)
            change(bad)
            async with PayrollMCPClient(transport=create_server(self.path, annual)) as client:
                client._client.call_tool = AsyncMock(return_value=SimpleNamespace(is_error=False, structured_content=bad))
                with self.assertRaisesRegex(PayrollMCPError, "invalid annual sample summary"):
                    await client.call("get_annual_payroll_summary", args)
                self.assertFalse(client.available)

    async def test_roster_schema_and_returned_pagination_are_validated(self):
        async with self.client() as client:
            result = await client.call("list_employees", {"as_of_date": "2026-10-19", "limit": 2})
            self.assertEqual(result["employees"][1]["annual_salary_cents"], 7500000)
            with self.assertRaises(PayrollMCPError):
                await client.call("list_employees", {"as_of_date": "2026-10-19", "sql": "SELECT * FROM employees"})
            bad = {**result, "next_offset": 0}
            client._client.call_tool = AsyncMock(return_value=SimpleNamespace(is_error=False, structured_content=bad))
            with self.assertRaisesRegex(PayrollMCPError, "invalid employee roster"):
                await client.call("list_employees", {"as_of_date": "2026-10-19", "limit": 2})
            self.assertFalse(client.available)
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "records with spaces.db"
        init_database(self.path)

    def client(self, **kwargs):
        return PayrollMCPClient(transport=create_server(self.path), **kwargs)

    async def test_discovery_call_and_session_cleanup(self):
        client = self.client()
        async with client:
            self.assertTrue(client.available)
            result = await client.call("get_employee", {"employee_id": "emp-001", "as_of_date": "2026-11-01"})
            self.assertEqual(result["compensation"]["annual_salary_cents"], 5000000)
        self.assertFalse(client.available)
        self.assertIsNone(client._client)
        with self.assertRaises(PayrollMCPError):
            await client.__aenter__()

    async def test_invalid_arguments_and_unapproved_tools_never_reach_server(self):
        async with self.client() as client:
            original = client._client.call_tool
            client._client.call_tool = AsyncMock(wraps=original)
            for name, args in [("update_salary", {}), ("find_employees", {"query": "Alex", "db_path": "secret"}),
                               ("find_employees", {"query": "Alex", "limit": True}), ("get_employee", {"employee_id": "emp-001"})]:
                with self.subTest(name=name, args=args), self.assertRaises(PayrollMCPError):
                    await client.call(name, args)
            client._client.call_tool.assert_not_called()

    async def test_tool_error_is_sanitized_and_connection_can_be_reused(self):
        async with self.client() as client:
            with self.assertRaisesRegex(PayrollMCPError, "check the employee ID"):
                await client.call("get_employee", {"employee_id": "emp-001", "as_of_date": "2026-02-30"})
            self.assertTrue(client.available)
            self.assertEqual((await client.call("find_employees", {"query": "Alex"}))["status"], "ambiguous")

    async def test_timeout_disables_connection_without_leaking_provider_details(self):
        async with self.client() as client:
            async def hang(*args):
                await asyncio.sleep(5)
            client._client.call_tool = hang
            client.timeout = .01
            with self.assertRaisesRegex(PayrollMCPError, "failed or timed out"):
                await client.call("find_employees", {"query": "Alex"})
            self.assertFalse(client.available)

    async def test_invalid_provenance_is_rejected(self):
        async with self.client() as client:
            client._client.call_tool = AsyncMock(return_value=SimpleNamespace(is_error=False, structured_content={"status": "ok", "provenance": {"is_fictional": False}}))
            with self.assertRaisesRegex(PayrollMCPError, "provenance"):
                await client.call("find_employees", {"query": "Alex"})
            self.assertFalse(client.available)

    async def test_incomplete_or_mutating_discovery_is_unavailable(self):
        async with self.client() as valid:
            definitions = list(valid.tools.values())
        for tools in (definitions[:-1], [t.model_copy(deep=True) for t in definitions]):
            if len(tools) == len(definitions):
                tools[0].annotations.read_only_hint = False
            fake = AsyncMock()
            fake.__aenter__.return_value = fake
            fake.list_tools.return_value = SimpleNamespace(tools=tools, next_cursor=None)
            with patch("payroll_mcp_client.Client", return_value=fake):
                async with self.client() as client:
                    self.assertFalse(client.available)
                fake.__aexit__.assert_awaited_once()

    async def test_summary_calls_reject_arbitrary_sql_and_validate_output_counts(self):
        async with self.client() as client:
            result = await client.call("get_workforce_summary", {"as_of_date": "2026-11-01"})
            self.assertEqual(result["employee_count"], 4)
            with self.assertRaises(PayrollMCPError):
                await client.call("get_workforce_summary", {"as_of_date": "2026-11-01", "sql": "DELETE FROM employees"})
            bad = {**result, "employee_count": True}
            client._client.call_tool = AsyncMock(return_value=SimpleNamespace(is_error=False, structured_content=bad))
            with self.assertRaisesRegex(PayrollMCPError, "invalid workforce summary"):
                await client.call("get_workforce_summary", {"as_of_date": "2026-11-01"})
            self.assertFalse(client.available)

    async def test_real_stdio_client_uses_local_database_without_writes(self):
        before = self.path.read_bytes()
        async with PayrollMCPClient(self.path) as client:
            self.assertTrue(client.available)
            result = await client.call("find_employees", {"query": "Alex"})
            self.assertTrue(result["requires_selection"])
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
