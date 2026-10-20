"""Session-scoped async client for approved local payroll record and summary reads."""

import asyncio
from contextlib import AsyncExitStack
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator
from mcp import Client

from init_employee_db import DEFAULT_DB
from smoke_payroll_mcp import TOOL_NAMES, server_parameters


class PayrollMCPError(RuntimeError):
    pass


class PayrollMCPClient:
    """Own the transport in one async context; connection failures allow RAG fallback."""
    def __init__(self, db_path: Path = DEFAULT_DB, *, annual_db_path: Path | None = None, timeout: float = 20, transport=None):
        if timeout <= 0:
            raise ValueError("MCP timeout must be positive")
        self.transport = transport if transport is not None else server_parameters(db_path, annual_db_path)
        self.timeout = timeout
        self.available = False
        self.tools = {}
        self._client = None
        self._stack = None
        self._used = False

    async def __aenter__(self):
        if self._used:
            raise PayrollMCPError("Create a new MCP client for each session")
        self._used = True
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        try:
            async with asyncio.timeout(self.timeout):
                self._client = await self._stack.enter_async_context(Client(self.transport, read_timeout_seconds=self.timeout))
                discovered = await self._client.list_tools()
            definitions = {tool.name: tool for tool in discovered.tools}
            if set(definitions) != TOOL_NAMES or len(discovered.tools) != len(TOOL_NAMES) or discovered.next_cursor is not None:
                raise PayrollMCPError("Employee MCP discovery does not match the approved read-only tool set")
            for tool in definitions.values():
                if (tool.annotations is None or tool.annotations.read_only_hint is not True
                        or tool.annotations.destructive_hint is not False
                        or tool.annotations.open_world_hint is not False):
                    raise PayrollMCPError("Employee MCP discovery lacks required read-only annotations")
            self.tools = definitions
            self.available = True
        except Exception:
            self.available = False
            await self._close()
        return self

    async def _close(self):
        self.available = False
        if self._stack is not None:
            stack, self._stack = self._stack, None
            try:
                await stack.aclose()
            except Exception:
                pass  # Never surface provider/process internals as payroll evidence.
        self._client = None

    async def __aexit__(self, exc_type, exc, traceback):
        await self._close()
        return False

    async def call(self, name: str, arguments: dict) -> dict:
        if not self.available or self._client is None:
            raise PayrollMCPError("Employee MCP connection is unavailable")
        if name not in TOOL_NAMES or name not in self.tools:
            raise PayrollMCPError("Employee MCP tool is not approved")
        # Reject extra arguments locally even when an SDK would ignore them.
        definition = self.tools[name]
        if not isinstance(arguments, dict) or set(arguments) - set(definition.input_schema.get("properties", {})):
            raise PayrollMCPError("Invalid employee tool arguments")
        if list(Draft202012Validator(definition.input_schema).iter_errors(arguments)):
            raise PayrollMCPError("Invalid employee tool arguments")
        try:
            async with asyncio.timeout(self.timeout):
                result = await self._client.call_tool(name, arguments)
        except Exception:
            self.available = False
            raise PayrollMCPError("Employee MCP request failed or timed out") from None
        if result.is_error:
            raise PayrollMCPError("Employee tool rejected the request; check the employee ID and explicit dates")
        data = result.structured_content
        if not isinstance(data, dict) or not isinstance(data.get("status"), str):
            self.available = False
            raise PayrollMCPError("Employee MCP returned an invalid structured result")
        provenance = data.get("provenance", {})
        if not isinstance(provenance, dict) or provenance.get("is_fictional") is not True or provenance.get("source") != "local_fictional_payroll_database":
            self.available = False
            raise PayrollMCPError("Employee MCP result lacks fictional database provenance")
        if name == "list_employees":
            valid = data["status"] in {"ok", "outside_snapshot"} and data.get("as_of_date") == arguments.get("as_of_date")
            if valid and data["status"] == "ok":
                rows = data.get("employees")
                valid = all(type(data.get(k)) is int and data[k] >= 0 for k in ("match_count", "returned_count", "limit", "offset"))
                valid = valid and data["limit"] == arguments.get("limit", 20) and data["offset"] == arguments.get("offset", 0)
                valid = valid and isinstance(rows, list) and len(rows) == data["returned_count"] <= data["limit"]
                if valid:
                    valid = all(isinstance(row, dict) and {"annual_salary_cents", "frequency", "ends_before"}.issubset(row)
                        and all(isinstance(row.get(k), str) and row[k] for k in (
                        "employee_id", "employee_number", "given_name", "family_name", "employment_id", "starts_on", "province_code", "currency"))
                        and row["currency"] == "CAD" and row["province_code"] == "ON"
                        and (row.get("annual_salary_cents") is None or type(row["annual_salary_cents"]) is int and row["annual_salary_cents"] > 0)
                        and (row["frequency"] is None or isinstance(row["frequency"], str) and row["frequency"] in {"weekly", "biweekly", "semimonthly", "monthly"})
                        and row["starts_on"] <= data["as_of_date"] and (row.get("ends_before") is None or isinstance(row["ends_before"], str) and data["as_of_date"] < row["ends_before"])
                        for row in rows)
                if valid:
                    valid = len({row["employee_id"] for row in rows}) == len(rows)
                    valid = valid and rows == sorted(rows, key=lambda r: (r["employee_number"], r["employee_id"]))
                    valid = valid and data["returned_count"] == min(data["limit"], max(0, data["match_count"] - data["offset"]))
                    more = data["offset"] + len(rows) < data["match_count"]
                    valid = valid and type(data.get("has_more")) is bool and data["has_more"] == more
                    valid = valid and data.get("next_offset") == (data["offset"] + len(rows) if more else None)
                    valid = valid and (type(data.get("next_offset")) is int if more else data.get("next_offset") is None)
            if not valid:
                self.available = False
                raise PayrollMCPError("Employee MCP returned an invalid employee roster")
        if name == "get_annual_payroll_summary":
            valid = self.valid_annual_summary(data, arguments)
            if not valid:
                self.available = False
                raise PayrollMCPError("Employee MCP returned an invalid annual sample summary")
        if name in {"get_workforce_summary", "get_employment_movements"}:
            date_fields = ("as_of_date",) if name == "get_workforce_summary" else ("start_date", "end_date")
            valid = data["status"] in {"ok", "data_gap", "outside_snapshot"} and all(data.get(k) == arguments.get(k) for k in date_fields)
            if data["status"] == "ok":
                keys = ("employee_count", "unfiltered_active_count", "total_employee_records", "missing_salary_count", "missing_pay_group_count") if name == "get_workforce_summary" else (
                    "joined_employee_count", "ended_employee_count", "join_event_count", "end_event_count")
                valid = valid and all(type(data.get(k)) is int and data[k] >= 0 for k in keys)
                if valid and name == "get_workforce_summary":
                    groups = data.get("pay_frequency_counts")
                    valid = isinstance(groups, list) and all(isinstance(g, dict) and g.get("frequency") in {
                        "weekly", "biweekly", "semimonthly", "monthly", "unknown"} and type(g.get("employee_count")) is int
                        and g["employee_count"] >= 0 for g in groups)
                    valid = valid and sum(g["employee_count"] for g in groups) == data["employee_count"] <= data["unfiltered_active_count"] <= data["total_employee_records"]
                    expected_filters = {"pay_frequency": arguments.get("pay_frequency"),
                        "salary_comparison": arguments.get("salary_comparison", "any"),
                        "salary_min_cents": arguments.get("salary_min_cents"), "salary_max_cents": arguments.get("salary_max_cents")}
                    valid = valid and data.get("filters") == expected_filters
                elif valid:
                    valid = data["joined_employee_count"] <= data["join_event_count"] and data["ended_employee_count"] <= data["end_event_count"]
            if not valid:
                self.available = False
                raise PayrollMCPError("Employee MCP returned an invalid workforce summary")
        return deepcopy(data)

    @staticmethod
    def valid_annual_summary(data, arguments):
        """Bind annual evidence to the requested identity/year and reconcile cents."""
        try:
            if data["status"] not in {"ok", "not_found", "incomplete_annual_records"}:
                return False
            if type(data.get("reporting_year")) is not int or data["reporting_year"] != arguments["reporting_year"]:
                return False
            if data.get("ready_for_filing") is not False:
                return False
            if data["status"] == "not_found":
                return data.get("employee_id") == arguments["employee_id"] and "box_candidates_cents" not in data
            employee = data.get("employee")
            if not isinstance(employee, dict) or employee.get("employee_id") != arguments["employee_id"]:
                return False
            if not all(isinstance(employee.get(k), str) and employee[k] for k in ("given_name", "family_name", "employee_number")):
                return False
            if data["status"] != "ok":
                return (data.get("is_complete_calendar_year") is False and data.get("ready_for_sample_breakdown") is False
                        and "box_candidates_cents" not in data and "totals" not in data)
            boxes, totals = data["box_candidates_cents"], data["totals"]
            if set(boxes) != {"14", "16", "16A", "18", "22", "24", "26"}:
                return False
            keys = {"payment_count", "employment_income_cents", "total_deductions_cents", "net_pay_cents",
                    "pensionable_earnings_cents", "insurable_earnings_cents"}
            if set(totals) != keys or not all(type(v) is int and 0 <= v <= 10**14 for v in [*boxes.values(), *totals.values()]):
                return False
            return (data.get("dataset_kind") == "simulated_year_end" and data.get("currency") == "CAD"
                and data.get("is_complete_calendar_year") is True and data.get("ready_for_sample_breakdown") is True
                and totals["payment_count"] > 0 and boxes["14"] == totals["employment_income_cents"]
                and sum(boxes[k] for k in ("16", "16A", "18", "22")) == totals["total_deductions_cents"]
                and totals["employment_income_cents"] - totals["total_deductions_cents"] == totals["net_pay_cents"]
                and boxes["24"] <= totals["insurable_earnings_cents"] <= boxes["14"]
                and boxes["26"] <= totals["pensionable_earnings_cents"] <= boxes["14"]
                and type(data.get("dental_benefits_code")) is int and 1 <= data["dental_benefits_code"] <= 5
                and isinstance(data.get("assumptions"), str) and bool(data["assumptions"].strip())
                and isinstance(data.get("rules_sha256"), str) and len(data["rules_sha256"]) == 64
                and all(c in "0123456789abcdef" for c in data["rules_sha256"])
                and isinstance(data.get("missing_filing_information"), list) and bool(data["missing_filing_information"])
                and all(isinstance(item, str) and item for item in data["missing_filing_information"])
                and data["provenance"]["dataset_snapshot_date"] >= f"{arguments['reporting_year']}-12-31")
        except (KeyError, TypeError, ValueError):
            return False
