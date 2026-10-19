"""Session-scoped async client for the four approved local payroll read tools."""

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
    def __init__(self, db_path: Path = DEFAULT_DB, *, timeout: float = 20, transport=None):
        if timeout <= 0:
            raise ValueError("MCP timeout must be positive")
        self.transport = transport if transport is not None else server_parameters(db_path)
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
        return deepcopy(data)
