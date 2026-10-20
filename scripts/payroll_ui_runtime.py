"""Turn-scoped bridge between Streamlit and the async payroll agent.

Only plain conversation/employee state survives a UI rerun. MCP transports are
created, used and closed in the same task and event loop on every submission.
"""
import asyncio
from copy import deepcopy
from pathlib import Path
import sys
from threading import Lock

from init_employee_db import DEFAULT_DB, ROOT
from payroll_agent import arun_turn, build_agent, create_model
from payroll_mcp_client import PayrollMCPClient
from search_documents import search_documents


_SEARCH_LOCK = Lock()


def ui_search(question: str, limit: int = 5, document_id: str | None = None, *,
              document_type: str | None = None, edition_year: int | None = None,
              pay_periods_per_year: int | None = None) -> dict:
    """Serialize UI searches because Qdrant local storage has an exclusive lock."""
    with _SEARCH_LOCK:
        return search_documents(question, limit, document_id, document_type=document_type,
                                edition_year=edition_year, pay_periods_per_year=pay_periods_per_year)


class LazyModel:
    """Employee-only turns require neither an API key nor a paid model call."""
    def __init__(self, factory=create_model):
        self.factory = factory
        self.model = None

    def bind_tools(self, tools, **kwargs):
        owner = self

        class BoundModel:
            def invoke(self, messages):
                if owner.model is None:
                    owner.model = owner.factory()
                return owner.model.bind_tools(tools, **kwargs).invoke(messages)

        return BoundModel()


async def _run_ui_turn(question, history, employee_session, db_path, model_factory, client_factory, retriever, annual_db_path):
    connection = client_factory(db_path, annual_db_path=annual_db_path) if annual_db_path is not None else client_factory(db_path)
    async with connection as client:
        graph = build_agent(LazyModel(model_factory), retriever, employee_client=client, enable_calculation=True)
        return await arun_turn(graph, question, history, employee_session)


def run_ui_turn(question: str, history: list, employee_session: dict, *,
                db_path: Path = DEFAULT_DB, annual_db_path: Path | None = None, model_factory=create_model,
                client_factory=PayrollMCPClient, retriever=ui_search) -> dict:
    """Run once from a Streamlit script thread; dependencies allow offline tests."""
    # Streamlit can select a Windows loop without subprocess support. Choose the
    # subprocess-capable loop locally, without modifying global asyncio policy.
    if annual_db_path is None and Path(db_path).resolve() == DEFAULT_DB.resolve():
        annual_db_path = ROOT / "data/payroll_t4_2026.db"
    loop_factory = asyncio.ProactorEventLoop if sys.platform == "win32" else asyncio.new_event_loop
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        return runner.run(_run_ui_turn(question, deepcopy(history), deepcopy(employee_session),
                                       db_path, model_factory, client_factory, retriever, annual_db_path))
