# Ontario Payroll AI Copilot

An agentic RAG assistant for payroll specialists, using fictional
Ontario employee records and retrieved government payroll guidance.

## Initial use cases

1. Calculate payroll deductions.
2. Retrieve an employee's next payday.
3. Prepare a bonus request pending human approval.
4. Answer payroll policy questions with source citations.

## Planned architecture

Payroll specialist
→ Streamlit chat interface
→ LangGraph agent workflow and LLM
→ MCP client
→ MCP server
→ Retrieval, employee, calculation, and request tools

The agent decides which tools to call, checks retrieved evidence,
and can search again when information is insufficient.

The payroll skill supplies instructions to the LangGraph agent.
Payroll arithmetic is reserved for a future deterministic calculation tool.

LangGraph manages conversation state, tool execution,
retrieval retries and stopping conditions. Request confirmation remains planned.

## Data storage

- SQLite: fictional employee/employment records, effective-dated compensation,
  company calendars, and historical payments. Pending requests remain planned.
- Source documents: verified CRA, Ontario, and sample company policies.
- Qdrant local mode: searchable document sections and their embeddings,
  persisted in `knowledge/vector_store/qdrant/`.

Each document section will retain its source, effective date,
and section identifier so answers can cite the evidence.

## Fictional employee database

The database is accessed through read-only MCP tools connected to the terminal
LangGraph agent. Those capabilities become available only after successful tool
discovery. This is a single-employer learning dataset, not a
production payroll system or an Ontario deduction calculator.

The schema follows company-style record separation:

| Table | Responsibility |
| --- | --- |
| `dataset_info` | Fictional employer, schema version, snapshot and history coverage |
| `employees` | Stable internal ID, unique employee number, names |
| `employments` | Hire/rehire periods and employment jurisdiction |
| `pay_groups` | Frequency and company timezone |
| `pay_group_assignments` | Effective-dated membership, without overwriting history |
| `compensation_rates` | Effective-dated annual salaries and change reasons |
| `pay_runs` | Explicit company pay periods, payment dates, UTC cutoffs and status |
| `payroll_payments` | Gross, deduction and net snapshots tied to historical compensation |
| `payroll_lines` | Separate earnings and deductions such as salary, CPP, EI and taxes |

This separation is informed by the [Dynamics 365 payroll employee model](https://learn.microsoft.com/en-us/dynamics365/human-resources/hr-admin-integration-payroll-api-payroll-employee)
and [pay-cycle/pay-period model](https://learn.microsoft.com/en-us/dynamics365/fin-ops-core/fin-ops/hr/localizations/usa/noam-usa-pay-cycle-pay-period-tasks-sample).
These are examples of enterprise data patterns, not sources of Ontario payroll rules.

Money is stored in integer CAD cents. Employment, assignment, and compensation
ranges include the start date and exclude the end date; a null end means an open
internal employment/rate interval. This differs from unknown effective dates in
the government-document catalog. Payroll-period start/end dates are inclusive.
Payment dates can precede period-end, as in the fictional monthly run. Cutoffs
are stored in UTC; the pay-group timezone is `America/Toronto`.

Posted payment headers, lines, and runs are protected against updates/deletion
by SQLite triggers. Posting checks line totals and historical employment,
compensation, and pay-group relationships. Referenced historical salary amounts
cannot be rewritten. This is integrity protection, not a complete audit system;
there is no reversal, correction, approval, or production posting workflow yet.

The fixtures contain five employees, two pay groups, six runs and seven payments.
Alex Morgan (`EMP-1001`) and Alex Chen (`EMP-1004`) deliberately share a first
name. Priya has a rate change effective 2026-11-01; Sam's employment ended before
that snapshot. Fixed deductions are fictional examples, not CRA-validated amounts.
Historical data covers **payments from 2026-10-16 through the 2026-11-01 snapshot**;
do not present those sums as complete year-to-date totals. Future dates are
explicit fictional company schedules, not inferred Ontario requirements.

Run these commands from the project directory on Windows, macOS, or Linux:

```sh
python scripts/init_employee_db.py --dry-run
python scripts/init_employee_db.py
python -m unittest discover -s tests -p test_employee_db.py -v
```

The script uses Python's standard `sqlite3` library (SQLite 3.37+ is required).
The dry run validates everything in memory. Initialization publishes a validated
`data/payroll.db` and refuses to overwrite an existing file, including concurrent
creation. Use `--db data/another_demo.db` for a separate dataset. Publication uses
a same-directory hard link, so use a local filesystem that supports hard links;
there is no destructive fallback for unsupported filesystems. Generated `.db`
files are ignored by Git; commit the schema and fixtures to reproduce them.

`connect_readonly()` opens an existing database in read-only/query-only mode for
the MCP layer. No SINs, banking details, tax elections, employee access
controls, encryption, backup policy, or full change audit are implemented. Those
need separate design before using real employee data. SQLite file access alone
does not implement authorization. No agent or live payroll calculation is enabled
by initializing the database.

## Knowledge ingestion

Verified documents
→ Extract text
→ Split into meaningful sections
→ Attach source metadata
→ Generate embeddings
→ Store in a vector database

Embeddings use Perplexity `pplx-embed-v1-0.6b` with 1,024 dimensions
and `base64_int8` output decoded to numeric vectors for cosine search.
The API key belongs in the local `.env` file and must not be committed.

PDF extraction writes to `knowledge/extracted_v2/`. Chunking reads those
files and writes to `knowledge/chunks/`, preserving source and review metadata.
The extraction directory name is retained because existing scripts use it.

Run from the project directory after activating the virtual environment:

```sh
python -m pip install -r requirements.txt
python scripts/index_documents.py --dry-run
python scripts/index_documents.py --limit 16
python scripts/index_documents.py --all
```

The dry run checks candidate inputs without API calls or database writes.
Other indexing commands call the paid embedding API for missing chunks.
Existing records are checked against their source chunks and skipped.
Keep the model, vector dimensions, and encoding consistent across runs.
Close other processes using the local Qdrant database before indexing.
Local mode is intended for this prototype; disk writes can be slow.

## Agent workflow

1. Understand the question and relevant conversation context.
2. Retrieve employee data or policy evidence as needed.
3. Check relevance, effective dates, and missing information.
4. Retrieve additional evidence or ask for clarification.
5. Call deterministic calculation tools when required.
6. Answer with sources, assumptions, and limitations.

Execution limits will prevent endless retrieval loops.
Creating payroll requests requires explicit user confirmation.
Requests remain pending human approval.

## Run the document agent

The implemented terminal agent uses an explicit LangGraph:
model decision -> sequential document tool calls -> model evidence review
-> response and citation validation. The model can refine a search or ask
for missing details. Model decisions are not hardcoded query routes.

It loads `skills/ontario-payroll/SKILL.md` as standing instructions.
The Perplexity Agent API supplies the tool-calling model; the same API key
also supplies embeddings for Qdrant searches. The default agent model is
`anthropic/claude-haiku-4-5`; set `PERPLEXITY_AGENT_MODEL` in your local
environment or `.env` to change it. Agent runs incur model and embedding costs.

```sh
python scripts/payroll_agent.py "What is reported in box 22 on a T4 slip?" --trace
python scripts/payroll_agent.py --trace
python -m unittest discover -s tests -v
```

Without a question, the CLI starts a conversation. Type `exit` to quit.
It retains up to six recent exchanges, bounded by 20,000 characters, in
memory while that process runs. This is not persistent conversation storage.
Each turn resets retrieved evidence and its budgets. It permits at most
five model calls and four searches, with a separate graph recursion limit.
Model API retries are disabled, requests have a 60-second timeout, and each
model call has a 1,600-output-token cap. These are execution limits, not
a guaranteed monetary cost cap.

`--trace` displays observable model/tool events without hidden reasoning.
Searches run sequentially because Qdrant local storage permits only one
client process to access its path at a time.

The model finishes by calling `submit_response` with `status`, `text`, and
`citations` as structured tool arguments. A Pydantic schema defines the tool
and validates its submission locally. Missing fields, wrong types, extra
fields, and unrecognized citation IDs are rejected. Submission must be the
only tool call in that response; it ends the turn without another model call.
Plain text or JSON messages do not substitute for a valid submission.
If a completed response is rejected, the graph may return local validation
feedback for one correction attempt per turn. Feedback pairs every rejected
tool call without executing it, or follows a plain response with an application
message. Retrieved evidence is retained. The correction consumes the same
five-model-call budget and does not reset the four-search budget. Incomplete
responses and API errors do not trigger correction. A second invalid response,
or an exhausted model budget, ends with the limitation message. The correction
counter resets for the next user turn.

Runtime capabilities are stated explicitly in the model prompt. Document search,
conversation and payroll intake are available. Employee records, company payroll
calendars and MCP transport are enabled after successful local discovery. Payroll
calculations and compensation requests remain disabled. With MCP unavailable,
employee requests return `unsupported`, explaining the missing connection rather
than collecting an employee ID. Salary intake may ask
for missing inputs with `clarification`, but must disclose that calculation is
unavailable and cannot promise to calculate after receiving those inputs.

The response validator rejects observed calculation/access promises, explicit
net-pay claims, and clarification responses that acknowledge unavailable employee
or calendar data. Rejections use the existing single correction opportunity.
These targeted language checks do not cover every possible wording; they do not
replace capability authorization or deterministic calculation tools. Live tests
check both salary turns for a calculation limitation and reject employee-ID
collection in the unavailable-payday scenario.

Payment-date context is derived only from user messages. Supply an explicit
`YYYY-MM-DD` date, either alone or labeled, for example `Payment date: 2026-11-13`.
It is retained across follow-ups even when its original exchange leaves the
model's recent-history window. A new labeled ISO date replaces it; `clear the
payment date` resets it. Invalid ISO dates, multiple candidate dates, and labeled
relative or ambiguous date formats clear the old date and require clarification.
This is conservative parsing, not general natural-language date understanding.
Dates in assistant replies and source documents cannot establish user context.
An illustrative date-format example in a reply likewise cannot establish it.

The model receives this context every call and must ask for a missing date before
date-sensitive intake. Explicit deduction-table or frequency-filter requests are
blocked before retrieval until a date is provided. Targeted output checks reject
observed assumed-date claims, including `as of` a different date. These language
checks are not complete semantic verification.

Retrieved sources carry `payment_applicability`. Known effective-date bounds are
compared inclusively against the provided date. A source outside a known bound,
or with invalid bounds, cannot support an `answer`. Missing bounds remain
unresolved; a null end date does not establish indefinite applicability. Final
cited responses display unknown/incomplete-date limitations. Edition year is
not a substitute for effective dates. For general definitions without a payment
date, the response can explain evidence limitations without demanding one.
Targeted validation also rejects observed definitive claims that guidance applies
to the user's payment date when cited date bounds are incomplete. Qualified
explanations of what a publication says remain allowed. This is a regression
guard for known language patterns, not complete claim-to-source verification.

This uses ordinary function tool calling. The separate API `response_format`
setting is not sent: the earlier native-output-schema configuration returned
HTTP 400 with the selected model. Incomplete API responses are rejected before
any partial tool call or submission is executed. The trace reports response
status, text length, token usage when available, and validation outcomes without
printing raw model output. Live acceptance tests print these traces as well.
API errors fail the run; submissions still invalid after the permitted correction
produce a limitation message. A successful schema check validates format and citations,
not the factual accuracy of the generated explanation.

A live T4 box 22 smoke test passed on October 15, 2026: the model searched
the indexed guide, submitted a structured answer, and passed citation checks
with pending-review and edition limitations displayed. It used two model calls
and one embedding request. This is one end-to-end check, not broad payroll
accuracy validation. Run the offline workflow suite to check routing, evidence,
validation, budgets and correction behavior; paid scenarios are skipped by default.

On October 16, 2026, 26 offline tests passed. The paid rerun of the salary
conversation and employee-payday scenarios used six model calls and no searches
or embedding requests. All three turns recovered from plain responses through
one correction each and passed submission validation. The salary conversation
passed its automated context checks, retaining the $50,000 salary and biweekly
frequency. The employee scenario still failed its acceptance check: the model
returned `clarification` rather than `unsupported` for unavailable employee and
schedule data. The salary replies also implied future calculation capability
that this prototype does not implement. That run established format correction,
not semantic correctness. After the subsequent capability rules and targeted
validation were added, 30 offline tests passed and both affected live scenarios
passed their capability acceptance checks. The rerun used six model calls and
no embedding requests. Alex's payday returned `unsupported` with the missing
employee/calendar connections explained. Both salary turns disclosed the missing
calculation tool, and the follow-up retained the $50,000 salary and biweekly pay.
Each turn used one response-format correction. These checks do not establish
full payroll correctness: the salary follow-up still assumed today's payment
date and current tax year despite the prompt asking for the intended date.
That run preceded explicit user-derived payment-date context and applicability
checks. A subsequent three-turn date conversation was executed with recorded
turns reused after test assertion corrections, avoiding duplicate model requests.
It retained the explicitly supplied `2026-11-13` date through a follow-up and
retrieval. The recorded rerun used ten model calls and four embedding requests.
Its original automated checks passed, but review found a definitive applicability
claim despite incomplete source dates. The new regression guard rejects that
observed claim and permits qualified uncertainty; 40 offline tests now pass.
That live reply also included unverified gross-pay arithmetic and answered a T4001-specific request
with T4127 excerpts. Calculation and source-grounding quality remain unresolved.
Full reproduction of the updated live acceptance test remains pending.

A subsequent focused applicability test passed all eight execution, citation,
metadata and targeted-wording checks. With payment date `2026-11-13`, it retrieved
the requested T4001 2025 guide, cited PDF pages 32 and 58, and explained that
incomplete effective-date metadata prevents confirmation of applicability. The
test used three model calls, one embedding request and one correction. It checks
the focused guard and source selection for this question, not every payroll fact
or the full multi-turn acceptance suite. Runtime logs remain local and ignored.

The final response must identify retrieved sources. The application rejects
unknown citation IDs and uncited answers and attaches links and PDF pages
from retrieved records. It adds pending-review and update-review notices.
Citation validation does not prove every claim follows from its source:
evidence relevance and applicability are currently assessed by the model.

The document model exposes search and response submission. A separate graph node
uses read-only employee MCP tools. Calculation tools and request confirmation
remain planned.

## Current status

Knowledge ingestion, filtered retrieval and a terminal LangGraph document
agent, fictional employee database and read-only employee MCP integration are implemented.

Completed:
- Created .venv using Python 3.11.4.
- Created .gitignore.
- Created the Ontario payroll skill and government document catalog.
- Downloaded 21 government PDFs and extracted 702 pages.
- Generated 3,473 chunks; indexing excludes 12 contents-page chunks.
- Selected Perplexity embeddings and Qdrant local storage.
- Migrated 96 existing embeddings from Chroma to Qdrant.
- Indexed all 3,461 eligible chunks and verified their saved vectors and metadata.
- Added document-type, edition-year and pay-frequency search filters.
- Added the bounded LangGraph document agent and offline workflow tests.
- Connected read-only employee MCP tools with selection and structured session context.

Document review and effective-date checks are still required before relying
on answers. This collection does not cover every Ontario payroll question.

The fictional employee database, standalone read-only MCP server, terminal agent
and local Streamlit browser chat are implemented (see below). Payroll calculations,
request submission and hosted deployment remain unimplemented.

## Read-only employee MCP server

`scripts/payroll_mcp_server.py` uses the official Python MCP SDK, pinned to
`mcp==2.3.0`. It runs locally over **stdio**: a client starts the Python server
process and exchanges MCP messages through stdin/stdout. No HTTP listener, model
API key, or paid AI call is required. See the [official SDK client documentation](https://py.sdk.modelcontextprotocol.io/client/).

The layers are:

```text
MCP client -> stdio MCP server -> employee_repository.py -> read-only payroll.db
```

The server exposes exactly four tools:

| Tool | Required inputs | Meaning |
| --- | --- | --- |
| `find_employees` | `query` | Name-prefix search or exact employee number/ID; ambiguity requires selection |
| `get_employee` | `employee_id`, `as_of_date` | Salary and employment/pay group effective on an explicit date |
| `get_payroll_schedule` | `employee_id`, `as_of_date` | Recorded scheduled payment dates strictly after that date |
| `get_payroll_history` | `employee_id`, `start_date`, `end_date` | Posted payments and earning/deduction lines in an inclusive payment-date range |

Dates must be real `YYYY-MM-DD` dates, not `today` or an inferred tax year.
`find_employees`, schedules and history accept `limit` (1-100) and `offset`;
follow `next_offset` while `has_more` is true. Ambiguity is based on all matches,
even if a one-row page is requested. Dates and status reflect the fictional
snapshot, not a live HR system. The schedule tool excludes same-day payments
and never extrapolates past the recorded calendar. A missing scheduled run does
not establish that no pay is owed.

Results include structured JSON with `status`, fictional-data provenance, the
dataset snapshot date, and history coverage. Money is integer CAD cents. Historical
amounts come from stored records; these tools do not calculate new deductions.
History always reports `is_complete_ytd: false`; check pagination and
`request_within_declared_coverage` before interpreting the result.

Install dependencies using the project's activated environment:

```sh
python -m pip install -r requirements.txt
```

If `data/payroll.db` does not exist yet, initialize it once with
`python scripts/init_employee_db.py`. Existing databases are never overwritten.
Then run the real subprocess demonstration:

```sh
python scripts/smoke_payroll_mcp.py
```

The smoke client discovers the tools and executes all four against explicit
fixture dates, checking both Alex matches, Alex Morgan's stored salary, the
recorded 2026-11-13 payment date, and the partial-history flag. It prints the
inputs, structured responses, and six checks. To use another initialized copy:

```sh
python scripts/smoke_payroll_mcp.py --db "data/another_demo.db"
```

The client uses `sys.executable` and absolute script/database paths, so paths
containing spaces work on Windows and macOS. It passes no `.env` credentials to
the server. The MCP SDK starts the Windows subprocess without a visible window.
Running `python scripts/payroll_mcp_server.py` directly starts a protocol server
waiting for messages, not an interactive payroll chat. stdout is reserved for
MCP; startup diagnostics go to stderr.

Run the local repository and protocol tests:

```sh
python -m unittest discover -s tests -p test_employee_repository.py -v
python -m unittest discover -s tests -p test_payroll_mcp_server.py -v
```

Tests cover date boundaries, salary changes, terminated employment, assignment
changes, ambiguous names, pagination, literal search input, error responses,
unchanged database bytes, discovery, and real subprocess tool calls. These
tests do not call Perplexity. For the whole offline suite, explicitly keep the
existing paid agent tests disabled:

```powershell
# Windows PowerShell
$env:RUN_LIVE_AGENT_TESTS='0'
python -m unittest discover -s tests -q
```

```sh
# macOS/Linux
RUN_LIVE_AGENT_TESTS=0 python -m unittest discover -s tests -q
```

The server opens existing databases in read-only/query-only mode, validates
the fictional marker/schema version, and offers no SQL-execution or write tool.
Its database path is server configuration, never a tool argument. Read-only tool
annotations describe the tools; database mode and fixed queries enforce the
restriction. This local demo does not implement per-user HR authorization or a
remote deployment.

## LangGraph with employee MCP

The terminal agent now owns a session-scoped `PayrollMCPClient`. It discovers the
four approved read-only tools before enabling employee/calendar capabilities.
Employee questions run through an asynchronous LangGraph node; the document RAG
loop still retrieves government PDF evidence. The employee node uses bounded
intent routing and deterministic rendering of tool results, with no model call
for the record workflow. It does not support arbitrary natural-language record queries.

```sh
python scripts/payroll_agent.py --trace
python scripts/payroll_agent.py --no-mcp "Explain T4 box 22"
python scripts/payroll_agent.py --db "data/payroll.db" "EMP-1001 salary as of 2026-11-01"
```

In one interactive session, try:

```text
You: Find Alex as of 2026-11-01
Agent: Alex Morgan (EMP-1001) and Alex Chen (EMP-1004); which employee number?
You: EMP-1001
Agent: Stored annual salary CAD $50,000.00; fictional database snapshot 2026-11-01.
You: When is their next payday?
Agent: Recorded dates strictly after 2026-11-01, starting 2026-11-13.
You: Show history from 2026-10-01 to 2026-11-01
Agent: Stored gross, deduction and net amounts; partial history, not complete YTD.
```

These are abbreviated fixture examples. No employee is selected when a name is
ambiguous. The selected identity and explicit lookup date persist in structured
session state; `reset` clears them and the chat history. A lookup date (`as of` or
`lookup date:`) is separate from a payroll payment date. History requires an
explicit inclusive date range. Name-prefix forms include `Find Alex` and `Alex's
salary`; exact employee numbers are the most reliable input.

Database facts use D references with tool/employee/provenance metadata. Government
PDF evidence uses S references with links and pages. Stored amounts are displayed,
never recalculated; payroll calculations and compensation writes remain disabled.
Each employee turn permits at most four MCP calls and shows at most the first
schedule/history page, with a warning when more records exist. Use narrower ranges
or the direct MCP pagination arguments for additional pages.

Missing/broken MCP leaves document RAG available and returns an explicit employee
connection limitation. `--no-mcp` avoids starting the MCP subprocess entirely.
The CLI currently creates the configured language-model client at startup, so its
API key is still required; document questions can incur API charges. Local employee
record turns and the tests below do not make model API calls.

```sh
python -m unittest discover -s tests -p test_payroll_mcp_client.py -v
python -m unittest discover -s tests -p test_payroll_agent_mcp.py -v
```

These tests use temporary fictional databases, real local MCP calls (including
stdio subprocesses), and mocked model responses. Existing paid tests stay opt-in.
Python callers should use `async with PayrollMCPClient(...)` and
`await arun_turn(graph, question, history, employee_session)` in the same async
session, passing the returned `employee_session` into the next turn. This context
is trusted application state, not an untrusted request field or assistant text.
The synchronous `run_turn` remains available for document-only graphs.

## Portability and learning

Target Windows, macOS, and Linux.
The local Streamlit interface runs in a browser. Hosted deployment remains planned.

Build incrementally and explain each component.
Obtain approval before every file change.

## Browser chat UI

Install and run with the project's virtual environment, rather than Anaconda base:

```powershell
# Windows PowerShell, from the project directory
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

```sh
# macOS/Linux, using a separately created local virtual environment
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

Open `http://localhost:8501`. If the fictional database has not been initialized,
run the existing `scripts/init_employee_db.py` once using the same environment.
Do not copy a Windows `.venv` to a Mac; recreate it from `requirements.txt`.

Type `Find Alex as of 2026-11-01`, choose **EMP-1001**, then ask
`When is their next payday?`.
You can also type `EMP-1001 salary as of 2026-11-01` or
`Show history from 2026-10-01 to 2026-11-01` after selecting an employee.
The sidebar shows selected employee context and the MCP status on the last turn.
Use **Reset conversation** to clear messages, selection and lookup context.
Session state lasts for the browser's active Streamlit session; this UI does not
provide durable saved conversations or HR authorization.

**Sources** separates fictional database facts from government PDF references.
**Tool activity** shows observable tool events, without hidden reasoning.
Payroll calculations and compensation changes remain unsupported.

Employee record turns use local MCP without creating a language-model client or
requiring an API key. Government guidance questions require the existing `.env`
configuration and can incur model and query-embedding API charges. Loading the
page and resetting the chat do not call the model or MCP server.

`scripts/payroll_ui_runtime.py` opens and closes MCP inside the same async task
and loop on each submitted turn. Only conversation/employee state is retained
between reruns. On Windows it uses a subprocess-capable Proactor event loop
locally, without changing global asyncio policy. UI retrieval calls are serialized
because Qdrant local storage uses an exclusive lock; other CLI/indexing processes
should not open the same store concurrently. MCP disconnection still allows
document retrieval, and failed turns preserve the last completed conversation.

Run the UI tests without paid APIs:

```sh
python -m unittest discover -s tests -p 'test_payroll_ui*.py' -v
```

The tests use Streamlit AppTest, mocked model responses and temporary fictional
databases, including real local stdio MCP calls. Streamlit's official
[AppTest documentation](https://docs.streamlit.io/develop/api-reference/app-testing)
describes the interaction testing API.

## Standalone payroll calculation engine

`scripts/payroll_calculator.py` implements an offline calculation engine for
**2026 Ontario regular cash salary**, with biweekly (exactly 26 periods) and
monthly (exactly 12 periods) payments. It returns gross pay, CPP (base plus first
additional), CPP2, EI, federal income tax, Ontario income tax, total deductions
and estimated net pay. It reads no employee database and makes no API calls.

**This engine is not connected to LangGraph, MCP or the UI yet.** Chat still
reports calculation as unavailable. Connecting the tested engine is a separate
step; no capability flags or existing agent safeguards were relaxed here.

The implementation follows [CRA T4127 January 2026](https://www.canada.ca/en/revenue-agency/services/forms-publications/payroll/t4127-payroll-deductions-formulas/t4127-jan/t4127-jan-payroll-deductions-formulas-computer-programs.html)
Option 1, using the optional year-to-date projection for CPP/EI tax credits.
The [July 2026 edition](https://www.canada.ca/en/revenue-agency/services/forms-publications/payroll/t4127-payroll-deductions-formulas/t4127-jul/t4127-jul-payroll-deductions-formulas.html)
states there is no Ontario change and refers back to January for unreproduced
formulas. Later-year payments retain both source references. Payments outside
2026 are rejected rather than reusing these rules. The explicit coverage limit
does not claim that CRA legislation cannot change; rate updates need review.

`data/payroll_rules_2026.json` stores rates, bracket constants, annual limits,
Ontario surtax, tax reduction, health-premium bands, source references, effective
dates and the rounding policy. It is separate from RAG evidence: the LLM does
not choose numbers from retrieved paragraphs or perform payroll arithmetic.
Results include the ruleset ID, SHA-256 of the exact rules file, formula method,
confirmed inputs, tax factors, limitations and source provenance.

Required input groups are:

| Inputs | Meaning |
| --- | --- |
| `annual_salary` | CAD dollars as a decimal string, such as `"50000.00"`; binary floats are rejected |
| `pay_frequency`, `pay_periods_per_year` | `biweekly`/26 or `monthly`/12, explicitly supplied; 27-period years are unsupported |
| `pay_period_number`, `payment_date` | Current calendar-year pay-period index and actual `YYYY-MM-DD` payment date; the index includes this payment |
| `province` | Province of employment: `ON` |
| `federal_claim`, `ontario_claim` | Explicit `{"mode":"basic_personal"}` choice, or `{"mode":"amount","amount":"..."}` using the confirmed TD1 total; claim codes are not amounts |
| `cpp_status`, `ei_status` | `eligible_all_year` or `exempt_all_year` for CPP; `insurable` or `exempt_all_year` for EI |
| `ytd` | Prior current-employer pensionable earnings, CPP, CPP2 and EI, excluding this payment; explicit `is_complete: true` and `employer_scope: "current_employer"` |
| Scope confirmations | `regular_salary_only`, `no_other_adjustments`, `ontario_only_employment` must explicitly be `true` |

No YTD value defaults to zero. Complete YTD confirmation is a caller assertion;
the engine cannot independently verify payroll completeness. The existing
partial database history cannot establish it. Use explicit zero prior values
only when they are appropriate for the supplied scenario. The engine checks
annual limits and basic consistency, but does not reconstruct historical payroll
or generate/validate an employer's calendar from the pay-period number.

The first release excludes bonuses, overtime/partial-period variations, benefits,
RRSP/RPP deductions, union dues, garnishments, extra withholding, Ontario dependent
tax reduction, TD1 tax-exempt declarations, non-resident cases, interprovincial
transfers and CPP eligibility changes during the year. These must not be hidden
behind `no_other_adjustments: true`. Missing required fields return `clarification`;
malformed data returns `invalid_input`; outside-scope cases return `unsupported`;
invalid/missing rules return `unavailable`. These responses contain no calculated
amounts. Salary changes in earlier periods can be reflected in complete supplied
YTD values; current gross is the current annual salary divided by the pay count.

Money uses `Decimal` with explicit cent rounding. The CPP period exemption is
truncated to cents; contributions and the F5 enhanced-CPP monetary deduction are
rounded to cents without rounding rate ratios. Tax factors retain precision;
the dynamic federal basic personal amount is rounded to cents. Ontario includes
surtax, tax reduction and the health premium. Federal and Ontario per-period tax
are each rounded to cents, then added. This reproduces the user-supplied PDOC
result for the first reference case; neither share is adjusted to force a rounded
combined total. `income_tax_rounding_difference` records the difference from
rounding combined tax once. Ruleset `cra-2026-on-option1-regular-v2` identifies
this rounding correction; it updates the internal version, not the rules filename.
This engine does not adjust the final year's gross payment to reconcile
salary-division rounding.

Run the dedicated tests with the project environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_payroll_calculator.py -v
```

From an activated Windows or macOS virtual environment, run an existing reference
request without creating a new input file:

```sh
python -c "import json; print(json.dumps(json.load(open('tests/fixtures/payroll_reference_cases.json'))['cases'][0]['input']))" | python scripts/payroll_calculator.py
```

Or supply a complete request JSON file:

```sh
python scripts/payroll_calculator.py --input request.json
```

The first reference case explicitly assumes a $50,000 annual salary, 26 periods,
the first payment on 2026-01-09, basic personal claims, full-year CPP eligibility,
EI insurability and complete zero prior YTD values. It returns gross **$1,923.08**,
CPP **$106.41**, CPP2 **$0.00**, EI **$31.35**, federal tax **$153.28**, Ontario tax
**$88.01**, total income tax **$241.29**, total deductions **$379.05** and estimated
net **$1,544.03**. Those amounts describe this stated scenario, not an
arbitrary employee's next paycheck.

`tests/fixtures/payroll_reference_cases.json` has 13 full-payroll reference cases,
computed separately with exact rational arithmetic and literal official constants,
plus published CRA CPP2/exemption component checks. Tests cover monetary and date
validation, TD1 choices, YTD completeness, caps, CPP2 crossing, health-premium and
tax-bracket boundaries, rounding reconciliation, rules failures, CLI behavior and
a full sequence of 26 payments. Expected values were not generated by the payroll
engine. A user-supplied PDOC results screenshot received on 2026-10-19 is recorded
in `pdoc_observations`, with the observed inputs, every displayed deduction and net
pay. The first reference case now matches those amounts exactly. The prior engine
returned Ontario tax $88.00, total income tax $241.28 and net $1,544.04 because
it rounded combined tax first. Rounding each tax separately corrects that one-cent
discrepancy. Regression tests check this observed result directly and preserve both
tax components when separate rounding changes the total in either direction.

**Only this one case has been compared with a user-supplied PDOC result.** There is
no claim of automated PDOC submission, independent screenshot capture, independent
human review, broad PDOC validation or CRA certification. Monthly cases, additional
TD1 configurations, and near-cap scenarios still need representative external
comparisons before treating the engine as payroll-ready. The reference arithmetic
for other cases remains separately computed; the calculator does not produce its
own test expectations.
