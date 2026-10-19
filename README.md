# Ontario Payroll AI Copilot

An agentic RAG assistant for payroll specialists, using fictional
Ontario employee records and verified payroll guidance.

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
Python performs payroll arithmetic.

LangGraph manages conversation state, tool execution,
retrieval retries, stopping conditions, and confirmation pauses.

## Data storage

- SQLite: fictional employee/employment records, effective-dated compensation,
  company calendars, and historical payments. Pending requests remain planned.
- Source documents: verified CRA, Ontario, and sample company policies.
- Qdrant local mode: searchable document sections and their embeddings,
  persisted in `knowledge/vector_store/qdrant/`.

Each document section will retain its source, effective date,
and section identifier so answers can cite the evidence.

## Fictional employee database

The database foundation is implemented separately from the agent. Employee and
calendar MCP tools are not connected yet, so the agent still reports those
capabilities as unavailable. This is a single-employer learning dataset, not a
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
the later MCP layer. No SINs, banking details, tax elections, employee access
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
calendars, payroll calculations, compensation requests and MCP transport are not.
Employee-specific payday requests must return `unsupported`, explaining the
missing connection rather than collecting an employee ID. Salary intake may ask
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

This first agent exposes document search and response submission. Employee data,
calculation tools, request confirmation, and MCP transport remain planned.

## Current status

Knowledge ingestion, filtered retrieval and a terminal LangGraph document
agent are implemented.

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

Document review and effective-date checks are still required before relying
on answers. This collection does not cover every Ontario payroll question.

The fictional employee database foundation is implemented; its agent/MCP
connection remains planned. The browser chat application, payroll calculations,
and MCP tools are not implemented yet.

## Portability and learning

Target Windows, macOS, and Linux.
Plan browser access through Streamlit Community Cloud.

Build incrementally and explain each component.
Obtain approval before every file change.
