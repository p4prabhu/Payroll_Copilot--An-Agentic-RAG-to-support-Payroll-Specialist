-- Single-employer fictional payroll dataset. Dates use ISO format.
-- Employment and effective-date ranges are [start, end): end is excluded.
-- Run period_start/period_end are inclusive calendar dates.
PRAGMA foreign_keys = ON;
PRAGMA user_version = 1;

CREATE TABLE dataset_info (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    employer_name TEXT NOT NULL,
    is_fictional INTEGER NOT NULL CHECK (is_fictional = 1),
    snapshot_date TEXT NOT NULL CHECK (date(snapshot_date, '+0 days') IS snapshot_date),
    history_complete_from TEXT NOT NULL CHECK (date(history_complete_from, '+0 days') IS history_complete_from),
    description TEXT NOT NULL
) STRICT;

CREATE TABLE employees (
    employee_id TEXT PRIMARY KEY,
    employee_number TEXT NOT NULL UNIQUE,
    given_name TEXT NOT NULL,
    family_name TEXT NOT NULL,
    created_at TEXT NOT NULL
) STRICT;

CREATE TABLE employments (
    employment_id TEXT PRIMARY KEY,
    employee_id TEXT NOT NULL REFERENCES employees(employee_id),
    starts_on TEXT NOT NULL CHECK (date(starts_on, '+0 days') IS starts_on),
    ends_before TEXT CHECK (ends_before IS NULL OR date(ends_before, '+0 days') IS ends_before),
    province_code TEXT NOT NULL CHECK (province_code = 'ON'),
    currency TEXT NOT NULL CHECK (currency = 'CAD'),
    CHECK (ends_before IS NULL OR starts_on < ends_before),
    UNIQUE (employee_id, starts_on)
) STRICT;

CREATE TABLE pay_groups (
    pay_group_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    frequency TEXT NOT NULL CHECK (frequency IN ('weekly', 'biweekly', 'semimonthly', 'monthly')),
    periods_per_year INTEGER NOT NULL,
    timezone TEXT NOT NULL CHECK (timezone = 'America/Toronto'),
    CHECK ((frequency = 'weekly' AND periods_per_year = 52)
        OR (frequency = 'biweekly' AND periods_per_year = 26)
        OR (frequency = 'semimonthly' AND periods_per_year = 24)
        OR (frequency = 'monthly' AND periods_per_year = 12))
) STRICT;

CREATE TABLE pay_group_assignments (
    assignment_id TEXT PRIMARY KEY,
    employment_id TEXT NOT NULL REFERENCES employments(employment_id),
    pay_group_id TEXT NOT NULL REFERENCES pay_groups(pay_group_id),
    valid_from TEXT NOT NULL CHECK (date(valid_from, '+0 days') IS valid_from),
    valid_to TEXT CHECK (valid_to IS NULL OR date(valid_to, '+0 days') IS valid_to),
    CHECK (valid_to IS NULL OR valid_from < valid_to),
    UNIQUE (employment_id, valid_from)
) STRICT;

CREATE TABLE compensation_rates (
    compensation_id TEXT PRIMARY KEY,
    employment_id TEXT NOT NULL REFERENCES employments(employment_id),
    pay_basis TEXT NOT NULL CHECK (pay_basis = 'annual_salary'),
    annual_salary_cents INTEGER NOT NULL CHECK (annual_salary_cents > 0),
    currency TEXT NOT NULL CHECK (currency = 'CAD'),
    valid_from TEXT NOT NULL CHECK (date(valid_from, '+0 days') IS valid_from),
    valid_to TEXT CHECK (valid_to IS NULL OR date(valid_to, '+0 days') IS valid_to),
    change_reason TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    CHECK (valid_to IS NULL OR valid_from < valid_to),
    UNIQUE (employment_id, valid_from)
) STRICT;

CREATE TABLE pay_runs (
    pay_run_id TEXT PRIMARY KEY,
    pay_group_id TEXT NOT NULL REFERENCES pay_groups(pay_group_id),
    period_start TEXT NOT NULL CHECK (date(period_start, '+0 days') IS period_start),
    period_end TEXT NOT NULL CHECK (date(period_end, '+0 days') IS period_end),
    payment_date TEXT NOT NULL CHECK (date(payment_date, '+0 days') IS payment_date),
    cutoff_at_utc TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('draft', 'scheduled', 'posted')),
    CHECK (period_start <= period_end),
    CHECK (substr(cutoff_at_utc, 1, 10) <= payment_date),
    UNIQUE (pay_group_id, period_start, period_end)
) STRICT;

CREATE TABLE payroll_payments (
    payment_id TEXT PRIMARY KEY,
    employment_id TEXT NOT NULL REFERENCES employments(employment_id),
    pay_run_id TEXT NOT NULL REFERENCES pay_runs(pay_run_id),
    compensation_id TEXT NOT NULL REFERENCES compensation_rates(compensation_id),
    payment_type TEXT NOT NULL CHECK (payment_type IN ('regular', 'bonus')),
    province_code TEXT NOT NULL CHECK (province_code = 'ON'),
    currency TEXT NOT NULL CHECK (currency = 'CAD'),
    gross_cents INTEGER NOT NULL CHECK (gross_cents >= 0),
    deduction_cents INTEGER NOT NULL CHECK (deduction_cents >= 0),
    net_cents INTEGER NOT NULL CHECK (net_cents >= 0 AND net_cents = gross_cents - deduction_cents),
    status TEXT NOT NULL CHECK (status IN ('draft', 'posted')),
    recorded_at TEXT NOT NULL,
    recorded_by TEXT NOT NULL,
    UNIQUE (employment_id, pay_run_id, payment_type)
) STRICT;

CREATE TABLE payroll_lines (
    payment_id TEXT NOT NULL REFERENCES payroll_payments(payment_id),
    line_number INTEGER NOT NULL CHECK (line_number > 0),
    category TEXT NOT NULL CHECK (category IN ('earning', 'deduction')),
    code TEXT NOT NULL,
    amount_cents INTEGER NOT NULL CHECK (amount_cents >= 0),
    PRIMARY KEY (payment_id, line_number),
    UNIQUE (payment_id, code),
    CHECK ((category = 'earning' AND code IN ('SALARY', 'BONUS'))
        OR (category = 'deduction' AND code IN ('FED_TAX', 'ON_TAX', 'CPP', 'CPP2', 'EI', 'BENEFITS')))
) STRICT;

CREATE INDEX employment_employee ON employments(employee_id, starts_on, ends_before);
CREATE INDEX assignment_as_of ON pay_group_assignments(employment_id, valid_from, valid_to);
CREATE INDEX compensation_as_of ON compensation_rates(employment_id, valid_from, valid_to);
CREATE INDEX run_payment_date ON pay_runs(pay_group_id, payment_date, status);
CREATE INDEX payment_history ON payroll_payments(employment_id, pay_run_id, status);

CREATE TRIGGER employment_no_overlap_insert BEFORE INSERT ON employments
WHEN EXISTS (SELECT 1 FROM employments e WHERE e.employee_id = NEW.employee_id
 AND NEW.starts_on < COALESCE(e.ends_before, '9999-12-31')
 AND e.starts_on < COALESCE(NEW.ends_before, '9999-12-31'))
BEGIN SELECT RAISE(ABORT, 'overlapping employment dates'); END;

CREATE TRIGGER employment_no_overlap_update BEFORE UPDATE ON employments
WHEN EXISTS (SELECT 1 FROM employments e WHERE e.employee_id = NEW.employee_id AND e.employment_id <> OLD.employment_id
 AND NEW.starts_on < COALESCE(e.ends_before, '9999-12-31')
 AND e.starts_on < COALESCE(NEW.ends_before, '9999-12-31'))
BEGIN SELECT RAISE(ABORT, 'overlapping employment dates'); END;

CREATE TRIGGER assignment_no_overlap_insert BEFORE INSERT ON pay_group_assignments
WHEN EXISTS (SELECT 1 FROM pay_group_assignments a WHERE a.employment_id = NEW.employment_id
 AND NEW.valid_from < COALESCE(a.valid_to, '9999-12-31')
 AND a.valid_from < COALESCE(NEW.valid_to, '9999-12-31'))
BEGIN SELECT RAISE(ABORT, 'overlapping pay-group assignments'); END;

CREATE TRIGGER assignment_no_overlap_update BEFORE UPDATE ON pay_group_assignments
WHEN EXISTS (SELECT 1 FROM pay_group_assignments a WHERE a.employment_id = NEW.employment_id AND a.assignment_id <> OLD.assignment_id
 AND NEW.valid_from < COALESCE(a.valid_to, '9999-12-31')
 AND a.valid_from < COALESCE(NEW.valid_to, '9999-12-31'))
BEGIN SELECT RAISE(ABORT, 'overlapping pay-group assignments'); END;

CREATE TRIGGER compensation_no_overlap_insert BEFORE INSERT ON compensation_rates
WHEN EXISTS (SELECT 1 FROM compensation_rates c WHERE c.employment_id = NEW.employment_id
 AND NEW.valid_from < COALESCE(c.valid_to, '9999-12-31')
 AND c.valid_from < COALESCE(NEW.valid_to, '9999-12-31'))
BEGIN SELECT RAISE(ABORT, 'overlapping compensation dates'); END;

CREATE TRIGGER compensation_no_overlap_update BEFORE UPDATE ON compensation_rates
WHEN EXISTS (SELECT 1 FROM compensation_rates c WHERE c.employment_id = NEW.employment_id AND c.compensation_id <> OLD.compensation_id
 AND NEW.valid_from < COALESCE(c.valid_to, '9999-12-31')
 AND c.valid_from < COALESCE(NEW.valid_to, '9999-12-31'))
BEGIN SELECT RAISE(ABORT, 'overlapping compensation dates'); END;

CREATE TRIGGER payment_start_draft BEFORE INSERT ON payroll_payments WHEN NEW.status <> 'draft'
BEGIN SELECT RAISE(ABORT, 'payments must be assembled as draft before posting'); END;
CREATE TRIGGER payment_identity_no_update BEFORE UPDATE OF payment_id ON payroll_payments
WHEN NEW.payment_id <> OLD.payment_id
BEGIN SELECT RAISE(ABORT, 'payment identity cannot change'); END;
CREATE TRIGGER payment_posted_no_update BEFORE UPDATE ON payroll_payments WHEN OLD.status = 'posted'
BEGIN SELECT RAISE(ABORT, 'posted payment is immutable'); END;
CREATE TRIGGER payment_posted_no_delete BEFORE DELETE ON payroll_payments WHEN OLD.status = 'posted'
BEGIN SELECT RAISE(ABORT, 'posted payment is immutable'); END;

CREATE TRIGGER payment_validate_posting BEFORE UPDATE OF status ON payroll_payments WHEN NEW.status = 'posted'
BEGIN
 SELECT CASE WHEN NEW.payment_id <> OLD.payment_id THEN RAISE(ABORT, 'payment identity cannot change') END;
 SELECT CASE WHEN NEW.gross_cents <> COALESCE((SELECT SUM(amount_cents) FROM payroll_lines WHERE payment_id = OLD.payment_id AND category = 'earning'), 0)
   OR NEW.deduction_cents <> COALESCE((SELECT SUM(amount_cents) FROM payroll_lines WHERE payment_id = OLD.payment_id AND category = 'deduction'), 0)
   OR NOT EXISTS (SELECT 1 FROM payroll_lines WHERE payment_id = OLD.payment_id AND category = 'earning')
 THEN RAISE(ABORT, 'payment totals do not match lines') END;
 SELECT CASE WHEN NOT EXISTS (
   SELECT 1 FROM pay_runs r JOIN employments e ON e.employment_id = NEW.employment_id
   JOIN compensation_rates c ON c.compensation_id = NEW.compensation_id AND c.employment_id = e.employment_id
   JOIN pay_group_assignments a ON a.employment_id = e.employment_id AND a.pay_group_id = r.pay_group_id
   WHERE r.pay_run_id = NEW.pay_run_id AND r.status = 'posted'
   AND e.starts_on <= r.period_end AND (e.ends_before IS NULL OR r.period_end < e.ends_before)
   AND a.valid_from <= r.period_end AND (a.valid_to IS NULL OR r.period_end < a.valid_to)
   AND c.valid_from <= r.period_end AND (c.valid_to IS NULL OR r.period_end < c.valid_to)
 ) THEN RAISE(ABORT, 'payment employment, compensation or pay group is invalid for run') END;
END;

CREATE TRIGGER line_no_insert_posted BEFORE INSERT ON payroll_lines
WHEN (SELECT status FROM payroll_payments WHERE payment_id = NEW.payment_id) = 'posted'
BEGIN SELECT RAISE(ABORT, 'posted payment lines are immutable'); END;
CREATE TRIGGER line_no_update_posted BEFORE UPDATE ON payroll_lines
WHEN (SELECT status FROM payroll_payments WHERE payment_id = OLD.payment_id) = 'posted'
 OR (SELECT status FROM payroll_payments WHERE payment_id = NEW.payment_id) = 'posted'
BEGIN SELECT RAISE(ABORT, 'posted payment lines are immutable'); END;
CREATE TRIGGER line_no_delete_posted BEFORE DELETE ON payroll_lines
WHEN (SELECT status FROM payroll_payments WHERE payment_id = OLD.payment_id) = 'posted'
BEGIN SELECT RAISE(ABORT, 'posted payment lines are immutable'); END;

CREATE TRIGGER posted_run_no_update BEFORE UPDATE ON pay_runs WHEN OLD.status = 'posted'
BEGIN SELECT RAISE(ABORT, 'posted payroll run is immutable'); END;
CREATE TRIGGER posted_run_no_delete BEFORE DELETE ON pay_runs WHEN OLD.status = 'posted'
BEGIN SELECT RAISE(ABORT, 'posted payroll run is immutable'); END;

CREATE TRIGGER posted_compensation_snapshot_no_update
BEFORE UPDATE OF compensation_id, employment_id, pay_basis, annual_salary_cents, currency, valid_from ON compensation_rates
WHEN EXISTS (SELECT 1 FROM payroll_payments p WHERE p.compensation_id = OLD.compensation_id AND p.status = 'posted')
BEGIN SELECT RAISE(ABORT, 'compensation referenced by posted history cannot be rewritten'); END;
CREATE TRIGGER posted_compensation_end_valid BEFORE UPDATE OF valid_to ON compensation_rates
WHEN NEW.valid_to IS NOT NULL AND EXISTS (
 SELECT 1 FROM payroll_payments p JOIN pay_runs r USING (pay_run_id)
 WHERE p.compensation_id = OLD.compensation_id AND p.status = 'posted' AND r.period_end >= NEW.valid_to)
BEGIN SELECT RAISE(ABORT, 'compensation end cannot invalidate posted history'); END;

CREATE TRIGGER posted_employment_owner_no_update BEFORE UPDATE OF employee_id ON employments
WHEN NEW.employee_id <> OLD.employee_id AND EXISTS (
 SELECT 1 FROM payroll_payments p WHERE p.employment_id = OLD.employment_id AND p.status = 'posted')
BEGIN SELECT RAISE(ABORT, 'posted employment history cannot be reassigned to another employee'); END;
CREATE TRIGGER posted_employment_dates_valid BEFORE UPDATE OF starts_on, ends_before ON employments
WHEN EXISTS (
 SELECT 1 FROM payroll_payments p JOIN pay_runs r USING (pay_run_id)
 WHERE p.employment_id = OLD.employment_id AND p.status = 'posted'
 AND (r.period_end < NEW.starts_on OR (NEW.ends_before IS NOT NULL AND r.period_end >= NEW.ends_before)))
BEGIN SELECT RAISE(ABORT, 'employment dates cannot invalidate posted history'); END;

-- Not a full YTD total: this view deliberately labels the fixture's partial coverage.
CREATE VIEW payroll_history AS
 SELECT p.*, e.employee_id, r.payment_date, r.period_start, r.period_end,
        d.is_fictional, d.history_complete_from, d.snapshot_date
 FROM payroll_payments p JOIN employments e USING (employment_id)
 JOIN pay_runs r USING (pay_run_id) CROSS JOIN dataset_info d
 WHERE p.status = 'posted';
