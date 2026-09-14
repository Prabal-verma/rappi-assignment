import type {
  Approval,
  ConstraintReport,
  Decision,
  Run,
  Step,
  Validation,
} from './api'

// ─────────────────────────────────────────────────────────────────────
// Small pieces
// ─────────────────────────────────────────────────────────────────────

export function Chip({ kind, children }: { kind: string; children: React.ReactNode }) {
  return <span className={`chip ${kind}`}>{children}</span>
}

const VERDICT_STYLE: Record<string, string> = {
  pass: 'ok',
  drift: 'warn',
  violation: 'bad',
  failed: 'bad',
}

const MODE_STYLE: Record<string, string> = {
  autonomous: 'ok',
  needs_approval: 'warn',
  blocked: 'bad',
}

export function money(n: number | undefined | null): string {
  if (n === undefined || n === null) return '—'
  return `$${n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}

// ─────────────────────────────────────────────────────────────────────
// Decision
// ─────────────────────────────────────────────────────────────────────

export function DecisionCard({ decision }: { decision: Decision }) {
  const verb = decision.decision_type
  return (
    <div className="panel">
      <h2>Decision</h2>
      <div className="body">
        <div className="decision-head">
          <span className={`decision-verb ${verb}`}>{verb.replace(/_/g, ' ').toUpperCase()}</span>
          {decision.recommended_units !== null && decision.recommended_units !== undefined && (
            <span className="qty">
              {decision.recommended_units.toLocaleString()}
              <span className="unit">units</span>
            </span>
          )}
          <div style={{ flex: 1 }} />
          <Chip kind="neutral">confidence {(decision.confidence * 100).toFixed(0)}%</Chip>
          <Chip kind={decision.risk_level === 'high' ? 'bad' : decision.risk_level === 'medium' ? 'warn' : 'ok'}>
            {decision.risk_level} risk
          </Chip>
        </div>

        <p className="rationale">{decision.rationale}</p>

        {decision.key_factors.length > 0 && (
          <>
            <div style={{ fontSize: 11, letterSpacing: '0.06em', color: 'var(--dim)', textTransform: 'uppercase', marginBottom: 4 }}>
              What drove it
            </div>
            <ul className="factors">
              {decision.key_factors.map((f, i) => (
                <li key={i}>{f}</li>
              ))}
            </ul>
          </>
        )}

        {decision.information_gaps.length > 0 && (
          <>
            <div style={{ fontSize: 11, letterSpacing: '0.06em', color: 'var(--warn)', textTransform: 'uppercase', margin: '14px 0 4px' }}>
              Acknowledged gaps
            </div>
            <ul className="factors">
              {decision.information_gaps.map((g, i) => (
                <li key={i}>{g}</li>
              ))}
            </ul>
          </>
        )}

        {decision.proposed_actions.length > 0 && (
          <table style={{ marginTop: 16 }}>
            <thead>
              <tr>
                <th>Proposed action</th>
                <th>Target</th>
                <th style={{ textAlign: 'right' }}>Units</th>
                <th>Why</th>
              </tr>
            </thead>
            <tbody>
              {decision.proposed_actions.map((a, i) => (
                <tr key={i}>
                  <td>
                    <code>{a.action_type}</code>
                  </td>
                  <td>
                    {a.supplier_id ?? a.po_id ?? a.from_node_id ?? '—'}
                    {a.node_id ? ` → ${a.node_id}` : ''}
                  </td>
                  <td className="num">{a.units?.toLocaleString() ?? '—'}</td>
                  <td style={{ color: 'var(--muted)' }}>{a.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────
// Expectation vs reality — the feedback loop, made visible
// ─────────────────────────────────────────────────────────────────────

export function ValidationCard({ validation }: { validation: Validation }) {
  const style = VERDICT_STYLE[validation.verdict] ?? 'neutral'
  return (
    <div className="panel">
      <h2>
        Validation
        <Chip kind={style}>{validation.verdict.toUpperCase()}</Chip>
      </h2>
      <div className="body">
        <div className="note" style={{ marginTop: 0, marginBottom: 12 }}>
          Ground truth was re-read from the database after the action and compared against the
          outcome the agent committed to <em>before</em> acting.
        </div>

        {validation.expectation_diffs.length > 0 && (
          <table>
            <thead>
              <tr>
                <th>Measure</th>
                <th style={{ textAlign: 'right' }}>Predicted</th>
                <th style={{ textAlign: 'right' }}>Actual</th>
                <th style={{ textAlign: 'right' }}>Tolerance</th>
                <th style={{ textAlign: 'right' }}>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {validation.expectation_diffs.map((d) => (
                <tr key={d.field} className={d.within_tolerance ? 'row-ok' : 'row-bad'}>
                  <td>{d.field.replace(/_/g, ' ')}</td>
                  <td className="num">{d.expected.toLocaleString()}</td>
                  <td className="num">{d.actual.toLocaleString()}</td>
                  <td className="num">±{d.tolerance.toLocaleString()}</td>
                  <td className="num">
                    <Chip kind={d.within_tolerance ? 'ok' : 'bad'}>
                      {d.within_tolerance ? 'within' : 'drift'}
                    </Chip>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {validation.findings.length > 0 && (
          <ul className="factors" style={{ marginTop: 12 }}>
            {validation.findings.map((f, i) => (
              <li key={i}>{f}</li>
            ))}
          </ul>
        )}

        {validation.corrective_guidance && (
          <div className="violation warn" style={{ marginTop: 12 }}>
            <div className="code">FED BACK TO THE AGENT</div>
            <div className="msg">{validation.corrective_guidance}</div>
          </div>
        )}

        <ConstraintChecks report={validation.constraint_report} />
      </div>
    </div>
  )
}

export function ConstraintChecks({ report }: { report: ConstraintReport }) {
  if (!report) return null
  const failed = new Set(report.blocking.map((v) => v.code))
  return (
    <>
      {report.blocking.map((v, i) => (
        <div className="violation" key={`b${i}`} style={{ marginTop: 12 }}>
          <div className="code">
            {v.code} · BLOCKING
            {v.observed !== null && v.limit !== null ? ` · ${v.observed} vs limit ${v.limit} ${v.unit}` : ''}
          </div>
          <div className="msg">{v.message}</div>
          {v.remedy_hint && <div className="hint">{v.remedy_hint}</div>}
        </div>
      ))}
      {report.warnings.map((v, i) => (
        <div className="violation warn" key={`w${i}`} style={{ marginTop: 12 }}>
          <div className="code">{v.code} · WARNING</div>
          <div className="msg">{v.message}</div>
          {v.remedy_hint && <div className="hint">{v.remedy_hint}</div>}
        </div>
      ))}
      {report.checks_run.length > 0 && (
        <>
          <div className="note">
            {report.checks_run.length} constraint checks ran against the persisted state:
          </div>
          <div className="checks">
            {report.checks_run.map((c) => (
              <span key={c} className={`check-pill ${failed.has(c) ? 'failed' : ''}`}>
                {c}
              </span>
            ))}
          </div>
        </>
      )}
    </>
  )
}

// ─────────────────────────────────────────────────────────────────────
// Authorization
// ─────────────────────────────────────────────────────────────────────

export function AuthorizationCard({ auth }: { auth: NonNullable<Run['result']>['authorization'] }) {
  if (!auth) return null
  return (
    <div className="panel">
      <h2>
        Authorization
        <Chip kind={MODE_STYLE[auth.mode] ?? 'neutral'}>{auth.mode.replace(/_/g, ' ')}</Chip>
      </h2>
      <div className="body">
        <ul className="factors" style={{ marginTop: 0 }}>
          {auth.reasons.map((r, i) => (
            <li key={i}>{r}</li>
          ))}
        </ul>
        {auth.policy_refs.length > 0 && (
          <div className="checks">
            {auth.policy_refs.map((p) => (
              <span key={p} className="check-pill">
                {p}
              </span>
            ))}
          </div>
        )}
        <div className="note">
          Decided in code by the constraint engine and the autonomy policy, not by the model. The
          agent proposes; this decides whether it may execute.
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────
// Trace
// ─────────────────────────────────────────────────────────────────────

export function TraceCard({ steps }: { steps: Step[] }) {
  return (
    <div className="panel">
      <h2>
        Decision trace
        <Chip kind="neutral">{steps.length} steps</Chip>
      </h2>
      <div className="body flush trace">
        {steps.map((s) => (
          <div className={`step kind-${s.kind}`} key={s.seq}>
            <span className="seq">{s.seq}</span>
            <span className={`node node-${s.graph_node}`}>{s.graph_node}</span>
            <span className="title">{s.title}</span>
            <span className="ms">{s.latency_ms ? `${s.latency_ms}ms` : ''}</span>
          </div>
        ))}
        {steps.length === 0 && <div className="empty">No steps recorded.</div>}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────
// Approvals
// ─────────────────────────────────────────────────────────────────────

export function ApprovalCard({
  approval,
  onDecide,
  busy,
}: {
  approval: Approval
  onDecide: (id: number, approve: boolean) => void
  busy: boolean
}) {
  const decision = approval.proposed_action?.decision
  const preflight = approval.proposed_action?.preflight_validation
  const pending = approval.status === 'pending'

  return (
    <div className="panel">
      <h2>
        Approval #{approval.id}
        <Chip kind={approval.risk_level === 'high' ? 'bad' : 'warn'}>{approval.risk_level} risk</Chip>
        <Chip kind={pending ? 'warn' : approval.status === 'approved' ? 'ok' : 'neutral'}>
          {approval.status}
        </Chip>
      </h2>
      <div className="body">
        <div style={{ fontSize: 11, letterSpacing: '0.06em', color: 'var(--dim)', textTransform: 'uppercase', marginBottom: 4 }}>
          Why a human is needed
        </div>
        <ul className="factors" style={{ marginTop: 0 }}>
          {approval.reason.split('\n').filter(Boolean).map((r, i) => (
            <li key={i}>{r}</li>
          ))}
        </ul>

        {approval.proposed_action?.actions?.length > 0 && (
          <table style={{ marginTop: 14 }}>
            <thead>
              <tr>
                <th>Executes on approval</th>
                <th>Target</th>
                <th style={{ textAlign: 'right' }}>Units</th>
              </tr>
            </thead>
            <tbody>
              {approval.proposed_action.actions.map((a, i) => (
                <tr key={i}>
                  <td>
                    <code>{a.action_type}</code>
                  </td>
                  <td>
                    {a.supplier_id ?? a.po_id ?? '—'} {a.node_id ? `→ ${a.node_id}` : ''}
                  </td>
                  <td className="num">{a.units?.toLocaleString() ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {preflight && (
          <div className="violation warn" style={{ marginTop: 14, borderLeftColor: 'var(--info)', background: 'rgba(88,166,255,0.07)' }}>
            <div className="code">PRE-FLIGHT PROJECTION · {preflight.verdict.toUpperCase()}</div>
            {preflight.findings.map((f, i) => (
              <div className="msg" key={i}>
                {f}
              </div>
            ))}
          </div>
        )}

        {decision && (
          <div className="note">
            <strong style={{ color: 'var(--text)' }}>Agent's reasoning:</strong> {decision.rationale}
          </div>
        )}

        {pending && (
          <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
            <button className="btn ok" disabled={busy} onClick={() => onDecide(approval.id, true)}>
              {busy ? 'Executing…' : 'Approve & execute'}
            </button>
            <button className="btn bad" disabled={busy} onClick={() => onDecide(approval.id, false)}>
              Reject
            </button>
          </div>
        )}

        <div className="note">
          The action above executes verbatim on approval — it is not re-derived by the model, so
          you are approving this exact purchase order and no other. It is then validated the same
          way an autonomous action would be.
        </div>
      </div>
    </div>
  )
}
