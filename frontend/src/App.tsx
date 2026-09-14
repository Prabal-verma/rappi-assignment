import { useCallback, useEffect, useState } from 'react'
import {
  api,
  type Approval,
  type Health,
  type PurchaseOrder,
  type Run,
  type Scenario,
} from './api'
import {
  ApprovalCard,
  AuthorizationCard,
  Chip,
  DecisionCard,
  TraceCard,
  ValidationCard,
  money,
} from './components'

type Tab = 'decision' | 'trace' | 'approvals' | 'world'

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [scenarios, setScenarios] = useState<Scenario[]>([])
  const [run, setRun] = useState<Run | null>(null)
  const [runs, setRuns] = useState<Run[]>([])
  const [approvals, setApprovals] = useState<Approval[]>([])
  const [pos, setPos] = useState<PurchaseOrder[]>([])
  const [budgets, setBudgets] = useState<Awaited<ReturnType<typeof api.budgets>>['budgets']>([])
  const [nodes, setNodes] = useState<Awaited<ReturnType<typeof api.nodes>>['nodes']>([])

  const [running, setRunning] = useState<string | null>(null)
  const [busyApproval, setBusyApproval] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [tab, setTab] = useState<Tab>('decision')

  const refresh = useCallback(async () => {
    try {
      const [a, p, b, n, r] = await Promise.all([
        api.approvals(),
        api.purchaseOrders(),
        api.budgets(),
        api.nodes(),
        api.runs(),
      ])
      setApprovals(a.approvals)
      setPos(p.purchase_orders)
      setBudgets(b.budgets)
      setNodes(n.nodes)
      setRuns(r.runs)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setError(String(e)))
    api.scenarios().then((d) => setScenarios(d.scenarios)).catch((e) => setError(String(e)))
    refresh()
  }, [refresh])

  async function start(scenarioId: string) {
    setRunning(scenarioId)
    setError(null)
    try {
      const result = await api.startRun(scenarioId)
      setRun(result)
      setTab('decision')
      await refresh()
    } catch (e) {
      setError(String(e))
    } finally {
      setRunning(null)
    }
  }

  async function reset() {
    setError(null)
    try {
      await api.reset()
      setRun(null)
      await refresh()
    } catch (e) {
      setError(String(e))
    }
  }

  async function decide(id: number, approve: boolean) {
    setBusyApproval(true)
    setError(null)
    try {
      await api.decideApproval(id, approve)
      await refresh()
      if (run) setRun(await api.run(run.run_id))
    } catch (e) {
      setError(String(e))
    } finally {
      setBusyApproval(false)
    }
  }

  const result = run?.result
  const pendingApprovals = approvals.filter((a) => a.status === 'pending')

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>AI Purchasing Agent</h1>
          <span className="sub">quick-commerce replenishment · decide, execute, validate</span>
        </div>
        <div className="spacer" />
        {health && (
          <>
            <Chip kind={health.llm_live ? 'violet' : 'neutral'}>
              {health.llm_live ? `${health.llm_provider}:${health.llm_model}` : 'rules engine (no LLM key)'}
            </Chip>
            <Chip kind="neutral">autonomy ≤ {money(health.autonomy_ceiling_usd)}</Chip>
          </>
        )}
        <button className="btn ghost" onClick={reset}>
          Reset world
        </button>
      </header>

      {error && <div className="err">{error}</div>}

      <div className="layout">
        {/* ---- left rail ---- */}
        <div>
          <div className="panel">
            <h2>Scenarios</h2>
            <div className="body flush">
              {scenarios.map((s) => (
                <button
                  key={s.scenario_id}
                  className={`scenario ${run?.scenario_id === s.scenario_id ? 'active' : ''}`}
                  disabled={running !== null}
                  onClick={() => start(s.scenario_id)}
                >
                  <span className="id">
                    {s.scenario_id}
                    {running === s.scenario_id && <span className="spin" style={{ marginLeft: 8 }} />}
                  </span>
                  <div className="title">{s.title}</div>
                  <div className="summary">{s.summary}</div>
                </button>
              ))}
              {scenarios.length === 0 && <div className="empty">Loading…</div>}
            </div>
          </div>

          {runs.length > 0 && (
            <div className="panel">
              <h2>Recent runs</h2>
              <div className="body flush">
                {runs.slice(0, 8).map((r) => (
                  <button
                    key={r.run_id}
                    className={`scenario ${run?.run_id === r.run_id ? 'active' : ''}`}
                    onClick={async () => {
                      setRun(await api.run(r.run_id))
                      setTab('decision')
                    }}
                  >
                    <span className="id">{r.scenario_id}</span>
                    <div className="title" style={{ fontSize: 12.5 }}>
                      {r.status.replace(/_/g, ' ')}
                    </div>
                    <div className="summary" style={{ fontSize: 11.5 }}>
                      {r.provider}:{r.model} · {r.attempts} replan(s) · {(r.duration_ms / 1000).toFixed(1)}s
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ---- main ---- */}
        <div>
          {!run && (
            <div className="panel">
              <h2>How this works</h2>
              <div className="body">
                <p style={{ marginTop: 0 }}>
                  Pick a scenario. The agent investigates with read-only tools, is held to a
                  required evidence checklist, commits to a numeric prediction of the outcome,
                  then acts only if the constraint engine and autonomy policy allow it. Afterwards
                  the system re-reads the database and holds reality against that prediction.
                </p>
                <p className="note" style={{ marginTop: 12 }}>
                  Every scenario is built so the obvious answer is wrong. The recommendation is a
                  hypothesis to test, not an instruction to rationalise.
                </p>
              </div>
            </div>
          )}

          {run && (
            <>
              <div className="tabs">
                <button className={`tab ${tab === 'decision' ? 'active' : ''}`} onClick={() => setTab('decision')}>
                  Decision
                </button>
                <button className={`tab ${tab === 'trace' ? 'active' : ''}`} onClick={() => setTab('trace')}>
                  Trace <span className="count">{run.steps.length}</span>
                </button>
                <button className={`tab ${tab === 'approvals' ? 'active' : ''}`} onClick={() => setTab('approvals')}>
                  Approvals
                  {pendingApprovals.length > 0 && <span className="count">{pendingApprovals.length}</span>}
                </button>
                <button className={`tab ${tab === 'world' ? 'active' : ''}`} onClick={() => setTab('world')}>
                  World state
                </button>
              </div>

              {tab === 'decision' && (
                <>
                  <div className="panel">
                    <h2>
                      Run {run.run_id}
                      <Chip kind={run.status === 'completed' ? 'ok' : run.status.includes('approval') ? 'warn' : 'neutral'}>
                        {run.status.replace(/_/g, ' ')}
                      </Chip>
                      {result?.llm_degraded && <Chip kind="warn">degraded to rules engine</Chip>}
                    </h2>
                    <div className="body flush">
                      <div className="metrics">
                        <div className="metric">
                          <div className="k">Scenario</div>
                          <div className="v small">{run.scenario_id}</div>
                        </div>
                        <div className="metric">
                          <div className="k">Reasoner</div>
                          <div className="v small">{run.provider}</div>
                        </div>
                        <div className="metric">
                          <div className="k">Replans</div>
                          <div className="v">{run.attempts}</div>
                        </div>
                        <div className="metric">
                          <div className="k">Evidence</div>
                          <div className="v">{result?.evidence_slots?.length ?? 0}</div>
                        </div>
                        <div className="metric">
                          <div className="k">Duration</div>
                          <div className="v">{(run.duration_ms / 1000).toFixed(1)}s</div>
                        </div>
                        <div className="metric">
                          <div className="k">Steps</div>
                          <div className="v">{run.steps.length}</div>
                        </div>
                      </div>
                      {result?.untrusted_inputs_seen && result.untrusted_inputs_seen.length > 0 && (
                        <div style={{ padding: '12px 16px', borderTop: '1px solid var(--line)' }}>
                          <Chip kind="warn">untrusted input handled</Chip>
                          <span className="note" style={{ marginLeft: 10 }}>
                            Third-party supplier content was read via{' '}
                            <code>{result.untrusted_inputs_seen.join(', ')}</code> and treated as
                            data, never as instruction.
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  {result?.decision && <DecisionCard decision={result.decision} />}
                  {result?.authorization && <AuthorizationCard auth={result.authorization} />}

                  {result?.action_results && result.action_results.length > 0 && (
                    <div className="panel">
                      <h2>What the world did back</h2>
                      <div className="body flush">
                        <table>
                          <thead>
                            <tr>
                              <th>Action</th>
                              <th>Entity</th>
                              <th>Outcome</th>
                              <th style={{ textAlign: 'right' }}>Asked</th>
                              <th style={{ textAlign: 'right' }}>Got</th>
                            </tr>
                          </thead>
                          <tbody>
                            {result.action_results.map((r, i) => {
                              const asked = r.action.units
                              const got = r.confirmed_units
                              const shortfall = asked && got !== null && got < asked
                              return (
                                <tr key={i} className={shortfall ? 'row-bad' : ''}>
                                  <td>
                                    <code>{r.action.action_type}</code>
                                  </td>
                                  <td>{r.entity_id ?? '—'}</td>
                                  <td>
                                    <Chip kind={r.outcome === 'confirmed' ? 'ok' : shortfall ? 'warn' : 'neutral'}>
                                      {r.outcome.replace(/_/g, ' ')}
                                    </Chip>
                                  </td>
                                  <td className="num">{asked?.toLocaleString() ?? '—'}</td>
                                  <td className="num">{got?.toLocaleString() ?? '—'}</td>
                                </tr>
                              )
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {result?.validation && <ValidationCard validation={result.validation} />}
                </>
              )}

              {tab === 'trace' && <TraceCard steps={run.steps} />}

              {tab === 'approvals' && (
                <>
                  {approvals.length === 0 && (
                    <div className="panel">
                      <div className="empty">
                        Nothing queued. Actions inside the autonomy policy execute without a human.
                      </div>
                    </div>
                  )}
                  {approvals.map((a) => (
                    <ApprovalCard key={a.id} approval={a} onDecide={decide} busy={busyApproval} />
                  ))}
                </>
              )}

              {tab === 'world' && (
                <>
                  <div className="panel">
                    <h2>Purchase orders</h2>
                    <div className="body flush">
                      <table>
                        <thead>
                          <tr>
                            <th>PO</th>
                            <th>Supplier</th>
                            <th>Node</th>
                            <th>Status</th>
                            <th style={{ textAlign: 'right' }}>Ordered</th>
                            <th style={{ textAlign: 'right' }}>Confirmed</th>
                            <th style={{ textAlign: 'right' }}>Committed</th>
                            <th>Raised by</th>
                          </tr>
                        </thead>
                        <tbody>
                          {pos.map((po) =>
                            po.lines.map((l, i) => (
                              <tr key={`${po.po_id}-${i}`} className={po.source_run_id ? 'row-ok' : ''}>
                                <td>
                                  <code>{po.po_id}</code>
                                </td>
                                <td>{po.supplier_id}</td>
                                <td>{po.node_id}</td>
                                <td>
                                  <Chip
                                    kind={
                                      po.status === 'confirmed'
                                        ? 'ok'
                                        : po.status === 'partially_confirmed'
                                          ? 'warn'
                                          : po.status === 'cancelled'
                                            ? 'bad'
                                            : 'neutral'
                                    }
                                  >
                                    {po.status.replace(/_/g, ' ')}
                                  </Chip>
                                </td>
                                <td className="num">{l.ordered_units.toLocaleString()}</td>
                                <td className="num">{l.confirmed_units?.toLocaleString() ?? '—'}</td>
                                <td className="num">{money(po.committed_value_usd)}</td>
                                <td style={{ color: 'var(--muted)', fontSize: 11.5 }}>
                                  {po.source_run_id ? 'agent' : po.created_by}
                                </td>
                              </tr>
                            )),
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  <div className="panel">
                    <h2>Budgets</h2>
                    <div className="body flush">
                      <table>
                        <thead>
                          <tr>
                            <th>Node</th>
                            <th>Category</th>
                            <th style={{ textAlign: 'right' }}>Allocated</th>
                            <th style={{ textAlign: 'right' }}>Committed</th>
                            <th style={{ textAlign: 'right' }}>Spent</th>
                            <th style={{ textAlign: 'right' }}>Available</th>
                          </tr>
                        </thead>
                        <tbody>
                          {budgets.map((b, i) => (
                            <tr key={i} className={b.available_usd < 0 ? 'row-bad' : ''}>
                              <td>{b.node_id}</td>
                              <td>{b.category}</td>
                              <td className="num">{money(b.allocated_usd)}</td>
                              <td className="num">{money(b.committed_usd)}</td>
                              <td className="num">{money(b.spent_usd)}</td>
                              <td className="num">{money(b.available_usd)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  <div className="panel">
                    <h2>Node capacity</h2>
                    <div className="body flush">
                      <table>
                        <thead>
                          <tr>
                            <th>Node</th>
                            <th>Name</th>
                            <th style={{ textAlign: 'right' }}>Capacity m³</th>
                            <th style={{ textAlign: 'right' }}>On hand</th>
                            <th style={{ textAlign: 'right' }}>Inbound</th>
                            <th style={{ textAlign: 'right' }}>Free</th>
                            <th style={{ textAlign: 'right' }}>Used</th>
                          </tr>
                        </thead>
                        <tbody>
                          {nodes.map((n) => (
                            <tr key={n.node_id} className={n.free_m3 < 0 ? 'row-bad' : ''}>
                              <td>
                                <code>{n.node_id}</code>
                              </td>
                              <td>{n.name}</td>
                              <td className="num">{n.capacity_m3.toFixed(1)}</td>
                              <td className="num">{n.used_m3.toFixed(2)}</td>
                              <td className="num">{n.inbound_m3.toFixed(2)}</td>
                              <td className="num">{n.free_m3.toFixed(2)}</td>
                              <td className="num">{n.utilisation_pct}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
