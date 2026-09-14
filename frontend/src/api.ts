const BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!response.ok) {
    const body = await response.text()
    throw new Error(`${response.status} ${response.statusText}: ${body.slice(0, 300)}`)
  }
  return response.json() as Promise<T>
}

// ---- types -----------------------------------------------------------

export interface Scenario {
  scenario_id: string
  title: string
  summary: string
  case_type: string
  sku: string
  node_id: string
  supplier_id?: string
  recommended_units?: number
  expected_behaviour: string
}

export interface Step {
  seq: number
  graph_node: string
  kind: string
  title: string
  tool_name: string | null
  payload_in: Record<string, unknown> | null
  payload_out: Record<string, unknown> | null
  latency_ms: number
  created_at: string
}

export interface Run {
  run_id: string
  scenario_id: string
  case_type: string
  status: string
  provider: string
  model: string
  attempts: number
  sku: string | null
  node_id: string | null
  created_at: string
  completed_at: string | null
  duration_ms: number
  result: RunResult | null
  steps: Step[]
}

export interface RunResult {
  outcome: string
  decision?: Decision
  authorization?: Authorization
  action_results?: ActionResult[]
  validation?: Validation
  evidence_slots?: string[]
  missing_slots?: string[]
  attempts?: number
  untrusted_inputs_seen?: string[]
  llm_degraded?: boolean
  degradation_reasons?: string[]
}

export interface Decision {
  decision_type: string
  recommended_units: number | null
  rationale: string
  key_factors: string[]
  information_gaps: string[]
  proposed_actions: ProposedAction[]
  expectation: Record<string, number | string>
  confidence: number
  risk_level: string
}

export interface ProposedAction {
  action_type: string
  sku?: string
  node_id?: string
  supplier_id?: string
  po_id?: string
  units?: number
  from_node_id?: string
  reason: string
}

export interface Authorization {
  mode: string
  reasons: string[]
  policy_refs: string[]
  risk_level: string
  constraint_report?: ConstraintReport
}

export interface ConstraintReport {
  passed: boolean
  blocking: Violation[]
  warnings: Violation[]
  checks_run: string[]
}

export interface Violation {
  code: string
  severity: string
  message: string
  observed: number | null
  limit: number | null
  unit: string
  remedy_hint: string
}

export interface ActionResult {
  action: ProposedAction
  succeeded: boolean
  entity_id: string | null
  outcome: string
  confirmed_units: number | null
  detail: Record<string, unknown>
  error: string | null
}

export interface Validation {
  verdict: string
  constraint_report: ConstraintReport
  expectation_diffs: ExpectationDiff[]
  observed_state: Record<string, unknown>
  findings: string[]
  corrective_guidance: string
}

export interface ExpectationDiff {
  field: string
  expected: number
  actual: number
  tolerance: number
  within_tolerance: boolean
}

export interface Approval {
  id: number
  run_id: string
  status: string
  risk_level: string
  reason: string
  proposed_action: {
    decision: Decision
    actions: ProposedAction[]
    policy_refs: string[]
    preflight_validation?: Validation
  }
  requested_at: string
  decided_at: string | null
  decided_by: string | null
}

export interface PurchaseOrder {
  po_id: string
  supplier_id: string
  node_id: string
  status: string
  created_by: string
  source_run_id: string | null
  expected_delivery_date: string
  eta_days: number
  ordered_value_usd: number
  committed_value_usd: number
  notes: string | null
  lines: {
    sku: string
    ordered_units: number
    confirmed_units: number | null
    received_units: number
    unit_price_usd: number
  }[]
}

export interface Health {
  status: string
  llm_provider: string
  llm_model: string
  llm_live: boolean
  autonomy_ceiling_usd: number
  max_replan_attempts: number
}

// ---- endpoints -------------------------------------------------------

export const api = {
  health: () => request<Health>('/health'),
  scenarios: () => request<{ scenarios: Scenario[] }>('/scenarios'),
  tools: () =>
    request<{ count: number; note: string; tools: { name: string; description: string; evidence_slot: string | null }[] }>(
      '/tools',
    ),
  runs: () => request<{ runs: Run[] }>('/runs'),
  run: (id: string) => request<Run>(`/runs/${id}`),
  startRun: (scenarioId: string, provider?: string) =>
    request<Run>('/runs', {
      method: 'POST',
      body: JSON.stringify({ scenario_id: scenarioId, provider }),
    }),
  reset: () => request<{ status: string }>('/admin/reset', { method: 'POST' }),
  approvals: () => request<{ approvals: Approval[] }>('/approvals'),
  decideApproval: (id: number, approve: boolean, note = '') =>
    request<{ status: string; validation: Validation | null }>(`/approvals/${id}/decide`, {
      method: 'POST',
      body: JSON.stringify({ approve, note, decided_by: 'buyer' }),
    }),
  purchaseOrders: () => request<{ purchase_orders: PurchaseOrder[] }>('/data/purchase-orders'),
  budgets: () =>
    request<{ budgets: { node_id: string; category: string; allocated_usd: number; committed_usd: number; spent_usd: number; available_usd: number }[] }>(
      '/data/budgets',
    ),
  nodes: () =>
    request<{ nodes: { node_id: string; name: string; capacity_m3: number; used_m3: number; inbound_m3: number; free_m3: number; utilisation_pct: number }[] }>(
      '/data/nodes',
    ),
  inventory: () =>
    request<{ inventory: { sku: string; name: string; node_id: string; on_hand_units: number; available_units: number; in_transit_units: number; inventory_position_units: number }[] }>(
      '/data/inventory',
    ),
}
