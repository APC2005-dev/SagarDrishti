import type {
  BasemapInfo,
  EnvField,
  EnvGroup,
  EnvRun,
  EnvStatus,
  Evaluation,
  EvaluationSummary,
  Feed,
  ForecastRun,
  ForecastSet,
  HealthCheck,
  IcebergDetail,
  IcebergEnvironment,
  IcebergSummary,
  IngestionStatus,
  ModelVersion,
  ModelVersionDetail,
  Observation,
  Overview,
  Page,
  PortSummary,
  RetrainingStatus,
  Route,
  RouteSummaryRow,
  SeaIceField,
  SeaIceStatus,
} from '../types/api';

const BASE = '/api/v1';
const TIMEOUT_MS = 20_000;

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
    readonly requestId: string | null,
  ) {
    super(`${status}: ${detail}`);
  }
  /** True when the backend itself is unreachable (network error / proxy 5xx). */
  get unavailable(): boolean {
    return this.status === 0 || this.status === 502 || this.status === 503 || this.status === 504;
  }
}

type Params = Record<string, string | number | boolean | null | undefined>;

async function get<T>(path: string, params?: Params): Promise<T> {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params ?? {})) if (v !== undefined && v !== null && v !== '') qs.set(k, String(v));
  const url = `${BASE}${path}${qs.size ? `?${qs}` : ''}`;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  let res: Response;
  try {
    res = await fetch(url, { signal: ctrl.signal, headers: { Accept: 'application/json' } });
  } catch {
    throw new ApiError(0, 'backend unreachable', null);
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === 'string') detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail, res.headers.get('X-Request-ID'));
  }
  return (await res.json()) as T;
}

/**
 * Route planning runs a bounded A* search server-side, so it needs a longer
 * budget than a read. Backend failures carry `{code, message}` in `detail`; the
 * code is preserved in `ApiError.detail` so the UI can explain it.
 */
async function post<T>(path: string, body: unknown, timeoutMs = 240_000): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      method: 'POST',
      signal: ctrl.signal,
      headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, 'backend unreachable', null);
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) {
    let detail = res.statusText;
    let code: string | null = null;
    try {
      const parsed = (await res.json()) as { detail?: unknown };
      const d = parsed.detail;
      if (typeof d === 'string') detail = d;
      else if (d && typeof d === 'object') {
        const obj = d as { code?: string; message?: string };
        code = obj.code ?? null;
        detail = obj.message ?? detail;
      }
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, code ? `${code}|${detail}` : detail, res.headers.get('X-Request-ID'));
  }
  return (await res.json()) as T;
}

/** Split the `CODE|message` form produced above. */
export function errorParts(error: unknown): { code: string | null; message: string } {
  if (error instanceof ApiError) {
    const [head, ...rest] = error.detail.split('|');
    return rest.length ? { code: head ?? null, message: rest.join('|') } : { code: null, message: error.detail };
  }
  return { code: null, message: error instanceof Error ? error.message : 'Unexpected error' };
}

export const api = {
  health: () => get<HealthCheck>('/health/ready'),
  overview: () => get<Overview>('/overview'),
  icebergs: (p: { status?: string; q?: string; limit?: number; offset?: number } = {}) =>
    get<Page<IcebergSummary>>('/icebergs', { limit: 500, ...p }),
  iceberg: (id: string) => get<IcebergDetail>(`/icebergs/${encodeURIComponent(id)}`),
  history: (id: string, includeHistorical = false, limit = 500) =>
    get<Page<Observation>>(`/icebergs/${encodeURIComponent(id)}/history`, { include_historical: includeHistorical, limit }),
  icebergForecast: (id: string, horizon = 7) =>
    get<ForecastSet>(`/icebergs/${encodeURIComponent(id)}/forecast`, { horizon }),
  icebergEvaluations: (id: string) => get<Evaluation[]>(`/icebergs/${encodeURIComponent(id)}/evaluations`),
  /** Always fetched at horizon 7: the 1/3/7-day views filter this single model run client-side. */
  latestForecasts: () => get<ForecastSet[]>('/forecasts/latest', { horizon: 7 }),
  models: () => get<ModelVersion[]>('/models'),
  currentModel: () => get<ModelVersion>('/models/current'),
  model: (v: string) => get<ModelVersionDetail>(`/models/${encodeURIComponent(v)}`),
  modelEvaluations: (v: string) => get<EvaluationSummary>(`/models/${encodeURIComponent(v)}/evaluations`),
  ingestion: () => get<IngestionStatus>('/operations/ingestion'),
  forecastRuns: () => get<ForecastRun[]>('/operations/forecasting'),
  retraining: () => get<RetrainingStatus>('/operations/retraining'),
  feeds: () => get<Feed[]>('/feeds'),
  basemap: () => get<BasemapInfo>('/basemap'),
  basemapTileUrl: (layer: string, z: number, row: number, col: number) =>
    `${BASE}/basemap/${encodeURIComponent(layer)}/${z}/${row}/${col}.jpeg`,
  environmentStatus: () => get<EnvStatus>('/environment/status'),
  environmentRuns: () => get<EnvRun[]>('/environment/runs', { limit: 30 }),
  environmentField: (group: EnvGroup) => get<EnvField>('/environment/field', { group }),
  icebergEnvironment: (id: string) => get<IcebergEnvironment>(`/icebergs/${encodeURIComponent(id)}/environment`),
  searchPorts: (q: string, domainOnly = true) =>
    get<PortSummary[]>('/ports/search', { q, limit: 10, domain_only: domainOnly }),
  planRoute: (departurePortId: string, destinationPortId: string, maxSic?: number) =>
    post<Route>('/routes', { departurePortId, destinationPortId, maxSic }),
  route: (id: string) => get<Route>(`/routes/${encodeURIComponent(id)}`),
  activeRoute: () => get<RouteSummaryRow | null>('/routes/active'),
  /** Full geometry of the active route, so a refresh can redraw it. */
  activeRouteDetail: async () => {
    const summary = await get<RouteSummaryRow | null>('/routes/active');
    return summary ? get<Route>(`/routes/${encodeURIComponent(summary.routeId)}`) : null;
  },
  routeHistory: () => get<RouteSummaryRow[]>('/routes', { limit: 20 }),
  seaIceStatus: () => get<SeaIceStatus>('/sea-ice/status'),
  seaIceField: (horizon: number | null, stride = 1) =>
    horizon == null
      ? get<SeaIceField>('/sea-ice/latest', { stride })
      : get<SeaIceField>('/sea-ice/forecast', { horizon, stride }),
};
