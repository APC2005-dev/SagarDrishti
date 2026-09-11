import type {
  BasemapInfo,
  EvaluationSummary,
  Evaluation,
  Feed,
  ForecastRun,
  ForecastSet,
  EnvField,
  EnvGroup,
  EnvRun,
  EnvStatus,
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
  RetrainingStatus,
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
};
