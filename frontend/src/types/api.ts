/**
 * API contract — mirrors backend/app/schemas (camelCase on the wire).
 * Coordinates are WGS84 / EPSG:4326 degrees unless a field says EPSG:3031.
 */

export type Provenance =
  | 'official_usnic'
  | 'historical_training_dataset'
  | 'derived'
  | 'interpolated'
  | 'predicted';

export type IcebergStatus = 'active' | 'not_in_latest_source' | 'historical_only';
export type FeedState = 'LIVE' | 'SYNCED' | 'DEGRADED' | 'FAILED' | 'UNKNOWN';
export type ForecastHorizon = 1 | 3 | 7;

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface IcebergSummary {
  icebergId: string;
  status: IcebergStatus;
  latitude: number | null;
  longitude: number | null;
  lastUpdate: string | null;
  lengthNm: number | null;
  widthNm: number | null;
  areaSqNm: number | null;
  areaSqKm: number | null;
  source: string | null;
  provenance: Provenance | null;
  latestObservationId: number | null;
  isStale: boolean;
  daysSinceUpdate: number | null;
  firstSeen: string | null;
  lastSeen: string | null;
  hasForecast: boolean;
  latestForecastModelVersion: string | null;
  latestForecastGeneratedAt: string | null;
}

export interface IcebergDetail extends IcebergSummary {
  firstOfficialSeen: string | null;
  lastOfficialSeen: string | null;
  officialObservationCount: number;
  historicalObservationCount: number;
  statusChangedAt: string | null;
}

export interface Observation {
  id: number;
  icebergId: string;
  observationDate: string;
  latitude: number;
  longitude: number;
  lengthNm: number | null;
  widthNm: number | null;
  areaSqNm: number | null;
  areaSqKm: number | null;
  source: string;
  sourceFile: string | null;
  fetchedAt: string | null;
  provenance: Provenance;
  revision: number;
  ingestionRunId: number | null;
}

export interface ForecastPoint {
  forecastId: number;
  horizonDays: number;
  forecastDate: string;
  predictedLatitude: number;
  predictedLongitude: number;
  predictedXM: number;
  predictedYM: number;
  riskRadiusKmP90: number | null;
  provenance: 'predicted';
}

export interface InputEntry {
  observationId: number | null;
  date: string;
  latitude: number;
  longitude: number;
  provenance: Provenance;
  elapsedDays: number;
}

export interface SequenceDiagnostics {
  official_entries: number;
  historical_entries: number;
  min_gap_days: number;
  median_gap_days: number;
  max_gap_days: number;
  span_days: number;
  daily_cadence: boolean;
}

export interface ForecastSet {
  forecastSetId: number;
  icebergId: string;
  modelVersion: string;
  isChampion: boolean;
  generatedAt: string;
  latestObservationDate: string;
  requestedHorizon: number;
  isStale: boolean;
  adapterStrategy: string;
  diagnostics: SequenceDiagnostics;
  anchor: {
    observationId: number;
    observationDate: string;
    latitude: number;
    longitude: number;
    provenance: Provenance;
  };
  points: ForecastPoint[];
  inputEntries: InputEntry[] | null;
}

export interface ForecastRow {
  forecastId: number;
  forecastSetId: number;
  icebergId: string;
  modelVersion: string;
  generatedAt: string;
  latestObservationDate: string;
  horizonDays: number;
  forecastDate: string;
  predictedLatitude: number;
  predictedLongitude: number;
  riskRadiusKmP90: number | null;
  provenance: 'predicted';
}

export interface ModelMetric {
  protocol: string;
  horizonDays: number | null;
  n: number;
  maeKm: number | null;
  rmseKm: number | null;
  medianKm: number | null;
  p90Km: number | null;
  source: string;
  computedAt: string;
}

export type ModelStatus = 'candidate' | 'validated' | 'deployed' | 'rejected' | 'archived';

export interface ModelVersion {
  version: string;
  versionNumber: number;
  parentVersion: string | null;
  architecture: string;
  architectureVersion: string;
  inputSequenceLength: number;
  inputSemantics: string;
  forecastHorizonDays: number;
  featureNames: string[];
  adapterStrategy: string | null;
  artifactOrigin: string;
  artifactSha256: Record<string, string>;
  trainingDataCutoff: string | null;
  trainingStartedAt: string | null;
  trainingCompletedAt: string | null;
  trainingSampleCount: number | null;
  day1Error: number | null;
  day3Error: number | null;
  day7Error: number | null;
  status: ModelStatus;
  statusReason: string | null;
  deployedAt: string | null;
  createdAt: string;
}

export interface ModelVersionDetail extends ModelVersion {
  metrics: ModelMetric[];
  statusHistory: { fromStatus: string | null; toStatus: string; reason: string; actor: string; createdAt: string }[];
  children: string[];
  datasetManifest: Record<string, unknown> | null;
  trainingConfig: Record<string, unknown> | null;
  testMetrics: Record<string, unknown> | null;
}

export interface HorizonMetrics {
  horizonDays: number;
  n: number;
  maeKm: number | null;
  rmseKm: number | null;
  medianKm: number | null;
  p90Km: number | null;
}

export interface Evaluation {
  evaluationId: number;
  forecastId: number;
  icebergId: string;
  modelVersion: string;
  horizonDays: number;
  forecastDate: string;
  predictedLatitude: number;
  predictedLongitude: number;
  actualLatitude: number;
  actualLongitude: number;
  errorKm: number;
  actualSource: Provenance;
  actualObservationId: number;
  actualObservationDate: string;
  evaluatedAt: string;
}

export interface EvaluationSummary {
  modelVersion: string | null;
  total: number;
  byHorizon: HorizonMetrics[];
  recent: Evaluation[];
}

export interface IngestionRun {
  id: number;
  source: string;
  sourceUrl: string | null;
  discoveryMethod: string | null;
  trigger: string;
  fetchedAt: string;
  completedAt: string | null;
  durationMs: number | null;
  httpStatus: number | null;
  checksumSha256: string | null;
  rowCount: number;
  validRows: number;
  newObservations: number;
  updatedObservations: number;
  duplicateObservations: number;
  failedRows: number;
  missingFromSource: number;
  sourceLatestUpdate: string | null;
  status: 'running' | 'success' | 'partial' | 'unchanged' | 'failed';
  errorMessage: string | null;
}

export interface IngestionStatus {
  feedState: FeedState;
  stateReasons: string[];
  latestRun: IngestionRun | null;
  lastSuccessfulRun: IngestionRun | null;
  latestOfficialObservationDate: string | null;
  recentRuns: IngestionRun[];
  latestRowErrors: { rowNumber: number; rawRow: Record<string, string>; errors: string[] }[];
}

export interface ForecastRun {
  id: number;
  modelVersion: string | null;
  trigger: string;
  ingestionRunId: number | null;
  status: 'running' | 'success' | 'partial' | 'failed' | 'skipped';
  icebergsConsidered: number;
  forecastSetsCreated: number;
  alreadyForecast: number;
  skipped: Record<string, number> | null;
  errorMessage: string | null;
  startedAt: string;
  completedAt: string | null;
}

export interface EligibilityCheck {
  value: number | string | null;
  threshold?: number;
  met: boolean;
}

export interface RetrainingRun {
  id: number;
  runId: string;
  trigger: string;
  status: 'running' | 'skipped' | 'promoted' | 'rejected' | 'failed';
  championVersion: string | null;
  candidateVersion: string | null;
  sourceDataCutoff: string | null;
  sampleCount: number | null;
  eligibility: { since: string | null; checks: Record<string, EligibilityCheck>; eligible: boolean } | null;
  decision: { promote: boolean; protocol: string; reasons: string[] } | null;
  metrics: Record<string, unknown> | null;
  startedAt: string;
  completedAt: string | null;
  failureReason: string | null;
}

export interface RetrainingStatus {
  policy: Record<string, Record<string, unknown>>;
  eligibility: { since: string | null; checks: Record<string, EligibilityCheck>; eligible: boolean } | null;
  latestRun: RetrainingRun | null;
  recentRuns: RetrainingRun[];
}

export interface Feed {
  id: string;
  name: string;
  provider: string;
  description: string;
  productUrl: string;
  sourceUrl: string | null;
  cadence: string;
  pollIntervalHours: number;
  state: FeedState;
  stateReasons: string[];
  lastFetchAt: string | null;
  lastSuccessAt: string | null;
  latestOfficialObservationDate: string | null;
  fetchDurationMs: number | null;
  checksumSha256: string | null;
  recordCount: number | null;
  discoveryMethod: string | null;
  errorMessage: string | null;
}

export interface Overview {
  generatedAt: string;
  trackedIcebergs: number;
  activeIcebergs: number;
  notInLatestSource: number;
  historicalOnlyIcebergs: number;
  staleIcebergs: number;
  officialObservations: number;
  newObservationsLastRun: number;
  latestUsnicUpdate: string | null;
  lastSyncAt: string | null;
  feedState: FeedState;
  feedStateReasons: string[];
  champion: {
    version: string;
    architectureVersion: string;
    adapterStrategy: string | null;
    deployedAt: string | null;
    day1Error: number | null;
    day3Error: number | null;
    day7Error: number | null;
  } | null;
  modelVersions: number;
  icebergsWithActiveForecasts: number;
  lastForecastRun: ForecastRun | null;
  operationalErrors: HorizonMetrics[];
  evaluationsTotal: number;
  retrainingStatus: string | null;
  pipelineState: FeedState;
}

/** Static EPSG:3031 basemap imagery (NASA Blue Marble via GIBS), proxied by the backend. */
export interface BasemapInfo {
  enabled: boolean;
  crs: string;
  extentM: number;
  tileSize: number;
  matrixSizeFormula: string;
  defaultLayer: string;
  layers: { id: string; title: string; maxZoom: number }[];
  attribution: string;
}

export interface HealthCheck {
  status: 'ok' | 'degraded' | 'unavailable';
  checks: Record<string, Record<string, unknown>>;
  version: string;
}
