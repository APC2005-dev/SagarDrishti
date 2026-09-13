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

/** Environmental value used for one entry of one group (JSON blob from the backend, snake_case). */
export interface EnvEntryValue {
  values: Record<string, number | null>;
  valid_date: string | null;
  provider: string;
  dataset_id: string;
  staleness_days: number | null;
  missing: boolean;
  as_of?: string;
  interpolation?: string;
  valid_neighbours?: number;
  reason?: string | null;
  quality_flags?: string[];
}

export interface InputEntry {
  observationId: number | null;
  date: string;
  latitude: number;
  longitude: number;
  provenance: Provenance;
  elapsedDays: number;
  environment?: Record<string, EnvEntryValue> | null;
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
  featureSchemaVersion: string;
  environmentAsOf: string | null;
  anchorEnvironment: Record<string, EnvEntryValue> | null;
  fallback: { champion: string; champion_schema: string; used_model: string; reason: string; missing: string[]; missing_count: number } | null;
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
  modelType: 'base' | 'trajectory' | 'environmental';
  featureSchemaVersion: string;
  environmentalDataSources: Record<string, { historical: Record<string, unknown> | null; operational: Record<string, unknown> | null }> | null;
  environmentalDataCutoff: string | null;
}

export interface ModelVersionDetail extends ModelVersion {
  featureSchema: { version: string; description: string; features: string[]; env_variables: string[]; groups: string[] } | null;
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
  experiment: ExperimentRecord | null;
}

export interface ExperimentCandidate {
  version: string;
  schema: string;
  architecture_version: string;
  train_samples: number;
  validation_samples: number;
  validation_primary_mae_km: number | null;
  status: string;
}

export interface ExperimentRecord {
  schemas?: string[];
  candidates?: ExperimentCandidate[];
  failures?: Record<string, string>;
  unavailable?: Record<string, string>;
  environmental_schemas_unavailable?: Record<string, string>;
  selection?: { criterion: string; selected: string; common_validation_samples: number };
}

export interface AblationEffect {
  without: string;
  with: string;
  verdict: Record<string, 'improved' | 'worse' | 'no significant difference' | 'no data'>;
  by_horizon: Record<string, { n: number; mae_diff_km: number | null; ci95_low_km: number | null; ci95_high_km: number | null; candidate_better_share: number | null }>;
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
  /** iceberg and sea_ice are CORE model feeds; environmental are feature sources. */
  category: 'iceberg' | 'sea_ice' | 'environmental';
  configured: boolean;
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
    modelType: string;
    featureSchemaVersion: string;
    featureSchemaDescription: string | null;
    environmentalSources: string[];
  } | null;
  environment: { enabled: boolean; configuredGroups: string[]; lastSyncAt: string | null } | null;
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

export type EnvGroup = 'wind' | 'current' | 'sea_ice';

export interface EnvSource {
  group: EnvGroup;
  role: 'operational' | 'historical';
  provider: string | null;
  configured: boolean;
  reason: string;
  authority: string | null;
  productId: string | null;
  datasetId: string | null;
  variables: Record<string, string> | null;
  units: Record<string, string> | null;
  spatialResolutionDeg: number | null;
  temporalResolution: string | null;
  aggregation: string | null;
  latencyDays: number | null;
  coverageStart: string | null;
  coverageEnd: string | null;
  depth: string | null;
  supportsForecast: boolean;
  forecastLeadDays: number;
  notes: string | null;
}

export interface EnvStatus {
  enabled: boolean;
  sources: EnvSource[];
  lastRuns: Record<string, { last_status: string | null; last_run_at: string | null; last_error: string | null; last_success_at: string | null }>;
  cacheEntries: number;
  cacheBytes: number;
  latestCacheFetch: string | null;
  schemas: { version: string; description: string; features: string[]; groups: string[]; trainable: boolean; reason: string }[];
  policy: Record<string, string | number>;
}

export interface EnvRun {
  id: number;
  kind: string;
  group: string | null;
  provider: string | null;
  datasetId: string | null;
  role: string | null;
  status: string;
  records: number;
  bytes: number | null;
  durationMs: number | null;
  errorMessage: string | null;
  startedAt: string;
  completedAt: string | null;
}

export interface EnvGroupValue {
  group: EnvGroup;
  provider: string;
  datasetId: string;
  values: Record<string, number | null>;
  units: Record<string, string>;
  validDate: string | null;
  asOf: string;
  stalenessDays: number | null;
  interpolation: string;
  validNeighbours: number;
  missing: boolean;
  reason: string | null;
  qualityFlags: string[];
  vector: { speedMS: number; directionDeg: number; directionConvention: 'from' | 'towards' } | null;
}

export interface IcebergEnvironment {
  icebergId: string;
  observationId: number | null;
  observationDate: string | null;
  aligned: boolean;
  reason: string | null;
  groups: EnvGroupValue[];
}

export interface EnvField {
  group: EnvGroup;
  variables: string[];
  validDate: string;
  provider: string | null;
  datasetId: string | null;
  resolutionDeg: number;
  /** [lat, lon, value1, value2?] */
  points: number[][];
}

export interface HealthCheck {
  status: 'ok' | 'degraded' | 'unavailable';
  checks: Record<string, Record<string, unknown>>;
  version: string;
}

/* --- Sea ice: a core model family with its own lineage, data and champion --- */

export interface SeaIceModel {
  version: string;
  shortVersion: string;
  family: string;
  versionNumber: number;
  parentVersion: string | null;
  architecture: string;
  architectureVersion: string;
  status: string;
  statusReason: string | null;
  artifactOrigin: string;
  /** Chronological database entries per input sequence — NOT calendar days. */
  inputWindowEntries: number;
  forecastHorizonsDays: number[];
  deployedAt: string | null;
  trainingDataCutoff: string | null;
  day1Rmse: number | null;
  day3Rmse: number | null;
  day7Rmse: number | null;
}

export interface SeaIceObservation {
  observationDate: string;
  provenance: string;
  authority: string;
  datasetId: string;
  variable: string;
  preprocessingVersion: string;
  gridShape: number[];
  gridResolutionDeg: number;
  crs: string;
  nValidCells: number;
  meanConcentration: number | null;
  fetchedAt: string;
  sourceTime: string | null;
}

export interface SeaIceWindowEntry {
  observationDate: string;
  nValidCells: number;
  meanConcentration: number | null;
}

export interface SeaIceForecast {
  modelVersion: string;
  anchorDate: string;
  generatedAt: string;
  horizonDays: number;
  targetDate: string;
  meanConcentration: number | null;
  inputWindowEntries: number;
  inputEntryDates: string[];
  inputSpanDays: number | null;
  dailyCadence: boolean | null;
}

export interface SeaIceEvaluation {
  modelVersion: string;
  horizonDays: number;
  anchorDate: string;
  targetDate: string;
  rmse: number;
  mae: number;
  persistenceRmse: number | null;
  persistenceMae: number | null;
  nValidCells: number;
  evaluatedAt: string;
}

export interface SeaIceRun {
  id: number;
  kind: string;
  trigger: string;
  status: string;
  startedAt: string;
  completedAt: string | null;
  durationMs: number | null;
  entriesNew: number;
  entriesDuplicate: number;
  sourceLatestDate: string | null;
  errorMessage: string | null;
}

export interface SeaIceField {
  kind: 'observation' | 'forecast';
  validDate: string;
  horizonDays: number | null;
  modelVersion: string | null;
  anchorDate: string | null;
  resolutionDeg: number;
  crs: string;
  minConcentration: number;
  meanConcentration: number | null;
  /** [lat, lon, concentration] per ice-covered cell */
  points: number[][];
}

export interface SeaIceStatus {
  enabled: boolean;
  sourceConfigured: boolean;
  sourceReason: string | null;
  datasetId: string;
  authority: string;
  model: SeaIceModel | null;
  modelUnavailableReason: string | null;
  latestObservation: SeaIceObservation | null;
  observationCount: number;
  windowEntriesRequired: number;
  window: SeaIceWindowEntry[];
  windowComplete: boolean;
  forecastUnavailableReason: string | null;
  latestForecasts: SeaIceForecast[];
  recentEvaluations: SeaIceEvaluation[];
  recentRuns: SeaIceRun[];
}

/* --- Route planning: ports (NGA WPI) and computed routes ------------------ */

export interface PortSummary {
  id: number;
  identifier: string;
  name: string;
  countryName: string | null;
  regionName: string | null;
  unlocode: string | null;
  latitude: number;
  longitude: number;
  source: string;
  inRoutingDomain: boolean;
}

export interface RouteWaypoint {
  waypoint: number;
  elapsedHours: number;
  arrivalTime: string;
  latitude: number;
  longitude: number;
  action: string;
}

export interface RouteEndpoint {
  port: PortSummary;
  requestedLatitude: number;
  requestedLongitude: number;
  resolvedLatitude: number | null;
  resolvedLongitude: number | null;
  connectorKm: number | null;
}

export interface RouteMetrics {
  distanceKm: number | null;
  durationHours: number | null;
  /** Dimensionless relative proxy — not litres or tonnes. */
  fuelProxy: number | null;
  /** Concentration-weighted hours — not a collision probability. */
  sicExposureHours: number | null;
  weightedObjective: number | null;
}

export interface RouteCurvature {
  routeLengthKm: number;
  gridEndpointGeodesicKm: number;
  detourRatio: number | null;
  extraDistanceVsGridGeodesicKm: number;
  extraDistanceVsGridGeodesicPct: number | null;
  totalAbsoluteTurnDeg: number;
  maximumTurnDeg: number;
  maximumDiscreteCurvatureRadPerKm: number;
  totalTurnRadiansPerRouteKm: number;
  removedStationaryWaypoints: number;
  curvatureDefinition: string;
  detourDefinition: string;
  requestedEndpointGeodesicKm?: number;
  requestedEndpointConnectorsValidated?: boolean;
  [key: string]: unknown;
}

export interface Route {
  routeId: string;
  status: string;
  errorCode: string | null;
  errorMessage: string | null;
  departure: RouteEndpoint;
  destination: RouteEndpoint;
  connectorsValidated: boolean | null;
  modelVersions: { trajectory: string | null; seaIce: string | null; routePlanner: string };
  trajectoryForecastRunId: number | null;
  seaIceForecastRunId: number | null;
  forecastReferenceTime: string | null;
  metrics: RouteMetrics;
  /** Status only: no calibrated route-safety probability is claimed. */
  confidence: { status: string; percent: null };
  curvature: RouteCurvature | null;
  waypoints: RouteWaypoint[];
  geometry: { type: string; coordinates: number[][] } | null;
  environmentSnapshot: Record<string, unknown> | null;
  expansions: number | null;
  runtimeSeconds: number | null;
  createdAt: string;
}

export interface RouteSummaryRow {
  routeId: string;
  status: string;
  departureName: string;
  destinationName: string;
  distanceKm: number | null;
  durationHours: number | null;
  fuelProxy: number | null;
  sicExposureHours: number | null;
  trajectoryModelVersion: string | null;
  seaIceModelVersion: string | null;
  routePlannerVersion: string;
  confidenceStatus: string;
  forecastReferenceTime: string | null;
  createdAt: string;
}

export interface ApiErrorDetail {
  code: string;
  message: string;
}
