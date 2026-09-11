import { useQuery } from '@tanstack/react-query';

import { ApiError, api } from '../api/client';

const MIN = 60_000;

/** Polling cadence is tuned to data change rates: USNIC is weekly, the backend polls every 72 h. */
const POLL = {
  status: MIN / 2,
  overview: MIN,
  icebergs: 5 * MIN,
  forecasts: 5 * MIN,
  models: 5 * MIN,
};

export const qk = {
  overview: ['overview'] as const,
  health: ['health'] as const,
  icebergs: ['icebergs'] as const,
  iceberg: (id: string) => ['iceberg', id] as const,
  history: (id: string, hist: boolean) => ['history', id, hist] as const,
  icebergForecast: (id: string) => ['icebergForecast', id] as const,
  icebergEvaluations: (id: string) => ['icebergEvaluations', id] as const,
  latestForecasts: ['latestForecasts'] as const,
  models: ['models'] as const,
  model: (v: string) => ['model', v] as const,
  modelEvaluations: (v: string) => ['modelEvaluations', v] as const,
  ingestion: ['ingestion'] as const,
  forecastRuns: ['forecastRuns'] as const,
  retraining: ['retraining'] as const,
  feeds: ['feeds'] as const,
};

export const retryPolicy = (count: number, error: unknown) =>
  !(error instanceof ApiError && (error.status === 404 || error.status === 422)) && count < 2;

export const useHealth = () => useQuery({ queryKey: qk.health, queryFn: api.health, refetchInterval: POLL.status, retry: 0 });
export const useOverview = () => useQuery({ queryKey: qk.overview, queryFn: api.overview, refetchInterval: POLL.overview });
export const useIcebergs = () =>
  useQuery({ queryKey: qk.icebergs, queryFn: () => api.icebergs({ status: 'current' }), refetchInterval: POLL.icebergs });
export const useIceberg = (id: string | null) =>
  useQuery({ queryKey: qk.iceberg(id ?? ''), queryFn: () => api.iceberg(id!), enabled: !!id });
export const useHistory = (id: string | null, includeHistorical = true) =>
  useQuery({
    queryKey: qk.history(id ?? '', includeHistorical),
    queryFn: () => api.history(id!, includeHistorical, 400),
    enabled: !!id,
    staleTime: 10 * MIN,
  });
export const useIcebergForecast = (id: string | null) =>
  useQuery({ queryKey: qk.icebergForecast(id ?? ''), queryFn: () => api.icebergForecast(id!, 7), enabled: !!id });
export const useIcebergEvaluations = (id: string | null) =>
  useQuery({ queryKey: qk.icebergEvaluations(id ?? ''), queryFn: () => api.icebergEvaluations(id!), enabled: !!id });
export const useLatestForecasts = () =>
  useQuery({ queryKey: qk.latestForecasts, queryFn: api.latestForecasts, refetchInterval: POLL.forecasts });
export const useModels = () => useQuery({ queryKey: qk.models, queryFn: api.models, refetchInterval: POLL.models });
export const useModel = (v: string | null) =>
  useQuery({ queryKey: qk.model(v ?? ''), queryFn: () => api.model(v!), enabled: !!v });
export const useModelEvaluations = (v: string | null) =>
  useQuery({ queryKey: qk.modelEvaluations(v ?? ''), queryFn: () => api.modelEvaluations(v!), enabled: !!v });
export const useIngestion = () => useQuery({ queryKey: qk.ingestion, queryFn: api.ingestion, refetchInterval: POLL.status });
export const useForecastRuns = () => useQuery({ queryKey: qk.forecastRuns, queryFn: api.forecastRuns, refetchInterval: POLL.status });
export const useRetraining = () => useQuery({ queryKey: qk.retraining, queryFn: api.retraining, refetchInterval: POLL.overview });
export const useBasemapInfo = () =>
  useQuery({ queryKey: ['basemap'], queryFn: api.basemap, staleTime: Infinity, retry: 1 });
export const useFeeds = () => useQuery({ queryKey: qk.feeds, queryFn: api.feeds, refetchInterval: POLL.overview });
