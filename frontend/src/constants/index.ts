import type { FeedState, ForecastHorizon } from '../types/api';

export const HORIZONS: ForecastHorizon[] = [1, 3, 7];

export const HORIZON_ROUTES: Record<ForecastHorizon, string> = { 1: '/forecast/1d', 3: '/forecast/3d', 7: '/forecast/7d' };

export const CRS = { data: 'EPSG:4326', render: 'EPSG:3031' } as const;

/** Scene colours (kept in sync with styles/tokens.css). */
export const COLORS = {
  official: '#00e5ff',
  officialStale: '#ff9800',
  forecast: '#b3a6ff',
  forecastLine: '#8f7ff0',
  risk: '#b3a6ff',
  selected: '#ffffff',
  history: '#6f7a99',
  land: '#d9dee9',
  landEdge: '#9fb0cc',
  grid: '#3a4160',
  ocean: '#0d1322',
} as const;

export const FEED_STATE_LABEL: Record<FeedState, string> = {
  LIVE: 'LIVE',
  SYNCED: 'SYNCED',
  DEGRADED: 'DEGRADED',
  FAILED: 'FAILED',
  UNKNOWN: 'NO DATA',
};
