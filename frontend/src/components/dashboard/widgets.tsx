import type { ReactNode } from 'react';

import type { Overview } from '../../types/api';
import { fmtDate, fmtKm, fmtNum, relTime } from '../../utils/format';
import { Metric } from '../common/Metric';
import { FeedStateChip, RunStatusChip } from '../common/StatusChip';

/**
 * Overview widget registry. Adding a widget = appending an entry; the page lays
 * them out in order. `planned` entries render as clearly-labelled empty slots
 * (no invented numbers) until their data source exists.
 */
export interface OverviewWidget {
  id: string;
  title: string;
  span?: 1 | 2;
  planned?: boolean;
  render: (o: Overview) => ReactNode;
}

const err = (o: Overview, h: number) => o.operationalErrors.find((e) => e.horizonDays === h);

export const OVERVIEW_WIDGETS: OverviewWidget[] = [
  {
    id: 'tracking',
    title: 'Tracking',
    span: 2,
    render: (o) => (
      <div className="grid-tiles" style={{ gridTemplateColumns: 'repeat(4, minmax(0,1fr))' }}>
        <Metric label="Tracked now" value={fmtNum(o.activeIcebergs)} tone="official" sub="in latest USNIC file" />
        <Metric label="Not in latest" value={fmtNum(o.notInLatestSource)} sub="history retained" />
        <Metric label="Stale" value={fmtNum(o.staleIcebergs)} tone={o.staleIcebergs ? 'warn' : 'default'} sub="old official fix" />
        <Metric label="Official obs." value={fmtNum(o.officialObservations)} sub="all time" />
      </div>
    ),
  },
  {
    id: 'source',
    title: 'Official source · USNIC',
    render: (o) => (
      <div className="grid-tiles">
        <Metric label="Latest USNIC update" value={fmtDate(o.latestUsnicUpdate)} tone="official" />
        <Metric label="Last backend sync" value={relTime(o.lastSyncAt)} />
        <Metric label="New obs. last run" value={fmtNum(o.newObservationsLastRun)} />
        <div className="metric">
          <span className="label">Feed state</span>
          <div style={{ marginTop: 6 }}>
            <FeedStateChip state={o.feedState} />
          </div>
          {o.feedStateReasons[0] && <div className="metric-sub">{o.feedStateReasons[0]}</div>}
        </div>
      </div>
    ),
  },
  {
    id: 'model',
    title: 'Forecast model',
    render: (o) =>
      o.champion ? (
        <div className="grid-tiles">
          <Metric label="Champion" value={o.champion.version} tone="forecast" sub={`${o.champion.architectureVersion} · ${o.champion.modelType}`} />
          <Metric
            label="Feature schema"
            value={o.champion.featureSchemaVersion}
            sub={o.champion.featureSchemaDescription ?? undefined}
            title={o.champion.environmentalSources.length ? `Sources: ${o.champion.environmentalSources.join(', ')}` : 'Trajectory features only'}
          />
          <Metric label="Adapter" value={o.champion.adapterStrategy ?? '—'} sub={`${o.modelVersions} versions registered`} />
          <Metric label="Active forecasts" value={fmtNum(o.icebergsWithActiveForecasts)} sub="from latest official fix" />
          <div className="metric">
            <span className="label">Pipeline</span>
            <div style={{ marginTop: 6 }}>
              <FeedStateChip state={o.pipelineState} />
            </div>
            <div className="metric-sub">
              last run {o.lastForecastRun ? <RunStatusChip status={o.lastForecastRun.status} /> : 'never'}
            </div>
          </div>
        </div>
      ) : (
        <div className="note warn">No model is deployed. Official positions are shown; forecasts are unavailable until the base artifact is registered.</div>
      ),
  },
  {
    id: 'accuracy',
    title: 'Model accuracy (mean great-circle error)',
    span: 2,
    render: (o) => (
      <div className="grid-tiles" style={{ gridTemplateColumns: 'repeat(3, minmax(0,1fr))' }}>
        {([1, 3, 7] as const).map((h) => {
          const bench = o.champion ? (h === 1 ? o.champion.day1Error : h === 3 ? o.champion.day3Error : o.champion.day7Error) : null;
          const live = err(o, h);
          return (
            <Metric
              key={h}
              label={`Day ${h}`}
              value={live?.maeKm != null ? fmtKm(live.maeKm) : '—'}
              tone="forecast"
              sub={
                <>
                  operational n={live?.n ?? 0}
                  <br />
                  benchmark {fmtKm(bench)}
                </>
              }
              title="Operational = scored against later official USNIC observations. Benchmark = historical test split of the model version."
            />
          );
        })}
      </div>
    ),
  },
  {
    id: 'learning',
    title: 'Evaluation & retraining',
    render: (o) => (
      <div className="grid-tiles">
        <Metric label="Evaluations" value={fmtNum(o.evaluationsTotal)} sub="prediction vs official" />
        <div className="metric">
          <span className="label">Last retraining</span>
          <div style={{ marginTop: 6 }}>{o.retrainingStatus ? <RunStatusChip status={o.retrainingStatus} /> : <span className="dim mono">none yet</span>}</div>
        </div>
      </div>
    ),
  },
  {
    id: 'environment',
    title: 'Environmental forcing (wind · current · sea ice)',
    render: (o) => (
      <div className="grid-tiles">
        <Metric
          label="Pipeline"
          value={o.environment?.enabled ? 'ENABLED' : 'DISABLED'}
          sub="environmental model versions only"
          tone={o.environment?.enabled ? 'ok' : 'default'}
        />
        <Metric label="Last env. sync" value={relTime(o.environment?.lastSyncAt)} sub={o.environment?.configuredGroups.join(' · ') || 'no source has delivered data yet'} />
        <Metric
          label="Champion env. sources"
          value={o.champion?.environmentalSources.length ? String(o.champion.environmentalSources.length) : 'none'}
          sub={o.champion?.environmentalSources.join(', ') || 'champion uses trajectory features only'}
        />
      </div>
    ),
  },
];
