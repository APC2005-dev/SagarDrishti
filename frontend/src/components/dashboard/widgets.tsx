import type { ReactNode } from 'react';

import type { Overview } from '../../types/api';
import { fmtDate, fmtKm, fmtNum, relTime } from '../../utils/format';
import { Metric } from '../common/Metric';
import { FeedStateChip } from '../common/StatusChip';

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
      <div className="grid-tiles" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))' }}>
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
    id: 'accuracy',
    title: 'Model accuracy (mean great-circle error)',
    span: 2,
    render: (o) => (
      <div className="grid-tiles" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))' }}>
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
];
