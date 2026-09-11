import { motion } from 'framer-motion';

import { OVERVIEW_WIDGETS } from '../../components/dashboard/widgets';
import { SectionTitle } from '../../components/common/Metric';
import { QueryState } from '../../components/common/QueryState';
import { AntarcticScene } from '../../components/globe/AntarcticScene';
import { useIcebergs, useLatestForecasts, useOverview } from '../../hooks/queries';
import { fmtDateTime } from '../../utils/format';

export default function OverviewPage() {
  const overview = useOverview();
  const icebergs = useIcebergs();
  const forecasts = useLatestForecasts();

  return (
    <div className="split" style={{ gridTemplateColumns: 'minmax(460px, 560px) 1fr' }}>
      <section className="split-left scroll pad" aria-label="Mission summary">
        <div className="page-head">
          <div>
            <h1>Mission overview</h1>
            <span className="dim mono" style={{ fontSize: 11 }}>
              {overview.data ? `generated ${fmtDateTime(overview.data.generatedAt)}` : ' '}
            </span>
          </div>
        </div>
        <QueryState query={overview}>
          {(o) => (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {OVERVIEW_WIDGETS.map((w, i) => (
                <motion.div
                  key={w.id}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.04, duration: 0.25 }}
                >
                  <SectionTitle>{w.title}</SectionTitle>
                  {w.planned ? <div className="placeholder-card mono">{w.render(o)}</div> : w.render(o)}
                </motion.div>
              ))}
            </div>
          )}
        </QueryState>
      </section>
      <section className="split-right" aria-label="Antarctic situation map">
        <AntarcticScene icebergs={icebergs.data?.items ?? []} forecasts={forecasts.data ?? []} horizon={7} />
        {(icebergs.isError || forecasts.isError) && (
          <div className="map-overlay" style={{ bottom: 14, right: 14 }}>
            <div className="glass qs-error" style={{ padding: '8px 12px' }}>
              {icebergs.isError ? 'Iceberg positions unavailable. ' : ''}
              {forecasts.isError ? 'Forecasts unavailable.' : ''}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
