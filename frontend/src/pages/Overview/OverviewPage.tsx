import { motion } from 'framer-motion';

import { SectionTitle } from '../../components/common/Metric';
import { QueryState } from '../../components/common/QueryState';
import { ResizeHandle } from '../../components/common/ResizeHandle';
import { OVERVIEW_WIDGETS } from '../../components/dashboard/widgets';
import { AntarcticScene } from '../../components/globe/AntarcticScene';
import { useIcebergs, useLatestForecasts, useOverview } from '../../hooks/queries';
import { useResizableSidebar } from '../../hooks/useResizableSidebar';
import { fmtDateTime } from '../../utils/format';

export default function OverviewPage() {
  const overview = useOverview();
  const icebergs = useIcebergs();
  const forecasts = useLatestForecasts();

  const { sidebarWidth, sidebarHeight, isDragging, isMobile, startResize, resetSize } =
    useResizableSidebar({
      storageKey: 'sagar_sidebar_overview',
      defaultWidth: 520,
      minWidth: 340,
    });

  return (
    <div
      className={`split-resizable${isMobile ? ' mobile-stacked' : ''}`}
      style={
        isMobile
          ? undefined
          : { gridTemplateColumns: `${sidebarWidth}px 6px 1fr` }
      }
    >
      <section
        className="split-left scroll pad"
        aria-label="Mission summary"
        style={isMobile ? { height: `${sidebarHeight}px`, flexShrink: 0 } : undefined}
      >
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

      <ResizeHandle
        width={sidebarWidth}
        height={sidebarHeight}
        isDragging={isDragging}
        isMobile={isMobile}
        onPointerDown={startResize}
        onDoubleClick={resetSize}
        label="Resize overview split"
      />

      <section
        className="split-right"
        aria-label="Antarctic situation map"
        style={isMobile ? { flex: 1, minHeight: 160 } : undefined}
      >
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
