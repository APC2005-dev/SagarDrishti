import { SectionTitle } from '../../components/common/Metric';
import { ResizeHandle } from '../../components/common/ResizeHandle';
import { Chip } from '../../components/common/StatusChip';
import { AntarcticScene } from '../../components/globe/AntarcticScene';
import { useIcebergs, useLatestForecasts } from '../../hooks/queries';
import { useResizableSidebar } from '../../hooks/useResizableSidebar';

/**
 * Structure for a future routing engine. The inputs it will consume are
 * listed with their real availability; only iceberg forecast danger zones
 * (p90 radii around D+1…D+7) exist today and are drawn on the map.
 */
const INPUTS = [
  { name: 'Iceberg forecast positions', status: 'available', detail: 'GET /api/v1/forecasts/latest — D+1…D+7 per iceberg with model version.' },
  { name: 'Danger zones', status: 'available', detail: 'p90 error radius around each forecast point (empirical, per horizon).' },
  { name: 'Sea-ice constraints', status: 'planned', detail: 'Requires a sea-ice concentration feed (not configured).' },
  { name: 'Vessel route', status: 'planned', detail: 'Waypoints input and ice-class parameters.' },
  { name: 'Routing engine', status: 'planned', detail: 'Not implemented in this release; no routes are computed or suggested.' },
] as const;

export default function RoutePlanningPage() {
  const icebergs = useIcebergs();
  const forecasts = useLatestForecasts();

  const { sidebarWidth, isDragging, isMobile, startResize, resetWidth } = useResizableSidebar({
    storageKey: 'sagar_sidebar_width_route_planning',
    defaultWidth: 420,
    minWidth: 320,
  });

  return (
    <div
      className="split-resizable"
      style={isMobile ? undefined : { gridTemplateColumns: `${sidebarWidth}px 6px 1fr` }}
    >
      <section className="split-left scroll pad" aria-label="Route planning inputs">
        <div className="page-head">
          <h1>Route planning</h1>
        </div>
        <div className="note">
          This screen defines the inputs a maritime routing engine will consume. It shows the danger zones derived from the
          current iceberg forecasts; it does not compute routes.
        </div>
        <SectionTitle>Inputs</SectionTitle>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {INPUTS.map((i) => (
            <div key={i.name} className={i.status === 'available' ? 'panel card' : 'placeholder-card'}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
                <span>{i.name}</span>
                <Chip tone={i.status === 'available' ? 'ok' : 'neutral'}>{i.status.toUpperCase()}</Chip>
              </div>
              <div className="dim" style={{ fontSize: 12, marginTop: 6 }}>{i.detail}</div>
            </div>
          ))}
        </div>
      </section>
      <ResizeHandle
        width={sidebarWidth}
        isDragging={isDragging}
        isMobile={isMobile}
        onPointerDown={startResize}
        onDoubleClick={resetWidth}
        label="Resize route planning sidebar"
      />
      <section className="split-right" aria-label="Route planning map">
        <AntarcticScene icebergs={icebergs.data?.items ?? []} forecasts={forecasts.data ?? []} horizon={7} riskMode="all" showHistory={false} />
      </section>
    </div>
  );
}
