import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useMemo, useState } from 'react';

import { api, errorParts } from '../../api/client';
import { Metric, SectionTitle } from '../../components/common/Metric';
import { ResizeHandle } from '../../components/common/ResizeHandle';
import { Chip } from '../../components/common/StatusChip';
import { AntarcticScene } from '../../components/globe/AntarcticScene';
import { RouteLayer } from '../../components/globe/RouteLayer';
import { PortInput } from '../../components/route/PortInput';
import { RouteScanner } from '../../components/route/RouteScanner';
import { useActiveRouteDetail, useIcebergs, useLatestForecasts } from '../../hooks/queries';
import { useResizableSidebar } from '../../hooks/useResizableSidebar';
import type { PortSummary, Route } from '../../types/api';
import { toScene } from '../../utils/projection';
import { fmtDateTime } from '../../utils/format';

/**
 * Two real World Port Index ports in, one time-dependent A* route out.
 *
 * The browser computes nothing: it posts the two canonical port identifiers and
 * renders the waypoints the backend returned, on the existing Antarctic scene
 * using that scene's own projection.
 */

/** Backend error codes -> what the user can actually do about them. */
const ERROR_HELP: Record<string, string> = {
  INVALID_DEPARTURE_PORT: 'Departure port is required.',
  INVALID_DESTINATION_PORT: 'Destination port is required.',
  PORT_NOT_FOUND: 'The specified port could not be found in the World Port Index.',
  SAME_PORT: 'Departure and destination ports must be different.',
  PORT_OUTSIDE_ROUTING_DOMAIN: 'This port is outside the currently supported Antarctic routing region.',
  PORT_SERVICE_UNAVAILABLE: 'The World Port Index service is unreachable. Try again shortly.',
  TRAJECTORY_FORECAST_UNAVAILABLE: 'No current iceberg forecast is available, so hazards cannot be evaluated.',
  SEA_ICE_FORECAST_UNAVAILABLE: 'No current sea-ice forecast is available. Routing without sea ice is not permitted.',
  ENVIRONMENT_DATA_UNAVAILABLE: 'Ocean-current data for this window is unavailable. Currents are never assumed to be zero.',
  ENVIRONMENT_FORECAST_MISMATCH: 'The iceberg and sea-ice forecasts do not share a reference time.',
  ROUTING_GRID_UNAVAILABLE: 'The routing grid could not be built for this region.',
  ROUTE_HORIZON_EXCEEDED: 'This voyage is longer than the available forecast horizon.',
  ROUTE_ENGINE_TIMEOUT: 'The search hit its time limit. This is not proof that no route exists.',
  NO_FEASIBLE_ROUTE: 'No feasible route exists under the current ice, iceberg and vessel constraints.',
  ROUTE_ENGINE_ERROR: 'The route engine failed.',
  CLIENT_TIMEOUT:
    'The browser gave up waiting. The search may still be running on the server — reopen this page shortly '
    + 'and use "Show it" if a voyage was recorded.',
};


/**
 * Vessel ice capability: the highest sea-ice concentration the ship may transit.
 * The route notebook's 0.30 is explicitly an illustrative software-test value,
 * not a certified ice class, so it is a vessel input rather than a constant.
 * Raising it does not relax the search — it describes a stronger ship.
 */
const ICE_CLASSES = [
  { value: 0.30, label: 'Standard (30%)' },
  { value: 0.50, label: 'Ice-strengthened (50%)' },
  { value: 0.70, label: 'Ice-class (70%)' },
] as const;

function hoursLabel(hours: number | null | undefined): string {
  if (hours == null) return '—';
  const days = Math.floor(hours / 24);
  const rest = Math.round(hours % 24);
  return days > 0 ? `${days}d ${rest}h` : `${rest}h`;
}

function ResultCard({ route }: { route: Route }) {
  const curvature = route.curvature;
  const num = (v: number | null | undefined, digits: number) =>
    typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '—';
  return (
    <div className="panel card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 10 }}>
        <h3 style={{ margin: 0 }}>Route found</h3>
        <Chip tone="ok">{route.status}</Chip>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10, marginTop: 12 }}>
        <Metric label="Departure" value={route.departure.port.name} />
        <Metric label="Destination" value={route.destination.port.name} />
        <Metric label="Distance" value={num(route.metrics.distanceKm, 1)} unit="km" />
        <Metric label="Travel duration" value={hoursLabel(route.metrics.durationHours)} />
        <Metric label="Fuel proxy" value={num(route.metrics.fuelProxy, 2)} sub="relative, not litres" />
        <Metric label="SIC exposure" value={num(route.metrics.sicExposureHours, 2)} sub="concentration-hours" />
        <Metric label="Route confidence" value="Not calibrated" sub="no calibrated probability" />
        <Metric label="Weighted objective" value={num(route.metrics.weightedObjective, 4)} />
      </div>
      {curvature && (
        <>
          <SectionTitle>Curvature (from the route geometry)</SectionTitle>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10 }}>
            <Metric label="Maximum turn" value={num(curvature.maximumTurnDeg, 2)} unit="°" />
            <Metric label="Max curvature" value={num(curvature.maximumDiscreteCurvatureRadPerKm, 5)} unit="rad/km" />
            <Metric label="Detour ratio" value={num(curvature.detourRatio, 4)} />
            <Metric label="Total turning" value={num(curvature.totalAbsoluteTurnDeg, 1)} unit="°" />
          </div>
          <div className="dim" style={{ fontSize: 11, marginTop: 8 }}>{curvature.curvatureDefinition}</div>
        </>
      )}
      <SectionTitle>Provenance</SectionTitle>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10 }}>
        <Metric label="Trajectory model" value={route.modelVersions.trajectory ?? '—'} />
        <Metric label="Sea-ice model" value={route.modelVersions.seaIce ?? '—'} />
        <Metric label="Route planner" value={route.modelVersions.routePlanner} />
        <Metric
          label="Forecast reference"
          value={route.forecastReferenceTime ? fmtDateTime(route.forecastReferenceTime) : '—'}
        />
      </div>
      <div className="dim mono" style={{ fontSize: 11, marginTop: 8 }}>
        {route.routeId} · {route.waypoints?.length ?? 0} waypoints · {route.expansions ?? 0} states expanded · connectors{' '}
        {route.connectorsValidated ? 'validated' : 'not validated'}
      </div>
    </div>
  );
}

export default function RoutePlanningPage() {
  const icebergs = useIcebergs();
  const forecasts = useLatestForecasts();
  const queryClient = useQueryClient();

  const [departure, setDeparture] = useState<PortSummary | null>(null);
  const [destination, setDestination] = useState<PortSummary | null>(null);
  const [horizon] = useState<number>(3);
  const [maxSic, setMaxSic] = useState<number>(0.3);
  const [route, setRoute] = useState<Route | null>(null);
  // The page opens as an empty form: a route is shown only once THIS visit has
  // produced one. The backend still holds the last voyage, but restoring it is
  // an explicit choice — auto-drawing it made the screen look like it always
  // had the same two ports baked in.
  const [restored, setRestored] = useState(false);
  const persisted = useActiveRouteDetail();
  const shown = route ?? (restored ? (persisted.data ?? null) : null);

  const plan = useMutation({
    mutationFn: () => api.planRoute(departure!.identifier, destination!.identifier, maxSic),
    onSuccess: (result) => {
      setRestored(false);
      setRoute(result);
      // The Feeds active-vessel card reads persisted backend state, not this one.
      void queryClient.invalidateQueries({ queryKey: ['activeRoute'] });
      void queryClient.invalidateQueries({ queryKey: ['activeRouteDetail'] });
    },
  });

  const resumable = !shown && !plan.isPending ? (persisted.data ?? null) : null;

  // Fly the camera to the center of Antarctica when scanning, or middle of route when loaded
  const routeFocus = useMemo<[number, number] | null>(() => {
    if (plan.isPending) {
      return [0, 0]; // Center of Antarctica polar stereographic projection
    }
    const points = shown?.waypoints;
    if (!points?.length) return null;
    const mid = points[Math.floor(points.length / 2)];
    return mid ? toScene(mid.latitude, mid.longitude) : null;
  }, [shown, plan.isPending]);

  const failure = plan.isError ? errorParts(plan.error) : null;
  const samePort = !!departure && !!destination && departure.id === destination.id;
  const canSubmit = !!departure && !!destination && !samePort && !plan.isPending;

  const {
    sidebarWidth,
    sidebarHeight,
    isDragging,
    isMobile,
    layoutMode,
    setLayoutMode,
    startResize,
    resetSize,
  } = useResizableSidebar({
    storageKey: 'sagar_sidebar_route_planning',
    defaultWidth: 440,
    minWidth: 320,
  });

  return (
    <div
      className={`split-resizable${isMobile ? ' mobile-stacked' : ''}`}
      style={isMobile ? undefined : { gridTemplateColumns: `${sidebarWidth}px 8px 1fr` }}
    >
      {/* Sidebar inputs panel */}
      {(!isMobile || layoutMode !== 'map') && (
        <section
          className="split-left scroll pad"
          aria-label="Route planning inputs"
          style={
            isMobile
              ? { height: layoutMode === 'content' ? '100%' : `${sidebarHeight}px`, flexShrink: 0 }
              : undefined
          }
        >
          <div className="page-head">
            <div>
              <h1>Route planning</h1>
            </div>

            {isMobile && (
              <div className="mobile-view-toggle">
                <button
                  type="button"
                  className={`mobile-view-btn${layoutMode === 'content' ? ' active' : ''}`}
                  onClick={() => setLayoutMode('content')}
                >
                  📋 Inputs
                </button>
                <button
                  type="button"
                  className={`mobile-view-btn${layoutMode === 'split' ? ' active' : ''}`}
                  onClick={() => setLayoutMode('split')}
                >
                  ◫ Split
                </button>
                <button
                  type="button"
                  className={`mobile-view-btn${layoutMode === 'map' ? ' active' : ''}`}
                  onClick={() => setLayoutMode('map')}
                >
                  🌐 Map
                </button>
              </div>
            )}
          </div>

          <div className="panel card">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <PortInput label="Departure port" value={departure} onChange={setDeparture} disabled={plan.isPending} />
              <PortInput
                label="Destination port"
                value={destination}
                onChange={setDestination}
                disabled={plan.isPending}
                invalidMessage={samePort ? 'Departure and destination ports must be different.' : null}
              />
              <div>
                <label className="label" htmlFor="vessel-ice" style={{ display: 'block', marginBottom: 4 }}>
                  Vessel ice capability
                </label>
                <select
                  id="vessel-ice"
                  className="input mono"
                  style={{ width: '100%' }}
                  value={maxSic}
                  disabled={plan.isPending}
                  onChange={(e) => setMaxSic(Number(e.target.value))}
                >
                  {ICE_CLASSES.map((c) => (
                    <option key={c.value} value={c.value}>{c.label}</option>
                  ))}
                </select>
                <div className="dim" style={{ fontSize: 11, marginTop: 4 }}>
                  Highest sea-ice concentration the vessel may transit. In late winter most
                  Antarctic crossings need an ice-class ship.
                </div>
              </div>
              <button
                type="button"
                className="btn primary"
                disabled={!canSubmit}
                onClick={() => plan.mutate()}
                style={{ width: '100%' }}
              >
                {plan.isPending ? 'Finding optimal route…' : 'Find Route'}
              </button>
              <div className="dim" style={{ fontSize: 11 }}>
                Ports are validated against the NGA World Port Index. If no route exists under the ice and iceberg
                constraints the backend says so — it never draws a straight line instead.
              </div>
            </div>
          </div>

          {resumable && !shown && (
            <div className="note" style={{ marginTop: 12, display: 'flex', justifyContent: 'space-between', gap: 10, alignItems: 'center' }}>
              <span>
                Last voyage on record: <strong>{resumable.departure.port.name}</strong> →{' '}
                <strong>{resumable.destination.port.name}</strong>
              </span>
              <button
                type="button"
                className="btn"
                onClick={() => {
                  setDeparture(resumable.departure.port);
                  setDestination(resumable.destination.port);
                  setRestored(true);
                }}
              >
                Load it
              </button>
            </div>
          )}

          {shown && !plan.isPending && (
            <div style={{ marginTop: 16 }}>
              <ResultCard route={shown} />
            </div>
          )}

          {plan.isPending && (
            <div className="note" style={{ marginTop: 12 }}>
              Assembling the forecast snapshot and running the route search. The first request for a forecast window
              also downloads its ocean currents, which takes a few minutes; later requests reuse them.
            </div>
          )}

          {failure && (
            <div className="panel card" style={{ marginTop: 12, borderColor: 'var(--bad)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
                <strong>Route not produced</strong>
                {failure.code && <Chip tone="bad">{failure.code}</Chip>}
              </div>
              <p className="muted" style={{ fontSize: 12, marginBottom: 0 }}>
                {(failure.code && ERROR_HELP[failure.code]) ?? failure.message}
              </p>
            </div>
          )}
        </section>
      )}

      {/* Resize Handle */}
      {(!isMobile || layoutMode === 'split') && (
        <ResizeHandle
          width={sidebarWidth}
          height={sidebarHeight}
          isDragging={isDragging}
          isMobile={isMobile}
          layoutMode={layoutMode}
          onLayoutModeChange={setLayoutMode}
          onPointerDown={startResize}
          onDoubleClick={resetSize}
          label="Resize route planning split"
        />
      )}

      {/* 3D Map panel */}
      {(!isMobile || layoutMode !== 'content') && (
        <section
          className="split-right"
          aria-label="Route planning map"
          style={
            isMobile
              ? { flex: 1, minHeight: 0, height: layoutMode === 'map' ? '100%' : 'auto' }
              : undefined
          }
        >
          {isMobile && layoutMode === 'map' && (
            <div className="mobile-floating-toggle">
              <button type="button" className="mobile-floating-btn" onClick={() => setLayoutMode('split')}>
                ◫ Show Inputs
              </button>
            </div>
          )}

          <RouteScanner active={plan.isPending} />
          <AntarcticScene
            icebergs={icebergs.data?.items ?? []}
            forecasts={forecasts.data ?? []}
            horizon={horizon}
            riskMode="all"
            showHistory={false}
            focusOverride={routeFocus}
            sceneChildren={shown?.waypoints?.length ? <RouteLayer waypoints={shown.waypoints} /> : null}
          />
        </section>
      )}
    </div>
  );
}
