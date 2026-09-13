import { ApiError } from '../../api/client';
import { useEnvironmentField } from '../../hooks/queries';
import { type MapLayers, useUi } from '../../stores/uiStore';

const LABELS: { key: keyof MapLayers; label: string; env?: 'wind' | 'current' | 'sea_ice' }[] = [
  { key: 'icebergs', label: 'ICEBERGS' },
  { key: 'forecast', label: 'FORECAST' },
  { key: 'wind', label: 'WIND', env: 'wind' },
  { key: 'current', label: 'CURRENT', env: 'current' },
];

function EnvToggle({ k, label, group }: { k: keyof MapLayers; label: string; group: 'wind' | 'current' | 'sea_ice' }) {
  const on = useUi((s) => s.layers[k]);
  const toggle = useUi((s) => s.toggleLayer);
  const field = useEnvironmentField(group, on);
  const unavailable = on && field.isError;
  const title = unavailable
    ? field.error instanceof ApiError ? field.error.detail : 'overlay unavailable'
    : field.data
      ? `${field.data.datasetId ?? ''} · valid ${field.data.validDate} · ${field.data.resolutionDeg}° grid`
      : 'Environmental overlay (cached analysis, not an observation of the iceberg)';
  return (
    <button className={`btn-ghost${on ? ' active' : ''}`} onClick={() => toggle(k)} title={title} style={unavailable ? { color: 'var(--warn)' } : undefined}>
      {label}
      {unavailable ? ' · N/A' : ''}
    </button>
  );
}

export function LayerToggles() {
  const layers = useUi((s) => s.layers);
  const toggle = useUi((s) => s.toggleLayer);
  return (
    <div className="segmented" role="group" aria-label="Map layers">
      {LABELS.map((l) =>
        l.env ? (
          <EnvToggle key={l.key} k={l.key} label={l.label} group={l.env} />
        ) : (
          <button key={l.key} className={`btn-ghost${layers[l.key] ? ' active' : ''}`} onClick={() => toggle(l.key)}>
            {l.label}
          </button>
        ),
      )}
    </div>
  );
}
