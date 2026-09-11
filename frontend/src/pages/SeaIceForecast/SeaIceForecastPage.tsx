import { SectionTitle } from '../../components/common/Metric';
import { Chip } from '../../components/common/StatusChip';

/**
 * Sea-ice forecasting is a different problem from iceberg trajectory
 * forecasting (gridded concentration fields vs. point trajectories). It gets
 * its own screen and, in future, its own feed and model lineage.
 */
export default function SeaIceForecastPage() {
  return (
    <div className="page-scroll" style={{ maxWidth: 980 }}>
      <div className="page-head">
        <h1>Sea-ice forecast</h1>
        <Chip tone="neutral">NO SOURCE CONFIGURED</Chip>
      </div>
      <div className="note">
        No sea-ice data is ingested in this release, so nothing is displayed here — this screen will never show simulated ice
        fields. Iceberg trajectories (Icebergs / 1-, 3-, 7-day views) are a separate product with separate models.
      </div>

      <SectionTitle>How the two products differ</SectionTitle>
      <div className="cards">
        <div className="panel card">
          <h3>Iceberg trajectory forecast</h3>
          <div className="muted" style={{ fontSize: 12 }}>
            Point positions of named icebergs · source USNIC Antarctic iceberg CSV (weekly) · model GRU entry14→day7 · output
            D+1…D+7 positions with p90 error radius.
          </div>
        </div>
        <div className="panel card">
          <h3>Sea-ice forecast</h3>
          <div className="muted" style={{ fontSize: 12 }}>
            Gridded concentration / extent fields · requires a sea-ice analysis feed (e.g. USNIC Antarctic sea-ice charts) · will
            be registered as its own feed and model family.
          </div>
        </div>
      </div>

      <SectionTitle>Planned integration contract</SectionTitle>
      <div className="panel pad mono" style={{ fontSize: 12, lineHeight: 1.8 }}>
        feed: tracking.ingestion_runs (source = sea_ice_*), raster/vector storage in PostGIS
        <br />
        api: GET /api/v1/sea-ice/latest · GET /api/v1/sea-ice/forecast?lead_days=
        <br />
        ui: concentration layer in the Antarctic scene · route-planning constraint input
      </div>
      <p className="dim" style={{ fontSize: 12 }}>
        Official USNIC Antarctic products:{' '}
        <a href="https://usicecenter.gov/Products/AntarcHome" target="_blank" rel="noreferrer">
          usicecenter.gov/Products/AntarcHome
        </a>
      </p>
    </div>
  );
}
