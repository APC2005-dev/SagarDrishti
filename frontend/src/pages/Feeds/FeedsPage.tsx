import { motion } from 'framer-motion';

import { KV } from '../../components/common/Metric';
import { QueryState } from '../../components/common/QueryState';
import { Chip, FeedStateChip } from '../../components/common/StatusChip';
import { useFeeds } from '../../hooks/queries';
import { fmtDate, fmtDateTime, fmtNum, relTime, shortHash } from '../../utils/format';

/** Sources the architecture anticipates. Listed as not configured — no data is shown for them. */
const PLANNED = [
  { name: 'Antarctic sea-ice concentration', note: 'Gridded daily concentration for the sea-ice screen and route constraints.' },
  { name: 'Ocean surface currents', note: 'Candidate exogenous feature for a future architecture version (explicitly versioned).' },
  { name: 'Vessel positions (AIS)', note: 'Route planning input.' },
];

export default function FeedsPage() {
  const feeds = useFeeds();
  return (
    <div className="page-scroll">
      <div className="page-head">
        <h1>Data feeds</h1>
        <span className="mono dim" style={{ fontSize: 11 }}>
          the browser never contacts sources directly — all data flows through the backend
        </span>
      </div>
      <QueryState query={feeds}>
        {(list) => (
          <div className="cards">
            {list.map((f, i) => (
              <motion.div key={f.id} className="panel card" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 10 }}>
                  <div>
                    <h3>{f.name}</h3>
                    <span className="dim">{f.provider}</span>
                  </div>
                  <FeedStateChip state={f.state} />
                </div>
                <p className="muted" style={{ fontSize: 12 }}>{f.description}</p>
                {f.stateReasons.length > 0 && <div className="note warn" style={{ marginBottom: 12 }}>{f.stateReasons.join(' · ')}</div>}
                <div className="kv-grid">
                  <KV k="Availability" v={f.errorMessage ? 'last fetch failed' : f.lastSuccessAt ? 'reachable' : 'unknown'} />
                  <KV k="Cadence" v={f.cadence} />
                  <KV k="Last fetch" v={`${fmtDateTime(f.lastFetchAt)} (${relTime(f.lastFetchAt)})`} />
                  <KV k="Last success" v={fmtDateTime(f.lastSuccessAt)} />
                  <KV k="Latest official obs." v={fmtDate(f.latestOfficialObservationDate)} />
                  <KV k="Fetch duration" v={f.fetchDurationMs != null ? `${fmtNum(f.fetchDurationMs)} ms` : '—'} />
                  <KV k="Records" v={fmtNum(f.recordCount)} />
                  <KV k="Discovery" v={f.discoveryMethod ?? '—'} />
                  <KV k="Checksum (sha256)" v={shortHash(f.checksumSha256, 20)} />
                  <KV k="Source URL" v={f.sourceUrl ? <a href={f.productUrl} target="_blank" rel="noreferrer">{f.sourceUrl}</a> : '—'} />
                </div>
                {f.errorMessage && <div className="note warn" style={{ marginTop: 12 }}>{f.errorMessage}</div>}
              </motion.div>
            ))}
            {PLANNED.map((p) => (
              <div key={p.name} className="placeholder-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
                  <span style={{ color: 'var(--text-2)' }}>{p.name}</span>
                  <Chip tone="neutral">NOT CONFIGURED</Chip>
                </div>
                <p style={{ fontSize: 12, marginBottom: 0 }}>{p.note}</p>
              </div>
            ))}
          </div>
        )}
      </QueryState>
    </div>
  );
}
