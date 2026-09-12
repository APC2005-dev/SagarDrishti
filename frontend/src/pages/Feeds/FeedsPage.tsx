import { motion } from 'framer-motion';
import { useMemo } from 'react';

import { KV } from '../../components/common/Metric';
import { QueryState } from '../../components/common/QueryState';
import { Chip, FeedStateChip } from '../../components/common/StatusChip';
import { useFeeds } from '../../hooks/queries';
import type { Feed } from '../../types/api';
import { fmtDate, fmtDateTime, fmtNum, relTime, shortHash } from '../../utils/format';

/** Sources the architecture anticipates. Listed as not configured — no data is shown for them. */
const PLANNED = [{ name: 'Vessel positions (AIS)', note: 'Route planning input.' }];

export default function FeedsPage() {
  const feeds = useFeeds();

  /**
   * Driven entirely by the backend list. Core model feeds (the iceberg CSV and
   * the sea-ice product the U-Net was trained on) come first; environmental
   * sources are the future feature inputs and are labelled as such. Nothing is
   * invented here when a feed is missing — an absent feed simply is not shown.
   */
  const { coreFeeds, environmentalFeeds } = useMemo(() => {
    const list = feeds.data ?? [];
    const rank = (f: Feed) => (f.category === 'iceberg' ? 0 : f.category === 'sea_ice' ? 1 : 2);
    const sorted = [...list].sort((a, b) => rank(a) - rank(b));
    return {
      coreFeeds: sorted.filter((f) => f.category !== 'environmental'),
      environmentalFeeds: sorted.filter((f) => f.category === 'environmental'),
    };
  }, [feeds.data]);
  const filteredFeeds = useMemo(
    () => [...coreFeeds, ...environmentalFeeds],
    [coreFeeds, environmentalFeeds],
  );

  return (
    <div className="page-scroll">
      <div className="page-head">
        <h1>Data feeds</h1>
        <span className="mono dim" style={{ fontSize: 11 }}>
          the browser never contacts sources directly — all data flows through the backend
        </span>
      </div>
      <QueryState query={feeds}>
        {() => {
          const configuredFeeds = filteredFeeds.filter((f) => f.configured);
          const unconfiguredFeeds = filteredFeeds.filter((f) => !f.configured);

          return (
            <div className="cards">
              {configuredFeeds.map((f, i) => (
                <motion.div
                  key={f.id}
                  className="panel card"
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 10 }}>
                    <div>
                      <h3>{f.name}</h3>
                      <span className="dim">{f.provider}</span>
                    </div>
                    <div style={{ display: 'flex', gap: 6, alignItems: 'flex-start' }}>
                      {f.category === 'environmental' ? (
                        <Chip tone="neutral">ENVIRONMENTAL</Chip>
                      ) : (
                        <Chip tone="info">CORE MODEL FEED</Chip>
                      )}
                      <FeedStateChip state={f.state} />
                    </div>
                  </div>
                  <p className="muted" style={{ fontSize: 12 }}>
                    {f.description}
                  </p>
                  {f.stateReasons.length > 0 && (
                    <div className="note warn" style={{ marginBottom: 12 }}>
                      {f.stateReasons.join(' · ')}
                    </div>
                  )}
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
                    <KV
                      k="Source URL"
                      v={f.sourceUrl ? <a href={f.productUrl} target="_blank" rel="noreferrer">{f.sourceUrl}</a> : '—'}
                    />
                  </div>
                  {f.errorMessage && <div className="note warn" style={{ marginTop: 12 }}>{f.errorMessage}</div>}
                </motion.div>
              ))}

              {unconfiguredFeeds.map((f) => (
                <div key={f.id} className="placeholder-card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
                    <span style={{ color: 'var(--text-2)' }}>{f.name}</span>
                    <Chip tone="neutral">NOT CONFIGURED</Chip>
                  </div>
                  <p style={{ fontSize: 12, marginBottom: 0 }}>{f.description}</p>
                  {f.stateReasons.length > 0 && (
                    <span className="mono dim" style={{ fontSize: 11, marginTop: 4, display: 'block' }}>
                      {f.stateReasons.join(' · ')}
                    </span>
                  )}
                </div>
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
          );
        }}
      </QueryState>
    </div>
  );
}
