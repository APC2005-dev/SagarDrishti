import { AnimatePresence, motion } from 'framer-motion';

import { FEED_STATE_LABEL } from '../../constants';
import type { FeedState, Provenance } from '../../types/api';

type Tone = 'ok' | 'info' | 'warn' | 'bad' | 'neutral' | 'official' | 'forecast';

const FEED_TONE: Record<FeedState, Tone> = { LIVE: 'ok', SYNCED: 'info', DEGRADED: 'warn', FAILED: 'bad', UNKNOWN: 'neutral' };

export function Chip({ tone, children, title, pulse }: { tone: Tone; children: React.ReactNode; title?: string; pulse?: boolean }) {
  return (
    <span className={`chip chip-${tone}`} title={title}>
      <span className={`chip-dot${pulse ? ' pulse' : ''}`} aria-hidden />
      {children}
    </span>
  );
}

export function FeedStateChip({ state, label }: { state: FeedState; label?: string }) {
  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.span key={state} initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.18 }}>
        <Chip tone={FEED_TONE[state]} pulse={state === 'LIVE'}>
          {label ?? FEED_STATE_LABEL[state]}
        </Chip>
      </motion.span>
    </AnimatePresence>
  );
}

const PROVENANCE: Record<Provenance, { tone: Tone; label: string; title: string }> = {
  official_usnic: { tone: 'official', label: 'OFFICIAL', title: 'Official USNIC observation' },
  historical_training_dataset: { tone: 'neutral', label: 'HISTORICAL', title: 'BYU consolidated database (training history)' },
  derived: { tone: 'neutral', label: 'DERIVED', title: 'Derived value' },
  interpolated: { tone: 'warn', label: 'INTERPOLATED', title: 'Interpolated — not an observation' },
  predicted: { tone: 'forecast', label: 'FORECAST', title: 'Model prediction — not an observation' },
};

export function ProvenanceChip({ provenance }: { provenance: Provenance }) {
  const p = PROVENANCE[provenance];
  return (
    <Chip tone={p.tone} title={p.title}>
      {p.label}
    </Chip>
  );
}

const RUN_TONE: Record<string, Tone> = {
  success: 'ok',
  promoted: 'ok',
  deployed: 'ok',
  unchanged: 'info',
  validated: 'info',
  candidate: 'info',
  running: 'info',
  partial: 'warn',
  skipped: 'neutral',
  archived: 'neutral',
  rejected: 'warn',
  failed: 'bad',
  active: 'official',
  not_in_latest_source: 'warn',
  historical_only: 'neutral',
};

export function RunStatusChip({ status }: { status: string }) {
  return <Chip tone={RUN_TONE[status] ?? 'neutral'}>{status.replace(/_/g, ' ').toUpperCase()}</Chip>;
}

export function StaleChip() {
  return (
    <Chip tone="warn" title="Latest official observation is older than the staleness threshold">
      STALE
    </Chip>
  );
}
