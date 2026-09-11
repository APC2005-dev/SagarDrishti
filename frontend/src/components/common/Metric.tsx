import { AnimatePresence, motion } from 'framer-motion';
import type { ReactNode } from 'react';

interface MetricProps {
  label: string;
  value: ReactNode;
  unit?: string;
  sub?: ReactNode;
  tone?: 'default' | 'official' | 'forecast' | 'warn' | 'bad' | 'ok';
  title?: string;
}

/** Compact metric tile; value changes cross-fade so updates are noticeable but calm. */
export function Metric({ label, value, unit, sub, tone = 'default', title }: MetricProps) {
  const key = typeof value === 'string' || typeof value === 'number' ? String(value) : undefined;
  return (
    <div className={`metric tone-${tone}`} title={title}>
      <span className="label">{label}</span>
      <div className="metric-value mono">
        <AnimatePresence mode="popLayout" initial={false}>
          <motion.span key={key} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={{ duration: 0.2 }}>
            {value}
          </motion.span>
        </AnimatePresence>
        {unit && <span className="metric-unit">{unit}</span>}
      </div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  );
}

export function KV({ k, v, mono = true }: { k: string; v: ReactNode; mono?: boolean }) {
  return (
    <div className="kv">
      <span className="label">{k}</span>
      <span className={mono ? 'mono' : undefined}>{v}</span>
    </div>
  );
}

export function SectionTitle({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="section-title">
      <span className="label">{children}</span>
      {right}
    </div>
  );
}
