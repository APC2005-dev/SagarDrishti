import { AnimatePresence, motion } from 'framer-motion';

import { useHealth, useIngestion } from '../../hooks/queries';

/** Global degraded-state strip: backend down, model unavailable, or feed not healthy. */
export function SystemBanner() {
  const health = useHealth();
  const ingestion = useIngestion();

  let message: string | null = null;
  let tone: 'bad' | 'warn' = 'warn';
  if (health.isError) {
    message = 'Backend API unreachable — displayed data may be outdated. No values are being substituted.';
    tone = 'bad';
  } else if (health.data?.checks.model && health.data.checks.model.ok === false) {
    message = `Forecast model unavailable (${String(health.data.checks.model.error ?? 'artifact missing')}). Official positions are still shown; forecasts are not generated.`;
  } else if (ingestion.data && (ingestion.data.feedState === 'DEGRADED' || ingestion.data.feedState === 'FAILED')) {
    tone = ingestion.data.feedState === 'FAILED' ? 'bad' : 'warn';
    message = `USNIC feed ${ingestion.data.feedState}: ${ingestion.data.stateReasons.join('; ')}`;
  }

  return (
    <AnimatePresence initial={false}>
      {message && (
        <motion.div
          className={`system-banner tone-${tone}`}
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          transition={{ duration: 0.2 }}
          role="status"
        >
          <span className="mono">{message}</span>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
