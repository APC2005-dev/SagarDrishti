import { motion } from 'framer-motion';

import { HORIZONS } from '../../constants';
import type { ForecastHorizon } from '../../types/api';

interface Props {
  value: ForecastHorizon;
  onChange: (h: ForecastHorizon) => void;
}

/** 1 / 3 / 7 DAY — filters over one 7-day model run, never separate models. */
export function HorizonSelector({ value, onChange }: Props) {
  return (
    <div className="segmented" role="radiogroup" aria-label="Forecast horizon">
      {HORIZONS.map((h) => (
        <button key={h} role="radio" aria-checked={value === h} className={`seg${value === h ? ' active' : ''}`} onClick={() => onChange(h)}>
          {value === h && <motion.span layoutId="horizon-pill" className="seg-pill" transition={{ type: 'spring', stiffness: 420, damping: 36 }} />}
          <span className="seg-text mono">
            {h} {h === 1 ? 'DAY' : 'DAYS'}
          </span>
        </button>
      ))}
    </div>
  );
}
