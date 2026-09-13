import { AnimatePresence, motion } from 'framer-motion';
import { useEffect, useState } from 'react';

const SCAN_MESSAGES = [
  'Analyzing sea-ice conditions...',
  'Checking iceberg hazards...',
  'Evaluating fuel efficiency...',
  'Calculating optimal route...',
  'Finalizing safest passage...',
];

interface RouteScannerProps {
  active: boolean;
  onCancel?: () => void;
}

export function RouteScanner({ active, onCancel }: RouteScannerProps) {
  const [msgIndex, setMsgIndex] = useState(0);

  useEffect(() => {
    if (!active) {
      setMsgIndex(0);
      return;
    }
    const timer = setInterval(() => {
      setMsgIndex((prev) => (prev + 1) % SCAN_MESSAGES.length);
    }, 2400);
    return () => clearInterval(timer);
  }, [active]);

  // Web Audio API high-tech radar scanning sound synthesis
  useEffect(() => {
    if (!active) return;
    let audioCtx: AudioContext | null = null;
    let isCancelled = false;

    try {
      const AudioContextClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      audioCtx = new AudioContextClass();
    } catch {
      return;
    }

    if (audioCtx.state === 'suspended') {
      void audioCtx.resume();
    }

    // Atmospheric low radar hum generator
    let humOsc: OscillatorNode | null = null;
    let humGain: GainNode | null = null;
    try {
      humOsc = audioCtx.createOscillator();
      humGain = audioCtx.createGain();
      humOsc.type = 'sine';
      humOsc.frequency.setValueAtTime(110, audioCtx.currentTime); // Low 110Hz hum
      humGain.gain.setValueAtTime(0.02, audioCtx.currentTime);
      humOsc.connect(humGain);
      humGain.connect(audioCtx.destination);
      humOsc.start();
    } catch {
      // Ignored
    }

    // Synthesize radar beam sweep ping sound (frequency chirp sweep + white noise burst)
    const playScannerSweepSound = () => {
      if (isCancelled || !audioCtx || audioCtx.state === 'closed') return;
      try {
        const now = audioCtx.currentTime;

        // 1. High-tech frequency sweep (Chirp: 1800Hz down to 800Hz)
        const sweepOsc = audioCtx.createOscillator();
        const sweepGain = audioCtx.createGain();
        sweepOsc.type = 'sine';
        sweepOsc.frequency.setValueAtTime(1800, now);
        sweepOsc.frequency.exponentialRampToValueAtTime(600, now + 0.18);

        sweepGain.gain.setValueAtTime(0.12, now);
        sweepGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.18);

        sweepOsc.connect(sweepGain);
        sweepGain.connect(audioCtx.destination);

        sweepOsc.start(now);
        sweepOsc.stop(now + 0.18);

        // 2. Textural radar sweep static burst (filtered noise burst)
        const bufferSize = audioCtx.sampleRate * 0.12;
        const noiseBuffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
        const output = noiseBuffer.getChannelData(0);
        for (let i = 0; i < bufferSize; i++) {
          output[i] = Math.random() * 2 - 1;
        }

        const whiteNoise = audioCtx.createBufferSource();
        whiteNoise.buffer = noiseBuffer;

        const filter = audioCtx.createBiquadFilter();
        filter.type = 'bandpass';
        filter.frequency.setValueAtTime(2400, now);
        filter.Q.setValueAtTime(3.0, now);

        const noiseGain = audioCtx.createGain();
        noiseGain.gain.setValueAtTime(0.05, now);
        noiseGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.12);

        whiteNoise.connect(filter);
        filter.connect(noiseGain);
        noiseGain.connect(audioCtx.destination);

        whiteNoise.start(now);
        whiteNoise.stop(now + 0.12);
      } catch {
        // Fallback
      }
    };

    // Play immediate sweep chirp sound, then repeat every 2.4 seconds in sync with radar sweep rotation
    playScannerSweepSound();
    const scanInterval = setInterval(playScannerSweepSound, 2400);

    return () => {
      isCancelled = true;
      clearInterval(scanInterval);
      if (humOsc) {
        try { humOsc.stop(); } catch { /* ignore */ }
      }
      if (audioCtx && audioCtx.state !== 'closed') {
        void audioCtx.close();
      }
    };
  }, [active]);

  return (
    <AnimatePresence>
      {active && (
        <motion.div
          className="route-scanner-overlay"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.35, ease: 'easeInOut' }}
        >
          {/* Radar Sweep Effect */}
          <div className="radar-container">
            <div className="radar-circle ring-1" />
            <div className="radar-circle ring-2" />
            <div className="radar-circle ring-3" />
            <div className="radar-crosshair-h" />
            <div className="radar-crosshair-v" />
            <div className="radar-sweep" />
            <div className="radar-center-ping" />
          </div>

          {/* Status HUD Overlay */}
          <div className="radar-hud-card glass">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
              <div className="radar-hud-header">
                <span className="radar-hud-dot pulse" />
                <span className="radar-hud-title mono">SCANNING ROUTE</span>
              </div>
              {onCancel && (
                <button
                  type="button"
                  className="btn-ghost"
                  style={{ padding: '2px 8px', fontSize: 10, borderColor: 'var(--bad)', color: 'var(--bad)' }}
                  onClick={onCancel}
                >
                  Cancel
                </button>
              )}
            </div>
            <div className="radar-hud-message mono">
              <motion.span
                key={msgIndex}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: 0.3 }}
              >
                {SCAN_MESSAGES[msgIndex]}
              </motion.span>
            </div>
            {/* Animated Indeterminate Progress Bar */}
            <div className="radar-progress-bar">
              <div className="radar-progress-indicator" />
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
