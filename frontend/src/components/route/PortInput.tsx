import { useEffect, useMemo, useRef, useState } from 'react';

import { api } from '../../api/client';
import type { PortSummary } from '../../types/api';

/**
 * Port name input backed by the backend's World Port Index search.
 *
 * The browser never contacts NGA — it calls `/api/v1/ports/search`. Requests are
 * debounced, and a sequence number discards responses that arrive out of order
 * so a slow earlier query can never overwrite a newer one. Selecting a
 * suggestion pins the canonical WPI identifier, but the backend re-validates the
 * request regardless: this component is convenience, never authorisation.
 */

const DEBOUNCE_MS = 250;

interface Props {
  label: string;
  value: PortSummary | null;
  onChange: (port: PortSummary | null) => void;
  disabled?: boolean;
  invalidMessage?: string | null;
}

export function PortInput({ label, value, onChange, disabled, invalidMessage }: Props) {
  const [text, setText] = useState(value?.name ?? '');
  const [suggestions, setSuggestions] = useState<PortSummary[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [open, setOpen] = useState(false);
  const sequence = useRef(0);
  const inputId = useMemo(() => `port-${label.toLowerCase().replace(/\s+/g, '-')}`, [label]);

  useEffect(() => {
    setText(value?.name ?? '');
  }, [value]);

  useEffect(() => {
    const query = text.trim();
    if (!open || query.length < 2 || (value && query === value.name)) {
      setSuggestions(null);
      setSearching(false);
      return;
    }
    setSearching(true);
    const ticket = ++sequence.current;
    const timer = setTimeout(() => {
      api
        .searchPorts(query)
        .then((rows) => {
          if (ticket !== sequence.current) return; // a newer query has superseded this one
          setSuggestions(rows);
          setSearching(false);
        })
        .catch(() => {
          if (ticket !== sequence.current) return;
          setSuggestions([]);
          setSearching(false);
        });
    }, DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [text, open, value]);

  return (
    <div style={{ position: 'relative' }}>
      <label className="label" htmlFor={inputId} style={{ display: 'block', marginBottom: 4 }}>
        {label}
      </label>
      <div style={{ position: 'relative' }}>
        <input
          id={inputId}
          className="input mono"
          style={{ width: '100%', paddingRight: value ? 28 : 12 }}
          autoComplete="off"
          placeholder="Type a port name…"
          value={text}
          disabled={disabled}
          onChange={(e) => {
            const val = e.target.value;
            setText(val);
            setOpen(true);
            if (value && val !== value.name) onChange(null); // typing invalidates a previous selection
          }}
          onFocus={() => {
            setOpen(true);
          }}
          onBlur={() => setTimeout(() => setOpen(false), 250)}
        />
        {value && !disabled && (
          <button
            type="button"
            style={{
              position: 'absolute',
              right: 8,
              top: 7,
              background: 'transparent',
              border: 'none',
              color: 'var(--text-3)',
              cursor: 'pointer',
              fontSize: 14,
              lineHeight: 1,
              padding: 0,
            }}
            title="Clear port selection"
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => {
              onChange(null);
              setText('');
              setOpen(true);
            }}
          >
            ✕
          </button>
        )}
      </div>
      {value && (
        <div className="dim mono" style={{ fontSize: 11, marginTop: 4 }}>
          WPI {value.identifier} · {value.countryName} · {value.latitude.toFixed(3)}, {value.longitude.toFixed(3)}
        </div>
      )}
      {invalidMessage && (
        <div className="qs-error" style={{ fontSize: 11, marginTop: 4 }}>
          {invalidMessage}
        </div>
      )}
      {open && !value && text.trim().length >= 2 && (
        <div
          className="glass"
          style={{ position: 'absolute', zIndex: 40, left: 0, right: 0, marginTop: 4, padding: 4, maxHeight: 260, overflowY: 'auto' }}
        >
          {searching && <div className="dim mono" style={{ fontSize: 11, padding: 6 }}>Searching the World Port Index…</div>}
          {!searching && suggestions?.length === 0 && (
            <div className="dim mono" style={{ fontSize: 11, padding: 6 }}>
              No matching port in the World Port Index.
            </div>
          )}
          {!searching &&
            suggestions?.map((port) => (
              <button
                key={port.id}
                type="button"
                className="suggestion"
                style={{
                  display: 'block', width: '100%', textAlign: 'left', background: 'transparent',
                  border: 0, padding: '6px 8px', cursor: 'pointer', borderRadius: 3,
                }}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => {
                  onChange(port);
                  setText(port.name);
                  setOpen(false);
                }}
              >
                <span>{port.name}</span>
                <span className="dim mono" style={{ fontSize: 11, marginLeft: 8 }}>
                  {port.countryName} · {port.latitude.toFixed(2)}, {port.longitude.toFixed(2)}
                </span>
              </button>
            ))}
        </div>
      )}
    </div>
  );
}
