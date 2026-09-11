import { create } from 'zustand';

import type { ForecastHorizon } from '../types/api';

export type RiskConeMode = 'selected' | 'all' | 'off';

/** Map layers. Environmental overlays are off by default so they never compete with trajectories. */
export interface MapLayers {
  icebergs: boolean;
  forecast: boolean;
  wind: boolean;
  current: boolean;
  seaIce: boolean;
}

interface UiState {
  selectedIcebergId: string | null;
  hoveredIcebergId: string | null;
  horizon: ForecastHorizon;
  riskCones: RiskConeMode;
  showHistory: boolean;
  layers: MapLayers;
  select: (id: string | null) => void;
  hover: (id: string | null) => void;
  setHorizon: (h: ForecastHorizon) => void;
  setRiskCones: (m: RiskConeMode) => void;
  toggleHistory: () => void;
  toggleLayer: (layer: keyof MapLayers) => void;
}

export const useUi = create<UiState>((set) => ({
  selectedIcebergId: null,
  hoveredIcebergId: null,
  horizon: 7,
  riskCones: 'selected',
  showHistory: true,
  layers: { icebergs: true, forecast: true, wind: false, current: false, seaIce: false },
  select: (id) => set({ selectedIcebergId: id }),
  hover: (id) => set({ hoveredIcebergId: id }),
  setHorizon: (horizon) => set({ horizon }),
  setRiskCones: (riskCones) => set({ riskCones }),
  toggleHistory: () => set((s) => ({ showHistory: !s.showHistory })),
  toggleLayer: (layer) => set((s) => ({ layers: { ...s.layers, [layer]: !s.layers[layer] } })),
}));
