import { create } from 'zustand';

import type { ForecastHorizon } from '../types/api';

export type RiskConeMode = 'selected' | 'all' | 'off';

interface UiState {
  selectedIcebergId: string | null;
  hoveredIcebergId: string | null;
  horizon: ForecastHorizon;
  riskCones: RiskConeMode;
  showHistory: boolean;
  select: (id: string | null) => void;
  hover: (id: string | null) => void;
  setHorizon: (h: ForecastHorizon) => void;
  setRiskCones: (m: RiskConeMode) => void;
  toggleHistory: () => void;
}

export const useUi = create<UiState>((set) => ({
  selectedIcebergId: null,
  hoveredIcebergId: null,
  horizon: 7,
  riskCones: 'selected',
  showHistory: true,
  select: (id) => set({ selectedIcebergId: id }),
  hover: (id) => set({ hoveredIcebergId: id }),
  setHorizon: (horizon) => set({ horizon }),
  setRiskCones: (riskCones) => set({ riskCones }),
  toggleHistory: () => set((s) => ({ showHistory: !s.showHistory })),
}));
