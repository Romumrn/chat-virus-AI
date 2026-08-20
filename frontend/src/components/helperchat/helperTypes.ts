/**
 * helperTypes.ts — shared types for the contagion helper. All stats are
 * normalized to [0,1]; the engine maps them onto growth and spread rates.
 */

export interface VirusStats {
  /** How fast it spreads locally and jumps to neighbours. */
  contagiosite: number;
  /** Share of the infected that die (higher = scarier, but visible sooner). */
  letalite: number;
  /** Resistance to the cure / recovery — slows eradication. */
  resistance: number;
}

export interface VirusConfig {
  virusName: string;
  presetId: string;
  stats: VirusStats;
}

/** Total "genetic points" the player can spread across the three stats. */
export const STAT_BUDGET = 1.5;

export function statTotal(s: VirusStats): number {
  return s.contagiosite + s.letalite + s.resistance;
}
