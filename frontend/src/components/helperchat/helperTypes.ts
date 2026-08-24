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

/**
 * Transmission routes, grounded in real epidemiology. Each interacts with the
 * environment differently (see helperEngine for the exact model):
 *  - respiratoire : droplets/aerosols, person-to-person; worse in cold seasons.
 *  - vectorielle  : arthropod vectors (mosquitoes); thrives in hot/humid biomes,
 *                   collapses in cold/arid ones.
 *  - hydrique     : water / faecal-oral (cholera-like); worse where sanitation
 *                   is poor (low-wealth cities).
 *  - zoonotique   : animal reservoir spillover; sparks fresh rural outbreaks
 *                   independently of the existing spread.
 *  - sexuelle     : sexual / bloodborne; slow, low-symptom, crosses borders with
 *                   travellers (ignores closures) but is detected late (delays
 *                   the vaccine).
 */
export type VectorId = "respiratoire" | "vectorielle" | "hydrique" | "zoonotique" | "sexuelle";

export interface VectorInfo {
  id: VectorId;
  label: string;
  emoji: string;
  desc: string;
}

export const VECTORS: VectorInfo[] = [
  {
    id: "respiratoire",
    label: "Respiratoire",
    emoji: "🫁",
    desc: "Gouttelettes & aérosols. Favorisée par le froid (promiscuité hivernale).",
  },
  {
    id: "vectorielle",
    label: "Vectorielle",
    emoji: "🦟",
    desc: "Moustiques. Explose en climat chaud/humide, s'effondre au froid et en zone aride.",
  },
  {
    id: "hydrique",
    label: "Hydrique",
    emoji: "💧",
    desc: "Eau & voie féco-orale. Frappe fort là où l'assainissement est faible.",
  },
  {
    id: "zoonotique",
    label: "Zoonotique",
    emoji: "🦇",
    desc: "Réservoir animal. Déclenche de nouveaux foyers ruraux imprévisibles.",
  },
  {
    id: "sexuelle",
    label: "Sexuelle / sanguine",
    emoji: "🩸",
    desc: "Lente et discrète. Ignore les frontières mais détectée tard (vaccin retardé).",
  },
];

export interface VirusConfig {
  virusName: string;
  presetId: string;
  stats: VirusStats;
  vectors: VectorId[];
}

/** Total "genetic points" the player can spread across the three stats. */
export const STAT_BUDGET = 1.5;

export function statTotal(s: VirusStats): number {
  return s.contagiosite + s.letalite + s.resistance;
}
