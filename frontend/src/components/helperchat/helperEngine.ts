/**
 * helperEngine.ts — pure, city-level contagion model. Each of the ~1000 cities
 * carries its own infection level in [0,1]; the world map shows them as dots.
 * No React, no DOM — takes a state + virus stats and returns the next state.
 *
 * Spread channels, one tick = one day:
 *   - local logistic growth inside an infected city,
 *   - short-range diffusion to the few nearest cities (precomputed neighbours),
 *   - stochastic long-distance travel (planes/boats) to a far city, weighted by
 *     population — the only way isolated cities/islands light up.
 *
 * Wealth resistance: richer countries (high GDP/capita → high `wealth`) grow
 * slower and are harder to infect, modelling stronger healthcare/hygiene.
 */
import { CITIES } from "./helperCities";
import { COUNTRY_BY_ISO } from "./helperData";
import type { VirusStats } from "./helperTypes";

// --- tuning knobs -----------------------------------------------------------
const GROWTH_K = 0.35; // local infection acceleration
const NEIGHBOR_K = 0.22; // short-range city-to-city transmission
const AIR_RATE = 0.02; // per-infected-city chance of a long jump each day
const AIR_SEED = 0.04; // infection injected by a plane/boat arrival
const DEATH_K = 0.015; // fraction of the infected that die per day, ×létalité
const SPREAD_MIN = 0.1; // a city must be this infected before it exports
const RICH_RESIST = 0.8; // how strongly wealth dampens spread (0..1)
const K_NEIGH = 6; // nearest cities each city can infect overland
const MAX_DAYS = 800;
const END_COVERAGE = 0.9; // population-weighted coverage that ends the game
const STALL_DAYS = 60;

const N = CITIES.length;

// Per-city constants.
const POP = CITIES.map((c) => c.pop);
const TOTAL_POP = POP.reduce((s, p) => s + p, 0);
// Wealth → a spread multiplier ≤ 1 (rich cities let less through).
const PERMEABILITY = CITIES.map((c) => 1 - RICH_RESIST * c.wealth);
const LANDLOCKED = CITIES.map((c) => COUNTRY_BY_ISO[c.iso3]?.landlocked ?? false);

// Population-weighted table for random long-jump destinations.
const JUMP_WEIGHTS = POP.slice();
const JUMP_WEIGHT_TOTAL = JUMP_WEIGHTS.reduce((s, w) => s + w, 0);

// Precompute each city's K nearest neighbours (once, at module load). A plain
// O(N²) scan is too slow at N≈10k, so bin cities into a lat/lng grid and only
// compare against nearby cells, widening the ring until we have enough
// candidates.
const NEIGHBOURS: number[][] = (() => {
  const latR = CITIES.map((c) => (c.lat * Math.PI) / 180);
  const lngR = CITIES.map((c) => (c.lng * Math.PI) / 180);
  const cosLat = latR.map(Math.cos);

  const CELL = 4; // degrees per grid cell
  const key = (la: number, lo: number) =>
    Math.floor(la / CELL) * 100000 + Math.floor(lo / CELL);
  const grid = new Map<number, number[]>();
  for (let i = 0; i < N; i++) {
    const k = key(CITIES[i].lat, CITIES[i].lng);
    const bucket = grid.get(k);
    if (bucket) bucket.push(i);
    else grid.set(k, [i]);
  }

  const result: number[][] = new Array(N);
  for (let i = 0; i < N; i++) {
    const cla = Math.floor(CITIES[i].lat / CELL);
    const clo = Math.floor(CITIES[i].lng / CELL);
    const candidates: number[] = [];
    // Widen the ring until we have a healthy candidate pool (or give up).
    for (let ring = 1; ring <= 12 && candidates.length < K_NEIGH * 4; ring++) {
      candidates.length = 0;
      for (let da = -ring; da <= ring; da++) {
        for (let db = -ring; db <= ring; db++) {
          const bucket = grid.get((cla + da) * 100000 + (clo + db));
          if (bucket) for (const j of bucket) if (j !== i) candidates.push(j);
        }
      }
    }
    // Pick the K nearest among candidates.
    const best: { j: number; d: number }[] = [];
    for (const j of candidates) {
      const dx = (lngR[j] - lngR[i]) * cosLat[i];
      const dy = latR[j] - latR[i];
      const d = dx * dx + dy * dy;
      if (best.length < K_NEIGH) {
        best.push({ j, d });
        if (best.length === K_NEIGH) best.sort((a, b) => a.d - b.d);
      } else if (d < best[K_NEIGH - 1].d) {
        best[K_NEIGH - 1] = { j, d };
        best.sort((a, b) => a.d - b.d);
      }
    }
    result[i] = best.map((b) => b.j);
  }
  return result;
})();

export type JumpMode = "air" | "sea";

/** A long-distance seeding event this tick — drawn as a plane or a boat. */
export interface Jump {
  from: number; // city index
  to: number;
  mode: JumpMode;
}

export interface SimState {
  day: number;
  inf: Float32Array; // per-city infection [0,1]
  dead: Float32Array; // per-city cumulative dead fraction [0,1]
  finished: boolean;
  patientZero: number;
  coverage: number;
  stallCount: number;
  lastJumps: Jump[];
}

export interface SimMetrics {
  day: number;
  infectedPct: number; // 0..1 population-weighted
  infectedPeople: number;
  citiesTouched: number;
  totalCities: number;
  countriesTouched: number;
  deaths: number;
}

/** The most populous city in a country — used as the click-to-pick target. */
export function biggestCityIn(iso3: string): number {
  let best = -1;
  let bestPop = -1;
  for (let i = 0; i < N; i++) {
    if (CITIES[i].iso3 === iso3 && POP[i] > bestPop) {
      bestPop = POP[i];
      best = i;
    }
  }
  return best;
}

export function initSim(patientZero: number): SimState {
  const inf = new Float32Array(N);
  inf[patientZero] = 0.08;
  return {
    day: 0,
    inf,
    dead: new Float32Array(N),
    finished: false,
    patientZero,
    coverage: 0,
    stallCount: 0,
    lastJumps: [],
  };
}

function pickJumpTarget(): number {
  let r = Math.random() * JUMP_WEIGHT_TOTAL;
  for (let i = 0; i < N; i++) {
    r -= JUMP_WEIGHTS[i];
    if (r <= 0) return i;
  }
  return N - 1;
}

export function stepSim(state: SimState, stats: VirusStats): SimState {
  const { contagiosite, letalite } = stats;
  const prev = state.inf;
  const next = new Float32Array(prev); // copy

  // 1. Local logistic growth (rich cities grow slower).
  for (let i = 0; i < N; i++) {
    const v = prev[i];
    if (v > 0 && v < 1) {
      next[i] = Math.min(1, v + contagiosite * GROWTH_K * PERMEABILITY[i] * v * (1 - v));
    }
  }

  // 2. Short-range diffusion to nearest neighbours (from pre-step levels).
  for (let i = 0; i < N; i++) {
    const v = prev[i];
    if (v < SPREAD_MIN) continue;
    const nb = NEIGHBOURS[i];
    for (let k = 0; k < nb.length; k++) {
      const j = nb[k];
      const cur = next[j];
      const transfer = v * contagiosite * NEIGHBOR_K * PERMEABILITY[j] * (1 - cur);
      if (transfer > 0) next[j] = Math.min(1, cur + transfer);
    }
  }

  // 3. Long-distance travel (planes/boats).
  const jumps: Jump[] = [];
  for (let i = 0; i < N; i++) {
    if (prev[i] < SPREAD_MIN) continue;
    if (Math.random() < contagiosite * AIR_RATE) {
      const dest = pickJumpTarget();
      if (dest === i) continue;
      next[dest] = Math.max(next[dest], AIR_SEED * PERMEABILITY[dest]);
      const canSea = !LANDLOCKED[i] && !LANDLOCKED[dest];
      const mode: JumpMode = canSea && Math.random() < 0.4 ? "sea" : "air";
      jumps.push({ from: i, to: dest, mode });
    }
  }

  // 4. Deaths accumulate, capped by the infected share.
  const dead = new Float32Array(state.dead);
  for (let i = 0; i < N; i++) {
    if (next[i] > 0) dead[i] = Math.min(next[i], dead[i] + next[i] * letalite * DEATH_K);
  }

  const day = state.day + 1;

  // Population-weighted coverage for the end condition.
  let popInf = 0;
  for (let i = 0; i < N; i++) popInf += next[i] * POP[i];
  const coverage = popInf / TOTAL_POP;
  const stallCount = coverage - state.coverage < 0.001 ? state.stallCount + 1 : 0;
  const finished = coverage >= END_COVERAGE || stallCount >= STALL_DAYS || day >= MAX_DAYS;

  return { day, inf: next, dead, finished, patientZero: state.patientZero, coverage, stallCount, lastJumps: jumps };
}

export function metrics(state: SimState): SimMetrics {
  let infectedPeople = 0;
  let deaths = 0;
  let citiesTouched = 0;
  const countries = new Set<string>();
  for (let i = 0; i < N; i++) {
    const v = state.inf[i];
    infectedPeople += v * POP[i];
    deaths += state.dead[i] * POP[i];
    if (v > 0.05) {
      citiesTouched++;
      countries.add(CITIES[i].iso3);
    }
  }
  return {
    day: state.day,
    infectedPct: infectedPeople / TOTAL_POP,
    infectedPeople,
    citiesTouched,
    totalCities: N,
    countriesTouched: countries.size,
    deaths,
  };
}
