/**
 * helperEngine.ts — pure, city-level SEIR contagion model. Each of the ~1800
 * cities carries S/E/I/R compartments (Susceptible, Exposed/latent, Infectious,
 * Recovered-immune). No React, no DOM — takes a state + virus stats and returns
 * the next state.
 *
 * Epidemiology (one tick = one day):
 *   - within a city: standard SEIR flow S→E→I→R with force of infection β·I,
 *     where β = R₀·γ and R₀ is derived from the virus stats, the transmission
 *     vectors, the local biome and the season;
 *   - herd immunity emerges naturally: as R grows, S shrinks and the effective
 *     reproduction number Rt = R₀·S falls below 1, ending the wave;
 *   - between cities: overland diffusion to nearest neighbours and stochastic
 *     long-distance travel (planes/boats) import *Exposed* cases elsewhere.
 *
 * Wealth resistance: richer countries (high GDP/capita → high `wealth`) transmit
 * less (healthcare/hygiene). Seasonality: respiratory spread rises in each
 * hemisphere's winter (six months out of phase).
 */
import { CITIES } from "./helperCities";
import { COUNTRY_BY_ISO } from "./helperData";
import { UNIVERSITIES } from "./helperUniversities";
import type { VirusStats, VectorId } from "./helperTypes";

// --- SEIR epidemiology ------------------------------------------------------
const SIGMA = 1 / 4; // 1 / latent period (≈4 days incubation before infectious)
const GAMMA = 1 / 6; // 1 / infectious period (≈6 days shedding)
const R0_MIN = 0.8; // R₀ at contagiosité 0
const R0_MAX = 5.5; // R₀ at contagiosité 1 (before biome/vector/season)
const SEASON_AMP = 0.35; // ± amplitude of the respiratory seasonal swing
const IMPORT_E = 0.02; // Exposed fraction injected by a long-distance import
const NEIGHBOR_K = 0.12; // overland coupling strength between neighbour cities
const AIR_RATE = 0.02; // per-infectious-city chance of a long jump each day
const DEATH_K = 0.5; // share of those leaving I who die = létalité × this (CFR cap)
const SPREAD_MIN = 0.02; // min infectious fraction before a city can export
const RICH_RESIST = 0.8; // how strongly wealth dampens spread (0..1)
const K_NEIGH = 6; // nearest cities each city can infect overland
const MAX_DAYS = 900;
const END_COVERAGE = 0.9; // ever-infected population fraction that ends the game
const STALL_DAYS = 60;

// --- vaccine race (Plague Inc style) ----------------------------------------
const VAC_RATE = 0.012; // base cure-research speed per day at full capacity
const VAC_RESIST = 0.85; // how much the "résistance" stat slows the vaccine
const CURE_RATE = 0.06; // how fast deployed vaccine pushes infection back down
const UNI_WEIGHT = 0.6; // share of research that comes from universities
const DISCOVERY_CITIES = 3; // active-outbreak cities before the world reacts

// --- random events (spice) --------------------------------------------------
const EVENT_CHANCE = 0.03; // per-day chance of a random gameplay event

// --- border closures --------------------------------------------------------
const CLOSE_COVERAGE = 0.03; // world coverage that triggers countries to react
const CLOSE_BASE = 0.02; // base per-day closing probability (×wealth-weighted)
const CLOSE_LEAK = 0.15; // base fraction of plane/boat jumps that slip through
const CLOSE_LAND = 0.35; // remaining overland transmission into closed countries
const SCREEN_STRENGTH = 0.8; // rich closed countries screen hubs → leak far less
const REOPEN_VACCINE = 0.7; // vaccine progress that lets countries reopen
const REOPEN_ACTIVE = 0.005; // local active infection under which a swept country reopens
const REOPEN_EVER = 0.5; // ...but only once it has already been through a wave
const TRACE_WEALTH = 0.6; // a wealthy infected city detects the pathogen at once

// --- transmission vectors ---------------------------------------------------
const ZOON_RATE = 0.06; // per-day chance of an animal-reservoir spillover
const SEX_RATE = 0.12; // per-day chance of a sexual/bloodborne cross-link
const SEX_VACCINE_DELAY = 0.7; // stealthy → research runs at 70% while active

const N = CITIES.length;

// --- biomes: harsh climates are harder to infect (Plague Inc environments) --
export type Biome = "Polaire" | "Froid" | "Tempéré" | "Chaud" | "Tropical" | "Aride";

const LANDLOCKED = CITIES.map((c) => COUNTRY_BY_ISO[c.iso3]?.landlocked ?? false);

function classifyBiome(lat: number, landlocked: boolean): Biome {
  const a = Math.abs(lat);
  if (a >= 60) return "Polaire"; // very cold
  if (a >= 50) return "Froid"; // cold
  if (a >= 15 && a <= 33 && landlocked) return "Aride"; // dry continental interiors
  if (a < 15) return "Tropical"; // hot & humid
  if (a < 23.5) return "Chaud"; // hot
  return "Tempéré"; // easiest
}

// Climate multiplier on transmission (1 = easy temperate; lower = harder).
const BIOME_FACTOR: Record<Biome, number> = {
  Tempéré: 1.0,
  Chaud: 0.75,
  Tropical: 0.6,
  Aride: 0.6,
  Froid: 0.65,
  Polaire: 0.4,
};

/** Per-city biome, exported for the map tooltip. */
export const CITY_BIOME: Biome[] = CITIES.map((c, i) => classifyBiome(c.lat, LANDLOCKED[i]));

// Per-city constants.
const POP = CITIES.map((c) => c.pop);
const TOTAL_POP = POP.reduce((s, p) => s + p, 0);
// Spread multiplier ≤ 1, combining wealth (healthcare) and climate (biome).
const PERMEABILITY = CITIES.map(
  (c, i) => (1 - RICH_RESIST * c.wealth) * BIOME_FACTOR[CITY_BIOME[i]],
);
// Research capacity of each city: wealthy, populous places fund the cure.
const RESEARCH = CITIES.map((c) => c.wealth * c.pop);
const RESEARCH_TOTAL = RESEARCH.reduce((s, r) => s + r, 0);

// Universities: each is tied to its nearest city (its staff get sick when that
// city is infected, cutting its research output).
const UNI_POWER = UNIVERSITIES.map((u) => u.power);
const UNI_POWER_TOTAL = UNI_POWER.reduce((s, p) => s + p, 0);
const UNI_NEAR_CITY = UNIVERSITIES.map((u) => {
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < CITIES.length; i++) {
    const dx = CITIES[i].lng - u.lng;
    const dy = CITIES[i].lat - u.lat;
    const d = dx * dx + dy * dy;
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
});

// Country mapping for border closures: each city → a country bucket, with the
// country's (representative) wealth used to decide how fast it seals borders.
const ISO_LIST: string[] = [];
const ISO_WEALTH: number[] = [];
const CITY_CIDX = new Int16Array(CITIES.length);
{
  const idx = new Map<string, number>();
  CITIES.forEach((c, i) => {
    let ci = idx.get(c.iso3);
    if (ci === undefined) {
      ci = ISO_LIST.length;
      idx.set(c.iso3, ci);
      ISO_LIST.push(c.iso3);
      ISO_WEALTH.push(c.wealth);
    }
    CITY_CIDX[i] = ci;
  });
}
const N_COUNTRY = ISO_LIST.length;
// Total urban population per country (for judging local threat).
const ISO_POP = new Float64Array(N_COUNTRY);
for (let i = 0; i < CITIES.length; i++) ISO_POP[CITY_CIDX[i]] += CITIES[i].pop;

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

/** Why the game ended. "infected" is the only virus victory. */
export type EndReason = "infected" | "vaccine" | "stall" | "timeout";

/** A long-distance seeding event this tick — drawn as a plane or a boat. */
export interface Jump {
  from: number; // city index
  to: number;
  mode: JumpMode;
}

export interface SimState {
  day: number;
  // SEIR compartments per city, each a fraction of that city's population.
  S: Float32Array; // susceptible
  E: Float32Array; // exposed / latent (infected, not yet infectious)
  I: Float32Array; // infectious
  R: Float32Array; // recovered & immune
  dead: Float32Array; // per-city cumulative dead fraction [0,1]
  finished: boolean;
  patientZero: number;
  coverage: number; // ever-infected population fraction (1 − S), pop-weighted
  stallCount: number;
  lastJumps: Jump[];
  vaccine: number; // global cure research progress [0,1]
  mod: number; // transient contagion multiplier from a random event
  modDays: number; // days left on `mod`
  lastEvent: string | null; // a random event fired this tick (for the ticker)
  closed: Uint8Array; // per-country border-closure flags
  closedCount: number; // how many countries have shut their borders
  r0: number; // representative basic reproduction number this tick
  rt: number; // effective reproduction number (R₀ × mean susceptibility)
  endReason: EndReason | null; // why the game finished (null while running)
}

/** Enabled transmission routes for the current game. */
export type VectorSet = Record<VectorId, boolean>;

export function toVectorSet(list: VectorId[]): VectorSet {
  return {
    respiratoire: list.includes("respiratoire"),
    vectorielle: list.includes("vectorielle"),
    hydrique: list.includes("hydrique"),
    zoonotique: list.includes("zoonotique"),
    sexuelle: list.includes("sexuelle"),
  };
}

export interface SimMetrics {
  day: number;
  infectedPct: number; // 0..1 ever-infected, population-weighted
  infectedPeople: number; // cumulative ever-infected people
  activePeople: number; // currently infectious people
  citiesTouched: number;
  totalCities: number;
  countriesTouched: number;
  deaths: number;
  deathsPct: number; // 0..1 of world urban population
  vaccine: number; // 0..1
  vaccineDeployed: boolean;
  closedCountries: number;
  r0: number;
  rt: number;
  endReason: EndReason | null;
  won: boolean; // virus victory = the world was infected before anything stopped it
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
  const S = new Float32Array(N).fill(1);
  const E = new Float32Array(N);
  const I = new Float32Array(N);
  const R = new Float32Array(N);
  // Patient zero: a small seed of infectious individuals.
  I[patientZero] = 0.02;
  S[patientZero] = 0.98;
  return {
    day: 0,
    S,
    E,
    I,
    R,
    dead: new Float32Array(N),
    finished: false,
    patientZero,
    coverage: 0,
    stallCount: 0,
    lastJumps: [],
    vaccine: 0,
    mod: 1,
    modDays: 0,
    lastEvent: null,
    closed: new Uint8Array(N_COUNTRY),
    closedCount: 0,
    r0: 0,
    rt: 0,
    endReason: null,
  };
}

/**
 * Seasonal transmission multiplier for a respiratory pathogen: peaks in each
 * hemisphere's winter, six months out of phase. `day` is days since start.
 */
function seasonFactor(day: number, lat: number, vec: VectorSet): number {
  if (!vec.respiratoire) return 1;
  // Northern winter ≈ day 0 (Jan-ish); southern winter is +182 days.
  const phase = lat >= 0 ? 0 : Math.PI;
  const season = Math.cos((2 * Math.PI * day) / 365 + phase); // +1 = deep winter
  return 1 + SEASON_AMP * season * Math.min(1, Math.abs(lat) / 40); // tropics: flat
}

/** Per-city climate/wealth multiplier for the enabled transmission vectors. */
function vectorMultiplier(i: number, vec: VectorSet): number {
  let m = 1;
  const b = CITY_BIOME[i];
  const w = CITIES[i].wealth;
  if (vec.respiratoire && (b === "Froid" || b === "Polaire" || b === "Tempéré")) {
    m *= 1.2; // cold-season indoor crowding
  }
  if (vec.vectorielle) {
    // mosquito-borne: loves hot & humid, dies off in cold/arid
    m *=
      b === "Tropical" || b === "Chaud"
        ? 1.9
        : b === "Froid" || b === "Polaire"
          ? 0.25
          : b === "Aride"
            ? 0.5
            : 0.85;
  }
  if (vec.hydrique) m *= 1 + (1 - w) * 0.9; // poor sanitation → faster
  return m;
}

function pickJumpTarget(): number {
  let r = Math.random() * JUMP_WEIGHT_TOTAL;
  for (let i = 0; i < N; i++) {
    r -= JUMP_WEIGHTS[i];
    if (r <= 0) return i;
  }
  return N - 1;
}

export function stepSim(state: SimState, stats: VirusStats, vec: VectorSet): SimState {
  const { contagiosite, letalite, resistance } = stats;
  const { S, E, I, R } = state;
  const nS = new Float32Array(S);
  const nE = new Float32Array(E);
  const nI = new Float32Array(I);
  const nR = new Float32Array(R);
  const dead = new Float32Array(state.dead);
  const day = state.day + 1;
  const vac = state.vaccine;

  // Intrinsic R₀ from contagiosité, then dampened by an active event modifier
  // and by the deployed vaccine.
  const r0Intrinsic = R0_MIN + (R0_MAX - R0_MIN) * contagiosite;
  const r0base = r0Intrinsic * state.mod * (1 - vac);
  const cfr = letalite * DEATH_K; // case-fatality among those leaving I

  // 0. Per-country active & ever-infected (pre-step), for border decisions.
  const cActive = new Float64Array(N_COUNTRY);
  const cEver = new Float64Array(N_COUNTRY);
  for (let i = 0; i < N; i++) {
    const c = CITY_CIDX[i];
    cActive[c] += I[i] * POP[i];
    cEver[c] += (1 - S[i]) * POP[i];
  }

  // Border closures AND reopenings. Richer states seal first once the pandemic
  // is recognised; they reopen once a vaccine is coming or their own wave has
  // passed (high ever-infected, low active).
  const closed = new Uint8Array(state.closed);
  let closedCount = state.closedCount;
  let closureEvent: string | null = null;
  let reopenEvent: string | null = null;
  if (state.coverage >= CLOSE_COVERAGE) {
    for (let c = 0; c < N_COUNTRY; c++) {
      const active = cActive[c] / ISO_POP[c];
      const ever = cEver[c] / ISO_POP[c];
      if (!closed[c]) {
        if (Math.random() < CLOSE_BASE * (0.3 + 0.7 * ISO_WEALTH[c])) {
          closed[c] = 1;
          closedCount++;
        }
      } else if (vac >= REOPEN_VACCINE || (ever > REOPEN_EVER && active < REOPEN_ACTIVE)) {
        closed[c] = 0;
        closedCount--;
      }
    }
    if (state.closedCount < 20 && closedCount >= 20) {
      closureEvent = "🛂 De nombreux pays ferment leurs frontières";
    }
    if (state.closedCount >= 20 && closedCount < 10) {
      reopenEvent = "🟢 Le monde rouvre progressivement ses frontières";
    }
  }

  // 1. Within-city SEIR flow (explicit Euler, one day). β = R₀·γ, modulated by
  //    wealth (healthcare), biome+vector climate fit, and season.
  for (let i = 0; i < N; i++) {
    const s = S[i];
    const e = E[i];
    const inf = I[i];
    if (s <= 0 && e <= 0 && inf <= 0) continue;
    const beta =
      r0base *
      GAMMA *
      PERMEABILITY[i] *
      vectorMultiplier(i, vec) *
      seasonFactor(day, CITIES[i].lat, vec);
    let newE = beta * s * inf; // S → E (force of infection)
    if (newE > s) newE = s;
    const prog = SIGMA * e; // E → I
    const rec = GAMMA * inf; // I → R (+ deaths)
    const vax = vac > 0 ? Math.min(s - newE, vac * CURE_RATE * s) : 0; // immunisation
    nS[i] = s - newE - vax;
    nE[i] = e + newE - prog;
    nI[i] = inf + prog - rec;
    nR[i] = R[i] + rec * (1 - cfr) + vax;
    dead[i] = dead[i] + rec * cfr;
  }

  // 2. Overland diffusion: infectious cities expose their nearest neighbours
  //    (contact persists across closed borders, just throttled).
  for (let i = 0; i < N; i++) {
    if (I[i] < SPREAD_MIN) continue;
    const nb = NEIGHBOURS[i];
    for (let k = 0; k < nb.length; k++) {
      const j = nb[k];
      const border = closed[CITY_CIDX[j]] ? CLOSE_LAND : 1;
      const imp =
        NEIGHBOR_K *
        I[i] *
        r0base *
        GAMMA *
        PERMEABILITY[j] *
        vectorMultiplier(j, vec) *
        border;
      const move = Math.min(nS[j], imp);
      if (move > 0) {
        nS[j] -= move;
        nE[j] += move;
      }
    }
  }

  // Helper: import Exposed cases into a city, drawing from its susceptibles.
  const seedExposed = (j: number, amount: number) => {
    const move = Math.min(nS[j], amount);
    if (move > 0) {
      nS[j] -= move;
      nE[j] += move;
    }
  };

  // 3. Long-distance travel (planes/boats). Closed destinations mostly refuse
  //    arrivals — only CLOSE_LEAK slips through, and no vehicle is drawn.
  const jumps: Jump[] = [];
  for (let i = 0; i < N; i++) {
    if (I[i] < SPREAD_MIN) continue;
    if (Math.random() < r0base * GAMMA * AIR_RATE) {
      const dest = pickJumpTarget();
      if (dest === i) continue;
      const destC = CITY_CIDX[dest];
      const destClosed = closed[destC];
      // Screening at hubs: wealthier closed countries let far fewer travellers
      // slip past quarantine.
      const leak = CLOSE_LEAK * (1 - ISO_WEALTH[destC] * SCREEN_STRENGTH);
      if (destClosed && Math.random() > leak) continue; // border holds
      seedExposed(dest, IMPORT_E);
      if (!destClosed) {
        const canSea = !LANDLOCKED[i] && !LANDLOCKED[dest];
        const mode: JumpMode = canSea && Math.random() < 0.4 ? "sea" : "air";
        jumps.push({ from: i, to: dest, mode });
      }
    }
  }

  // 3b. Zoonotic spillover: an animal reservoir sparks a fresh outbreak in a
  //     random (rural/low-wealth-leaning) city, independent of current spread.
  if (vec.zoonotique && Math.random() < ZOON_RATE) {
    let dest = Math.floor(Math.random() * N);
    const alt = Math.floor(Math.random() * N);
    if (CITIES[alt].wealth < CITIES[dest].wealth) dest = alt;
    seedExposed(dest, 0.03);
  }

  // 3c. Sexual / bloodborne channel: slow link that ignores borders/climate.
  if (vec.sexuelle && Math.random() < SEX_RATE) {
    let hasInfected = false;
    for (let t = 0; t < 6; t++) {
      if (I[Math.floor(Math.random() * N)] > SPREAD_MIN) {
        hasInfected = true;
        break;
      }
    }
    if (hasInfected) seedExposed(Math.floor(Math.random() * N), 0.03);
  }

  // 4. Vaccine research: infected wealthy cities + universities, gated by spread,
  //    slowed by résistance and by stealthy (sexual) strains.
  let cityResearch = 0;
  let infectedCount = 0;
  let richDetected = false; // tracing: a wealthy country spots it immediately
  for (let i = 0; i < N; i++) {
    if (I[i] > 0.02) {
      cityResearch += RESEARCH[i];
      infectedCount++;
      if (CITIES[i].wealth >= TRACE_WEALTH) richDetected = true;
    }
  }
  let uniResearch = 0;
  for (let u = 0; u < UNI_POWER.length; u++) {
    uniResearch += UNI_POWER[u] * (1 - I[UNI_NEAR_CITY[u]]); // sick staff → less
  }
  const outbreak = Math.min(1, infectedCount / 40);
  const researchFactor = Math.min(
    1,
    (1 - UNI_WEIGHT) * (cityResearch / RESEARCH_TOTAL) +
      UNI_WEIGHT * (uniResearch / UNI_POWER_TOTAL),
  );
  const stealth = vec.sexuelle ? SEX_VACCINE_DELAY : 1;
  // Research begins once the pathogen is identified — established in a few
  // cities, OR spotted at once by a well-resourced (wealthy) country's tracing.
  const discovered = state.vaccine > 0 || richDetected || infectedCount >= DISCOVERY_CITIES;
  const vacGain = discovered
    ? VAC_RATE * researchFactor * outbreak * stealth * (1 - resistance * VAC_RESIST)
    : 0;
  const vaccine = Math.min(1, state.vaccine + Math.max(0, vacGain));

  // 5. Random events. Decrement any active modifier first.
  let mod = state.modDays > 0 ? state.mod : 1;
  let modDays = Math.max(0, state.modDays - 1);
  let lastEvent: string | null = null;
  let vaccineAfterEvent = vaccine;
  if (Math.random() < EVENT_CHANCE && infectedCount > 3) {
    const roll = Math.random();
    if (roll < 0.28) {
      mod = 1.7;
      modDays = 18;
      lastEvent = "🧬 Mutation : le virus devient bien plus contagieux !";
    } else if (roll < 0.5) {
      mod = 0.45;
      modDays = 18;
      lastEvent = "🔒 Confinement mondial : la propagation ralentit";
    } else if (roll < 0.68) {
      const dest = pickJumpTarget();
      seedExposed(dest, 0.2);
      lastEvent = `🎉 Rassemblement de masse à ${CITIES[dest].name} : flambée soudaine !`;
    } else if (roll < 0.84) {
      vaccineAfterEvent = Math.min(1, vaccine + 0.1);
      lastEvent = "🧪 Percée en laboratoire : le vaccin accélère";
    } else {
      vaccineAfterEvent = Math.max(0, vaccine - 0.12);
      lastEvent = "🧬 Variant d'échappement : le vaccin perd du terrain";
    }
  }
  if (!lastEvent && closureEvent) lastEvent = closureEvent;
  if (!lastEvent && reopenEvent) lastEvent = reopenEvent;

  // 6. Aggregates: ever-infected coverage (1−S), susceptibility, Rt.
  let everInf = 0;
  let susceptible = 0;
  for (let i = 0; i < N; i++) {
    everInf += (1 - nS[i]) * POP[i];
    susceptible += nS[i] * POP[i];
  }
  const coverage = everInf / TOTAL_POP;
  const meanS = susceptible / TOTAL_POP;
  const rt = r0base * meanS;

  const stallCount = Math.abs(coverage - state.coverage) < 0.0005 ? state.stallCount + 1 : 0;
  // End reason (priority: infecting the world wins even if the vaccine lands the
  // same day).
  let endReason: EndReason | null = null;
  if (coverage >= END_COVERAGE) endReason = "infected";
  else if (vaccineAfterEvent >= 1) endReason = "vaccine";
  else if (stallCount >= STALL_DAYS) endReason = "stall";
  else if (day >= MAX_DAYS) endReason = "timeout";
  const finished = endReason !== null;

  return {
    endReason,
    day,
    S: nS,
    E: nE,
    I: nI,
    R: nR,
    dead,
    finished,
    patientZero: state.patientZero,
    coverage,
    stallCount,
    lastJumps: jumps,
    vaccine: vaccineAfterEvent,
    mod,
    modDays,
    lastEvent,
    closed,
    closedCount,
    r0: r0Intrinsic,
    rt,
  };
}

export function metrics(state: SimState): SimMetrics {
  let infectedPeople = 0; // ever infected (1 − S)
  let activePeople = 0; // currently infectious (I)
  let deaths = 0;
  let citiesTouched = 0;
  const countries = new Set<string>();
  for (let i = 0; i < N; i++) {
    const ever = 1 - state.S[i];
    infectedPeople += ever * POP[i];
    activePeople += state.I[i] * POP[i];
    deaths += state.dead[i] * POP[i];
    if (ever > 0.05) {
      citiesTouched++;
      countries.add(CITIES[i].iso3);
    }
  }
  return {
    day: state.day,
    infectedPct: infectedPeople / TOTAL_POP,
    infectedPeople,
    activePeople,
    citiesTouched,
    totalCities: N,
    countriesTouched: countries.size,
    deaths,
    deathsPct: deaths / TOTAL_POP,
    vaccine: state.vaccine,
    vaccineDeployed: state.vaccine >= 1,
    closedCountries: state.closedCount,
    r0: state.r0,
    rt: state.rt,
    endReason: state.endReason,
    won: state.endReason === "infected",
  };
}
