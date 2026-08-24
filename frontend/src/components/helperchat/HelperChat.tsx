/**
 * HelperChat — lightweight in-app assistant overlay mounted from the composer.
 * Opened by a specific command typed in the chat box. Self-contained: nothing
 * here touches the conversation state or the backend chat pipeline.
 *
 * (Internally this drives a small contagion simulation the user can play with;
 * kept intentionally isolated so it can be removed without touching Chat.tsx
 * beyond a single mount point.)
 *
 * Flow: setup (choose virus) → pick (click patient zero) → running (animated
 * spread) → end (final tally).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { X, Pause, Play, RotateCcw } from "lucide-react";
import HelperMap from "./HelperMap";
import HelperSetup from "./HelperSetup";
import HelperVehicles, { type VehicleDot } from "./HelperVehicles";
import {
  initSim,
  stepSim,
  metrics,
  toVectorSet,
  type SimState,
  type Jump,
  type VectorSet,
} from "./helperEngine";
import { CITIES } from "./helperCities";
import { COUNTRY_BY_ISO } from "./helperData";
import type { VirusConfig } from "./helperTypes";

const ZERO_INF: number[] = CITIES.map(() => 0);

// Per-country aggregation for the news ticker: map each city to a country
// bucket, and track each country's total urban population.
const COUNTRY_LIST: { iso3: string; nameFr: string; pop: number }[] = [];
const CITY_COUNTRY = new Int16Array(CITIES.length);
{
  const idxByIso = new Map<string, number>();
  CITIES.forEach((c, i) => {
    let ci = idxByIso.get(c.iso3);
    if (ci === undefined) {
      ci = COUNTRY_LIST.length;
      idxByIso.set(c.iso3, ci);
      COUNTRY_LIST.push({ iso3: c.iso3, nameFr: COUNTRY_BY_ISO[c.iso3]?.nameFr ?? c.iso3, pop: 0 });
    }
    CITY_COUNTRY[i] = ci;
    COUNTRY_LIST[ci].pop += c.pop;
  });
}
// Lyon gets a bespoke message (SHAPE-Med@Lyon in-joke).
const LYON_IDX = CITIES.findIndex((c) => c.iso3 === "FRA" && c.name === "Lyon");
import { api } from "@/lib/api";

interface ScoreRow {
  name: string;
  virus: string;
  days: number;
  infected_pct: number;
  dead: number;
  score: number;
}

interface ScoreBreakdown {
  coverage_pts: number;
  mortality_pts: number;
  speed_pts: number;
  vaccine_penalty: boolean;
  total: number;
}

interface ScoreOut {
  score: number;
  breakdown: ScoreBreakdown;
  leaderboard: ScoreRow[];
}

type Phase = "setup" | "pick" | "running" | "end";

const TICK_MS = 650; // simulated-day cadence — slow enough to watch spread
const MAP_H = 480;
const MAX_VEHICLES = 40; // cap so the overlay never clutters
const AIR_DUR = 1500; // ms a plane spends crossing
const SEA_DUR = 2600; // boats are slower

interface Vehicle {
  id: number;
  from: [number, number]; // [lat, lng]
  to: [number, number];
  mode: "air" | "sea";
  start: number; // performance.now() at spawn
  dur: number;
}

let vehicleSeq = 0;
let eventSeq = 0;
const COUNTRY_TH = 0.3; // share of a country's urban pop infected → announce it
const LYON_TH = 0.4; // infection level at which the Lyon easter egg fires
const FLASH_CHANCE = 0.06; // per-tick chance of a random flavour headline

// Varied phrasings for a country falling (suffix form keeps French genders safe).
const COUNTRY_TEMPLATES: ((n: string) => string)[] = [
  (n) => `Nouveau foyer : ${n}`,
  (n) => `Alerte rouge : ${n}`,
  (n) => `État d'urgence : ${n}`,
  (n) => `Frontières fermées : ${n}`,
  (n) => `Hôpitaux saturés : ${n}`,
  (n) => `Confinement décrété : ${n}`,
  (n) => `Couvre-feu imposé : ${n}`,
  (n) => `Panique générale : ${n}`,
  (n) => `Premiers décès : ${n}`,
  (n) => `Écoles fermées : ${n}`,
];

// Random flavour headlines sprinkled in for atmosphere.
const FLASHES: string[] = [
  "😷 Les masques en rupture de stock",
  "🧻 Ruée sur le papier toilette",
  "🕵️ Les théories du complot explosent",
  "💻 Le télétravail devient la norme",
  "📉 Les bourses mondiales dévissent",
  "🏪 Les supermarchés dévalisés",
  "🧴 Pénurie de gel hydroalcoolique",
  "📺 Les chaînes d'info en continu s'affolent",
  "🙈 Les autorités appellent au calme",
  "🧪 Un laboratoire annonce une piste de vaccin",
  "🚑 Les services d'urgence débordés",
  "🍺 Bars et restaurants ferment leurs portes",
  "🐒 On accuse une chauve-souris",
  "🛌 Le pays entier se met au lit",
  "📊 Les modélisateurs revoient leurs courbes",
];

// One-shot milestone headlines based on the running totals.
const MILESTONES: { key: string; test: (people: number, deaths: number, countries: number) => boolean; text: string }[] = [
  { key: "who", test: (p) => p >= 1e6, text: "🚨 L'OMS déclare une urgence sanitaire mondiale" },
  { key: "p100m", test: (p) => p >= 1e8, text: "🌐 Plus de 100 millions de contaminés" },
  { key: "p1md", test: (p) => p >= 1e9, text: "🌍 Un milliard de personnes touchées" },
  { key: "p4md", test: (p) => p >= 4e9, text: "😱 La moitié de l'humanité est infectée" },
  { key: "d1m", test: (_p, d) => d >= 1e6, text: "⚰️ Le bilan dépasse le million de morts" },
  { key: "d100m", test: (_p, d) => d >= 1e8, text: "🕯️ Plus de 100 millions de victimes" },
  { key: "c50", test: (_p, _d, c) => c >= 50, text: "✈️ Vols internationaux suspendus" },
  { key: "c120", test: (_p, _d, c) => c >= 120, text: "🌎 La pandémie touche le monde entier" },
];

function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

interface FeedEvent {
  id: number;
  text: string;
}

function easeInOut(p: number): number {
  return p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
}

const D2R = Math.PI / 180;
const R2D = 180 / Math.PI;

/**
 * Great-circle interpolation between two [lat,lng] points — the real path a
 * plane or ship follows on the globe, drawn as a curve on the flat map. Uses
 * spherical linear interpolation of the two points' unit vectors.
 */
function greatCircle(
  from: [number, number],
  to: [number, number],
  f: number,
): { lat: number; lng: number } {
  const la1 = from[0] * D2R;
  const lo1 = from[1] * D2R;
  const la2 = to[0] * D2R;
  const lo2 = to[1] * D2R;
  const x1 = Math.cos(la1) * Math.cos(lo1);
  const y1 = Math.cos(la1) * Math.sin(lo1);
  const z1 = Math.sin(la1);
  const x2 = Math.cos(la2) * Math.cos(lo2);
  const y2 = Math.cos(la2) * Math.sin(lo2);
  const z2 = Math.sin(la2);
  let dot = x1 * x2 + y1 * y2 + z1 * z2;
  dot = Math.max(-1, Math.min(1, dot));
  const omega = Math.acos(dot);
  if (omega < 1e-6) return { lat: from[0], lng: from[1] };
  const s = Math.sin(omega);
  const a = Math.sin((1 - f) * omega) / s;
  const b = Math.sin(f * omega) / s;
  const x = a * x1 + b * x2;
  const y = a * y1 + b * y2;
  const z = a * z1 + b * z2;
  return { lat: Math.atan2(z, Math.hypot(x, y)) * R2D, lng: Math.atan2(y, x) * R2D };
}

function fmt(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + " Md";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + " M";
  if (n >= 1e3) return (n / 1e3).toFixed(0) + " k";
  return Math.round(n).toString();
}

export default function HelperChat({ onClose }: { onClose: () => void }) {
  const [phase, setPhase] = useState<Phase>("setup");
  const [config, setConfig] = useState<VirusConfig | null>(null);
  const [sim, setSim] = useState<SimState | null>(null);
  const [paused, setPaused] = useState(false);
  const [vehicles, setVehicles] = useState<Vehicle[]>([]);
  const [, setFrame] = useState(0); // drives the per-frame re-render
  // End screen: the recap card can be dismissed to inspect the final map.
  const [endCardOpen, setEndCardOpen] = useState(true);
  // Bottom news ticker: notable cities announced as they get contaminated.
  const [feed, setFeed] = useState<FeedEvent[]>([]);

  // End-of-game scoring.
  const [myScore, setMyScore] = useState<number | null>(null);
  const [breakdown, setBreakdown] = useState<ScoreBreakdown | null>(null);
  const [leaderboard, setLeaderboard] = useState<ScoreRow[] | null>(null);
  const [scoreError, setScoreError] = useState(false);
  const submittedRef = useRef(false);

  // Which countries have already been announced (+ Lyon flag + fired milestones
  // + last flash to avoid repeats), per game.
  const announcedRef = useRef<Set<number>>(new Set());
  const lyonFiredRef = useRef(false);
  const milestonesRef = useRef<Set<string>>(new Set());
  const lastFlashRef = useRef<string>("");

  // Latest sim in a ref so the tick loop can read it without re-subscribing.
  const simRef = useRef<SimState | null>(null);
  useEffect(() => {
    simRef.current = sim;
  }, [sim]);

  function spawnVehicles(jumps: Jump[]) {
    if (!jumps.length) return;
    const now = performance.now();
    const born = jumps
      .map((j): Vehicle | null => {
        const a = CITIES[j.from];
        const b = CITIES[j.to];
        if (!a || !b) return null;
        return {
          id: vehicleSeq++,
          from: [a.lat, a.lng],
          to: [b.lat, b.lng],
          mode: j.mode,
          start: now,
          dur: j.mode === "sea" ? SEA_DUR : AIR_DUR,
        };
      })
      .filter(Boolean) as Vehicle[];
    setVehicles((vs) => [...vs, ...born].slice(-MAX_VEHICLES));
  }

  // Tick loop: advance the simulation while running and not paused.
  useEffect(() => {
    if (phase !== "running" || paused || !config) return;
    const vecSet: VectorSet = toVectorSet(config.vectors);
    const id = setInterval(() => {
      const prev = simRef.current;
      if (!prev || prev.finished) return;
      const next = stepSim(prev, config.stats, vecSet);
      setSim(next);
      spawnVehicles(next.lastJumps);

      // One O(N) pass: per-country infected pop + global totals.
      const fresh: FeedEvent[] = [];
      const cInf = new Float64Array(COUNTRY_LIST.length);
      let people = 0;
      let deaths = 0;
      for (let i = 0; i < next.S.length; i++) {
        const p = CITIES[i].pop;
        const ever = 1 - next.S[i]; // ever-infected fraction
        cInf[CITY_COUNTRY[i]] += ever * p;
        people += ever * p;
        deaths += next.dead[i] * p;
      }

      // Countries crossing the threshold, with a varied headline each.
      for (let ci = 0; ci < COUNTRY_LIST.length; ci++) {
        if (announcedRef.current.has(ci)) continue;
        if (cInf[ci] / COUNTRY_LIST[ci].pop >= COUNTRY_TH) {
          announcedRef.current.add(ci);
          fresh.push({ id: eventSeq++, text: `🦠 ${pick(COUNTRY_TEMPLATES)(COUNTRY_LIST[ci].nameFr)}` });
        }
      }

      // One-shot milestone headlines.
      const nc = announcedRef.current.size;
      for (const ms of MILESTONES) {
        if (!milestonesRef.current.has(ms.key) && ms.test(people, deaths, nc)) {
          milestonesRef.current.add(ms.key);
          fresh.push({ id: eventSeq++, text: ms.text });
        }
      }

      // Vaccine research milestones.
      if (prev.vaccine < 0.001 && next.vaccine >= 0.001 && !milestonesRef.current.has("vac0")) {
        milestonesRef.current.add("vac0");
        fresh.push({ id: eventSeq++, text: "🔬 Les laboratoires lancent la course au vaccin" });
      }
      if (prev.vaccine < 0.5 && next.vaccine >= 0.5 && !milestonesRef.current.has("vac50")) {
        milestonesRef.current.add("vac50");
        fresh.push({ id: eventSeq++, text: "💉 Vaccin développé à 50 %" });
      }
      if (next.vaccine >= 1 && !milestonesRef.current.has("vac100")) {
        milestonesRef.current.add("vac100");
        fresh.push({ id: eventSeq++, text: "💉 Vaccin déployé — l'humanité contre-attaque !" });
      }

      // Lyon easter egg.
      if (LYON_IDX >= 0 && !lyonFiredRef.current && 1 - next.S[LYON_IDX] >= LYON_TH) {
        lyonFiredRef.current = true;
        fresh.push({ id: eventSeq++, text: "🏢 Le Prabi est contaminé !" });
      }

      // Gameplay event this tick (mutation, lockdown, super-spreader…).
      if (next.lastEvent) fresh.push({ id: eventSeq++, text: next.lastEvent });

      // Occasional random flavour headline (never the same twice in a row).
      if (Math.random() < FLASH_CHANCE) {
        let flash = pick(FLASHES);
        if (flash === lastFlashRef.current) flash = pick(FLASHES);
        lastFlashRef.current = flash;
        fresh.push({ id: eventSeq++, text: flash });
      }

      if (fresh.length) setFeed((f) => [...f, ...fresh].slice(-40));

      if (next.finished) setPhase("end");
    }, TICK_MS);
    return () => clearInterval(id);
  }, [phase, paused, config]);

  // Animation loop: re-render each frame and retire vehicles that have arrived.
  useEffect(() => {
    if (phase !== "running" || paused) return;
    let raf = 0;
    const tick = () => {
      const now = performance.now();
      setVehicles((vs) =>
        vs.some((v) => now - v.start > v.dur) ? vs.filter((v) => now - v.start <= v.dur) : vs,
      );
      setFrame((f) => f + 1);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [phase, paused]);

  // Submit the finished game once and pull the global leaderboard back.
  useEffect(() => {
    if (phase !== "end" || !sim || !config || submittedRef.current) return;
    submittedRef.current = true;
    const m = metrics(sim);
    api
      .post<ScoreOut>("/api/helper/score", {
        virus: config.virusName,
        days: m.day,
        infected_pct: m.infectedPct,
        deaths_pct: m.deathsPct,
        dead: Math.round(m.deaths),
        won: m.won,
        vaccine_deployed: m.vaccineDeployed,
      })
      .then((res) => {
        setMyScore(res.score);
        setBreakdown(res.breakdown);
        setLeaderboard(res.leaderboard);
      })
      .catch(() => setScoreError(true));
  }, [phase, sim, config]);

  function launch(cfg: VirusConfig) {
    setConfig(cfg);
    setPhase("pick");
  }

  const pickPatientZero = useCallback((cityIndex: number) => {
    if (cityIndex < 0 || cityIndex >= CITIES.length) return;
    setSim(initSim(cityIndex));
    setVehicles([]);
    setFeed([]);
    announcedRef.current = new Set();
    lyonFiredRef.current = false;
    milestonesRef.current = new Set();
    lastFlashRef.current = "";
    setPaused(false);
    setEndCardOpen(true);
    setPhase("running");
  }, []);

  function restart() {
    setSim(null);
    setConfig(null);
    setVehicles([]);
    setFeed([]);
    announcedRef.current = new Set();
    lyonFiredRef.current = false;
    milestonesRef.current = new Set();
    lastFlashRef.current = "";
    setPaused(false);
    setMyScore(null);
    setBreakdown(null);
    setLeaderboard(null);
    setScoreError(false);
    submittedRef.current = false;
    setPhase("setup");
  }

  // Interpolate live vehicle positions for the current frame.
  const vehicleDots: VehicleDot[] = (() => {
    if (phase !== "running") return [];
    const now = performance.now();
    return vehicles.map((v) => {
      const e = easeInOut(Math.min(1, (now - v.start) / v.dur));
      const p = greatCircle(v.from, v.to, e); // curved real-world trajectory
      return { lat: p.lat, lng: p.lng, mode: v.mode };
    });
  })();

  const m = sim ? metrics(sim) : null;
  // New array only when the sim advances (per tick), so the memoized map does
  // not redraw on every animation frame — critical with ~10k dots.
  const cityInf = useMemo(
    () => (sim ? Array.from(sim.S, (s) => 1 - s) : ZERO_INF),
    [sim],
  );
  // Per-city cumulative infected people & deaths, for the hover tooltip.
  const cityInfPeople = useMemo(
    () => (sim ? Array.from(sim.S, (s, i) => (1 - s) * CITIES[i].pop) : ZERO_INF),
    [sim],
  );
  const cityDeaths = useMemo(
    () => (sim ? Array.from(sim.dead, (d, i) => d * CITIES[i].pop) : ZERO_INF),
    [sim],
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="relative flex h-[85vh] w-[90vw] max-w-5xl flex-col overflow-hidden rounded-2xl border border-white/10 bg-neutral-950 text-neutral-100 shadow-2xl">
        <button
          onClick={onClose}
          className="absolute right-3 top-3 z-10 rounded-lg p-2 text-neutral-400 hover:bg-white/10 hover:text-white"
          title="Fermer"
        >
          <X className="h-5 w-5" />
        </button>

        {phase === "setup" && <HelperSetup onLaunch={launch} />}

        {(phase === "pick" || phase === "running" || phase === "end") && (
          <div className="flex flex-1 flex-col overflow-hidden p-4">
            {/* Header / counters */}
            {phase === "pick" ? (
              <div className="mb-2 text-center">
                <div className="text-lg font-semibold text-white">
                  Clique sur ta ville de départ 🎯
                </div>
                <div className="text-xs text-neutral-400">
                  {config?.virusName} — choisis ton patient zéro (une ville ou un pays)
                </div>
                <div className="mt-1 text-xs text-red-300/80">
                  🎯 But : faire le <span className="font-semibold">maximum de morts</span>. (La
                  partie s'arrête à 90% de contaminés, ou si le vaccin est déployé.)
                </div>
              </div>
            ) : (
              m && (
                <>
                  <div className="mb-2 grid grid-cols-5 gap-2 text-center">
                    <Stat label="Jour" value={String(m.day)} />
                    <Stat label="Contaminés" value={fmt(m.infectedPeople)} accent />
                    <Stat label="Actifs" value={fmt(m.activePeople)} />
                    <Stat label="Villes" value={`${m.citiesTouched}/${m.totalCities}`} />
                    <Stat label="Morts" value={fmt(m.deaths)} danger />
                  </div>
                  {/* R0 / Rt (herd immunity pushes Rt down over time) */}
                  <div className="mb-2 text-center text-[11px] text-neutral-400">
                    <InfoTip text="R₀ (« R zéro ») : nombre moyen de personnes qu'un malade contamine dans une population entièrement susceptible. Fixé par ton virus. Au-dessus de 1, l'épidémie peut décoller.">
                      R₀
                    </InfoTip>{" "}
                    <span className="font-mono text-neutral-200">{m.r0.toFixed(2)}</span>
                    {"  ·  "}
                    <InfoTip text="Rt : nombre effectif de contaminations par malade à l'instant t. Rt = R₀ × part de gens encore susceptibles. Il baisse quand les gens deviennent immunisés ou vaccinés. Sous 1, l'épidémie reflue.">
                      Rt
                    </InfoTip>{" "}
                    <span
                      className={
                        "font-mono " + (m.rt >= 1 ? "text-red-400" : "text-emerald-400")
                      }
                    >
                      {m.rt.toFixed(2)}
                    </span>
                    <span className="text-neutral-600">
                      {" "}
                      {m.rt >= 1 ? "(épidémie en expansion)" : "(reflux — immunité/vaccin)"}
                    </span>
                  </div>
                  {/* Vaccine research race — only shown once research has begun
                      (i.e. the pathogen has been identified). */}
                  {m.vaccine > 0 && (
                    <div className="mb-2 flex items-center gap-2">
                      <span className="w-16 text-[11px] uppercase tracking-wide text-cyan-300/80">
                        Vaccin
                      </span>
                      <div className="h-2 flex-1 overflow-hidden rounded-full bg-neutral-800">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-sky-300 transition-all"
                          style={{ width: `${Math.round(m.vaccine * 100)}%` }}
                        />
                      </div>
                      <span className="w-10 text-right font-mono text-xs text-cyan-300">
                        {Math.round(m.vaccine * 100)}%
                      </span>
                    </div>
                  )}
                  {m.closedCountries > 0 && (
                    <div className="mb-2 text-center text-[11px] text-amber-300/80">
                      🛂 {m.closedCountries} pays ont fermé leurs frontières
                    </div>
                  )}
                </>
              )
            )}

            {/* Map */}
            <div className="relative flex-1 overflow-hidden rounded-xl border border-white/5 bg-black/30">
              <HelperMap
                cityInf={cityInf}
                infPeople={cityInfPeople}
                deaths={cityDeaths}
                height={MAP_H}
                onPick={phase === "pick" ? pickPatientZero : undefined}
              />
              {phase === "running" && vehicleDots.length > 0 && (
                <HelperVehicles vehicles={vehicleDots} height={MAP_H} />
              )}
              {phase === "end" && !endCardOpen && (
                <button
                  onClick={() => setEndCardOpen(true)}
                  className="absolute left-1/2 top-3 z-10 -translate-x-1/2 rounded-full border border-red-500/40 bg-neutral-900/90 px-4 py-1.5 text-sm font-medium text-red-300 shadow-lg hover:bg-neutral-800"
                >
                  🏆 Revoir le résultat
                </button>
              )}
              {phase === "end" && endCardOpen && (
                <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/40">
                  <div className="pointer-events-auto relative rounded-2xl border border-red-500/40 bg-neutral-900/95 px-8 py-6 text-center shadow-2xl">
                    <button
                      onClick={() => setEndCardOpen(false)}
                      className="absolute right-2 top-2 rounded-lg p-1.5 text-neutral-400 hover:bg-white/10 hover:text-white"
                      title="Voir la carte finale"
                    >
                      <X className="h-4 w-4" />
                    </button>
                    {m?.won ? (
                      <div className="text-4xl font-extrabold tracking-wide text-red-500">
                        VICTOIRE 🦠
                      </div>
                    ) : (
                      <div className="text-4xl font-extrabold tracking-wide text-cyan-300">
                        DÉFAITE
                      </div>
                    )}
                    <div className="mt-1 text-sm font-medium text-neutral-200">
                      {m?.endReason === "infected" &&
                        `${config?.virusName} a submergé le monde : plus de ${Math.round(
                          (m?.infectedPct ?? 0) * 100,
                        )}% contaminés.`}
                      {m?.endReason === "vaccine" &&
                        "💉 Vaccin déployé — l'humanité est sauvée."}
                      {m?.endReason === "stall" &&
                        "🛡️ Le virus s'est éteint : l'immunité collective l'a stoppé."}
                      {m?.endReason === "timeout" && "⏳ Temps écoulé — le monde a tenu."}
                    </div>
                    <div className="mt-2 text-sm text-neutral-300">
                      <span className="font-semibold text-white">{config?.virusName}</span> a
                      contaminé{" "}
                      <span className="font-semibold text-white">
                        {m ? Math.round(m.infectedPct * 100) : 0}%
                      </span>{" "}
                      du monde en{" "}
                      <span className="font-semibold text-white">{m?.day}</span> jours ·{" "}
                      <span className="font-semibold text-white">{m ? fmt(m.deaths) : 0}</span> morts.
                    </div>

                    {/* Score + global top 5 */}
                    <div className="mt-4">
                      {myScore !== null && (
                        <div className="text-sm text-neutral-300">
                          Ton score :{" "}
                          <span className="font-mono text-2xl font-bold text-amber-400">
                            {myScore}
                          </span>
                        </div>
                      )}
                      {breakdown && m && (
                        <div className="mx-auto mt-2 max-w-xs rounded-lg border border-white/5 bg-white/5 px-3 py-2 text-left text-xs text-neutral-400">
                          <div className="mb-1 text-center font-medium text-neutral-300">
                            Comment est calculé ton score
                          </div>
                          <div className="flex justify-between">
                            <span>Couverture ({Math.round(m.infectedPct * 100)}%)</span>
                            <span className="text-neutral-200">+{breakdown.coverage_pts}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Mortalité ({Math.round(m.deathsPct * 100)}%)</span>
                            <span className="text-neutral-200">+{breakdown.mortality_pts}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Rapidité ({m.day} j)</span>
                            <span className="text-neutral-200">+{breakdown.speed_pts}</span>
                          </div>
                          {breakdown.vaccine_penalty && (
                            <div className="flex justify-between text-cyan-300">
                              <span>Malus vaccin</span>
                              <span>× 0.4</span>
                            </div>
                          )}
                          <div className="mt-1 border-t border-white/10 pt-1 text-center text-neutral-300">
                            Total ={" "}
                            <span className="font-mono font-semibold text-amber-400">{myScore}</span>
                          </div>
                          <div className="mt-1 text-center text-[11px] text-neutral-500">
                            La <em>mortalité</em> rapporte le plus — tue un maximum, vite, avant le
                            vaccin.
                          </div>
                        </div>
                      )}
                      {scoreError && (
                        <div className="text-xs text-red-400">
                          Score non enregistré (backend indisponible).
                        </div>
                      )}
                      {leaderboard && leaderboard.length > 0 && (
                        <div className="mx-auto mt-3 max-w-xs text-left">
                          <div className="mb-1 text-center text-xs uppercase tracking-wide text-neutral-500">
                            🏆 Top 5 mondial
                          </div>
                          <ol className="space-y-1">
                            {leaderboard.map((row, i) => (
                              <li
                                key={i}
                                className="flex items-center justify-between rounded-md bg-white/5 px-2 py-1 text-sm"
                              >
                                <span className="flex items-center gap-2 truncate">
                                  <span className="w-4 text-right font-mono text-neutral-500">
                                    {i + 1}
                                  </span>
                                  <span className="truncate text-neutral-200">{row.name}</span>
                                  <span className="text-xs text-neutral-500">({row.virus})</span>
                                </span>
                                <span className="font-mono font-semibold text-amber-400">
                                  {row.score}
                                </span>
                              </li>
                            ))}
                          </ol>
                        </div>
                      )}
                    </div>

                    <button
                      onClick={restart}
                      className="mt-4 inline-flex items-center gap-2 rounded-xl bg-red-600 px-5 py-2 font-semibold text-white hover:bg-red-500"
                    >
                      <RotateCcw className="h-4 w-4" />
                      Rejouer
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Breaking-news ticker: notable cities as they fall. */}
            {(phase === "running" || phase === "end") && (
              <div className="mt-2 overflow-hidden rounded-lg border border-white/5 bg-black/40">
                {feed.length === 0 ? (
                  <div className="py-1.5 text-center text-xs text-neutral-600">
                    En attente des premières contaminations…
                  </div>
                ) : (
                  <div
                    className="whitespace-nowrap py-1.5 text-sm font-medium text-red-300"
                    style={{ animation: "helperMarquee 65s linear infinite" }}
                  >
                    {[0, 1].map((dup) => (
                      <span key={dup}>
                        {feed.map((e, i) => (
                          <span key={`${dup}-${i}`} className="mx-6">
                            {e.text}
                          </span>
                        ))}
                      </span>
                    ))}
                  </div>
                )}
                <style>{"@keyframes helperMarquee{from{transform:translateX(0)}to{transform:translateX(-50%)}}"}</style>
              </div>
            )}

            {/* Controls */}
            {phase === "running" && (
              <div className="mt-2 flex justify-center">
                <button
                  onClick={() => setPaused((p) => !p)}
                  className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-4 py-2 text-sm text-neutral-200 hover:bg-white/10"
                >
                  {paused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
                  {paused ? "Reprendre" : "Pause"}
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/** A term with a hover bubble explaining it. */
function InfoTip({ children, text }: { children: React.ReactNode; text: string }) {
  return (
    <span className="group relative cursor-help underline decoration-dotted underline-offset-2">
      {children}
      <span className="pointer-events-none absolute bottom-full left-1/2 z-20 mb-1 hidden w-56 -translate-x-1/2 rounded-lg border border-white/10 bg-neutral-900 px-3 py-2 text-left text-[11px] font-normal leading-snug normal-case text-neutral-200 shadow-xl group-hover:block">
        {text}
      </span>
    </span>
  );
}

function Stat({
  label,
  value,
  accent,
  danger,
}: {
  label: string;
  value: string;
  accent?: boolean;
  danger?: boolean;
}) {
  return (
    <div className="rounded-lg border border-white/5 bg-white/5 px-2 py-1.5">
      <div className="text-[10px] uppercase tracking-wide text-neutral-500">{label}</div>
      <div
        className={[
          "font-mono text-lg font-semibold",
          danger ? "text-red-500" : accent ? "text-orange-400" : "text-neutral-100",
        ].join(" ")}
      >
        {value}
      </div>
    </div>
  );
}
