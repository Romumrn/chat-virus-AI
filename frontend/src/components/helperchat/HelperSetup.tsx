/**
 * HelperSetup — the "name your virus" screen. Type a name, spread 3 stats
 * within a shared budget, then launch. Emits a VirusConfig to the parent.
 */
import { useState } from "react";
import { Play } from "lucide-react";
import { STAT_BUDGET, statTotal, type VirusConfig, type VirusStats } from "./helperTypes";

const STAT_META: { key: keyof VirusStats; label: string }[] = [
  { key: "contagiosite", label: "Contagiosité" },
  { key: "letalite", label: "Létalité" },
  { key: "resistance", label: "Résistance" },
];

const DEFAULT_STATS: VirusStats = { contagiosite: 0.5, letalite: 0.3, resistance: 0.2 };

export default function HelperSetup({ onLaunch }: { onLaunch: (cfg: VirusConfig) => void }) {
  const [name, setName] = useState("");
  const [stats, setStats] = useState<VirusStats>(DEFAULT_STATS);

  function setStat(key: keyof VirusStats, value: number) {
    setStats((s) => {
      // Clamp to the shared budget: a slider can only rise as far as the points
      // left over from the other two stats allow.
      const others = statTotal(s) - s[key];
      const maxForKey = Math.min(1, STAT_BUDGET - others);
      return { ...s, [key]: Math.max(0, Math.min(value, maxForKey)) };
    });
  }

  const used = statTotal(stats);
  const remaining = Math.max(0, STAT_BUDGET - used);
  const virusName = name.trim() || "Virus sans nom";

  return (
    <div className="flex flex-1 flex-col overflow-y-auto p-6">
      <div className="mx-auto w-full max-w-2xl">
        <h2 className="text-center text-2xl font-bold text-white">Crée ton virus</h2>
        <p className="mb-6 text-center text-sm text-neutral-400">
          Donne-lui un nom, règle ses caractéristiques, puis lâche-le sur le monde.
        </p>

        {/* Virus name */}
        <div className="mb-6">
          <label className="mb-1 block text-sm font-medium text-neutral-200">Nom du virus</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            maxLength={40}
            placeholder="ex. Virus X, La Peste 2.0, Grippe du singe…"
            className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-white placeholder:text-neutral-600 focus:border-red-500 focus:outline-none focus:ring-1 focus:ring-red-500"
          />
        </div>

        {/* Genetic-points budget */}
        <div className="mb-4 rounded-xl border border-white/10 bg-white/5 p-3">
          <div className="mb-1 flex items-baseline justify-between text-sm">
            <span className="font-medium text-neutral-200">Budget génétique</span>
            <span className="font-mono text-neutral-300">
              {Math.round(used * 100)}
              <span className="text-neutral-500"> / {Math.round(STAT_BUDGET * 100)}%</span>
            </span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-700">
            <div
              className="h-full rounded-full bg-gradient-to-r from-amber-500 to-red-500 transition-all"
              style={{ width: `${(used / STAT_BUDGET) * 100}%` }}
            />
          </div>
          <div className="mt-1 text-xs text-neutral-500">
            {remaining > 0.005
              ? `${Math.round(remaining * 100)}% de points restants à répartir`
              : "Budget épuisé — baisse une caractéristique pour en monter une autre"}
          </div>
        </div>

        {/* Stat sliders */}
        <div className="space-y-5">
          {STAT_META.map(({ key, label }) => (
            <div key={key}>
              <div className="mb-1 flex items-baseline justify-between">
                <span className="text-sm font-medium text-neutral-200">{label}</span>
                <span className="font-mono text-sm text-red-400">
                  {Math.round(stats[key] * 100)}%
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={100}
                value={Math.round(stats[key] * 100)}
                onChange={(e) => setStat(key, Number(e.target.value) / 100)}
                className="h-2 w-full cursor-pointer appearance-none rounded-full bg-neutral-700 accent-red-500"
              />
            </div>
          ))}
        </div>

        {/* Launch */}
        <div className="mt-8 flex justify-center">
          <button
            onClick={() => onLaunch({ virusName, presetId: "custom", stats })}
            className="inline-flex items-center gap-2 rounded-xl bg-red-600 px-6 py-3 font-semibold text-white shadow-lg shadow-red-900/40 transition hover:bg-red-500"
          >
            <Play className="h-5 w-5" />
            Lâcher le virus
          </button>
        </div>
      </div>
    </div>
  );
}
