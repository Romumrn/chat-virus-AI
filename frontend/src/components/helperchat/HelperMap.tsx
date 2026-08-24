/**
 * HelperMap — hybrid world view: faded country shapes for geographic context
 * (still clickable, to pick a patient zero), with the ~1000 cities drawn on top
 * as dots that light up and redden as their infection level rises.
 */
import { memo } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
// @ts-expect-error — dist-min ships no type declarations
import Plotly from "plotly.js-dist-min";
import { COUNTRIES } from "./helperData";
import { CITIES } from "./helperCities";
import { UNIVERSITIES } from "./helperUniversities";
import { biggestCityIn, CITY_BIOME } from "./helperEngine";
import { geoBlock, MAP_MARGIN } from "./helperMapLayout";

const Plot = createPlotlyComponent(Plotly);

// City infection ramp: faint blue-grey when clean → neon red when saturated.
const DOT_SCALE: [number, string][] = [
  [0.0, "#33405a"],
  [0.12, "#7a5a2a"],
  [0.4, "#d9720b"],
  [0.7, "#f23a1d"],
  [1.0, "#ff1e3c"],
];

const CITY_LAT = CITIES.map((c) => c.lat);
const CITY_LON = CITIES.map((c) => c.lng);
const CITY_TEXT = CITIES.map((c) => c.name);
// Marker size by population (sqrt so megacities don't dwarf everything).
const CITY_SIZE = CITIES.map((c) => Math.max(3, Math.min(18, Math.sqrt(c.pop) / 500)));

// Faded base countries, uniform colour, drawn only for context + click target.
const COUNTRY_LOC = COUNTRIES.map((c) => c.iso3);
const COUNTRY_Z = COUNTRIES.map(() => 0);

// Research universities (static overlay).
const UNI_LAT = UNIVERSITIES.map((u) => u.lat);
const UNI_LON = UNIVERSITIES.map((u) => u.lng);
const UNI_TEXT = UNIVERSITIES.map((u) => u.short);

function HelperMap({
  cityInf,
  infPeople,
  deaths,
  height = 480,
  onPick,
}: {
  /** per-city infection level [0,1], aligned with CITIES (used for colour). */
  cityInf: number[];
  /** per-city cumulative infected people (for the tooltip). */
  infPeople: number[];
  /** per-city cumulative deaths (for the tooltip). */
  deaths: number[];
  height?: number;
  /** Called with a CITIES index when a country or a city dot is clicked. */
  onPick?: (cityIndex: number) => void;
}) {
  return (
    <Plot
      data={[
        {
          type: "choropleth",
          locationmode: "ISO-3",
          locations: COUNTRY_LOC,
          z: COUNTRY_Z,
          zmin: 0,
          zmax: 1,
          colorscale: [
            [0, "#161a20"],
            [1, "#161a20"],
          ] as any,
          showscale: false,
          marker: { line: { color: "#0a0b0e", width: 0.4 } },
          hoverinfo: "skip",
        } as any,
        {
          type: "scattergeo",
          mode: "markers",
          lat: CITY_LAT,
          lon: CITY_LON,
          text: CITY_TEXT,
          customdata: cityInf.map((_v, i) => [CITY_BIOME[i], infPeople[i], deaths[i]]),
          marker: {
            size: CITY_SIZE,
            color: cityInf,
            colorscale: DOT_SCALE as any,
            cmin: 0,
            cmax: 1,
            opacity: 0.9,
            line: { width: 0 },
          },
          hovertemplate:
            "<b>%{text}</b><br>Biome : %{customdata[0]}" +
            "<br>Contaminés : %{customdata[1]:.3s}" +
            "<br>Morts : %{customdata[2]:.3s}<extra></extra>",
        } as any,
        {
          type: "scattergeo",
          mode: "text",
          lat: UNI_LAT,
          lon: UNI_LON,
          text: UNIVERSITIES.map(() => "🎓"),
          customdata: UNI_TEXT,
          textfont: { size: 13 },
          hovertemplate: "🎓 %{customdata}<br>Recherche un vaccin<extra></extra>",
        } as any,
      ]}
      layout={
        {
          height,
          margin: MAP_MARGIN,
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          clickmode: "event",
          dragmode: false,
          showlegend: false,
          geo: geoBlock(true),
          font: { color: "#cbd5e1" },
        } as any
      }
      config={{ displayModeBar: false, responsive: true } as any}
      style={{ width: "100%" }}
      useResizeHandler
      onClick={
        onPick
          ? (e: any) => {
              const pt = e?.points?.[0];
              if (!pt) return;
              // curveNumber 1 = the city dots → that exact city; curveNumber 0 =
              // a country (choropleth) → its most populous city. The university
              // layer (curveNumber 2) is not a valid patient zero.
              if (pt.curveNumber === 1 && typeof pt.pointNumber === "number") {
                onPick(pt.pointNumber);
              } else if (pt.location) {
                const idx = biggestCityIn(pt.location);
                if (idx >= 0) onPick(idx);
              }
            }
          : undefined
      }
    />
  );
}

// Memoized: the map only needs to redraw when the infection array or handlers
// change (once per simulation tick), not on every animation frame.
export default memo(HelperMap);
