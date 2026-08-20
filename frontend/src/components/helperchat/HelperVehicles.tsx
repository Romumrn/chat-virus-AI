/**
 * HelperVehicles — a transparent scattergeo layer stacked on top of HelperMap.
 * It draws the in-flight planes ✈️ and boats 🚢 at their current positions.
 * Same geo block as the base map ⇒ pixel-perfect alignment with the countries
 * underneath. staticPlot keeps it cheap to redraw every animation frame.
 */
import createPlotlyComponent from "react-plotly.js/factory";
// @ts-expect-error — dist-min ships no type declarations
import Plotly from "plotly.js-dist-min";
import { geoBlock, MAP_MARGIN } from "./helperMapLayout";

const Plot = createPlotlyComponent(Plotly);

export interface VehicleDot {
  lat: number;
  lng: number;
  mode: "air" | "sea";
}

export default function HelperVehicles({
  vehicles,
  height = 480,
}: {
  vehicles: VehicleDot[];
  height?: number;
}) {
  return (
    <div className="pointer-events-none absolute inset-0">
      <Plot
        data={[
          {
            type: "scattergeo",
            mode: "text",
            lat: vehicles.map((v) => v.lat),
            lon: vehicles.map((v) => v.lng),
            text: vehicles.map((v) => (v.mode === "sea" ? "🚢" : "✈️")),
            textfont: { size: 17 },
            hoverinfo: "skip",
          } as any,
        ]}
        layout={
          {
            height,
            margin: MAP_MARGIN,
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            geo: geoBlock(false), // transparent: only the vehicles show
            showlegend: false,
          } as any
        }
        config={{ displayModeBar: false, staticPlot: true, responsive: true } as any}
        style={{ width: "100%" }}
        useResizeHandler
      />
    </div>
  );
}
