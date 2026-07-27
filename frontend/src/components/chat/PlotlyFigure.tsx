/**
 * PlotlyFigure — render a Plotly figure JSON (as produced by the MCP
 * create_visualization / create_map tools). Uses the factory + dist-min build
 * so we don't pull the full plotly.js bundle.
 */
import createPlotlyComponent from "react-plotly.js/factory";
// @ts-expect-error — dist-min ships no type declarations
import Plotly from "plotly.js-dist-min";

const Plot = createPlotlyComponent(Plotly);

export default function PlotlyFigure({ figure }: { figure: any }) {
  if (!figure) return null;
  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  const layout = {
    ...(figure.layout || {}),
    autosize: true,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: isDark ? "#e5e7eb" : "#111827" },
    margin: { t: 40, r: 20, b: 40, l: 50, ...(figure.layout?.margin || {}) },
    modebar: { orientation: "h", ...(figure.layout?.modebar || {}) },
  };
  return (
    <div className="my-3 overflow-x-auto rounded-lg border border-border p-2">
      <Plot
        data={figure.data || []}
        layout={layout}
        config={{ responsive: true, displaylogo: false, scrollZoom: true }}
        style={{ width: "100%", height: "400px" }}
        useResizeHandler
      />
    </div>
  );
}
