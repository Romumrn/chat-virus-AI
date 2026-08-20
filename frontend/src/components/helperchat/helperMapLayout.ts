/**
 * helperMapLayout.ts — the shared Plotly `geo` block used by both the base
 * choropleth (HelperMap) and the vehicle overlay (HelperVehicles). Keeping it
 * in one place guarantees the two stacked maps use the exact same projection
 * and framing, so overlaid planes/boats line up with the countries beneath.
 */
export function geoBlock(showLand: boolean) {
  return {
    bgcolor: "rgba(0,0,0,0)",
    showframe: false,
    showcoastlines: false,
    showland: showLand,
    landcolor: "#1c1f26",
    showocean: showLand,
    oceancolor: "#0c0e13",
    showlakes: false,
    projection: { type: "natural earth" as const },
    lonaxis: { range: [-180, 180] },
    lataxis: { range: [-60, 85] },
  };
}

export const MAP_MARGIN = { l: 0, r: 0, t: 0, b: 0 };
