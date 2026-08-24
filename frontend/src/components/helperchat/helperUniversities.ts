/**
 * helperUniversities.ts — real research institutions that work on the cure.
 * Each one adds to global vaccine research; if its home city becomes heavily
 * infected, its staff fall ill and its output drops (so they're worth
 * contaminating). `power` roughly reflects biomedical/virology weight.
 */
export interface University {
  name: string;
  short: string;
  lat: number;
  lng: number;
  iso3: string;
  power: number; // 1 (small) … 3 (major biomedical hub)
}

export const UNIVERSITIES: University[] = [
  { name: "Harvard University", short: "Harvard", lat: 42.377, lng: -71.116, iso3: "USA", power: 3 },
  { name: "MIT", short: "MIT", lat: 42.36, lng: -71.092, iso3: "USA", power: 2 },
  { name: "Johns Hopkins", short: "Johns Hopkins", lat: 39.329, lng: -76.62, iso3: "USA", power: 3 },
  { name: "Stanford University", short: "Stanford", lat: 37.427, lng: -122.17, iso3: "USA", power: 2 },
  { name: "University of Oxford", short: "Oxford", lat: 51.754, lng: -1.254, iso3: "GBR", power: 3 },
  { name: "University of Cambridge", short: "Cambridge", lat: 52.205, lng: 0.117, iso3: "GBR", power: 2 },
  { name: "Institut Pasteur", short: "Institut Pasteur", lat: 48.84, lng: 2.312, iso3: "FRA", power: 3 },
  { name: "Université Lyon 1", short: "Lyon 1", lat: 45.782, lng: 4.865, iso3: "FRA", power: 2 },
  { name: "Sorbonne Université", short: "Sorbonne", lat: 48.847, lng: 2.356, iso3: "FRA", power: 1 },
  { name: "ETH Zürich", short: "ETH Zürich", lat: 47.376, lng: 8.548, iso3: "CHE", power: 2 },
  { name: "Heidelberg University", short: "Heidelberg", lat: 49.41, lng: 8.707, iso3: "DEU", power: 1 },
  { name: "Karolinska Institutet", short: "Karolinska", lat: 59.349, lng: 18.03, iso3: "SWE", power: 3 },
  { name: "University of Tokyo", short: "Tokyo", lat: 35.713, lng: 139.762, iso3: "JPN", power: 2 },
  { name: "Tsinghua University", short: "Tsinghua", lat: 40.0, lng: 116.326, iso3: "CHN", power: 2 },
  { name: "Seoul National University", short: "Seoul Nat'l", lat: 37.459, lng: 126.952, iso3: "KOR", power: 1 },
  { name: "National University of Singapore", short: "NUS", lat: 1.296, lng: 103.776, iso3: "SGP", power: 2 },
  { name: "University of Toronto", short: "Toronto", lat: 43.663, lng: -79.395, iso3: "CAN", power: 2 },
  { name: "University of Cape Town", short: "Cape Town", lat: -33.957, lng: 18.461, iso3: "ZAF", power: 1 },
  { name: "University of São Paulo", short: "USP", lat: -23.559, lng: -46.731, iso3: "BRA", power: 1 },
  { name: "University of Melbourne", short: "Melbourne", lat: -37.797, lng: 144.961, iso3: "AUS", power: 1 },
];
