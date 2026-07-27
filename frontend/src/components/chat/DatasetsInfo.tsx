/**
 * DatasetsInfo — the "About" content (project link, Datasets, Tools) shown in
 * the Info tab of the chat sidebar. Kept as its own component so the copy lives
 * in one place.
 */
import { Database, Wrench } from "lucide-react";

export default function DatasetsInfo() {
  return (
    <div className="space-y-4 p-3 text-sm">
      <p className="text-muted-foreground">
        A chatbot to explore viral metagenomic data from the{" "}
        <a
          href="http://shape-med-lyon.fr/projets/structurants-vague-1/virometlas/"
          target="_blank"
          rel="noreferrer"
          className="text-primary hover:underline"
        >
          Virome@tlas project
        </a>
        .
      </p>

      <div>
        <div className="flex items-center gap-2 font-medium">
          <Database className="h-4 w-4 text-primary" /> Datasets
        </div>
        <ul className="mt-2 space-y-2 text-muted-foreground">
          <li>
            <span className="font-medium text-foreground">Taxonomy</span> — NCBI
            Taxonomy, enriched with genome assembly availability, SRA sequencing
            activity, and GBIF biodiversity observations.
          </li>
          <li>
            <span className="font-medium text-foreground">Virus-host occurrences</span> —
            SRA/GenBank/BioSample samples linked to host &amp; virus taxonomy,
            geographic location, and disease status.
          </li>
        </ul>
      </div>

      <div>
        <div className="flex items-center gap-2 font-medium">
          <Wrench className="h-4 w-4 text-primary" /> Tools
        </div>
        <ul className="mt-2 space-y-1 text-muted-foreground">
          <li className="flex gap-2"><span className="text-primary">•</span> SQL &amp; pandas queries over both datasets</li>
          <li className="flex gap-2"><span className="text-primary">•</span> Interactive maps and charts</li>
          <li className="flex gap-2"><span className="text-primary">•</span> Wikipedia &amp; PubMed search for biological background</li>
        </ul>
      </div>
    </div>
  );
}
