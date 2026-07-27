/**
 * DatasetsInfo — the "About" content (project link, Datasets, Tools) shown in
 * the Info tab of the chat sidebar. Kept as its own component so the copy lives
 * in one place.
 */
import {
  Database,
  Wrench,
  Search,
  Code2,
  BarChart3,
  Map,
  BookOpen,
  FileText,
  ShieldCheck,
} from "lucide-react";
import { Card } from "@/components/ui";

export default function Info() {
  return (
    <div className="space-y-6 p-3 text-sm">
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

      <Card className="p-4">
        <div className="flex items-center gap-2 font-medium">
          <Database className="h-4 w-4 text-primary" /> Datasets
        </div>
        <div className="mt-3 space-y-4 text-muted-foreground">
          <div>
            <div className="font-medium text-foreground">Taxonomy</div>
            <p className="mt-1">
              NCBI Taxonomy — one row per taxon (species, subspecies, genus, family,
              order, class, phylum...), enriched with genome assembly availability, SRA
              sequencing activity, and GBIF biodiversity observations. Includes
              ecological flags such as endangered, extinct, invasive, domestic, and
              model organism.
            </p>
          </div>
          <div>
            <div className="font-medium text-foreground">Virus-host occurrences</div>
            <p className="mt-1">
              SRA/GenBank/BioSample samples (~65 columns) linking virus and host
              taxonomy, collection date, and geographic location — country, coordinates,
              land/ocean, distance to rivers, lakes and coastline. Also covers sample
              type, sequencing platform/assay, and disease context via the MONDO disease
              and UBERON anatomical-site ontologies.
            </p>
          </div>
        </div>
      </Card>

      <Card className="p-4">
        <div className="flex items-center gap-2 font-medium">
          <Wrench className="h-4 w-4 text-primary" /> Tools
        </div>
        <ul className="mt-3 space-y-2.5 text-muted-foreground">
          <li className="flex gap-2">
            <Search className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
            <span>
              <span className="font-medium text-foreground">Taxonomy lookup</span> —
              resolves acronyms, common names or synonyms (e.g. "HIV") to the
              authoritative NCBI scientific name, rank, and lineage.
            </span>
          </li>
          <li className="flex gap-2">
            <Database className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
            <span>
              <span className="font-medium text-foreground">SQL queries</span> —
              read-only, filtered queries over the virus-host occurrences dataset.
            </span>
          </li>
          <li className="flex gap-2">
            <Code2 className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
            <span>
              <span className="font-medium text-foreground">Pandas queries</span> —
              filters, aggregates, and combines the taxonomy dataset with the results of
              a previous SQL query.
            </span>
          </li>
          <li className="flex gap-2">
            <BarChart3 className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
            <span>
              <span className="font-medium text-foreground">Charts</span> — interactive
              bar charts, pie charts, and other visualizations of the data.
            </span>
          </li>
          <li className="flex gap-2">
            <Map className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
            <span>
              <span className="font-medium text-foreground">Maps</span> — interactive
              geographic maps of virus-host observation locations.
            </span>
          </li>
          <li className="flex gap-2">
            <BookOpen className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
            <span>
              <span className="font-medium text-foreground">Wikipedia search</span> —
              biological background on a species, family, or disease.
            </span>
          </li>
          <li className="flex gap-2">
            <FileText className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
            <span>
              <span className="font-medium text-foreground">PubMed search</span> —
              scientific literature, with titles, abstracts, authors, and PMIDs/DOIs.
            </span>
          </li>
        </ul>
      </Card>

      <Card className="p-4">
        <div className="flex items-center gap-2 font-medium">
          <ShieldCheck className="h-4 w-4 text-primary" /> Good to know
        </div>
        <ul className="mt-3 space-y-1.5 text-muted-foreground">
          <li className="flex gap-2">
            <span className="text-primary">•</span> Every answer that runs code or looks
            something up shows a "Sources" panel, so you can check the exact SQL/pandas
            code or Wikipedia/PubMed links used.
          </li>
          <li className="flex gap-2">
            <span className="text-primary">•</span> The assistant remembers the last few
            exchanges of a conversation, so you can ask natural follow-up questions.
          </li>
          <li className="flex gap-2">
            <span className="text-primary">•</span> Full-table dumps (SELECT *) are
            blocked — ask for specific columns, filters, or aggregates instead.
          </li>
        </ul>
      </Card>
    </div>
  );
}
