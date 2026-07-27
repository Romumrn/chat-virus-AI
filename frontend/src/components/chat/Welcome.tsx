/**
 * Welcome — the empty-state shown on a new chat. Carries the guidance and
 * clickable example questions from the old Streamlit landing, plus an "About"
 * panel describing the Virome@tlas datasets and tools. Clicking an example
 * sends it straight away.
 */
import { Lightbulb, Sparkles } from "lucide-react";
import { Card } from "@/components/ui";

const TIPS = [
  "Use English and provide as much relevant detail as you can",
  "Ask precise and clearly formulated questions, avoid acronyms or abbreviations",
  "A vague question leads to a vague answer ;)",
];

const EXAMPLES = [
  "Give me information about Orthopoxvirus. Is it a family or a genus? How many species does it include?",
  "Show me a summary in piechart of the dataframe in term of viral family repartition",
  "World repartition of poxviridae",
  "Tell me more about Polyomavirus infection way",
];

export default function Welcome({
  name,
  onExample,
}: {
  name: string;
  onExample: (q: string) => void;
}) {
  return (
    <div className="mx-auto max-w-3xl py-6">
      <div className="text-center">
        <div className="text-5xl">🦠</div>
        <h1 className="mt-3 text-2xl font-semibold">
          Welcome{name ? ` ${name}` : ""}!
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
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
      </div>

      <Card className="mt-6 p-5">
        <div className="flex items-center gap-2 font-medium">
          <Lightbulb className="h-4 w-4 text-primary" /> For best results
        </div>
        <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
          {TIPS.map((t) => (
            <li key={t} className="flex gap-2">
              <span className="text-primary">•</span> {t}
            </li>
          ))}
        </ul>
      </Card>

      <div className="mt-6">
        <div className="mb-2 flex items-center gap-2 text-sm font-medium">
          <Sparkles className="h-4 w-4 text-primary" /> Try an example
        </div>
        <div className="grid gap-2 sm:grid-cols-2">
          {EXAMPLES.map((q) => (
            <button
              key={q}
              onClick={() => onExample(q)}
              className="rounded-lg border border-border bg-card p-3 text-left text-sm transition-colors hover:bg-accent hover:text-accent-foreground"
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      <p className="mt-6 text-center text-xs text-muted-foreground">
        See the <span className="font-medium">Info</span> tab in the sidebar for the datasets and tools available.
      </p>
    </div>
  );
}
