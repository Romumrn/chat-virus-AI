/** Minimal className joiner (shadcn uses clsx+tailwind-merge; this is enough). */
export function cn(...classes: (string | false | null | undefined)[]): string {
  return classes.filter(Boolean).join(" ");
}
