"use client";

import { Cpu, Flag, GitFork, LayoutDashboard, Search } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { cn } from "@/util/cn";
import type { Overview } from "@/util/types";
import { useApi } from "@/util/use-api";

const NAV = [
  { href: "/", label: "Tree", icon: GitFork },
  { href: "/overview", label: "Overview", icon: LayoutDashboard },
  { href: "/runs", label: "Runs", icon: Flag },
  { href: "/policies", label: "Policies", icon: Cpu },
];

const TABS = [
  { href: "/", label: "Tree" },
  { href: "/evals", label: "Evals" },
  { href: "/wandb", label: "wandb" },
  { href: "/files", label: "Files" },
];

type SearchHit = {
  runs: { id: string; name: string; slug: string }[];
  checkpoints: { id: string; step: number; run_name: string; run_id: string }[];
};

type Note = { id: string; body: string; created_at: string };

function activeNav(pathname: string, href: string) {
  if (href === "/") return pathname === "/";
  return pathname === href || pathname.startsWith(`${href}/`);
}

export function Shell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const { data } = useApi<Overview>("/api/overview", 15000);
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit | null>(null);
  const [noteOpen, setNoteOpen] = useState(false);

  useEffect(() => {
    if (!query.trim()) {
      setHits(null);
      return;
    }
    const handle = setTimeout(() => {
      fetch(`/api/search?q=${encodeURIComponent(query.trim())}`)
        .then((response) => response.json())
        .then((body: SearchHit) => setHits(body))
        .catch(() => setHits(null));
    }, 200);
    return () => clearTimeout(handle);
  }, [query]);

  return (
    <div className="flex min-h-screen text-xs">
      <aside className="sticky top-0 flex h-screen w-44 shrink-0 flex-col border-r border-white/[0.06] bg-elevated">
        <div className="flex h-9 items-center gap-1.5 px-3">
          <span className="h-2.5 w-2.5 bg-gradient-to-br from-accent to-accent-2" />
          <span className="text-sm tracking-tight">Garage</span>
        </div>
        <nav className="flex flex-col">
          {NAV.map((item) => {
            const Icon = item.icon;
            const active = activeNav(pathname, item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-2 border-l-2 px-3 py-1.5",
                  active
                    ? "border-accent-2 bg-white/[0.04] text-text"
                    : "border-transparent text-muted hover:bg-white/[0.03] hover:text-text",
                )}
              >
                <Icon size={12} />
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="mt-auto border-t border-white/[0.06] px-3 py-2 text-[11px] text-muted">
          <div>Best {data?.best ? data.best.name : "—"}</div>
          <div className="tabular-nums text-text">
            {data?.best ? `${data.best.lap.toFixed(1)}s` : "—"}
          </div>
        </div>
      </aside>
      <div className="min-w-0 flex-1">
        <header className="sticky top-0 z-20 flex h-9 items-center gap-2 border-b border-white/[0.06] bg-elevated px-3">
          <div className="relative w-full max-w-md">
            <Search size={12} className="pointer-events-none absolute left-2 top-1.5 text-muted" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search"
              className="h-6 w-full border border-white/[0.08] bg-bg pl-6 pr-2 text-xs outline-none placeholder:text-muted"
            />
            {hits ? (
              <div className="absolute mt-1 w-full border border-white/[0.08] bg-card">
                {hits.runs.map((run) => (
                  <Link key={run.id} href={`/runs/${run.id}`} className="block px-2 py-1 hover:bg-white/5" onClick={() => setQuery("")}>
                    {run.name}
                  </Link>
                ))}
                {hits.checkpoints.map((ckpt) => (
                  <Link key={ckpt.id} href={`/runs/${ckpt.run_id}`} className="block px-2 py-1 text-muted hover:bg-white/5" onClick={() => setQuery("")}>
                    {ckpt.run_name} · {ckpt.step.toLocaleString()}
                  </Link>
                ))}
                {!hits.runs.length && !hits.checkpoints.length ? (
                  <div className="px-2 py-1 text-muted">No matches</div>
                ) : null}
              </div>
            ) : null}
          </div>
          <a href="/api/export" className="ml-auto">
            <Button variant="ghost">Export</Button>
          </a>
          <Button onClick={() => setNoteOpen(true)}>Note</Button>
        </header>
        <div className="flex gap-3 border-b border-white/[0.06] px-3">
          {TABS.map((tab) => {
            const active = pathname === tab.href;
            return (
              <Link
                key={tab.href}
                href={tab.href}
                className={cn(
                  "border-b-2 py-1.5",
                  active ? "border-accent-2 text-text" : "border-transparent text-muted hover:text-text",
                )}
              >
                {tab.label}
              </Link>
            );
          })}
        </div>
        <main className="px-3 py-2">{children}</main>
      </div>
      <NoteDialog open={noteOpen} onOpenChange={setNoteOpen} />
    </div>
  );
}

function NoteDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [body, setBody] = useState("");
  const [saved, setSaved] = useState<Note[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    fetch("/api/notes")
      .then((response) => response.json())
      .then((payload: { notes: Note[] }) => setSaved(payload.notes))
      .catch(() => setSaved([]));
  }, [open]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange} title="New note">
      <p className="mb-2 text-xs text-muted">A note only. This does not start training.</p>
      <textarea
        value={body}
        onChange={(event) => setBody(event.target.value)}
        className="h-24 w-full border border-white/[0.08] bg-bg p-2 text-xs outline-none"
        placeholder="What changed"
      />
      {error ? <p className="mt-1 text-xs text-red">{error}</p> : null}
      <Button
        className="mt-2"
        onClick={() => {
          setError(null);
          fetch("/api/notes", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ body }),
          })
            .then(async (response) => {
              if (!response.ok) throw new Error(await response.text());
              return response.json() as Promise<Note>;
            })
            .then((note) => {
              setSaved((current) => [note, ...current]);
              setBody("");
            })
            .catch(() => setError("Could not save the note."));
        }}
      >
        Save note
      </Button>
      <ul className="mt-2 flex flex-col gap-1">
        {saved.map((note) => (
          <li key={note.id} className="text-xs text-muted">{note.body}</li>
        ))}
      </ul>
    </Dialog>
  );
}
