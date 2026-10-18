"use client";

import {
  Clapperboard,
  Cpu,
  Flag,
  GitFork,
  LayoutDashboard,
  Search,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Dialog } from "@/components/ui/dialog";
import { cn } from "@/util/cn";
import { downloadCheckpoint } from "@/util/download";
import { fmtDelta, fmtStat, trimNum } from "@/util/format";
import type { Overview } from "@/util/types";
import { useApi } from "@/util/use-api";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/tree", label: "Tree", icon: GitFork },
  { href: "/runs", label: "Runs", icon: Flag },
  { href: "/videos", label: "Videos", icon: Clapperboard },
  { href: "/policies", label: "Policies", icon: Cpu },
];

const TABS = [
  { href: "/", label: "Overview" },
  { href: "/tree", label: "Tree" },
  { href: "/evals", label: "Evals" },
  { href: "/videos", label: "Videos" },
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
  const { data } = useApi<Overview>("/api/overview");
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
    <div className="flex min-h-screen">
      <aside className="sticky top-0 flex h-screen w-[240px] shrink-0 flex-col bg-elevated">
        <div className="flex h-16 items-center gap-2 px-5">
          <span className="h-3.5 w-3.5 rounded-[4px] bg-orange shadow-[0_0_16px_rgba(255,90,31,0.85)]" />
          <span className="text-lg tracking-tight">Garage</span>
        </div>
        <nav className="flex flex-col gap-1 px-3">
          {NAV.map((item) => {
            const Icon = item.icon;
            const active = activeNav(pathname, item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 rounded-full px-3 py-2 text-sm",
                  active ? "bg-orange text-white" : "text-muted hover:text-text",
                )}
              >
                <Icon size={16} />
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="mt-auto p-3">
          <Card hot>
            <div className="text-xs text-muted">Current best</div>
            <div className="mt-2 text-3xl tabular-nums">{data?.best ? `${trimNum(data.best.lap, 1)}s` : "—"}</div>
            <div className="mt-1 text-sm text-muted tabular-nums">
              {data?.best ? trimNum(data.best.completion) : "—"} completion
            </div>
            <div className="mt-1 truncate text-xs text-muted">{data?.best?.name}</div>
            <Button
              className="mt-4 w-full"
              disabled={!data?.best}
              onClick={() => {
                if (!data?.best) return;
                downloadCheckpoint(data.best.checkpoint_id, data.best.staged).catch((reason: unknown) => {
                  window.alert(reason instanceof Error ? reason.message : "Download failed");
                });
              }}
            >
              Download
            </Button>
          </Card>
        </div>
      </aside>
      <div className="min-w-0 flex-1">
        <header className="sticky top-0 z-20 flex items-center gap-3 bg-elevated/90 px-6 py-4 backdrop-blur">
          <div className="relative w-full max-w-xl">
            <Search size={16} className="pointer-events-none absolute left-3 top-3 text-muted" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search runs and checkpoints"
              className="h-10 w-full rounded-full border border-white/[0.06] bg-bg pl-9 pr-4 text-sm outline-none placeholder:text-muted"
            />
            {hits ? (
              <div className="absolute mt-2 w-full rounded-2xl border border-white/[0.06] bg-card p-2">
                {hits.runs.map((run) => (
                  <Link key={run.id} href={`/runs/${run.id}`} className="block rounded-xl px-3 py-2 text-sm hover:bg-white/5" onClick={() => setQuery("")}>
                    {run.name}
                  </Link>
                ))}
                {hits.checkpoints.map((ckpt) => (
                  <Link key={ckpt.id} href={`/runs/${ckpt.run_id}`} className="block rounded-xl px-3 py-2 text-sm text-muted hover:bg-white/5" onClick={() => setQuery("")}>
                    {ckpt.run_name} · {ckpt.step.toLocaleString()}
                  </Link>
                ))}
                {!hits.runs.length && !hits.checkpoints.length ? (
                  <div className="px-3 py-2 text-sm text-muted">No matches</div>
                ) : null}
              </div>
            ) : null}
          </div>
          <a href="/api/export" className="ml-auto">
            <Button variant="ghost">Export</Button>
          </a>
          <Button onClick={() => setNoteOpen(true)}>New note</Button>
        </header>
        <main className="px-6 pb-10">
          <section className="grid grid-cols-2 gap-4 xl:grid-cols-4">
            {(data?.stats ?? []).map((stat) => (
              <Card key={stat.label}>
                <div className="text-sm text-muted">{stat.label}</div>
                <div className="mt-3 text-4xl tabular-nums tracking-tight">{fmtStat(stat.value, stat.unit)}</div>
                <div className="mt-3 text-sm tabular-nums">
                  <span className={stat.delta != null && stat.delta > 0 ? "text-green" : stat.delta != null && stat.delta < 0 ? "text-red" : "text-muted"}>
                    {fmtDelta(stat.delta, stat.unit)}
                  </span>
                  {stat.versus ? <span className="ml-2 text-muted">vs {stat.versus}</span> : null}
                </div>
              </Card>
            ))}
          </section>
          <div className="mt-4 flex flex-wrap gap-2">
            {TABS.map((tab) => {
              const active = pathname === tab.href;
              return (
                <Link
                  key={tab.href}
                  href={tab.href}
                  className={cn(
                    "rounded-full px-4 py-2 text-sm",
                    active ? "bg-orange text-white" : "text-muted hover:text-text",
                  )}
                >
                  {tab.label}
                </Link>
              );
            })}
          </div>
          <div className="mt-4">{children}</div>
        </main>
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
      <p className="mb-3 text-sm text-muted">A note only. This does not start training.</p>
      <textarea
        value={body}
        onChange={(event) => setBody(event.target.value)}
        className="h-28 w-full rounded-2xl border border-white/[0.06] bg-bg p-3 text-sm outline-none"
        placeholder="What changed"
      />
      {error ? <p className="mt-2 text-sm text-red">{error}</p> : null}
      <Button
        className="mt-3"
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
      <ul className="mt-4 flex flex-col gap-2">
        {saved.map((note) => (
          <li key={note.id} className="text-sm text-muted">{note.body}</li>
        ))}
      </ul>
    </Dialog>
  );
}
