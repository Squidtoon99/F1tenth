"use client";

import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { useApi } from "@/util/use-api";

type RaceModel = {
  id: string | null;
  run: string;
  step: number | null;
  name: string;
  path: string | null;
  kind: string | null;
  available: boolean;
};

const KIND_LABEL: Record<string, string> = {
  "gigaflow-gru": "Gigaflow GRU",
  "gigaflow-feedforward": "Gigaflow feedforward",
  "gigaflow-frame-stack": "Gigaflow frame stack",
  "cnn-gru": "CNN-GRU",
  "action-chunk": "Action chunk",
  tinylidarnet: "TinyLidarNet",
  "lidar-dreamer": "LiDAR Dreamer",
};

function kindLabel(kind: string | null) {
  if (!kind) return "Unread";
  return KIND_LABEL[kind] ?? kind;
}

export default function RacePage() {
  const models = useApi<{ models: RaceModel[]; viewer: boolean; cache: boolean }>(
    "/api/race/models",
  );
  const tracks = useApi<{ tracks: string[]; cache: boolean }>("/api/race/tracks");
  const [picked, setPicked] = useState<string[]>([]);
  const [track, setTrack] = useState("");
  const [status, setStatus] = useState<string | null>(null);

  const groups = useMemo(() => {
    const rows = models.data?.models ?? [];
    const order: string[] = [];
    const byKind = new Map<string, RaceModel[]>();
    for (const row of rows) {
      const key = row.kind ?? "unread";
      if (!byKind.has(key)) {
        byKind.set(key, []);
        order.push(key);
      }
      byKind.get(key)?.push(row);
    }
    return order.map((key) => ({ key, rows: byKind.get(key) ?? [] }));
  }, [models.data]);

  const trackValue = track || tracks.data?.tracks[0] || "";

  function toggle(path: string) {
    setPicked((current) => {
      if (current.includes(path)) return current.filter((item) => item !== path);
      if (current.length >= 8) return current;
      return [...current, path];
    });
  }

  async function race() {
    setStatus(null);
    const response = await fetch("/api/race/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        v: 4,
        type: "set_lineup",
        checkpoints: picked,
        track: trackValue,
      }),
    });
    if (!response.ok) {
      const body = (await response.json().catch(() => null)) as { error?: string } | null;
      setStatus(body?.error ?? `Race failed (${response.status})`);
      return;
    }
    setStatus(`${picked.length} model${picked.length === 1 ? "" : "s"} on ${trackValue}`);
  }

  return (
    <div className="-mx-3 -my-2 flex h-[calc(100vh-2.25rem)] flex-col">
      <div className="flex h-8 shrink-0 items-center gap-2 border-b border-white/[0.06] px-2">
        <label className="flex items-center gap-1 text-muted">
          Track
          <select
            value={trackValue}
            onChange={(event) => setTrack(event.target.value)}
            className="h-6 border border-white/[0.08] bg-bg px-1 text-xs text-text"
          >
            {(tracks.data?.tracks ?? []).map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <details className="relative">
          <summary className="cursor-pointer list-none border border-white/[0.08] px-1.5 py-0.5">
            Models ({picked.length})
          </summary>
          <div className="absolute left-0 z-20 mt-1 max-h-64 w-72 overflow-auto border border-white/[0.08] bg-bg p-1">
            {groups.map((group) => (
              <div key={group.key} className="mb-1">
                <div className="text-muted">
                  {kindLabel(group.key === "unread" ? null : group.key)}
                </div>
                {group.rows.map((row) => {
                  const disabled = !row.available || !row.path;
                  const checked = row.path ? picked.includes(row.path) : false;
                  return (
                    <label
                      key={`${row.run}-${row.name}-${row.step ?? "disk"}`}
                      className="flex items-start gap-1 py-0.5"
                    >
                      <input
                        type="checkbox"
                        className="mt-0.5"
                        disabled={disabled}
                        checked={checked}
                        onChange={() => row.path && toggle(row.path)}
                      />
                      <span>
                        <span className="block">{row.name}</span>
                        <span className="block text-[11px] text-muted">
                          {row.run}
                          {row.step != null ? ` · ${row.step.toLocaleString()}` : ""}
                          {disabled ? " · file missing" : ""}
                        </span>
                      </span>
                    </label>
                  );
                })}
              </div>
            ))}
            {models.error ? <p className="text-red">Garage API is not running.</p> : null}
            {!models.data && !models.error ? <p className="text-muted">Loading</p> : null}
          </div>
        </details>
        <Button onClick={() => void race()} disabled={picked.length === 0 || !trackValue}>
          Race
        </Button>
        {status ? <span className="text-muted">{status}</span> : null}
        {models.data && !models.data.viewer ? (
          <span className="text-red">Viewer build is missing.</span>
        ) : null}
        {tracks.data && !tracks.data.cache ? (
          <span className="text-muted">Track cache is not built yet.</span>
        ) : null}
      </div>
      <iframe
        title="Race"
        src="/api/race/ui/index.html?transport=http&embed=1"
        allowFullScreen
        className="min-h-0 w-full flex-1 border-0 bg-black"
      />
    </div>
  );
}
