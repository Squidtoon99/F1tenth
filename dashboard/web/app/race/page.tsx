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
    <div className="grid grid-cols-[220px_minmax(0,1fr)] gap-2">
      <div>
        <label className="mb-2 block text-muted">
          Track
          <select
            value={trackValue}
            onChange={(event) => setTrack(event.target.value)}
            className="mt-1 h-6 w-full border border-white/[0.08] bg-bg px-1 text-xs"
          >
            {(tracks.data?.tracks ?? []).map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <div className="max-h-[calc(100vh-160px)] overflow-auto">
          {groups.map((group) => (
            <div key={group.key} className="mb-2">
              <div className="text-muted">{kindLabel(group.key === "unread" ? null : group.key)}</div>
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
        <Button onClick={() => void race()} disabled={picked.length === 0 || !trackValue}>
          Race
        </Button>
        {status ? <p className="mt-1 text-muted">{status}</p> : null}
        {models.data && !models.data.viewer ? (
          <p className="mt-1 text-red">Viewer build is missing.</p>
        ) : null}
        {tracks.data && !tracks.data.cache ? (
          <p className="mt-1 text-muted">Track cache is not built yet.</p>
        ) : null}
      </div>
      <iframe
        title="Race"
        src="/api/race/ui/index.html?transport=http&embed=1"
        className="h-[calc(100vh-88px)] w-full border border-white/[0.06] bg-black"
      />
    </div>
  );
}
