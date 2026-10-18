"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { downloadCheckpoint } from "@/util/download";
import { fmtBytes } from "@/util/format";
import type { CheckpointRow } from "@/util/types";
import { useApi } from "@/util/use-api";

export default function PoliciesPage() {
  const { data, error } = useApi<{ checkpoints: CheckpointRow[] }>("/api/policies");
  const [message, setMessage] = useState<string | null>(null);
  if (error) return <p className="text-sm text-muted">Garage API is not running.</p>;
  if (!data) return <p className="text-sm text-muted">Loading</p>;
  return (
    <Card className="overflow-x-auto p-0">
      {message ? <p className="px-4 pt-4 text-sm text-red">{message}</p> : null}
      <table className="w-full text-left text-sm">
        <thead className="text-muted">
          <tr>
            {["Run", "Step", "Flags", "Size", "sha256", ""].map((label) => (
              <th key={label || "download"} className="px-4 py-3 font-normal">{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.checkpoints.map((row) => (
            <tr key={row.id} className="border-t border-white/[0.06]">
              <td className="px-4 py-3">{row.run_name}</td>
              <td className="px-4 py-3 tabular-nums">{row.step.toLocaleString()}</td>
              <td className="px-4 py-3">
                {row.is_best ? <span className="mr-2 rounded-full border border-orange/40 px-2 py-0.5 text-xs text-orange">Best</span> : null}
                {row.is_final ? <span className="rounded-full border border-white/10 px-2 py-0.5 text-xs text-muted">Final</span> : null}
              </td>
              <td className="px-4 py-3 tabular-nums">{fmtBytes(row.bytes)}</td>
              <td className="max-w-[240px] truncate px-4 py-3 font-mono text-xs text-muted">{row.sha256 ?? "—"}</td>
              <td className="px-4 py-3">
                <Button
                  onClick={() => {
                    setMessage(null);
                    downloadCheckpoint(row.id, row.staged).catch((reason: unknown) => {
                      setMessage(reason instanceof Error ? reason.message : "Download failed");
                    });
                  }}
                >
                  Download
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}
