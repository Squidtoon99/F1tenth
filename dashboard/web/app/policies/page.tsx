"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import { downloadCheckpoint } from "@/util/download";
import { fmtBytes } from "@/util/format";
import type { CheckpointRow } from "@/util/types";
import { useApi } from "@/util/use-api";

export default function PoliciesPage() {
  const { data, error } = useApi<{ checkpoints: CheckpointRow[] }>("/api/policies");
  const [message, setMessage] = useState<string | null>(null);
  if (error) return <p className="text-muted">Garage API is not running.</p>;
  if (!data) return <p className="text-muted">Loading</p>;
  return (
    <div>
      {message ? <p className="mb-1 text-red">{message}</p> : null}
      <table className="w-full text-left">
        <thead className="text-muted">
          <tr>
            {["Run", "Step", "Flags", "Size", "sha256", ""].map((label) => (
              <th key={label || "dl"} className="px-1.5 py-1 font-normal">{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.checkpoints.map((row) => (
            <tr key={row.id}>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5">{row.run_name}</td>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5 tabular-nums">{row.step.toLocaleString()}</td>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5">
                {row.is_best ? "best " : ""}
                {row.is_final ? "final" : ""}
              </td>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5 tabular-nums">{fmtBytes(row.bytes)}</td>
              <td className="max-w-[220px] truncate border-t border-white/[0.04] px-1.5 py-0.5 font-mono text-[11px] text-muted">
                {row.sha256 ?? "—"}
              </td>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5">
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
    </div>
  );
}
