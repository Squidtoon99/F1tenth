"use client";

import Link from "next/link";

import { Card } from "@/components/ui/card";
import { fmtMillions } from "@/util/format";
import type { RunRow } from "@/util/types";
import { useApi } from "@/util/use-api";

export default function RunsPage() {
  const { data, error } = useApi<{ runs: RunRow[] }>("/api/runs");
  if (error) return <p className="text-sm text-muted">Garage API is not running.</p>;
  if (!data) return <p className="text-sm text-muted">Loading</p>;
  return (
    <Card className="overflow-x-auto p-0">
      <table className="w-full text-left text-sm">
        <thead className="text-muted">
          <tr>
            {["Run", "Seed", "Plant", "Transitions", "Stop", "Verdict"].map((label) => (
              <th key={label} className="px-4 py-3 font-normal">{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.runs.map((run) => (
            <tr key={run.id} className="border-t border-white/[0.06]">
              <td className="px-4 py-3">
                <Link href={`/runs/${run.id}`} className="hover:text-accent-2">{run.name}</Link>
              </td>
              <td className="px-4 py-3 tabular-nums">{run.seed ?? "—"}</td>
              <td className="px-4 py-3">{run.plant ?? "—"}</td>
              <td className="px-4 py-3 tabular-nums">{fmtMillions(run.transitions)}</td>
              <td className={`px-4 py-3 ${run.stop_reason === "collapse" ? "text-red" : ""}`}>
                {run.stop_reason ?? "running"}
              </td>
              <td className="px-4 py-3">{run.verdict}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}
