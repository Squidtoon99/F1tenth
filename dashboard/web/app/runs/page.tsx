"use client";

import Link from "next/link";

import { fmtMillions } from "@/util/format";
import type { RunRow } from "@/util/types";
import { useApi } from "@/util/use-api";

export default function RunsPage() {
  const { data, error } = useApi<{ runs: RunRow[] }>("/api/runs");
  if (error) return <p className="text-muted">Garage API is not running.</p>;
  if (!data) return <p className="text-muted">Loading</p>;
  return (
    <table className="w-full text-left">
      <thead className="text-muted">
        <tr>
          {["Run", "Seed", "Plant", "Transitions", "Stop", "Result"].map((label) => (
            <th key={label} className="px-1.5 py-1 font-normal">{label}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.runs.map((run) => (
          <tr key={run.id}>
            <td className="border-t border-white/[0.04] px-1.5 py-0.5">
              <Link href={`/runs/${run.id}`} className="hover:text-accent-2">{run.name}</Link>
            </td>
            <td className="border-t border-white/[0.04] px-1.5 py-0.5 tabular-nums">{run.seed ?? "—"}</td>
            <td className="border-t border-white/[0.04] px-1.5 py-0.5">{run.plant ?? "—"}</td>
            <td className="border-t border-white/[0.04] px-1.5 py-0.5 tabular-nums">{fmtMillions(run.transitions)}</td>
            <td className={`border-t border-white/[0.04] px-1.5 py-0.5 ${run.stop_reason === "collapse" ? "text-red" : ""}`}>
              {run.stop_reason ?? "running"}
            </td>
            <td className="border-t border-white/[0.04] px-1.5 py-0.5">{run.verdict}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
