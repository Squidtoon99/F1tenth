"use client";

import { useApi } from "@/util/use-api";

type Chart = { id: string; name: string; wandb_url: string | null; src: string };

export default function WandbPage() {
  const { data, error } = useApi<{ runs: Chart[] }>("/api/wandb");
  if (error) return <p className="text-muted">Garage API is not running.</p>;
  if (!data) return <p className="text-muted">Loading</p>;
  return (
    <table className="w-full text-left">
      <thead className="text-muted">
        <tr>
          <th className="px-1.5 py-1 font-normal">Run</th>
          <th className="px-1.5 py-1 font-normal">wandb</th>
        </tr>
      </thead>
      <tbody>
        {data.runs.map((run) => (
          <tr key={run.id}>
            <td className="border-t border-white/[0.04] px-1.5 py-0.5">{run.name}</td>
            <td className="border-t border-white/[0.04] px-1.5 py-0.5">
              {run.wandb_url ? (
                <a className="text-accent-2" href={run.wandb_url}>{run.wandb_url}</a>
              ) : "—"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
