"use client";

import { Card } from "@/components/ui/card";
import { useApi } from "@/util/use-api";

type Chart = { id: string; name: string; wandb_url: string | null; src: string };

export default function WandbPage() {
  const { data, error } = useApi<{ runs: Chart[] }>("/api/wandb");
  if (error) return <p className="text-sm text-muted">Garage API is not running.</p>;
  if (!data) return <p className="text-sm text-muted">Loading</p>;
  return (
    <div className="grid grid-cols-12 gap-4">
      {data.runs.map((run) => (
        <Card key={run.id} className="col-span-12 xl:col-span-6">
          <div className="text-lg">{run.name}</div>
          <img src={run.src} alt={run.name} className="mt-3 w-full rounded-2xl" />
          {run.wandb_url ? (
            <a className="mt-3 inline-block text-sm text-accent-2" href={run.wandb_url}>
              {run.wandb_url}
            </a>
          ) : null}
        </Card>
      ))}
    </div>
  );
}
