"use client";

import { useParams } from "next/navigation";

import { EvalTable } from "@/components/eval-table";
import { Card } from "@/components/ui/card";
import { VideoCard } from "@/components/video-card";
import { fmtMillions } from "@/util/format";
import type { RunDetail } from "@/util/types";
import { useApi } from "@/util/use-api";

export default function RunPage() {
  const params = useParams<{ id: string }>();
  const { data, error } = useApi<RunDetail>(`/api/runs/${params.id}`);
  if (error) return <p className="text-sm text-muted">Garage API is not running.</p>;
  if (!data) return <p className="text-sm text-muted">Loading</p>;
  const run = data.run;
  return (
    <div className="flex flex-col gap-4">
      <Card>
        <div className="text-3xl tracking-tight">{run.name}</div>
        <div className="mt-2 text-sm text-muted">
          seed {run.seed ?? "—"} · {run.plant ?? "—"} · {fmtMillions(run.transitions)} transitions ·{" "}
          <span className={run.stop_reason === "collapse" ? "text-red" : ""}>{run.stop_reason ?? "running"}</span>
        </div>
        {data.chart.wandb_url ? (
          <a className="mt-3 inline-block text-sm text-orange" href={data.chart.wandb_url}>
            {data.chart.wandb_url}
          </a>
        ) : null}
      </Card>
      <Card className="p-0">
        <div className="px-4 pt-4 text-sm text-muted">Evals</div>
        <EvalTable rows={data.evals.map((row) => ({ ...row, run_name: run.name }))} />
      </Card>
      <div className="grid grid-cols-12 gap-4">
        {data.videos.map((video) => (
          <div key={video.id} className="col-span-12 md:col-span-4">
            <VideoCard video={video} />
          </div>
        ))}
      </div>
      <Card>
        <div className="text-sm text-muted">wandb</div>
        <img src={data.chart.src} alt={data.chart.title} className="mt-3 w-full rounded-2xl" />
        {data.chart.wandb_url ? (
          <a className="mt-3 inline-block text-sm text-orange" href={data.chart.wandb_url}>
            Open the run
          </a>
        ) : null}
      </Card>
    </div>
  );
}
