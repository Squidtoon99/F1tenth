"use client";

import Link from "next/link";
import { useParams } from "next/navigation";

import { EvalTable } from "@/components/eval-table";
import { AreaSeries } from "@/components/run-chart";
import { Card } from "@/components/ui/card";
import { VideoCard } from "@/components/video-card";
import { fmtMillions } from "@/util/format";
import type { RunDetail } from "@/util/types";
import { useApi } from "@/util/use-api";

function resultClass(verdict: string) {
  if (verdict === "pass") return "text-green";
  if (verdict === "fail") return "text-red";
  if (verdict === "running") return "text-accent-2";
  return "text-muted";
}

export default function RunPage() {
  const params = useParams<{ id: string }>();
  const { data, error } = useApi<RunDetail>(`/api/runs/${params.id}`, 12000);
  if (error) return <p className="text-sm text-muted">Garage API is not running.</p>;
  if (!data) return <p className="text-sm text-muted">Loading</p>;
  const run = data.run;
  return (
    <div className="flex flex-col gap-4">
      <Card>
        <div className="text-3xl tracking-tight">{run.name}</div>
        <div className="mt-2 text-sm text-muted">
          seed {run.seed ?? "—"} · {run.plant ?? "—"} · {fmtMillions(run.transitions)} transitions ·{" "}
          <span className={resultClass(run.verdict)}>{run.verdict}</span>
        </div>
        <div className="mt-2 text-sm text-muted">{run.one_change}</div>
        {data.chart.wandb_url ? (
          <a className="mt-3 inline-block text-sm text-accent-2" href={data.chart.wandb_url}>
            {data.chart.wandb_url}
          </a>
        ) : null}
      </Card>
      <div className="grid grid-cols-12 gap-4">
        <Card className="col-span-12 md:col-span-6">
          <div className="text-sm text-muted">Model inputs</div>
          <div className="mt-1 text-lg">{data.model_io.kind}</div>
          <ul className="mt-3 flex flex-col gap-2 text-sm">
            {data.model_io.inputs.map((item) => (
              <li key={item.name} className="flex justify-between gap-4">
                <span>{item.name}</span>
                <span className="text-muted">{item.size ?? "—"} · {item.note}</span>
              </li>
            ))}
          </ul>
        </Card>
        <Card className="col-span-12 md:col-span-6">
          <div className="text-sm text-muted">Model outputs</div>
          <div className="mt-1 text-lg">{data.model_io.action_dim ?? "—"}-d action</div>
          <ul className="mt-3 flex flex-col gap-2 text-sm">
            {data.model_io.outputs.map((item) => (
              <li key={item.name} className="flex justify-between gap-4">
                <span>{item.name}</span>
                <span className="text-muted">{item.size ?? "—"} · {item.note}</span>
              </li>
            ))}
          </ul>
        </Card>
      </div>
      <Card className="p-0">
        <div className="px-4 pt-4 text-sm text-muted">Eval metrics</div>
        <EvalTable rows={data.evals.map((row) => ({ ...row, run_name: run.name }))} />
      </Card>
      <div>
        <div className="mb-2 text-sm text-muted">Videos</div>
        {data.videos.length ? (
          <div className="grid grid-cols-12 gap-4">
            {data.videos.map((video) => (
              <div key={video.id} className="col-span-12 md:col-span-4">
                <VideoCard video={video} />
              </div>
            ))}
          </div>
        ) : (
          <Card>
            <p className="text-sm text-muted">No eval clip for this attempt.</p>
          </Card>
        )}
      </div>
      <Card>
        <div className="text-sm text-muted">Ancestor path</div>
        <ol className="mt-3 flex flex-col gap-3">
          {data.timeline.map((item, index) => (
            <li key={item.slug} className="flex items-start gap-3">
              <span className="mt-0.5 w-5 shrink-0 text-xs text-muted tabular-nums">{index + 1}</span>
              <div className="min-w-0">
                <Link href={`/runs/${item.run_id}`} className="text-sm hover:text-accent-2">
                  {item.name}
                </Link>
                <div className="text-xs text-muted">{item.one_change}</div>
              </div>
              <span className={`ml-auto text-xs uppercase ${resultClass(item.result)}`}>
                {item.result}
              </span>
            </li>
          ))}
        </ol>
      </Card>
      <Card className="min-h-[280px]">
        <div className="text-sm text-muted">Score</div>
        <div className="mt-4 h-56">
          {data.series.episode.length ? (
            <AreaSeries points={data.series.episode} xKey="step" yKey="seconds" unit="s" />
          ) : (
            <AreaSeries points={data.series.laps} xKey="step" yKey="lap" unit="s" />
          )}
        </div>
      </Card>
    </div>
  );
}
