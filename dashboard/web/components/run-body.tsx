"use client";

import Link from "next/link";

import { EvalTable } from "@/components/eval-table";
import { AreaSeries } from "@/components/run-chart";
import { fmtMillions, trimNum } from "@/util/format";
import type { RunDetail } from "@/util/types";
import { useApi } from "@/util/use-api";

function resultClass(verdict: string) {
  if (verdict === "pass") return "text-green";
  if (verdict === "fail") return "text-red";
  if (verdict === "running") return "text-accent-2";
  return "text-muted";
}

export default function RunBody({ id }: { id: string }) {
  const { data, error } = useApi<RunDetail>(`/api/runs/${id}`, 12000);
  if (error) return <p className="mt-2 text-muted">Could not load run metrics.</p>;
  if (!data) return <p className="mt-2 text-muted">Loading metrics…</p>;
  const run = data.run;
  const extra = data.videos.slice(1);
  return (
    <div className="mt-3 flex flex-col gap-3">
      <div className="text-muted">
        seed {run.seed ?? "—"} · {run.plant ?? "—"} · {fmtMillions(run.transitions)} · score{" "}
        <span className="text-text tabular-nums">{run.score == null && data.score == null ? "—" : trimNum((data.score ?? run.score) as number, 1)}</span>
        {data.chart.wandb_url ? (
          <>
            {" · "}
            <a className="text-accent-2" href={data.chart.wandb_url}>wandb</a>
          </>
        ) : null}
      </div>
      {data.score_formula ? (
        <p className="text-[11px] text-muted">
          score = {data.score_formula}. {data.score_note}
        </p>
      ) : null}
      <table className="w-full text-left">
        <thead className="text-muted">
          <tr>
            <th className="px-1.5 py-1 font-normal">Side</th>
            <th className="px-1.5 py-1 font-normal">Channel</th>
            <th className="px-1.5 py-1 font-normal">N</th>
            <th className="px-1.5 py-1 font-normal">Note</th>
          </tr>
        </thead>
        <tbody>
          {data.model_io.inputs.map((item) => (
            <tr key={`in-${item.name}`}>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5 text-muted">in</td>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5">{item.name}</td>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5 tabular-nums">{item.size ?? "—"}</td>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5 text-muted">{item.note}</td>
            </tr>
          ))}
          {data.model_io.outputs.map((item) => (
            <tr key={`out-${item.name}`}>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5 text-muted">out</td>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5">{item.name}</td>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5 tabular-nums">{item.size ?? "—"}</td>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5 text-muted">{item.note}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <EvalTable rows={data.evals.map((row) => ({ ...row, run_name: run.name }))} />
      {extra.length ? (
        <div className="grid grid-cols-2 gap-2">
          {extra.map((video) => (
            <video key={video.id} className="w-full bg-black" controls preload="metadata" src={video.stream} />
          ))}
        </div>
      ) : null}
      <table className="w-full text-left">
        <thead className="text-muted">
          <tr>
            <th className="px-1.5 py-1 font-normal">#</th>
            <th className="px-1.5 py-1 font-normal">Ancestor</th>
            <th className="px-1.5 py-1 font-normal">Change</th>
            <th className="px-1.5 py-1 font-normal">Result</th>
          </tr>
        </thead>
        <tbody>
          {data.timeline.map((item, index) => (
            <tr key={item.slug}>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5 tabular-nums text-muted">{index + 1}</td>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5">
                <Link href={`/runs/${item.run_id}`} className="hover:text-accent-2">{item.name}</Link>
              </td>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5 text-muted">{item.one_change}</td>
              <td className={`border-t border-white/[0.04] px-1.5 py-0.5 ${resultClass(item.result)}`}>
                {item.result}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="h-48">
        {data.series.episode.length ? (
          <AreaSeries points={data.series.episode} xKey="step" yKey="seconds" unit="s" />
        ) : (
          <AreaSeries points={data.series.laps} xKey="step" yKey="lap" unit="s" />
        )}
      </div>
    </div>
  );
}
