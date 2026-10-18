"use client";

import dynamic from "next/dynamic";
import { useParams } from "next/navigation";

import type { RunPreview } from "@/util/types";
import { useApi } from "@/util/use-api";

const RunBody = dynamic(() => import("@/components/run-body"), {
  ssr: false,
  loading: () => <p className="mt-2 text-muted">Loading metrics…</p>,
});

function resultClass(verdict: string) {
  if (verdict === "pass") return "text-green";
  if (verdict === "fail") return "text-red";
  if (verdict === "running") return "text-accent-2";
  return "text-muted";
}

export default function RunPage() {
  const params = useParams<{ id: string }>();
  const { data, error } = useApi<RunPreview>(`/api/runs/${params.id}/preview`);
  if (error) return <p className="text-muted">Garage API is not running.</p>;
  const run = data?.run;
  const video = data?.videos?.[0];
  return (
    <div>
      <div className="mb-1 flex items-baseline gap-2">
        <h1 className="text-sm">{run?.name ?? "Run"}</h1>
        {run ? <span className={resultClass(run.verdict)}>{run.verdict}</span> : null}
        {run ? <span className="text-muted">{run.one_change}</span> : null}
      </div>
      {video?.stream ? (
        <video
          className="max-h-[70vh] w-full bg-black"
          controls
          playsInline
          preload="metadata"
          src={video.stream}
        />
      ) : (
        <div className="flex h-40 items-center justify-center border border-dashed border-white/10 text-muted">
          {data ? "No eval clip for this attempt." : "Loading clip…"}
        </div>
      )}
      {data ? <RunBody id={params.id} /> : null}
    </div>
  );
}
