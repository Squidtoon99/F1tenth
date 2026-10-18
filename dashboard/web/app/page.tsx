"use client";

import { EpisodeMarks, LapChart } from "@/components/lap-chart";
import { HalfGauge, ResourceBars, RoundGauge } from "@/components/gauges";
import { Card } from "@/components/ui/card";
import type { Overview } from "@/util/types";
import { useApi } from "@/util/use-api";

export default function OverviewPage() {
  const { data, error } = useApi<Overview>("/api/overview");
  if (error) return <p className="text-sm text-muted">Garage API is not running.</p>;
  if (!data) return <p className="text-sm text-muted">Loading</p>;
  return (
    <div className="grid grid-cols-12 gap-4">
      <Card className="col-span-12 min-h-[420px] xl:col-span-6">
        <div className="text-sm text-muted">Lap time</div>
        <div className="mt-1 text-lg">Over attempts</div>
        <div className="mt-4 h-[320px]">
          <LapChart points={data.laps} />
        </div>
      </Card>
      <div className="col-span-12 flex flex-col gap-4 md:col-span-6 xl:col-span-3">
        <Card>
          <div className="text-sm text-muted">Episode length</div>
          <div className="mt-1 text-lg">{data.episode?.run_name ?? "—"}</div>
          <div className="mt-4 max-h-64 overflow-auto">
            {data.episode ? <EpisodeMarks marks={data.episode.marks} /> : null}
          </div>
        </Card>
        <Card>
          <div className="mb-3 text-sm text-muted">Scenarios</div>
          <ResourceBars rows={data.scenarios} />
        </Card>
      </div>
      <div className="col-span-12 flex flex-col gap-4 md:col-span-6 xl:col-span-3">
        <Card>
          <div className="text-sm text-muted">Transitions</div>
          <div className="mt-1 truncate text-lg">{data.gauge?.label ?? "—"}</div>
          {data.gauge ? <RoundGauge used={data.gauge.used} budget={data.gauge.budget} /> : null}
        </Card>
        <Card>
          <div className="text-sm text-muted">Crash rate</div>
          <div className="mt-1 truncate text-lg">{data.crash.label ?? "—"}</div>
          <HalfGauge rate={data.crash.rate} />
        </Card>
      </div>
    </div>
  );
}
