"use client";

import { useState } from "react";

import { Card } from "@/components/ui/card";
import type { VideoRow } from "@/util/types";

export function VideoCard({ video }: { video: VideoRow }) {
  const [playing, setPlaying] = useState(false);
  const size = video.width && video.height ? `${video.width}×${video.height}` : null;
  return (
    <Card className="transition duration-200 hover:-translate-y-0.5">
      <button type="button" className="block w-full text-left" onClick={() => setPlaying(true)}>
        <div className="text-xs uppercase tracking-wide text-muted">{video.kind}</div>
        <div className="mt-2 text-lg">{video.run_name ?? "Clip"}</div>
        <div className="mt-1 text-sm text-muted tabular-nums">
          {video.duration_s == null ? "—" : `${video.duration_s}s`}
          {size ? ` · ${size}` : ""}
        </div>
      </button>
      {playing ? (
        <video className="mt-3 w-full rounded-xl bg-black" controls autoPlay src={video.stream} />
      ) : null}
    </Card>
  );
}
