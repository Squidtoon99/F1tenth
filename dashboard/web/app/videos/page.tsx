"use client";

import { VideoCard } from "@/components/video-card";
import type { VideoRow } from "@/util/types";
import { useApi } from "@/util/use-api";

export default function VideosPage() {
  const { data, error } = useApi<{ videos: VideoRow[] }>("/api/videos");
  if (error) return <p className="text-sm text-muted">Garage API is not running.</p>;
  if (!data) return <p className="text-sm text-muted">Loading</p>;
  return (
    <div className="grid grid-cols-12 gap-4">
      {data.videos.map((video) => (
        <div key={video.id} className="col-span-12 md:col-span-6 xl:col-span-4">
          <VideoCard video={video} />
        </div>
      ))}
    </div>
  );
}
