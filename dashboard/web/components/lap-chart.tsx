export function LapChart({
  points,
}: {
  points: { label: string; lap: number }[];
}) {
  if (!points.length) {
    return <p className="text-sm text-muted">No scored attempts yet.</p>;
  }
  const width = 640;
  const height = 320;
  const pad = { l: 44, r: 12, t: 28, b: 28 };
  const min = Math.min(...points.map((point) => point.lap));
  const max = Math.max(...points.map((point) => point.lap));
  const y0 = min * 0.85;
  const y1 = max * 1.08;
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  const coords = points.map((point, index) => {
    const x = pad.l + (points.length === 1 ? innerW / 2 : (index / (points.length - 1)) * innerW);
    const y = pad.t + (1 - (point.lap - y0) / (y1 - y0)) * innerH;
    return { ...point, x, y };
  });
  const line = coords.map((point) => `${point.x},${point.y}`).join(" ");
  const area = `${pad.l},${height - pad.b} ${line} ${coords[coords.length - 1].x},${height - pad.b}`;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-full w-full">
      <defs>
        <linearGradient id="lap-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#FF8A3D" />
          <stop offset="100%" stopColor="#FF5A1F" stopOpacity="0.4" />
        </linearGradient>
      </defs>
      <polygon points={area} fill="url(#lap-fill)" />
      <polyline points={line} fill="none" stroke="#FF5A1F" strokeWidth="3" />
      {coords.map((point) => (
        <g key={point.label}>
          <circle cx={point.x} cy={point.y} r="4" fill="#FF8A3D" />
          <text
            x={point.x}
            y={point.y - 10}
            textAnchor="middle"
            fill="#F4F4F5"
            fontSize="12"
            className="tabular-nums"
          >
            {point.lap.toFixed(1)}
          </text>
        </g>
      ))}
    </svg>
  );
}

export function EpisodeMarks({
  marks,
}: {
  marks: { step: number; seconds: number }[];
}) {
  const max = Math.max(...marks.map((mark) => mark.seconds), 1);
  return (
    <div className="flex flex-col gap-1.5">
      {marks.map((mark) => (
        <div key={mark.step} className="flex items-center gap-3">
          <span className="w-10 shrink-0 text-xs text-muted tabular-nums">
            {Math.round(mark.step / 1_000_000)}M
          </span>
          <div
            className="h-[3px] rounded-full bg-orange"
            style={{ width: `${Math.max(8, (mark.seconds / max) * 100)}%` }}
          />
        </div>
      ))}
    </div>
  );
}
