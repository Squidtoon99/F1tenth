"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Point = { label?: string; step?: number; lap?: number; seconds?: number };

const tooltipStyle = {
  background: "#161618",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 12,
  color: "#F4F4F5",
};

export function AreaSeries({
  points,
  xKey,
  yKey,
  unit = "",
}: {
  points: Point[];
  xKey: "label" | "step";
  yKey: "lap" | "seconds";
  unit?: string;
}) {
  if (!points.length) {
    return <p className="text-sm text-muted">No scored points yet.</p>;
  }
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={points} margin={{ top: 12, right: 8, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="garage-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#6EE7F0" stopOpacity={0.55} />
            <stop offset="100%" stopColor="#3AA0C8" stopOpacity={0.08} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
        <XAxis dataKey={xKey} hide={xKey === "label"} tick={{ fill: "#9A9A9E", fontSize: 11 }} />
        <YAxis
          tick={{ fill: "#9A9A9E", fontSize: 11 }}
          width={40}
          tickFormatter={(value: number) => `${value}${unit}`}
        />
        <Tooltip
          contentStyle={tooltipStyle}
          formatter={(value) => [`${value}${unit}`, yKey]}
        />
        <Area
          type="monotone"
          dataKey={yKey}
          stroke="#6EE7F0"
          strokeWidth={2}
          fill="url(#garage-fill)"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
