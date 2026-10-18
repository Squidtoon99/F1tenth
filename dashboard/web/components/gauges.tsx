import { fmtMillions } from "@/util/format";

export function RoundGauge({ used, budget }: { used: number; budget: number }) {
  const pct = Math.max(0, Math.min(used / budget, 1));
  const radius = 72;
  const circ = 2 * Math.PI * radius;
  return (
    <svg viewBox="0 0 200 200" className="mx-auto h-52 w-52">
      <circle cx="100" cy="100" r={radius} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="12" />
      <circle
        cx="100"
        cy="100"
        r={radius}
        fill="none"
        stroke="#3AA0C8"
        strokeWidth="12"
        strokeLinecap="round"
        strokeDasharray={`${circ * pct} ${circ}`}
        transform="rotate(-90 100 100)"
      />
      <text x="100" y="96" textAnchor="middle" fill="#F4F4F5" fontSize="28" className="tabular-nums">
        {fmtMillions(used)}
      </text>
      <text x="100" y="118" textAnchor="middle" fill="#9A9A9E" fontSize="12">
        of {fmtMillions(budget)}
      </text>
    </svg>
  );
}

export function HalfGauge({ rate }: { rate: number | null }) {
  const pct = rate == null ? 0 : Math.max(0, Math.min(rate, 1));
  const radius = 78;
  const length = Math.PI * radius;
  const percent = rate == null ? "—" : `${(rate * 100).toFixed(1)}%`;
  return (
    <svg viewBox="0 0 200 120" className="mx-auto w-full max-w-xs">
      <path
        d="M20 100 A80 80 0 0 1 180 100"
        fill="none"
        stroke="rgba(255,255,255,0.08)"
        strokeWidth="14"
        strokeLinecap="round"
      />
      <path
        d="M20 100 A80 80 0 0 1 180 100"
        fill="none"
        stroke="#3AA0C8"
        strokeWidth="14"
        strokeLinecap="round"
        strokeDasharray={`${length * pct} ${length}`}
      />
      <text x="100" y="92" textAnchor="middle" fill="#F4F4F5" fontSize="32" className="tabular-nums">
        {percent}
      </text>
    </svg>
  );
}

export function ResourceBars({
  rows,
}: {
  rows: { name: string; fraction: number }[];
}) {
  return (
    <div className="flex flex-col gap-3">
      {rows.map((row) => (
        <div key={row.name}>
          <div className="mb-1 flex justify-between text-sm">
            <span>{row.name}</span>
            <span className="text-muted tabular-nums">{Math.round(row.fraction * 100)}%</span>
          </div>
          <div className="h-2 rounded-full bg-white/5">
            <div
              className="h-2 rounded-full bg-gradient-to-r from-accent to-accent-2"
              style={{ width: `${Math.round(row.fraction * 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
