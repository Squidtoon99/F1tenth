import { trimNum } from "@/util/format";
import type { EvalRow } from "@/util/types";

function cell(value: number | null) {
  return value == null ? "—" : trimNum(value);
}

export function EvalTable({ rows }: { rows: EvalRow[] }) {
  return (
    <table className="w-full text-left">
      <thead className="text-muted">
        <tr>
          {["Run", "Step", "Scenario", "Lap", "Comp", "Crash", "Ovt", "Rev", "Spawn"].map((label) => (
            <th key={label} className="px-1.5 py-1 font-normal">{label}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.id}>
            <td className="border-t border-white/[0.04] px-1.5 py-0.5">{row.run_name ?? "—"}</td>
            <td className="border-t border-white/[0.04] px-1.5 py-0.5 tabular-nums">{row.step.toLocaleString()}</td>
            <td className="border-t border-white/[0.04] px-1.5 py-0.5">{row.scenario}</td>
            <td className="border-t border-white/[0.04] px-1.5 py-0.5 tabular-nums">
              {row.flying_lap_s == null ? "—" : `${trimNum(row.flying_lap_s, 1)}s`}
            </td>
            <td className="border-t border-white/[0.04] px-1.5 py-0.5 tabular-nums">{cell(row.completion)}</td>
            <td className="border-t border-white/[0.04] px-1.5 py-0.5 tabular-nums">{cell(row.crash)}</td>
            <td className="border-t border-white/[0.04] px-1.5 py-0.5 tabular-nums">{cell(row.overtakes)}</td>
            <td className="border-t border-white/[0.04] px-1.5 py-0.5 tabular-nums">{cell(row.steer_reversal)}</td>
            <td className="border-t border-white/[0.04] px-1.5 py-0.5 tabular-nums">
              {row.spawn_mps == null ? "—" : `${trimNum(row.spawn_mps)}`}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
