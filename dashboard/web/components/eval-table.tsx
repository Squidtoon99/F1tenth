import { trimNum } from "@/util/format";
import type { EvalRow } from "@/util/types";

function cell(value: number | null) {
  return value == null ? "—" : trimNum(value);
}

export function EvalTable({ rows }: { rows: EvalRow[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="text-muted">
          <tr>
            {["Run", "Step", "Scenario", "Flying lap", "Completion", "Crash", "Overtakes", "Steer reversal", "Spawn"].map((label) => (
              <th key={label} className="px-3 py-2 font-normal">{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id} className="border-t border-white/[0.06]">
              <td className="px-3 py-3">{row.run_name ?? "—"}</td>
              <td className="px-3 py-3 tabular-nums">{row.step.toLocaleString()}</td>
              <td className="px-3 py-3">{row.scenario}</td>
              <td className="px-3 py-3 tabular-nums">{row.flying_lap_s == null ? "—" : `${trimNum(row.flying_lap_s)}s`}</td>
              <td className="px-3 py-3 tabular-nums">{cell(row.completion)}</td>
              <td className="px-3 py-3 tabular-nums">{cell(row.crash)}</td>
              <td className="px-3 py-3 tabular-nums">{cell(row.overtakes)}</td>
              <td className="px-3 py-3 tabular-nums">{cell(row.steer_reversal)}</td>
              <td className="px-3 py-3 tabular-nums">{row.spawn_mps == null ? "—" : `${trimNum(row.spawn_mps)} m/s`}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
