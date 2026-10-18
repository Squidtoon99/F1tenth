"use client";

import Link from "next/link";

import { trimNum } from "@/util/format";
import type { TreeNode } from "@/util/types";
import { useApi } from "@/util/use-api";

type TreePayload = {
  nodes: TreeNode[];
  live: string[];
  score_formula: string;
  score_note: string;
};

export default function OverviewPage() {
  const { data, error } = useApi<TreePayload>("/api/tree", 15000);
  if (error) return <p className="text-muted">Garage API is not running.</p>;
  if (!data) return <p className="text-muted">Loading</p>;
  const live = new Set(data.live);
  const rows = data.nodes.filter((node) => live.has(node.slug) || node.verdict === "running");
  return (
    <div>
      <table className="w-full text-left">
        <thead className="text-muted">
          <tr>
            {["Live run", "Result", "Score", "Lap", "Comp", "Crash", "Steps"].map((label) => (
              <th key={label} className="px-1.5 py-1 font-normal">{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((node) => (
            <tr key={node.id}>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5">
                <Link href={`/runs/${node.run_id}`} className="hover:text-accent-2">{node.name}</Link>
              </td>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5 text-accent-2">{node.verdict}</td>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5 tabular-nums">
                {node.score == null ? "—" : trimNum(node.score, 1)}
              </td>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5 tabular-nums">
                {node.lap == null ? "—" : `${trimNum(node.lap, 1)}s`}
              </td>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5 tabular-nums">
                {node.completion == null ? "—" : trimNum(node.completion)}
              </td>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5 tabular-nums">
                {node.crash == null ? "—" : trimNum(node.crash)}
              </td>
              <td className="border-t border-white/[0.04] px-1.5 py-0.5 tabular-nums">
                {node.live?.transitions == null ? "—" : Math.round(node.live.transitions).toLocaleString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-[11px] text-muted">
        score = {data.score_formula}. {data.score_note}
      </p>
    </div>
  );
}
