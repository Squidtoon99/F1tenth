"use client";

import Link from "next/link";

import { trimNum } from "@/util/format";
import type { TreeNode } from "@/util/types";
import { useApi } from "@/util/use-api";

function resultClass(verdict: string) {
  if (verdict === "pass") return "text-green";
  if (verdict === "fail") return "text-red";
  if (verdict === "running") return "text-accent-2";
  return "text-muted";
}

function cell(value: number | null | undefined, digits = 3) {
  return value == null ? "—" : trimNum(value, digits);
}

type TreePayload = {
  nodes: TreeNode[];
  score_formula: string;
  score_note: string;
};

export function TreeView() {
  const { data, error } = useApi<TreePayload>("/api/tree", 12000);
  if (error) return <p className="text-muted">Garage API is not running.</p>;
  if (!data) return <p className="text-muted">Loading</p>;
  const children = new Map<string, TreeNode[]>();
  data.nodes.forEach((node) => {
    const key = node.parent_id ?? "root";
    children.set(key, [...(children.get(key) ?? []), node]);
  });
  const rows: { node: TreeNode; depth: number }[] = [];
  const walk = (id: string, depth: number) => {
    (children.get(id) ?? []).forEach((child) => {
      rows.push({ node: child, depth });
      walk(child.id, depth + 1);
    });
  };
  walk("root", 0);
  return (
    <div>
      <table className="w-full border-collapse text-left">
        <thead className="sticky top-9 bg-elevated text-muted">
          <tr>
            {["Attempt", "Result", "Score", "Lap", "Comp", "Crash", "Ovt", "Rev", "Clip"].map((label) => (
              <th key={label} className="border-b border-white/[0.08] px-1.5 py-1 font-normal">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ node, depth }) => {
            const live = node.live?.state === "running" || node.verdict === "running";
            return (
              <tr key={node.id} className="hover:bg-white/[0.03]">
                <td className="border-b border-white/[0.04] px-1.5 py-0.5">
                  <Link href={`/runs/${node.run_id}`} className="block hover:text-accent-2">
                    <span style={{ paddingLeft: depth * 12 }}>{node.name}</span>
                    <span className="ml-2 text-muted">{node.one_change}</span>
                  </Link>
                </td>
                <td className={`border-b border-white/[0.04] px-1.5 py-0.5 ${resultClass(node.verdict)}`}>
                  {live ? "running" : node.verdict}
                </td>
                <td className="border-b border-white/[0.04] px-1.5 py-0.5 tabular-nums">
                  {cell(node.score, 1)}
                </td>
                <td className="border-b border-white/[0.04] px-1.5 py-0.5 tabular-nums">
                  {node.lap == null ? "—" : `${trimNum(node.lap, 1)}s`}
                </td>
                <td className="border-b border-white/[0.04] px-1.5 py-0.5 tabular-nums">{cell(node.completion)}</td>
                <td className="border-b border-white/[0.04] px-1.5 py-0.5 tabular-nums">{cell(node.crash)}</td>
                <td className="border-b border-white/[0.04] px-1.5 py-0.5 tabular-nums">{cell(node.overtakes)}</td>
                <td className="border-b border-white/[0.04] px-1.5 py-0.5 tabular-nums">{cell(node.steer_reversal)}</td>
                <td className="border-b border-white/[0.04] px-1.5 py-0.5 text-muted">
                  {node.video?.stream ? "yes" : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-2 text-[11px] text-muted">
        score = {data.score_formula}. {data.score_note}
      </p>
    </div>
  );
}
