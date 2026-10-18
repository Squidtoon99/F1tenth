"use client";

import Link from "next/link";

import { Card } from "@/components/ui/card";
import { trimNum } from "@/util/format";
import type { TreeNode } from "@/util/types";
import { useApi } from "@/util/use-api";

const NODE_W = 220;
const NODE_H = 148;
const GAP_X = 72;
const GAP_Y = 28;

function verdictClass(verdict: string) {
  if (verdict === "collapsed" || verdict === "revert") return "text-red";
  return "text-muted";
}

export default function TreePage() {
  const { data, error } = useApi<{ nodes: TreeNode[] }>("/api/tree");
  if (error) return <p className="text-sm text-muted">Garage API is not running.</p>;
  if (!data) return <p className="text-sm text-muted">Loading</p>;
  const nodes = data.nodes;
  const bestLap = Math.min(...nodes.map((node) => node.lap ?? Number.POSITIVE_INFINITY));
  const children = new Map<string, TreeNode[]>();
  nodes.forEach((node) => {
    const key = node.parent_id ?? "root";
    children.set(key, [...(children.get(key) ?? []), node]);
  });
  const measure = (id: string): number => {
    const kids = children.get(id) ?? [];
    if (!kids.length) return NODE_H;
    return kids.reduce((sum, kid) => sum + measure(kid.id), 0) + (kids.length - 1) * GAP_Y;
  };
  const positions = new Map<string, { x: number; y: number }>();
  const place = (id: string, depth: number, top: number) => {
    const kids = children.get(id) ?? [];
    const x = depth * (NODE_W + GAP_X);
    if (!kids.length) {
      positions.set(id, { x, y: top });
      return;
    }
    let cursor = top;
    kids.forEach((kid) => {
      const height = measure(kid.id);
      place(kid.id, depth + 1, cursor);
      cursor += height + GAP_Y;
    });
    const first = positions.get(kids[0].id)!;
    const last = positions.get(kids[kids.length - 1].id)!;
    positions.set(id, { x, y: (first.y + last.y) / 2 });
  };
  let top = 0;
  (children.get("root") ?? []).forEach((root) => {
    const height = measure(root.id);
    place(root.id, 0, top);
    top += height + GAP_Y;
  });
  const width = Math.max(...[...positions.values()].map((pos) => pos.x), 0) + NODE_W;
  const height = Math.max(...[...positions.values()].map((pos) => pos.y), 0) + NODE_H;

  return (
    <div className="overflow-x-auto">
      <div className="relative" style={{ width, height }}>
        <svg className="absolute inset-0 h-full w-full">
          {nodes.map((node) => {
            if (!node.parent_id) return null;
            const parent = positions.get(node.parent_id);
            const child = positions.get(node.id);
            if (!parent || !child) return null;
            const x1 = parent.x + NODE_W;
            const y1 = parent.y + NODE_H / 2;
            const x2 = child.x;
            const y2 = child.y + NODE_H / 2;
            const mid = (x1 + x2) / 2;
            return (
              <path
                key={node.id}
                d={`M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`}
                fill="none"
                stroke="rgba(255,90,31,0.45)"
                strokeWidth="1.5"
              />
            );
          })}
        </svg>
        {nodes.map((node) => {
          const pos = positions.get(node.id);
          if (!pos) return null;
          const hot = node.lap != null && node.lap === bestLap && node.verdict === "keep";
          return (
            <Link
              key={node.id}
              href={`/runs/${node.run_id}`}
              className="absolute block"
              style={{ left: pos.x, top: pos.y, width: NODE_W }}
            >
              <Card hot={hot} className="h-[148px]">
                <div className="truncate text-base">{node.name}</div>
                <div className="mt-1 line-clamp-2 text-sm text-muted">{node.one_change}</div>
                <div className={`mt-2 text-xs uppercase tracking-wide ${verdictClass(node.verdict)}`}>
                  {node.verdict}
                </div>
                <div className="mt-1 text-2xl tabular-nums">
                  {node.lap == null ? "—" : `${trimNum(node.lap)}s`}
                </div>
              </Card>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
