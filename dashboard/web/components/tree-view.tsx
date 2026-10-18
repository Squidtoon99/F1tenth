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

function NodeVideo({ node }: { node: TreeNode }) {
  if (node.video?.stream) {
    return (
      <video
        className="h-14 w-[88px] shrink-0 rounded-md bg-black object-cover"
        muted
        playsInline
        preload="metadata"
        src={node.video.stream}
      />
    );
  }
  return (
    <div className="flex h-14 w-[88px] shrink-0 items-center justify-center rounded-md border border-dashed border-white/10 text-[10px] text-muted">
      No clip
    </div>
  );
}

function NodeRow({ node, depth }: { node: TreeNode; depth: number }) {
  const live = node.live?.state === "running" || node.verdict === "running";
  return (
    <Link
      href={`/runs/${node.run_id}`}
      className="flex items-center gap-3 rounded-lg px-2 py-1.5 hover:bg-white/[0.04]"
      style={{ paddingLeft: 8 + depth * 18 }}
    >
      <NodeVideo node={node} />
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          <span className="truncate text-sm">{node.name}</span>
          <span className={`text-[10px] uppercase tracking-wide ${resultClass(node.verdict)}`}>
            {live ? "running" : node.verdict}
          </span>
        </div>
        <div className="truncate text-xs text-muted">{node.one_change}</div>
      </div>
      <div className="w-16 shrink-0 text-right text-sm tabular-nums">
        {node.lap == null ? "—" : `${trimNum(node.lap, 1)}s`}
      </div>
    </Link>
  );
}

export function TreeView() {
  const { data, error } = useApi<{ nodes: TreeNode[] }>("/api/tree", 12000);
  if (error) return <p className="text-sm text-muted">Garage API is not running.</p>;
  if (!data) return <p className="text-sm text-muted">Loading</p>;
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
    <div className="rounded-2xl border border-white/[0.06] bg-card py-1">
      {rows.map(({ node, depth }) => (
        <NodeRow key={node.id} node={node} depth={depth} />
      ))}
    </div>
  );
}
