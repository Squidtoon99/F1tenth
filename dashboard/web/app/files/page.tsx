"use client";

import { Card } from "@/components/ui/card";
import { fmtBytes } from "@/util/format";
import { useApi } from "@/util/use-api";

type FileRow = {
  kind: string;
  name: string;
  run_name: string | null;
  bytes: number | null;
  key: string | null;
  staged: boolean;
};

export default function FilesPage() {
  const { data, error } = useApi<{ files: FileRow[] }>("/api/files");
  if (error) return <p className="text-sm text-muted">Garage API is not running.</p>;
  if (!data) return <p className="text-sm text-muted">Loading</p>;
  return (
    <Card className="overflow-x-auto p-0">
      <table className="w-full text-left text-sm">
        <thead className="text-muted">
          <tr>
            {["Kind", "Name", "Run", "Size", "Key"].map((label) => (
              <th key={label} className="px-4 py-3 font-normal">{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.files.map((row) => (
            <tr key={`${row.kind}-${row.key}`} className="border-t border-white/[0.06]">
              <td className="px-4 py-3">{row.kind}</td>
              <td className="px-4 py-3">{row.name}</td>
              <td className="px-4 py-3">{row.run_name ?? "—"}</td>
              <td className="px-4 py-3 tabular-nums">{fmtBytes(row.bytes)}</td>
              <td className="px-4 py-3 font-mono text-xs text-muted">{row.key ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}
