"use client";

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
  if (error) return <p className="text-muted">Garage API is not running.</p>;
  if (!data) return <p className="text-muted">Loading</p>;
  return (
    <table className="w-full text-left">
      <thead className="text-muted">
        <tr>
          {["Kind", "Name", "Run", "Size", "Key"].map((label) => (
            <th key={label} className="px-1.5 py-1 font-normal">{label}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.files.map((row) => (
          <tr key={`${row.kind}-${row.key}`}>
            <td className="border-t border-white/[0.04] px-1.5 py-0.5">{row.kind}</td>
            <td className="border-t border-white/[0.04] px-1.5 py-0.5">{row.name}</td>
            <td className="border-t border-white/[0.04] px-1.5 py-0.5">{row.run_name ?? "—"}</td>
            <td className="border-t border-white/[0.04] px-1.5 py-0.5 tabular-nums">{fmtBytes(row.bytes)}</td>
            <td className="border-t border-white/[0.04] px-1.5 py-0.5 font-mono text-[11px] text-muted">{row.key ?? "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
