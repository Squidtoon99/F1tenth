"use client";

import { EvalTable } from "@/components/eval-table";
import { Card } from "@/components/ui/card";
import type { EvalRow } from "@/util/types";
import { useApi } from "@/util/use-api";

export default function EvalsPage() {
  const { data, error } = useApi<{ evals: EvalRow[] }>("/api/evals");
  if (error) return <p className="text-sm text-muted">Garage API is not running.</p>;
  if (!data) return <p className="text-sm text-muted">Loading</p>;
  return (
    <Card className="p-0">
      <EvalTable rows={data.evals} />
    </Card>
  );
}
