"use client";

import { useEffect, useState } from "react";

export function useApi<T>(path: string) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    fetch(path)
      .then((response) => {
        if (!response.ok) throw new Error(`${response.status}`);
        return response.json() as Promise<T>;
      })
      .then((body) => {
        if (live) setData(body);
      })
      .catch((reason: unknown) => {
        if (live) setError(reason instanceof Error ? reason.message : "unavailable");
      });
    return () => {
      live = false;
    };
  }, [path]);

  return { data, error };
}
