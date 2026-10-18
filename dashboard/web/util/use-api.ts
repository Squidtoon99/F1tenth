"use client";

import { useEffect, useState } from "react";

export function useApi<T>(path: string, intervalMs = 0) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    const load = () => {
      fetch(path)
        .then((response) => {
          if (!response.ok) throw new Error(`${response.status}`);
          return response.json() as Promise<T>;
        })
        .then((body) => {
          if (live) {
            setData(body);
            setError(null);
          }
        })
        .catch((reason: unknown) => {
          if (live) setError(reason instanceof Error ? reason.message : "unavailable");
        });
    };
    load();
    if (intervalMs <= 0) {
      return () => {
        live = false;
      };
    }
    const handle = setInterval(load, intervalMs);
    return () => {
      live = false;
      clearInterval(handle);
    };
  }, [path, intervalMs]);

  return { data, error };
}
