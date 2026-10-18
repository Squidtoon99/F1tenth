import type { ReactNode } from "react";

import { cn } from "@/util/cn";

export function Card({
  hot = false,
  className,
  children,
}: {
  hot?: boolean;
  className?: string;
  children: ReactNode;
}) {
  return (
    <div
      className={cn(
        "rounded-2xl border p-4",
        hot
          ? "border-[rgba(255,90,31,0.45)] bg-hot shadow-[0_0_48px_rgba(255,90,31,0.28)]"
          : "border-white/[0.06] bg-card",
        className,
      )}
    >
      {children}
    </div>
  );
}
