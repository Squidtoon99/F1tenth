"use client";

import type { ReactNode } from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X } from "lucide-react";

import { cn } from "@/util/cn";

export function Dialog({
  open,
  onOpenChange,
  title,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  children: ReactNode;
}) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/60" />
        <DialogPrimitive.Content
          className={cn(
            "fixed left-1/2 top-1/2 z-50 w-[min(480px,calc(100%-2rem))] -translate-x-1/2 -translate-y-1/2",
            "rounded-2xl border border-white/[0.06] bg-card p-5 shadow-[0_0_48px_rgba(255,90,31,0.28)]",
          )}
        >
          <div className="mb-4 flex items-center justify-between">
            <DialogPrimitive.Title className="text-lg">{title}</DialogPrimitive.Title>
            <DialogPrimitive.Close className="text-muted hover:text-text" aria-label="Close">
              <X size={16} />
            </DialogPrimitive.Close>
          </div>
          {children}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
