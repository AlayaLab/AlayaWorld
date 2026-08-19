"use client";

import { useEffect, useState } from "react";

import { cx } from "./ui";

export type ToastTone = "info" | "error";

interface Toast {
  id: number;
  tone: ToastTone;
  text: string;
}

type Listener = (toast: Toast) => void;

const listeners = new Set<Listener>();
let nextId = 1;

/** Show a short-lived message. Safe to call from anywhere in the tree. */
export function notify(tone: ToastTone, text: string): void {
  const toast = { id: nextId++, tone, text };
  for (const listener of listeners) listener(toast);
}

export function Toasts() {
  const [toasts, setToasts] = useState<Toast[]>([]);

  useEffect(() => {
    const listener: Listener = (toast) => {
      setToasts((current) => [...current, toast]);
      const lifetime = toast.tone === "error" ? 8000 : 4000;
      window.setTimeout(
        () => setToasts((current) => current.filter((t) => t.id !== toast.id)),
        lifetime,
      );
    };
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  }, []);

  if (toasts.length === 0) return null;

  return (
    <div className="pointer-events-none fixed bottom-4 left-1/2 z-50 flex w-[min(28rem,calc(100vw-2rem))] -translate-x-1/2 flex-col gap-2">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={cx(
            "rounded-md border px-3 py-2 text-xs shadow-lg backdrop-blur",
            toast.tone === "error"
              ? "border-alert/40 bg-alert/10 text-alert"
              : "border-edge bg-panel/90 text-ink-dim",
          )}
        >
          {toast.text}
        </div>
      ))}
    </div>
  );
}
