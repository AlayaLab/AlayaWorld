"use client";

import type { DemoConfig } from "@/lib/config";

import { Button, Dot, Pill } from "./ui";

/** How the SDK's connection status reads in the UI. */
const STATUS_LABEL: Record<string, string> = {
  disconnected: "Not connected",
  connecting: "Connecting",
  waiting: "Waiting for a worker",
  ready: "Live",
  error: "Connection failed",
};

export function Header({
  config,
  status,
  onConnect,
  onDisconnect,
}: {
  config: DemoConfig;
  status: string;
  onConnect: () => void;
  onDisconnect: () => void;
}) {
  const busy = status === "connecting" || status === "waiting";
  const local = config.mode === "local";

  return (
    <header className="flex shrink-0 flex-wrap items-center gap-3 border-b border-edge px-4 py-3">
      <h1 className="text-sm font-semibold tracking-tight">AlayaWorld</h1>

      <Pill
        title={
          local
            ? "Local mode ignores the model name: the container you started is the only model there is."
            : "The session is created for this model name. It has to match the name you deployed the model under."
        }
      >
        <span className="text-ink-faint">model</span>
        <code className="text-ink-dim">{config.modelName}</code>
      </Pill>

      <Pill tone={local ? "neutral" : "live"}>
        {local ? `local · ${config.apiUrl}` : "hosted · API key"}
      </Pill>

      <div className="ml-auto flex items-center gap-3">
        <span className="flex items-center gap-2 text-xs text-ink-dim">
          <Dot
            tone={
              status === "ready"
                ? "live"
                : status === "error"
                  ? "alert"
                  : busy
                    ? "pending"
                    : "idle"
            }
            pulse={busy}
          />
          {STATUS_LABEL[status] ?? status}
        </span>
        {status === "ready" ? (
          <Button onClick={onDisconnect}>Disconnect</Button>
        ) : (
          <Button tone="primary" onClick={onConnect} disabled={busy}>
            {busy ? "Connecting…" : "Connect"}
          </Button>
        )}
      </div>
    </header>
  );
}
