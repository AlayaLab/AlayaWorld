"use client";

import { useReactorMessage } from "@reactor-team/js-sdk";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  useAlayaWorld,
  useAlayaWorldImageSelected,
  useAlayaWorldRolloutResetQueued,
  useAlayaWorldStateUpdate,
} from "@/lib/alayaworld/client.react";
import type { DemoConfig } from "@/lib/config";
import type { WorldState } from "@/lib/model";

import { CameraBar } from "./CameraBar";
import { Header } from "./Header";
import { RolloutPanel } from "./RolloutPanel";
import { ScenePanel } from "./ScenePanel";
import { Stage } from "./Stage";
import { Steps } from "./Steps";
import { notify } from "./Toasts";

/**
 * Put a readable sentence in front of the failures a reader is likely to hit.
 *
 * A container serves one session at a time and holds a closed one open for about
 * a minute in case the client comes back, so leaving the page without
 * disconnecting is the most common reason a reconnect is refused.
 */
function explain(message: string): string {
  if (message.includes("orphaned")) {
    return "An earlier session is still closing. The model accepts one session at a time and releases a dropped one after about a minute — try again shortly, and use Disconnect to leave straight away.";
  }
  return message;
}

export function Workspace({ config }: { config: DemoConfig }) {
  const model = useAlayaWorld();
  const { status, lastError } = model;
  const [world, setWorld] = useState<WorldState | null>(null);

  // The model broadcasts a complete snapshot whenever anything observable
  // changes, and sends one to every viewer as it joins. Treating that snapshot
  // as the only source of truth keeps the UI honest about what the model will
  // actually do next, rather than guessing from the commands this tab sent.
  useAlayaWorldStateUpdate(setWorld);

  useAlayaWorldImageSelected((message) => {
    notify(
      "info",
      message.source === "built_in"
        ? `Loaded built-in scene ${message.filename}`
        : `Loaded ${message.filename}`,
    );
  });

  useAlayaWorldRolloutResetQueued((message) => {
    if (message.trigger === "automatic_chunk_limit") {
      notify(
        "info",
        `Rollout limit reached after ${message.completed_chunks} chunks — restarting from the same image.`,
      );
    }
  });

  // A rejected command comes back correlated to the request that caused it, not
  // as one of the model's own messages, so it arrives here without a `type`.
  useReactorMessage((raw: unknown) => {
    const error = (raw as { error?: { code?: string; message?: string } })
      ?.error;
    if (error?.message || error?.code) {
      notify("error", error.message || `Command rejected (${error.code})`);
    }
  });

  const previousStatus = useRef(status);
  useEffect(() => {
    if (previousStatus.current !== status && status === "disconnected") {
      setWorld(null);
    }
    previousStatus.current = status;
  }, [status]);

  useEffect(() => {
    if (lastError) notify("error", explain(lastError.message));
  }, [lastError]);

  const connected = status === "ready";
  const hasImage = Boolean(world?.image_name);

  const connect = useCallback(() => {
    void model.connect().catch((error: unknown) => {
      notify(
        "error",
        error instanceof Error ? error.message : "Could not connect",
      );
    });
  }, [model]);

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <Header
        config={config}
        status={status}
        onConnect={connect}
        onDisconnect={() => void model.disconnect()}
      />

      <main className="flex min-h-0 flex-1 flex-col-reverse lg:flex-row">
        <aside className="flex w-full shrink-0 flex-col gap-3 overflow-y-auto border-edge p-3 lg:w-[24rem] lg:border-r">
          <Steps connected={connected} hasImage={hasImage} />
          <ScenePanel model={model} world={world} connected={connected} />
          <RolloutPanel model={model} world={world} enabled={hasImage} />
        </aside>

        <section className="flex min-h-0 min-w-0 flex-1 flex-col">
          <Stage config={config} status={status} world={world} />
          <CameraBar model={model} world={world} enabled={hasImage} />
        </section>
      </main>
    </div>
  );
}
