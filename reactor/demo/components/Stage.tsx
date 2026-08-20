"use client";

import { AlayaWorldMainVideoView } from "@/lib/alayaworld/client.react";
import type { DemoConfig } from "@/lib/config";
import type { WorldState } from "@/lib/model";

import { Dot } from "./ui";

/** What to say over the video when there is nothing to watch yet. */
function placeholder(
  status: string,
  config: DemoConfig,
  world: WorldState | null,
): { title: string; detail: string } | null {
  if (status === "ready") {
    if (!world?.image_name) {
      return {
        title: "Choose a starting image",
        detail:
          "Pick a built-in scene or upload your own. The world is generated from that first frame.",
      };
    }
    return null;
  }
  if (status === "connecting" || status === "waiting") {
    return {
      title: "Opening a session",
      detail:
        config.mode === "local"
          ? "A local container serves one session at a time, so this waits for any earlier session to close."
          : "The first frames arrive once the model has a scene to generate from.",
    };
  }
  if (status === "error") {
    return {
      title: "Could not connect",
      detail:
        config.mode === "local"
          ? `Check that a container is serving on ${config.apiUrl}, the port passed to \`reactor run --port\`. A session left open by an earlier page also refuses a new one for about a minute.`
          : "Check the API key and that the model name matches a deployed model.",
    };
  }
  return {
    title: "Not connected",
    detail:
      config.mode === "local"
        ? `Connect to the model container on ${config.apiUrl}.`
        : `Connect to ${config.modelName}.`,
  };
}

export function Stage({
  config,
  status,
  world,
}: {
  config: DemoConfig;
  status: string;
  world: WorldState | null;
}) {
  const overlay = placeholder(status, config, world);
  const live = status === "ready" && Boolean(world?.image_name);

  return (
    <div className="relative min-h-0 min-w-0 flex-1 bg-black">
      <AlayaWorldMainVideoView
        className="absolute inset-0 size-full"
        videoObjectFit="contain"
      />

      {overlay ? (
        <div className="absolute inset-0 flex items-center justify-center p-8">
          <div className="max-w-sm text-center">
            <p className="text-sm font-medium text-ink-dim">{overlay.title}</p>
            <p className="mt-2 text-xs leading-relaxed text-ink-faint">
              {overlay.detail}
            </p>
          </div>
        </div>
      ) : null}

      {live ? (
        <div className="pointer-events-none absolute bottom-3 left-3 flex items-center gap-3 rounded-md border border-white/10 bg-black/50 px-2.5 py-1.5 text-[11px] text-white/70 backdrop-blur">
          <span className="flex items-center gap-1.5">
            <Dot
              tone={world?.generating ? "live" : "idle"}
              pulse={world?.generating}
            />
            chunk {world?.completed_chunks ?? 0}
          </span>
          {world?.paused ? <span className="text-pending">paused</span> : null}
        </div>
      ) : null}
    </div>
  );
}
