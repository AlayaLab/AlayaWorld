"use client";

import { useState } from "react";

import type { Model, WorldState } from "@/lib/model";

import { Button, Dot, Panel, StatRow } from "./ui";

/**
 * Playback and the life of the current world.
 *
 * Each chunk is 32 frames, a third over a second of video, and costs real GPU
 * time — so pausing, stepping one chunk at a time, and restarting from the same
 * image with a chosen seed are all first-class controls rather than debug aids.
 */
export function RolloutPanel({
  model,
  world,
  enabled,
}: {
  model: Model;
  world: WorldState | null;
  enabled: boolean;
}) {
  const [seed, setSeed] = useState("");

  const paused = world?.paused ?? true;
  const completed = world?.completed_chunks ?? 0;
  const max = world?.max_chunks ?? 0;
  const progress = max > 0 ? Math.min(100, (completed / max) * 100) : 0;

  return (
    <Panel
      title="Playback"
      hint={
        enabled
          ? "Pausing also releases every camera axis, so the world stops where you left it."
          : "Available once the world has a starting image."
      }
      disabled={!enabled}
    >
      <div className="flex gap-2">
        <Button
          full
          tone={paused ? "primary" : "default"}
          onClick={() => void model.setPaused({ paused: !paused })}
        >
          {paused ? "Resume" : "Pause"}
        </Button>
        <Button
          full
          disabled={!paused}
          title="Generate exactly one 32-frame chunk"
          onClick={() => void model.step()}
        >
          Step once
        </Button>
      </div>

      <div className="mt-3 flex gap-2">
        <input
          value={seed}
          onChange={(event) =>
            setSeed(event.target.value.replace(/[^0-9]/g, ""))
          }
          inputMode="numeric"
          placeholder="seed (optional)"
          className="h-8 min-w-0 flex-1 rounded-md border border-edge bg-panel-raised px-2.5 text-xs tabular-nums text-ink placeholder:text-ink-faint focus:border-ink-faint focus:outline-none"
        />
        <Button
          tone="danger"
          title="Rebuild the world from the same image and prompt"
          onClick={() => {
            void model.reset(seed ? { seed: Number(seed) } : { seed: -1 });
            setSeed("");
          }}
        >
          Restart world
        </Button>
      </div>

      <div className="mt-3 border-t border-edge pt-2">
        <StatRow
          label="Chunks"
          value={max > 0 ? `${completed} / ${max}` : String(completed)}
          title="The world restarts by itself once it reaches the chunk limit."
        />
        <div className="h-1 overflow-hidden rounded-full bg-panel-raised">
          <div
            className="h-full rounded-full bg-ink-faint transition-[width] duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
        <StatRow label="Next chunk" value={world?.next_chunk ?? "—"} />
        <StatRow label="Seed" value={world?.seed ?? "—"} />
        <StatRow
          label="State"
          value={
            <span className="inline-flex items-center gap-1.5">
              <Dot
                tone={
                  world?.generating ? "live" : paused ? "idle" : "pending"
                }
                pulse={world?.generating}
              />
              {world?.generating
                ? "generating"
                : world?.reset_queued
                  ? "restarting"
                  : paused
                    ? world?.step_queued
                      ? "step queued"
                      : "paused"
                    : "waiting"}
            </span>
          }
        />
      </div>
    </Panel>
  );
}
