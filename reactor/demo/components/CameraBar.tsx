"use client";

import { useCallback } from "react";

import { type Axis, BINDING_BY_CODE, PAD_ROWS } from "@/lib/controls";
import type { Model, WorldState } from "@/lib/model";

import { useCameraKeys } from "./useCameraKeys";
import { AxisMeter, cx } from "./ui";

/** One command per axis, so a key press maps to exactly one call. */
function sendAxis(model: Model, axis: Axis, value: number): Promise<void> {
  switch (axis) {
    case "forward":
      return model.setForward({ forward: value });
    case "strafe":
      return model.setStrafe({ strafe: value });
    case "vertical":
      return model.setVertical({ vertical: value });
    case "pitch":
      return model.setPitch({ pitch: value });
    case "yaw":
      return model.setYaw({ yaw: value });
    case "roll":
      return model.setRoll({ roll: value });
  }
}

const METERS: {
  axis: Axis;
  label: string;
  negative: string;
  positive: string;
}[] = [
  { axis: "forward", label: "forward", negative: "back", positive: "fwd" },
  { axis: "strafe", label: "strafe", negative: "left", positive: "right" },
  { axis: "vertical", label: "vertical", negative: "down", positive: "up" },
  { axis: "pitch", label: "pitch", negative: "down", positive: "up" },
  { axis: "yaw", label: "yaw", negative: "left", positive: "right" },
  { axis: "roll", label: "roll", negative: "ccw", positive: "cw" },
];

/**
 * The six camera axes, kept under the video where their effect is visible.
 *
 * The meters read from the model's own snapshot rather than from local key
 * state, so they show the velocity the next chunk will actually use — including
 * the zeroes the model applies on its own when playback pauses.
 */
export function CameraBar({
  model,
  world,
  enabled,
}: {
  model: Model;
  world: WorldState | null;
  enabled: boolean;
}) {
  const onAxis = useCallback(
    (axis: Axis, value: number) => {
      void sendAxis(model, axis, value);
    },
    [model],
  );

  const { pressed, press, release } = useCameraKeys({ enabled, onAxis });

  return (
    <div
      className={cx(
        "shrink-0 border-t border-edge bg-panel px-4 py-3 transition-opacity",
        !enabled && "pointer-events-none opacity-40",
      )}
    >
      <div className="flex flex-wrap items-start gap-x-8 gap-y-4">
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-dim">
            Camera
          </p>
          <div className="space-y-1.5">
            {PAD_ROWS.map((row, index) => (
              <div key={index} className="flex gap-1.5">
                {row.map((code) => {
                  const binding = BINDING_BY_CODE[code];
                  const down = pressed.has(code);
                  return (
                    <button
                      key={code}
                      type="button"
                      title={binding.action}
                      onPointerDown={(event) => {
                        event.preventDefault();
                        press(code);
                      }}
                      onPointerUp={() => release(code)}
                      onPointerLeave={() => release(code)}
                      onPointerCancel={() => release(code)}
                      className={cx(
                        "flex w-[4.5rem] flex-col items-center gap-0.5 rounded-md border px-1 py-1 text-[10px] transition-colors select-none",
                        down
                          ? "border-accent bg-accent/15 text-accent"
                          : "border-edge bg-panel-raised text-ink-dim hover:border-ink-faint",
                      )}
                    >
                      <span className="font-semibold">{binding.key}</span>
                      <span className="text-[9px] text-ink-faint">
                        {binding.action}
                      </span>
                    </button>
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        <div className="grid min-w-[18rem] flex-1 grid-cols-1 gap-x-6 gap-y-2 sm:grid-cols-2 xl:grid-cols-3">
          {METERS.map((meter) => (
            <AxisMeter
              key={meter.axis}
              label={meter.label}
              negative={meter.negative}
              positive={meter.positive}
              value={world?.[meter.axis] ?? 0}
            />
          ))}
        </div>

        <p className="max-w-xs text-[11px] leading-relaxed text-ink-faint">
          {enabled
            ? "Each axis holds its value until you let go. The model samples all six when the next chunk starts, so a change lands on the chunk the panel calls next."
            : "Choose a starting image to unlock the controls."}
        </p>
      </div>
    </div>
  );
}
