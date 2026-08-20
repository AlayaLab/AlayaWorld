"use client";

import { cx } from "./ui";

/**
 * The three things that have to happen before the world can be driven, and
 * which one is next. A session opens with no scene chosen, so without this the
 * disabled controls below have no explanation.
 */
export function Steps({
  connected,
  hasImage,
}: {
  connected: boolean;
  hasImage: boolean;
}) {
  const steps = [
    { title: "Connect", detail: "Open a session with the model." },
    {
      title: "Choose a starting image",
      detail: "A built-in scene, or upload your own.",
    },
    {
      title: "Drive the world",
      detail: "Steer with the keyboard and change the prompt as you go.",
    },
  ];
  const current = !connected ? 0 : !hasImage ? 1 : 2;

  return (
    <ol className="rounded-lg border border-edge bg-panel p-4">
      {steps.map((step, index) => {
        const done = index < current;
        const active = index === current;
        return (
          <li
            key={step.title}
            className={cx(
              "flex gap-3 py-1.5",
              !done && !active && "opacity-40",
            )}
          >
            <span
              className={cx(
                "mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full border text-[10px] font-semibold",
                done && "border-live/40 bg-live/10 text-live",
                active && "border-ink bg-ink text-canvas",
                !done && !active && "border-edge text-ink-faint",
              )}
            >
              {done ? "✓" : index + 1}
            </span>
            <span className="min-w-0">
              <span
                className={cx(
                  "block text-xs font-medium",
                  active ? "text-ink" : "text-ink-dim",
                )}
              >
                {step.title}
              </span>
              {active ? (
                <span className="block text-[11px] leading-relaxed text-ink-faint">
                  {step.detail}
                </span>
              ) : null}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
