"use client";

import type { ReactNode } from "react";

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

export function Panel({
  title,
  hint,
  children,
  disabled,
}: {
  title: string;
  hint?: ReactNode;
  children: ReactNode;
  disabled?: boolean;
}) {
  return (
    <section
      className={cx(
        "rounded-lg border border-edge bg-panel p-4 transition-opacity",
        disabled && "pointer-events-none opacity-40",
      )}
    >
      <header className="mb-3">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-dim">
          {title}
        </h2>
        {hint ? (
          <p className="mt-1 text-[11px] leading-relaxed text-ink-faint">
            {hint}
          </p>
        ) : null}
      </header>
      {children}
    </section>
  );
}

type ButtonTone = "default" | "primary" | "danger";

export function Button({
  children,
  onClick,
  disabled,
  tone = "default",
  full,
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  tone?: ButtonTone;
  full?: boolean;
  title?: string;
}) {
  const tones: Record<ButtonTone, string> = {
    default:
      "border-edge bg-panel-raised text-ink hover:border-ink-faint hover:bg-edge",
    primary:
      "border-transparent bg-ink text-canvas hover:bg-white disabled:bg-ink-faint",
    danger:
      "border-edge bg-panel-raised text-alert hover:border-alert hover:bg-edge",
  };
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      disabled={disabled}
      className={cx(
        "inline-flex h-8 items-center justify-center gap-2 rounded-md border px-3 text-xs font-medium transition-colors",
        "disabled:cursor-not-allowed disabled:opacity-40",
        tones[tone],
        full && "w-full",
      )}
    >
      {children}
    </button>
  );
}

export function Pill({
  children,
  tone = "neutral",
  title,
}: {
  children: ReactNode;
  tone?: "neutral" | "live" | "pending" | "alert";
  title?: string;
}) {
  const tones = {
    neutral: "border-edge text-ink-dim",
    live: "border-live/40 text-live",
    pending: "border-pending/40 text-pending",
    alert: "border-alert/40 text-alert",
  } as const;
  return (
    <span
      title={title}
      className={cx(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium",
        tones[tone],
      )}
    >
      {children}
    </span>
  );
}

export function Dot({
  tone,
  pulse,
}: {
  tone: "idle" | "live" | "pending" | "alert";
  pulse?: boolean;
}) {
  const tones = {
    idle: "bg-ink-faint",
    live: "bg-live",
    pending: "bg-pending",
    alert: "bg-alert",
  } as const;
  return (
    <span
      className={cx(
        "inline-block size-1.5 shrink-0 rounded-full",
        tones[tone],
        pulse && "animate-pulse",
      )}
    />
  );
}

export function StatRow({
  label,
  value,
  title,
}: {
  label: string;
  value: ReactNode;
  title?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1" title={title}>
      <span className="text-[11px] text-ink-faint">{label}</span>
      <span className="truncate text-right text-xs tabular-nums text-ink-dim">
        {value}
      </span>
    </div>
  );
}

/** A signed -1..1 velocity drawn as a bar that fills from the centre. */
export function AxisMeter({
  label,
  negative,
  positive,
  value,
}: {
  label: string;
  negative: string;
  positive: string;
  value: number;
}) {
  const magnitude = Math.abs(value) * 50;
  return (
    <div>
      <div className="flex items-baseline justify-between text-[10px] text-ink-faint">
        <span>{negative}</span>
        <span className="font-medium uppercase tracking-[0.1em] text-ink-dim">
          {label}
        </span>
        <span>{positive}</span>
      </div>
      <div className="relative mt-1 h-1.5 overflow-hidden rounded-full bg-panel-raised">
        <div className="absolute left-1/2 top-0 h-full w-px bg-edge" />
        <div
          className="absolute top-0 h-full rounded-full bg-accent transition-[width,left] duration-100"
          style={
            value >= 0
              ? { left: "50%", width: `${magnitude}%` }
              : { left: `${50 - magnitude}%`, width: `${magnitude}%` }
          }
        />
      </div>
    </div>
  );
}
