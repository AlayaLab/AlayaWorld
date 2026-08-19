/**
 * Keyboard bindings for the model's six camera axes.
 *
 * Each axis takes a velocity between -1 and 1 that the model holds until a new
 * value arrives, so a key press is a single command and a key release is a
 * single command back to zero. Nothing is sent while a key is simply held down.
 */

export const AXES = [
  "forward",
  "strafe",
  "vertical",
  "pitch",
  "yaw",
  "roll",
] as const;

export type Axis = (typeof AXES)[number];

/** The six velocities as the client last sent them. */
export type Motion = Record<Axis, number>;

export const NEUTRAL_MOTION: Motion = {
  forward: 0,
  strafe: 0,
  vertical: 0,
  pitch: 0,
  yaw: 0,
  roll: 0,
};

export interface Binding {
  /** `KeyboardEvent.code`, so the layout is independent of keyboard language. */
  code: string;
  axis: Axis;
  /** Velocity applied while the key is down. */
  value: 1 | -1;
  /** Key name shown in the on-screen pad. */
  key: string;
  /** What the key does, in the model's terms. */
  action: string;
}

/**
 * Movement on the left hand, looking on the right hand.
 *
 * W/A/S/D translate and I/J/K/L rotate, which keeps looking around on the
 * keyboard rather than the mouse: the model samples one velocity per chunk, so
 * mouse deltas would be averaged away rather than felt.
 */
export const BINDINGS: readonly Binding[] = [
  { code: "KeyW", axis: "forward", value: 1, key: "W", action: "Forward" },
  { code: "KeyS", axis: "forward", value: -1, key: "S", action: "Back" },
  { code: "KeyA", axis: "strafe", value: -1, key: "A", action: "Left" },
  { code: "KeyD", axis: "strafe", value: 1, key: "D", action: "Right" },
  { code: "Space", axis: "vertical", value: 1, key: "Space", action: "Rise" },
  { code: "KeyC", axis: "vertical", value: -1, key: "C", action: "Descend" },
  { code: "KeyI", axis: "pitch", value: 1, key: "I", action: "Look up" },
  { code: "KeyK", axis: "pitch", value: -1, key: "K", action: "Look down" },
  { code: "KeyJ", axis: "yaw", value: -1, key: "J", action: "Turn left" },
  { code: "KeyL", axis: "yaw", value: 1, key: "L", action: "Turn right" },
  { code: "KeyQ", axis: "roll", value: -1, key: "Q", action: "Roll left" },
  { code: "KeyE", axis: "roll", value: 1, key: "E", action: "Roll right" },
];

export const BINDING_BY_CODE: Record<string, Binding> = Object.fromEntries(
  BINDINGS.map((binding) => [binding.code, binding]),
);

/** The pad layout: movement keys first, then looking, then roll. */
export const PAD_ROWS: readonly (readonly string[])[] = [
  ["KeyW", "KeyS", "KeyA", "KeyD"],
  ["KeyI", "KeyK", "KeyJ", "KeyL"],
  ["Space", "KeyC", "KeyQ", "KeyE"],
];

/**
 * Resolve the six velocities from the set of keys currently down.
 *
 * Opposite keys on one axis cancel, which keeps the axis at zero instead of
 * letting whichever key was pressed last win.
 */
export function motionFrom(pressed: ReadonlySet<string>): Motion {
  const motion: Motion = { ...NEUTRAL_MOTION };
  for (const code of pressed) {
    const binding = BINDING_BY_CODE[code];
    if (binding) motion[binding.axis] += binding.value;
  }
  for (const axis of AXES) {
    motion[axis] = Math.max(-1, Math.min(1, motion[axis]));
  }
  return motion;
}

/** The axes whose velocity changed, so only those are sent. */
export function changedAxes(from: Motion, to: Motion): Axis[] {
  return AXES.filter((axis) => from[axis] !== to[axis]);
}

/** True while a text field has focus and keystrokes belong to it. */
export function isTypingTarget(target: EventTarget | null): boolean {
  const element = target as HTMLElement | null;
  if (!element) return false;
  const tag = element.tagName;
  return (
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "SELECT" ||
    element.isContentEditable
  );
}
