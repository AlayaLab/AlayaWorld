"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  type Axis,
  BINDING_BY_CODE,
  type Motion,
  NEUTRAL_MOTION,
  changedAxes,
  isTypingTarget,
  motionFrom,
} from "@/lib/controls";

interface Options {
  /** Keys are ignored until the model has a world to steer. */
  enabled: boolean;
  /** Called once per axis whose velocity changed. */
  onAxis: (axis: Axis, value: number) => void;
}

/**
 * Track which movement keys are down and push each axis change to the model.
 *
 * The model holds a velocity until it is replaced, so a key press and a key
 * release are one command each and holding a key sends nothing. `press` and
 * `release` are exposed so the on-screen pad drives the same key set as the
 * keyboard, and both light up the same way.
 */
export function useCameraKeys({ enabled, onAxis }: Options) {
  const [pressed, setPressed] = useState<ReadonlySet<string>>(new Set());
  const pressedRef = useRef<Set<string>>(new Set());
  const motionRef = useRef<Motion>({ ...NEUTRAL_MOTION });
  const onAxisRef = useRef(onAxis);

  useEffect(() => {
    onAxisRef.current = onAxis;
  });

  const commit = useCallback(() => {
    const next = motionFrom(pressedRef.current);
    for (const axis of changedAxes(motionRef.current, next)) {
      onAxisRef.current(axis, next[axis]);
    }
    motionRef.current = next;
    setPressed(new Set(pressedRef.current));
  }, []);

  const press = useCallback(
    (code: string) => {
      if (pressedRef.current.has(code)) return;
      pressedRef.current.add(code);
      commit();
    },
    [commit],
  );

  const release = useCallback(
    (code: string) => {
      if (!pressedRef.current.delete(code)) return;
      commit();
    },
    [commit],
  );

  const releaseAll = useCallback(() => {
    if (pressedRef.current.size === 0) return;
    pressedRef.current.clear();
    commit();
  }, [commit]);

  useEffect(() => {
    if (!enabled) {
      releaseAll();
      return;
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.repeat || isTypingTarget(event.target)) return;
      if (!BINDING_BY_CODE[event.code]) return;
      event.preventDefault();
      press(event.code);
    };
    const onKeyUp = (event: KeyboardEvent) => {
      if (!BINDING_BY_CODE[event.code]) return;
      event.preventDefault();
      release(event.code);
    };
    // A window that loses focus keeps no key state, so the model would keep
    // moving on a velocity the user can no longer cancel.
    const onBlur = () => releaseAll();

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
    document.addEventListener("visibilitychange", onBlur);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
      document.removeEventListener("visibilitychange", onBlur);
      releaseAll();
    };
  }, [enabled, press, release, releaseAll]);

  return useMemo(
    () => ({ pressed, press, release, releaseAll }),
    [pressed, press, release, releaseAll],
  );
}
