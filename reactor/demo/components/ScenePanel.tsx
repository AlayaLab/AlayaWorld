"use client";

import { useEffect, useRef, useState } from "react";

import type { Model, WorldState } from "@/lib/model";

import { notify } from "./Toasts";
import { Button, Panel, StatRow } from "./ui";

/**
 * Pick what the world starts from, and steer it with text once it is running.
 *
 * Choosing an image restarts the world from scratch; changing the prompt does
 * not. Both take effect at the next chunk boundary, so the panel reports which
 * chunk will be the first to use what was just queued.
 */
export function ScenePanel({
  model,
  world,
  connected,
}: {
  model: Model;
  world: WorldState | null;
  connected: boolean;
}) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [draft, setDraft] = useState("");
  const [dirty, setDirty] = useState(false);
  const [uploading, setUploading] = useState(false);

  // Follow the model's prompt until the user starts writing their own, so the
  // box shows what is actually queued rather than a stale copy.
  const modelPrompt = world?.prompt ?? "";
  useEffect(() => {
    if (!dirty) setDraft(modelPrompt);
  }, [modelPrompt, dirty]);

  const hasImage = Boolean(world?.image_name);
  const trimmed = draft.trim();

  const upload = async (file: File) => {
    setUploading(true);
    try {
      const reference = await model.uploadFile(file);
      await model.setImage(
        trimmed ? { image: reference, prompt: trimmed } : { image: reference },
      );
      setDirty(false);
    } catch (error) {
      notify(
        "error",
        error instanceof Error ? error.message : "Upload failed",
      );
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  return (
    <Panel
      title="Scene"
      hint={
        connected
          ? "A new image restarts the world. Uploads are cropped to the stream's 960×544 frame."
          : "Connect first."
      }
      disabled={!connected}
    >
      <div className="flex gap-2">
        <Button full onClick={() => void model.randomImage()}>
          Random scene
        </Button>
        <Button
          full
          onClick={() => fileInput.current?.click()}
          disabled={uploading}
        >
          {uploading ? "Uploading…" : "Upload image"}
        </Button>
      </div>
      <input
        ref={fileInput}
        type="file"
        accept="image/jpeg,image/png,image/webp,image/bmp"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) void upload(file);
        }}
      />

      <label className="mt-4 block text-[11px] text-ink-faint">
        Prompt
        <textarea
          value={draft}
          onChange={(event) => {
            setDraft(event.target.value);
            setDirty(true);
          }}
          rows={3}
          placeholder="Describe the scene the model should keep generating"
          className="mt-1 w-full resize-none rounded-md border border-edge bg-panel-raised px-2.5 py-2 text-xs text-ink placeholder:text-ink-faint focus:border-ink-faint focus:outline-none"
        />
      </label>
      <Button
        full
        tone="primary"
        disabled={!hasImage || !trimmed || trimmed === modelPrompt}
        onClick={() => {
          void model.setPrompt({ prompt: trimmed });
          setDirty(false);
        }}
      >
        {hasImage ? "Queue prompt" : "Choose an image first"}
      </Button>

      <div className="mt-3 border-t border-edge pt-2">
        <StatRow label="Image" value={world?.image_name ?? "none"} />
        <StatRow
          label="Source"
          value={
            world?.image_source === "built_in"
              ? "built-in"
              : (world?.image_source ?? "—")
          }
        />
        {world?.active_prompt ? (
          <p
            className="mt-1 line-clamp-2 text-[11px] leading-relaxed text-ink-faint"
            title={world.active_prompt}
          >
            <span className="text-ink-dim">Generating now: </span>
            {world.active_prompt}
          </p>
        ) : null}
      </div>
    </Panel>
  );
}
