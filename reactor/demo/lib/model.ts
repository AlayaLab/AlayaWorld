import type { AlayaWorldStateUpdateMessage } from "./alayaworld/client";
import type { useAlayaWorld } from "./alayaworld/client.react";

/** The typed command surface the generated hook returns. */
export type Model = ReturnType<typeof useAlayaWorld>;

/** The model's own snapshot of everything a client can observe. */
export type WorldState = AlayaWorldStateUpdateMessage;
