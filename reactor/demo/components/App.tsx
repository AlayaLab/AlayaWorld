"use client";

import { ReactorProvider } from "@reactor-team/js-sdk";

import { AlayaWorldTracks } from "@/lib/alayaworld/client";
import type { DemoConfig } from "@/lib/config";

import { Toasts } from "./Toasts";
import { Workspace } from "./Workspace";

/**
 * Fetch a session token from this app's own server.
 *
 * The SDK takes a resolver rather than a string so it can ask for a fresh token
 * on every request it makes, including uploads and stream renegotiation.
 */
async function fetchToken(): Promise<string> {
  const response = await fetch("/api/token");
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as {
      error?: string;
    };
    throw new Error(body.error ?? `Token request failed: ${response.status}`);
  }
  const { jwt } = (await response.json()) as { jwt: string };
  return jwt;
}

/**
 * Configure the connection and hand the rest of the app a live session.
 *
 * The provider comes from the SDK rather than the generated client so the model
 * name stays configurable: normally this runs against a container started on this
 * machine, where the name is whatever it was deployed under.
 */
export function App({ config }: { config: DemoConfig }) {
  const local = config.mode === "local";

  return (
    <>
      <ReactorProvider
        modelName={config.modelName}
        modelTracks={[...AlayaWorldTracks]}
        local={local}
        {...(config.apiUrl ? { apiUrl: config.apiUrl } : {})}
        {...(local ? {} : { getJwt: fetchToken })}
        connectOptions={{ autoConnect: false }}
      >
        <Workspace config={config} />
      </ReactorProvider>
      <Toasts />
    </>
  );
}
