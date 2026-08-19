/**
 * The two ways this app can reach the model.
 *
 * `local` talks straight to a container started with `reactor run`: no account,
 * no API key, no session token. `hosted` goes through the Reactor API, which
 * mints a short-lived session token for the browser.
 */
export type Mode = "local" | "hosted";

export interface DemoConfig {
  mode: Mode;
  /** Model name sent when a hosted session is created. */
  modelName: string;
  /** Base URL the SDK connects to, or `undefined` to accept the SDK default. */
  apiUrl?: string;
}

/** Matches `model.name` in the example's reactor.yaml. */
const DEFAULT_MODEL_NAME = "alaya-world";

/** Where `reactor run` serves when `--port` is not passed. */
const DEFAULT_LOCAL_URL = "http://localhost:8080";

/**
 * Read the environment once, on the server.
 *
 * `REACTOR_API_KEY` is the switch: with a key the app creates hosted sessions,
 * without one it connects to a local container. The key itself never leaves the
 * server — only the resolved mode, model name, and base URL are handed to the
 * browser.
 */
export function readConfig(): DemoConfig {
  const modelName = process.env.REACTOR_MODEL_NAME?.trim() || DEFAULT_MODEL_NAME;

  if (process.env.REACTOR_API_KEY?.trim()) {
    return {
      mode: "hosted",
      modelName,
      apiUrl: process.env.REACTOR_API_URL?.trim() || undefined,
    };
  }

  return {
    mode: "local",
    modelName,
    apiUrl: process.env.REACTOR_LOCAL_URL?.trim() || DEFAULT_LOCAL_URL,
  };
}
