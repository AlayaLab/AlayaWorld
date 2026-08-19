import { App } from "@/components/App";
import { readConfig } from "@/lib/config";

/**
 * Read the environment per request rather than at build time, so the same build
 * can be pointed at a different container or a hosted model by changing the
 * environment alone.
 */
export const dynamic = "force-dynamic";

/**
 * Resolve the connection settings on the server and hand the browser only what
 * it needs. Reading the environment here keeps `REACTOR_API_KEY` server-side
 * while the mode it selects is still visible in the UI.
 */
export default function Page() {
  return <App config={readConfig()} />;
}
