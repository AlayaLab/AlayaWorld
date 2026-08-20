import { NextResponse } from "next/server";

import { readConfig } from "@/lib/config";

const TOKEN_LIFETIME_SECONDS = 60 * 60;
const MAX_SESSIONS = 10;
/** Expire the browser's cached copy before the token itself expires. */
const CACHE_SKEW_SECONDS = 60;
const DEFAULT_API_URL = "https://api.reactor.inc";

/**
 * Exchange the server's API key for a short-lived session token.
 *
 * The token is scoped to this one model and to sessions it created itself, so
 * handing it to the browser grants nothing else on the account. Only hosted mode
 * calls this route; in local mode there is no token to mint.
 */
export async function GET() {
  const apiKey = process.env.REACTOR_API_KEY?.trim();
  if (!apiKey) {
    return NextResponse.json(
      {
        error:
          "REACTOR_API_KEY is not set. Leave it unset to connect to a local `reactor run` container instead.",
      },
      { status: 500 },
    );
  }

  const { modelName, apiUrl } = readConfig();
  const response = await fetch(`${apiUrl ?? DEFAULT_API_URL}/tokens`, {
    method: "POST",
    headers: {
      "Reactor-API-Key": apiKey,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      expires_after: TOKEN_LIFETIME_SECONDS,
      authorization_details: [
        {
          type: "session",
          resources: { models: { match: [modelName] } },
          constraints: { max_sessions: MAX_SESSIONS },
        },
      ],
    }),
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    return NextResponse.json(
      {
        error: `Reactor API rejected the token request (${response.status}). ${detail}`.trim(),
      },
      { status: 502 },
    );
  }

  const { jwt, expires_at } = (await response.json()) as {
    jwt: string;
    expires_at: number;
  };
  const maxAge = Math.max(
    0,
    expires_at - Math.floor(Date.now() / 1000) - CACHE_SKEW_SECONDS,
  );

  return NextResponse.json(
    { jwt },
    { headers: { "Cache-Control": `private, max-age=${maxAge}` } },
  );
}
