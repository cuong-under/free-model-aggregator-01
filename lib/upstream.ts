// lib/upstream.ts — one call to one provider.
//
// The entry reaches a provider at `$MAKEFUN_ENV_URL` joined to the provider's base path, on
// the wire the directory names for it. The auth header is the scheme the directory names —
// the wire does not decide it. The call is buffered in full and parsed into a plain
// answer; the entry never streams a provider's response through, so a refused first call
// leaves nothing half-sent to the client and failover stays clean.
//
// Outcomes:
//   - `answer`: a 2xx response, parsed (or salvaged when its shape is unexpected — a 2xx
//     answer must always be used, never discarded for another provider).
//   - `refused`: 402/429/5xx — the provider's budget, failure schedule or outage. Counts
//     toward retirement and is worth retrying on a later call.
//   - `bad_request`: anything else non-2xx (a wrong key or model name) — a construction
//     fault, not a budget event; the provider is skipped for this request and the refusal
//     is not held against it.
//   - `network`: the call never completed (timeout or transport fault).

import type { ProviderEntry } from './brief.ts';
import type { CanonicalRequest, ProviderAnswer } from './wires.ts';
import { parseAnthropicResponse, parseGeminiResponse, parseOpenAIResponse, toAnthropic, toGemini, toOpenAI } from './wires.ts';

export type AttemptOutcome =
  | { kind: 'answer'; answer: ProviderAnswer }
  | { kind: 'refused'; status: number }
  | { kind: 'bad_request'; status: number }
  | { kind: 'network'; message: string };

function joinUrl(origin: string, path: string): string {
  const base = origin.endsWith('/') ? origin.slice(0, -1) : origin;
  return path.startsWith('/') ? `${base}${path}` : `${base}/${path}`;
}

function authHeaders(provider: ProviderEntry): Record<string, string> {
  switch (provider.auth) {
    case 'bearer':
      return { authorization: `Bearer ${provider.key}` };
    case 'x-api-key':
      return { 'x-api-key': provider.key };
    case 'x-goog-api-key':
      return { 'x-goog-api-key': provider.key };
    default:
      return {};
  }
}

/** A last-resort read of a 2xx body whose shape the parser did not recognise. */
function salvage(bodyText: string): ProviderAnswer {
  let text = bodyText;
  try {
    const value: unknown = JSON.parse(bodyText);
    if (typeof value === 'object' && value !== null) {
      const record = value as Record<string, unknown>;
      if (typeof record.text === 'string') text = record.text;
    }
  } catch {
    // Keep the raw body.
  }
  return { text, inputTokens: 0, outputTokens: 0, stopFamily: 0 };
}

export async function callProvider(
  envUrl: string,
  provider: ProviderEntry,
  model: string,
  canon: CanonicalRequest,
  timeoutMs: number,
): Promise<AttemptOutcome> {
  let url: string;
  let body: string;
  const headers: Record<string, string> = { 'content-type': 'application/json', accept: 'application/json', ...authHeaders(provider) };
  if (provider.wire === 'openai') {
    url = joinUrl(envUrl, `${provider.base}/v1/chat/completions`);
    body = JSON.stringify(toOpenAI(canon, model));
    if (canon.stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = joinUrl(envUrl, `${provider.base}/v1/messages`);
    body = JSON.stringify(toAnthropic(canon, model));
    headers['anthropic-version'] = '2023-06-01';
    if (canon.stream) headers.accept = 'text/event-stream';
  } else {
    const action = canon.stream ? 'streamGenerateContent' : 'generateContent';
    url = joinUrl(envUrl, `${provider.base}/v1beta/models/${encodeURIComponent(model)}:${action}`);
    if (canon.stream) url += '?alt=sse';
    body = JSON.stringify(toGemini(canon));
  }
  let response: Response;
  try {
    response = await fetch(url, { method: 'POST', headers, body, signal: AbortSignal.timeout(timeoutMs) });
  } catch (err) {
    return { kind: 'network', message: (err as Error).message };
  }
  const bodyText = await response.text();
  if (response.ok) {
    const parsed =
      provider.wire === 'openai'
        ? parseOpenAIResponse(bodyText, canon.stream)
        : provider.wire === 'anthropic'
          ? parseAnthropicResponse(bodyText, canon.stream)
          : parseGeminiResponse(bodyText, canon.stream);
    return { kind: 'answer', answer: parsed ?? salvage(bodyText) };
  }
  if (response.status === 402 || response.status === 429 || response.status >= 500) {
    return { kind: 'refused', status: response.status };
  }
  return { kind: 'bad_request', status: response.status };
}
