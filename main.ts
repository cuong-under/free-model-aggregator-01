// main.ts — the free-model aggregator: one server answering OpenAI and Anthropic chat
// requests with the best free model meeting each request's requirements.
//
// Trial flow: the brief (MAKEFUN_PROMPT_FILE) names the providers and their quality
// table. The entry listens on a picked port, registers its catalog with the env
// ($MAKEFUN_ENV_URL/register), serves the client's requests one at a time — each on
// the best live provider, failing over when one refuses — and polls the env's /work
// until the workload is done, then exits.
//
// `node main.ts --list` prints the catalog document byte-for-byte identical to the one
// registered, and exits 0; with no brief set it prints {"providers":[]}.
//
// Everything the entry chooses derives from the brief, the requests and the providers'
// answers: no wall clock and no unseeded randomness anywhere in a decision path, so a
// replay leaves the same state export byte for byte.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { buildCatalog, freeModelIds, isRecord, readBrief, requirementsFromBody, type Brief, type CatalogReport, type Json, type Requirement } from './lib/brief.ts';
import { anthropicError, anthropicModelList, fromAnthropic, fromOpenAI, openAIError, openAIModelList, renderAnthropic, renderOpenAI, type CanonicalRequest, type ClientWire } from './lib/wires.ts';
import { attemptOrder, candidatesFor, ProviderHealth } from './lib/router.ts';
import { callProvider } from './lib/upstream.ts';

const UPSTREAM_TIMEOUT_MS = 5000;
const ATTEMPTS_PER_PROVIDER = 3;
const WORK_POLL_MS = 100;
const MAX_BODY_BYTES = 8 * 1024 * 1024;

function log(message: string): void {
  process.stderr.write(`[fma01] ${message}\n`);
}

function joinUrl(origin: string, path: string): string {
  const base = origin.endsWith('/') ? origin.slice(0, -1) : origin;
  return path.startsWith('/') ? `${base}${path}` : `${base}/${path}`;
}

async function readBody(request: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of request) {
    const buffer = chunk as Buffer;
    total += buffer.length;
    if (total > MAX_BODY_BYTES) throw new Error('request body too large');
    chunks.push(buffer);
  }
  return Buffer.concat(chunks).toString('utf8');
}

function sendJson(response: ServerResponse, status: number, body: string): void {
  response.writeHead(status, { 'content-type': 'application/json' });
  response.end(body);
}

function sendStream(response: ServerResponse, status: number, body: string): void {
  response.writeHead(status, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  response.end(body);
}

interface Entry {
  brief: Brief;
  catalog: CatalogReport;
  modelList: { id: string; provider: string }[];
  health: ProviderHealth;
  /** Requests served so far: the workload manifest is sent in order, one at a time. */
  servedCount: number;
}

/** The requirements for one request: the manifest's row by arrival order, or the body's. */
function requirementsFor(entry: Entry, wire: ClientWire, body: Json): Requirement {
  const row = entry.brief.manifest[entry.servedCount];
  if (row !== undefined && row.api === wire) return row.requires;
  return requirementsFromBody(body);
}

/**
 * Whether the provider must echo JSON mode: only an OpenAI request whose response_format
 * asks for JSON. The Anthropic wire can never echo it, and an Anthropic client never
 * expects it.
 */
function jsonEchoFor(wire: ClientWire, canon: CanonicalRequest): boolean {
  return wire === 'openai' && canon.jsonMode;
}

async function handleChat(entry: Entry, envUrl: string, wire: ClientWire, body: Json, response: ServerResponse): Promise<void> {
  const canon = wire === 'openai' ? fromOpenAI(body) : fromAnthropic(body);
  const requires = requirementsFor(entry, wire, body);
  entry.servedCount += 1;
  const candidates = candidatesFor(entry.brief, requires, jsonEchoFor(wire, canon));
  const order = attemptOrder(candidates, entry.health, ATTEMPTS_PER_PROVIDER);
  const skipped = new Set<string>();
  let attempts = 0;
  for (const candidate of order) {
    const providerId = candidate.provider.id;
    if (skipped.has(providerId) || entry.health.isDead(providerId)) continue;
    attempts += 1;
    entry.health.recordAttempt(providerId);
    const outcome = await callProvider(envUrl, candidate.provider, candidate.model.model, canon, UPSTREAM_TIMEOUT_MS);
    if (outcome.kind === 'answer') {
      entry.health.recordSuccess(providerId);
      const rendered = wire === 'openai' ? renderOpenAI(outcome.answer, candidate.model.model, canon.stream) : renderAnthropic(outcome.answer, candidate.model.model, canon.stream);
      log(`request ${entry.servedCount} ${wire} -> ${providerId}/${candidate.model.model} (attempt ${attempts})`);
      if (canon.stream) sendStream(response, 200, rendered);
      else sendJson(response, 200, rendered);
      return;
    }
    if (outcome.kind === 'refused') {
      entry.health.recordRefusal(providerId);
      continue;
    }
    // A construction fault or a transport fault: not a budget event, and retrying the
    // same construction cannot help. Skip the provider for this request only.
    skipped.add(providerId);
  }
  log(`request ${entry.servedCount} ${wire}: no provider answered after ${attempts} attempts`);
  const error = wire === 'openai' ? openAIError(503, 'no provider available') : anthropicError(503, 'no provider available');
  sendJson(response, error.status, error.body);
}

// Requests are served strictly one at a time: a promise chain serialises the handlers,
// so the manifest index follows arrival order and no provider ever sees two calls at
// once. The chain is never left rejected — a handler fault must not stall the requests
// behind it.
let queue: Promise<void> = Promise.resolve();

function handleRequest(entry: Entry, envUrl: string, request: IncomingMessage, response: ServerResponse): void {
  const url = new URL(request.url ?? '/', 'http://127.0.0.1');
  if (request.method === 'GET' && url.pathname === '/v1/models') {
    // The wire is the auth header: x-api-key names Anthropic, anything else OpenAI.
    const anthropic = request.headers['x-api-key'] !== undefined;
    sendJson(response, 200, anthropic ? anthropicModelList(entry.modelList) : openAIModelList(entry.modelList));
    return;
  }
  const wire: ClientWire | null =
    request.method === 'POST' && url.pathname === '/v1/chat/completions'
      ? 'openai'
      : request.method === 'POST' && url.pathname === '/v1/messages'
        ? 'anthropic'
        : null;
  if (wire === null) {
    sendJson(response, 404, JSON.stringify({ error: 'not found' }));
    return;
  }
  const run = async (): Promise<void> => {
    try {
      await serveChat(entry, envUrl, wire, request, response);
    } catch (err) {
      log(`request ${entry.servedCount + 1} ${wire}: handler fault: ${(err as Error).message}`);
    }
  };
  queue = queue.then(run, run);
}

async function serveChat(entry: Entry, envUrl: string, wire: ClientWire, request: IncomingMessage, response: ServerResponse): Promise<void> {
  let text: string;
  try {
    text = await readBody(request);
  } catch {
    const error = wire === 'openai' ? openAIError(400, 'request body too large') : anthropicError(400, 'request body too large');
    if (!response.writableEnded) sendJson(response, error.status, error.body);
    return;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    const error = wire === 'openai' ? openAIError(400, 'invalid JSON') : anthropicError(400, 'invalid JSON');
    if (!response.writableEnded) sendJson(response, error.status, error.body);
    return;
  }
  if (!isRecord(parsed)) {
    const error = wire === 'openai' ? openAIError(400, 'request body must be an object') : anthropicError(400, 'request body must be an object');
    if (!response.writableEnded) sendJson(response, error.status, error.body);
    return;
  }
  try {
    await handleChat(entry, envUrl, wire, parsed, response);
  } catch (err) {
    log(`request ${entry.servedCount + 1} ${wire}: ${(err as Error).message}`);
    const error = wire === 'openai' ? openAIError(500, 'internal error') : anthropicError(500, 'internal error');
    if (!response.writableEnded) sendJson(response, error.status, error.body);
  }
}

async function register(envUrl: string, port: number, catalog: CatalogReport): Promise<void> {
  const url = joinUrl(envUrl, '/register');
  let lastError = '';
  for (let attempt = 0; attempt < 40; attempt += 1) {
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      const text = await response.text();
      if (response.ok) {
        log(`registered on port ${port}: ${text}`);
        return;
      }
      lastError = `${response.status} ${text}`;
    } catch (err) {
      lastError = (err as Error).message;
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`registration failed: ${lastError}`);
}

async function waitForDone(envUrl: string): Promise<void> {
  const url = joinUrl(envUrl, '/work');
  for (;;) {
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(5000) });
      const text = await response.text();
      const value: unknown = JSON.parse(text);
      if (isRecord(value) && value.done === true) return;
    } catch {
      // The env may not answer yet; keep polling.
    }
    await new Promise((resolve) => setTimeout(resolve, WORK_POLL_MS));
  }
}

async function main(): Promise<void> {
  const listOnly = process.argv.includes('--list');
  const briefPath = process.env.MAKEFUN_PROMPT_FILE ?? '';
  const brief = readBrief(briefPath);
  const catalog = buildCatalog(brief);
  if (listOnly) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }
  if (brief.error !== null) {
    process.stderr.write(`[fma01] ${brief.error}\n`);
    process.exitCode = 1;
    return;
  }
  const envUrl = process.env.MAKEFUN_ENV_URL ?? '';
  if (envUrl === '') {
    process.stderr.write('[fma01] MAKEFUN_ENV_URL is not set\n');
    process.exitCode = 1;
    return;
  }
  const entry: Entry = {
    brief,
    catalog,
    modelList: freeModelIds(catalog),
    health: new ProviderHealth(),
    servedCount: 0,
  };
  const server = createServer((request, response) => handleRequest(entry, envUrl, request, response));
  const port = await new Promise<number>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      resolve(typeof address === 'object' && address !== null ? address.port : 0);
    });
  });
  log(`listening on 127.0.0.1:${port} with ${entry.brief.providers.length} providers`);
  await register(envUrl, port, catalog);
  await waitForDone(envUrl);
  log('workload done; exiting');
  server.close();
  process.exitCode = 0;
}

process.on('SIGTERM', () => process.exit(0));
process.on('SIGINT', () => process.exit(0));

main().catch((err: Error) => {
  process.stderr.write(`[fma01] ${err.message}\n`);
  process.exitCode = 1;
});
