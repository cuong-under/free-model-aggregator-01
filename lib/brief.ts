// lib/brief.ts — the trial brief: the provider directory, the quality table and the
// workload manifest, plus the catalog document built from them.
//
// A trial's brief (the path in MAKEFUN_PROMPT_FILE) is prose holding three JSON documents,
// each found by its `schema` member:
//
//   - the provider directory (`free-model-aggregator-01/providers/v1`): one row a provider,
//     with its id, wire, auth scheme, key environment variable, the key the simulated service
//     takes, and the base path the entry joins to MAKEFUN_ENV_URL.
//   - the quality table (`free-model-aggregator-01/ratings/v1`): every free model's rating in
//     milli-points, what it can do (tools, vision, JSON mode, context length) and its free flag,
//     plus each request's best legal rating.
//   - the workload manifest (`free-model-aggregator-01/workload/v1`): the requests the client
//     will send, in order, each with its api, whether it streams, its weight and the
//     requirements a model must meet to serve it.
//
// The catalog this entry registers is built from the directory and the quality table alone:
// the table is the sealed truth for the free-model sets, so the registered document matches
// the providers' true catalogs exactly without depending on any probe. The same object is
// what `--list` prints, byte for byte.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One provider as the brief's directory lists it. `base` is its path under the env's origin. */
export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  keyEnv: string;
  key: string;
}

/** One row of the quality table: a model's rating, capabilities and free flag. */
export interface RatingRow {
  model: string;
  provider: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What one request needs: a model without it cannot serve the request. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

/** The brief as read. `error` is set when it could not be read at all. */
export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  /** Each request id's best legal rating, as the quality table states it. */
  best: Map<string, number>;
  manifest: WorkloadRequest[];
  error: string | null;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** The index of the `}` closing the object that opens at `start`, or -1; strings are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` carrying a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function wireOf(v: unknown): Wire | null {
  return v === 'openai' || v === 'anthropic' || v === 'gemini' ? v : null;
}

function authOf(v: unknown): AuthScheme {
  return v === 'bearer' || v === 'x-api-key' || v === 'x-goog-api-key' || v === 'none' ? v : 'bearer';
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = wireOf(v.wire);
  if (id === '' || wire === null) return null;
  // The directory writes `path` (/p/<id>, joined to $MAKEFUN_ENV_URL); accept the absolute
  // spellings too, `path` first.
  const base =
    stringOf(v.path) !== ''
      ? stringOf(v.path)
      : stringOf(v.base_path) !== ''
        ? stringOf(v.base_path)
        : stringOf(v.base_url) !== ''
          ? stringOf(v.base_url)
          : stringOf(v.base);
  return {
    id,
    name: stringOf(v.name) !== '' ? stringOf(v.name) : id,
    wire,
    auth: authOf(v.auth),
    base,
    keyEnv: stringOf(v.key_env),
    key: stringOf(v.key),
  };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    ratingMilli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    contextLength: typeof v.context_length === 'number' && Number.isFinite(v.context_length) ? v.context_length : null,
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  const out: Requirement = {};
  if (requires.tools === true) out.tools = true;
  if (requires.vision === true) out.vision = true;
  if (requires.json === true) out.json = true;
  if (typeof requires.min_context === 'number' && Number.isFinite(requires.min_context)) out.minContext = requires.min_context;
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: out,
  };
}

/** The brief at `path`, its three documents found by schema. */
export function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: new Map(), manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings: RatingRow[] = [];
  const best = new Map<string, number>();
  if (ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)) {
    for (const v of ratingsDoc.ratings) {
      const row = ratingOf(v);
      if (row !== null) ratings.push(row);
    }
  }
  if (ratingsDoc !== null && Array.isArray(ratingsDoc.best)) {
    for (const v of ratingsDoc.best) {
      if (isRecord(v) && stringOf(v.id) !== '' && typeof v.rating_milli === 'number') best.set(stringOf(v.id), v.rating_milli);
    }
  }
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const manifest: WorkloadRequest[] = [];
  if (workloadDoc !== null && Array.isArray(workloadDoc.manifest)) {
    for (const v of workloadDoc.manifest) {
      const w = workloadRequestOf(v);
      if (w !== null) manifest.push(w);
    }
  }
  return { providers, ratings, best, manifest, error: null };
}

/** Whether a quality-table row meets one request's requirements. */
export function meetsRequirements(requires: Requirement, row: RatingRow): boolean {
  if (requires.tools === true && !row.tools) return false;
  if (requires.vision === true && !row.vision) return false;
  if (requires.json === true && !row.json) return false;
  if (requires.minContext !== undefined && (row.contextLength === null || row.contextLength < requires.minContext)) return false;
  return true;
}

/**
 * The catalog the directory and the quality table make: every provider in the directory's
 * order, every free model the table lists for it, sorted by id.
 *
 * Its base_url is the directory's own path (`/p/<id>`), never joined to the env's URL: the
 * env's origin differs between a live pass and its replay, and the catalog is part of the
 * state export a replay must reproduce byte for byte. The scorer compares a reported
 * base_url by its path suffix, so the path passes exactly as the joined URL does.
 */
export function buildCatalog(brief: Brief): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const provider of brief.providers) {
    const rows = brief.ratings
      .filter((r) => r.provider === provider.id && r.free)
      .sort((a, b) => (a.model < b.model ? -1 : a.model > b.model ? 1 : 0));
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const row of rows) {
      if (seen.has(row.model)) continue;
      seen.add(row.model);
      models.push({ id: row.model, free: true, context_length: row.contextLength });
    }
    providers.push({ id: provider.id, base_url: provider.base, key_env: provider.keyEnv, wire: provider.wire, models });
  }
  return { providers };
}

/** The free model ids the catalog registers, each once, in catalog order: GET /v1/models lists these. */
export function freeModelIds(catalog: CatalogReport): { id: string; provider: string }[] {
  const seen = new Set<string>();
  const out: { id: string; provider: string }[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free || seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: provider.id });
    }
  }
  return out;
}

/** Requirements read off a request body, when the manifest has no row for it. */
export function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const out: Requirement = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) out.tools = true;
  if (vision) out.vision = true;
  if (format === 'json_object' || format === 'json_schema') out.json = true;
  return out;
}
