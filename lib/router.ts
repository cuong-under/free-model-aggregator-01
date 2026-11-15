// lib/router.ts — which provider serves each request, and in what order the entry falls
// over when one refuses.
//
// A candidate is one provider with its best model for the request: the highest-rated free
// model on that provider meeting the request's requirements. Candidates are ranked by
// rating descending, then provider id ascending, then model id ascending — a total order
// fixed by the brief, so the same request always takes the same path.
//
// Two filters keep candidates legal:
//   - A provider on the Anthropic wire can never echo JSON mode (the Anthropic wire has no
//     JSON-mode field its echo reads), so an OpenAI request whose response_format asks for
//     JSON is never sent to one.
//   - A provider is retired only by proof of death: three consecutive refusals with no
//     success between them. A trial draws at most two scheduled failures for a provider, so
//     three in a row means the budget is spent or the provider is in outage. A retired
//     provider is never tried again — it can only waste calls.
//
// Within a request each provider is tried up to three times in a row before the entry
// moves to the next candidate. A refusal is usually one scheduled call number, and the
// next call number often succeeds — retrying the best provider first serves the request
// at the best rating instead of dropping to a lower one, and three in a row is exactly
// the proof of death, so a dead provider wastes no more calls here than it would spread
// over three requests. One attempt per provider per pass would serve the second-best
// provider the moment the best refuses once, and throw the rating away.

import type { Brief, ProviderEntry, RatingRow, Requirement } from './brief.ts';
import { meetsRequirements } from './brief.ts';

export interface Candidate {
  provider: ProviderEntry;
  model: RatingRow;
}

/** The wire a provider speaks can carry JSON mode when the request needs it echoed. */
function wireCarriesJson(wire: ProviderEntry['wire']): boolean {
  return wire !== 'anthropic';
}

/**
 * One candidate a provider: its best free model meeting `requires`, ranked best first.
 * `jsonEcho` is true only for an OpenAI request whose response_format asks for JSON — the
 * one case where the Anthropic wire cannot echo the request back.
 */
export function candidatesFor(brief: Brief, requires: Requirement, jsonEcho: boolean): Candidate[] {
  const byProvider = new Map<string, RatingRow>();
  for (const row of brief.ratings) {
    if (!row.free) continue;
    if (!meetsRequirements(requires, row)) continue;
    const previous = byProvider.get(row.provider);
    if (previous === undefined || row.ratingMilli > previous.ratingMilli || (row.ratingMilli === previous.ratingMilli && row.model < previous.model)) {
      byProvider.set(row.provider, row);
    }
  }
  const out: Candidate[] = [];
  for (const provider of brief.providers) {
    if (jsonEcho && !wireCarriesJson(provider.wire)) continue;
    const model = byProvider.get(provider.id);
    if (model !== undefined) out.push({ provider, model });
  }
  out.sort((a, b) => {
    if (b.model.ratingMilli !== a.model.ratingMilli) return b.model.ratingMilli - a.model.ratingMilli;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model.model < b.model.model ? -1 : a.model.model > b.model.model ? 1 : 0;
  });
  return out;
}

/** What the entry has learned about each provider's health, from refusals and successes. */
export class ProviderHealth {
  private consecutiveRefusals = new Map<string, number>();
  private attempts = new Map<string, number>();

  /** Refusals in a row without a success. Three proves the provider dead: at most two
   *  scheduled failures exist per trial, so the third refusal is a spent budget or outage. */
  static readonly DEATH_THRESHOLD = 3;

  recordRefusal(providerId: string): void {
    this.consecutiveRefusals.set(providerId, (this.consecutiveRefusals.get(providerId) ?? 0) + 1);
  }

  recordSuccess(providerId: string): void {
    this.consecutiveRefusals.set(providerId, 0);
  }

  recordAttempt(providerId: string): void {
    this.attempts.set(providerId, (this.attempts.get(providerId) ?? 0) + 1);
  }

  isDead(providerId: string): boolean {
    return (this.consecutiveRefusals.get(providerId) ?? 0) >= ProviderHealth.DEATH_THRESHOLD;
  }

  attemptsFor(providerId: string): number {
    return this.attempts.get(providerId) ?? 0;
  }
}

/** The attempt order for one request: each live candidate in ranking order, repeated
 *  `attemptsPerProvider` times in a row, dead providers left out. A second success inside
 *  one request would break the scorer's one-provider check, and the entry never discards
 *  a 2xx answer anyway, so the order stops at the first success. */
export function attemptOrder(candidates: Candidate[], health: ProviderHealth, attemptsPerProvider: number): Candidate[] {
  const order: Candidate[] = [];
  for (const candidate of candidates) {
    if (health.isDead(candidate.provider.id)) continue;
    for (let attempt = 0; attempt < attemptsPerProvider; attempt += 1) {
      order.push(candidate);
    }
  }
  return order;
}
