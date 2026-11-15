# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/cuong-under/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/cuong-under/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server answering OpenAI and Anthropic chat requests with the best free model meeting each request's requirements, failing over across ten simulated free-tier providers when one refuses.

## How it works

The entry reads the trial brief (the provider directory, the quality table and the workload manifest), registers a catalog built from the directory and the quality table alone — no probes, so the catalog matches the providers' true free-model sets exactly — and serves each request on the highest-rated free model that meets its requirements (tools, vision, JSON mode, context length), on whichever wire that provider speaks: OpenAI, Anthropic or Gemini.

Every upstream call is buffered in full and parsed into a plain answer before the client's response is emitted, so a refused first call leaves nothing half-sent and failover stays clean: requests are served strictly one at a time, a provider is retired only after three consecutive refusals (at most two scheduled failures exist per trial, so three proves a spent budget or an outage), and a 2xx answer is always used, never discarded. All choices derive from the brief, the requests and the providers' answers — no wall clock, no unseeded randomness — so a replay leaves the same state export byte for byte.

## Layout

- `main.ts` — the server and the trial flow (register, serve, poll `/work`, exit).
- `lib/brief.ts` — the brief: provider directory, quality table, workload manifest, and the catalog document built from them.
- `lib/wires.ts` — the canonical request model, the six translations between the client wires and the provider wires, the three response parsers, and the two response emitters.
- `lib/router.ts` — candidate ranking (best free model per provider, by rating) and provider health.
- `lib/upstream.ts` — one buffered call to one provider, on its own wire and auth scheme.

Standard library only; TypeScript runs directly via type stripping. `node main.ts --list` prints the catalog document byte-for-byte identical to the registered catalog.

## Run it

```sh
node main.ts
```

## License

MIT — see [LICENSE](LICENSE).

---

<sub>Entered on [make.fun](https://make.fun/?ref=envcarcq&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
