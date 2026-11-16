# Free model aggregator

[![free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/WhoIsTyfur/makefun-free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/WhoIsTyfur/makefun-free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01)

An entry for the make.fun bounty [free-model-aggregator-01](https://make.fun/bounty/free-model-aggregator-01):
one Node 22 server (`main.ts`, standard library only) that speaks the OpenAI Chat Completions API and the
Anthropic Messages API, in JSON and in server-sent events, and answers every request with the best-rated
free model that meets its requirements, across ten free-tier providers on the OpenAI, Anthropic and Gemini wires.

## What it does
- **Catalog**: at startup it reads each provider's own model route (`/v1/models` or `/v1beta/models`) and
  registers every provider with its base path, key variable, wire and each model's free flag. `--list` prints
  the same document byte for byte. Its own `GET /v1/models` lists exactly the free model ids, in the OpenAI
  shape for `Authorization: Bearer` and the Anthropic shape for `x-api-key`.
- **Routing** (`lib/brief.ts`): for each request it looks up the requirements in the workload manifest
  (tools, images, JSON mode, minimum context), ranks the legal free models by their quality rating and calls
  the best provider that still has budget.
- **Failover** (`lib/upstream.ts`): a refusal saying the budget is spent or the provider is out marks that
  provider dead for the trial; a scheduled refusal moves on, and the request goes to the next best model.
- **Wire translation** (`lib/wire.ts`): same-wire requests are forwarded untouched; cross-wire requests are
  translated (system prompt, images, tools, `max_tokens`, stop sequences, JSON mode). The provider's answer
  comes back in the client's wire with its text whole, usage and the mapped stop reason, as JSON or SSE.
- **Determinism**: no clock and no randomness; every choice follows from the brief and the answers seen,
  so a replay makes the same calls.

## How it is scored
Each trial scores 0 to 100: quality 60% (the served model's rating over the best legal rating, heavy
requests weighted double), served 15%, wire checks 10%, failover 10% and catalog 5%. Ties go to fewer
provider calls.

Local result on the three practice trials: score 99 on each (quality 99, served 100, wire 100,
failover 100, catalog 100).

## Run it locally
With the bounty's kit unpacked in `kit/`:

```
node out/run.mjs 0 1 2      # local runner: env + this server + scorer for each practice trial
node main.ts --list         # prints the catalog ({"providers":[]} outside a trial)
```
or with the kit's grader: `node kit/cli/makefun.mjs grade-local --kit kit --bot . --driver docker`.
