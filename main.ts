// Free model aggregator: serves the OpenAI and Anthropic APIs from the best free model that answers.
// No clock and no randomness reach a request: every choice follows from the brief and the answers seen.

import { createServer } from 'node:http';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { readBrief, modelsFromListing, modelsFromRatings, catalogText, freeIds, candidates } from './lib/brief.ts';
import type { CatalogEntry, Requires } from './lib/brief.ts';
import { neutralOf, providerBody, answerOf, clientAnswer, clientError } from './lib/wire.ts';
import { listModels, infer } from './lib/upstream.ts';

const ENV = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const brief = readBrief(process.env.MAKEFUN_PROMPT_FILE ?? '');
const providers = new Map(brief.providers.map((p) => [p.id, p]));

async function buildCatalog(): Promise<CatalogEntry[]> {
  const out: CatalogEntry[] = [];
  for (const p of brief.providers) {
    const listing = ENV === '' ? null : await listModels(ENV, p);
    const models = (listing === null ? null : modelsFromListing(p.wire, listing)) ?? modelsFromRatings(brief, p.id);
    out.push({ id: p.id, base_url: p.path, key_env: p.key_env, wire: p.wire, models });
  }
  return out;
}

const dead = new Set<string>();
let served = 0;

async function envJson(path: string, init?: RequestInit): Promise<any> {
  const res = await fetch(`${ENV}${path}`, init);
  return res.json();
}

function requiresFor(id: string | null, n: ReturnType<typeof neutralOf>): Requires {
  const row = id === null ? undefined : brief.manifest.find((r) => r.id === id);
  if (row !== undefined) return row.requires ?? {};
  const vision = n.messages.some((m) => m.parts.some((p) => p.type === 'image'));
  return { tools: n.tools.length > 0, vision, json: n.json };
}

async function serve(api: string, body: any, version: string): Promise<{ status: number; type: string; body: string }> {
  let id: string | null = null;
  try { const w = await envJson('/work'); if (w && w.done === false && typeof w.next === 'string') id = w.next; } catch { id = null; }
  const n = neutralOf(api, body);
  const stream = body.stream === true;
  const list = candidates(brief, requiresFor(id, n), api, n.json, freeSet);
  for (const r of list) {
    const p = providers.get(r.provider);
    if (p === undefined || dead.has(p.id)) continue;
    for (let attempt = 0; attempt < 2; attempt++) {
      const out = await infer(ENV, p, r.model, providerBody(p.wire, api, body, n, r.model), version);
      if (out.kind === 'ok') {
        const a = answerOf(p.wire, out.body);
        if (a === null) break;
        served += 1;
        const reply = clientAnswer(api, stream, a, r.model, served);
        return { status: 200, ...reply };
      }
      if (out.kind === 'dead') dead.add(p.id);
      if (out.kind !== 'refused') break;
    }
  }
  return { status: 503, type: 'application/json', body: clientError(api, 503, 'no free model could answer this request') };
}

function modelsAnswer(anthropic: boolean): string {
  const ids = freeIds(catalog);
  if (anthropic) {
    return JSON.stringify({ data: ids.map((id) => ({ type: 'model', id, display_name: id, created_at: '2026-01-01T00:00:00Z' })), has_more: false, first_id: ids[0] ?? null, last_id: ids[ids.length - 1] ?? null });
  }
  return JSON.stringify({ object: 'list', data: ids.map((id) => ({ id, object: 'model', created: 0, owned_by: 'aggregator' })) });
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve) => {
    const chunks: Buffer[] = [];
    req.on('data', (c: Buffer) => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', () => resolve(''));
  });
}

// Requests are handled one at a time, in arrival order: never two calls in flight to one provider.
let queue: Promise<void> = Promise.resolve();

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const path = (req.url ?? '/').split('?')[0];
  const send = (status: number, type: string, text: string) => { res.writeHead(status, { 'content-type': type }); res.end(text); };
  if (req.method === 'GET' && path === '/v1/models') {
    const anthropic = typeof req.headers['x-api-key'] === 'string' && typeof req.headers.authorization !== 'string';
    return send(200, 'application/json', modelsAnswer(anthropic));
  }
  if (req.method === 'POST' && (path === '/v1/chat/completions' || path === '/v1/messages')) {
    const api = path === '/v1/messages' ? 'anthropic' : 'openai';
    const raw = await readBody(req);
    let body: any;
    try { body = JSON.parse(raw); } catch { return send(400, 'application/json', clientError(api, 400, 'the body is not JSON')); }
    const version = typeof req.headers['anthropic-version'] === 'string' ? req.headers['anthropic-version'] : '2023-06-01';
    const done = queue.then(() => serve(api, body, version));
    queue = done.then(() => undefined, () => undefined);
    const out = await done.catch(() => ({ status: 500, type: 'application/json', body: clientError(api, 500, 'internal error') }));
    send(out.status, out.type, out.body);
    watchDone();
    return;
  }
  send(404, 'application/json', JSON.stringify({ error: 'not_found' }));
}

let watching = false;
function watchDone(): void {
  if (watching || ENV === '') return;
  watching = true;
  const tick = async () => {
    try { const w = await envJson('/work'); if (w && w.done === true) process.exit(0); } catch { /* keep polling */ }
    setTimeout(tick, 250);
  };
  setTimeout(tick, 100);
}

let catalog: CatalogEntry[] = [];
let freeSet = new Set<string>();

async function main(): Promise<void> {
  catalog = await buildCatalog();
  const text = catalogText(catalog);
  if (process.argv.includes('--list')) { process.stdout.write(`${text}\n`); return; }
  for (const e of catalog) for (const m of e.models) if (m.free) freeSet.add(`${e.id}\u0000${m.id}`);
  if (ENV === '') return;
  const server = createServer((req, res) => { void handle(req, res); });
  await new Promise<void>((resolve) => server.listen(0, '0.0.0.0', () => resolve()));
  const addr = server.address();
  const port = typeof addr === 'object' && addr !== null ? addr.port : 0;
  const reg = `{"protocol_version":1,"port":${port},"catalog":${text}}`;
  await envJson('/register', { method: 'POST', headers: { 'content-type': 'application/json' }, body: reg });
  watchDone();
}

void main();
