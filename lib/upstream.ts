// Provider calls: each provider's own auth scheme and routes, one call at a time.

import type { Provider } from './brief.ts';

export function authHeaders(p: Provider): Record<string, string> {
  if (p.auth === 'bearer') return { authorization: `Bearer ${p.key}` };
  if (p.auth === 'x-api-key') return { 'x-api-key': p.key };
  if (p.auth === 'x-goog-api-key') return { 'x-goog-api-key': p.key };
  return {};
}

export function baseOf(envUrl: string, p: Provider): string {
  return `${envUrl}${p.path}`;
}

export async function listModels(envUrl: string, p: Provider): Promise<any | null> {
  const route = p.wire === 'gemini' ? '/v1beta/models' : '/v1/models';
  const headers: Record<string, string> = { ...authHeaders(p) };
  if (p.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(`${baseOf(envUrl, p)}${route}`, { headers });
    if (res.status < 200 || res.status >= 300) { await res.arrayBuffer(); return null; }
    return await res.json();
  } catch {
    return null;
  }
}

export type Outcome =
  | { kind: 'ok'; body: any }
  | { kind: 'dead' }        // budget spent or outage: the provider serves no more this trial
  | { kind: 'refused' }     // a refusal on the provider's schedule: a later call may pass
  | { kind: 'busy' }        // concurrent_requests
  | { kind: 'bad' };        // 400/401/other: this model or call cannot serve

export async function infer(envUrl: string, p: Provider, model: string, body: any, anthropicVersion: string): Promise<Outcome> {
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(p) };
  let url: string;
  if (p.wire === 'openai') url = `${baseOf(envUrl, p)}/v1/chat/completions`;
  else if (p.wire === 'anthropic') { url = `${baseOf(envUrl, p)}/v1/messages`; headers['anthropic-version'] = anthropicVersion; }
  else url = `${baseOf(envUrl, p)}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
  let res: Response;
  let text = '';
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body) });
    text = await res.text();
  } catch {
    return { kind: 'bad' };
  }
  if (res.status >= 200 && res.status < 300) {
    try { return { kind: 'ok', body: JSON.parse(text) }; } catch { return { kind: 'bad' }; }
  }
  if (res.status === 402) return { kind: 'dead' };
  if (res.status === 429 || res.status >= 500) {
    if (text.includes('concurrent_requests')) return { kind: 'busy' };
    if (/budget is spent|is out\b/.test(text)) return { kind: 'dead' };
    return { kind: 'refused' };
  }
  return { kind: 'bad' };
}
