// The brief: provider directory, quality table and workload manifest, plus the catalog and routing order.

import { readFileSync } from 'node:fs';

export interface Provider { id: string; name: string; wire: string; auth: string; key_env: string; key: string; path: string }
export interface Rating { model: string; provider: string; rating_milli: number; context_length: number | null; tools: boolean; vision: boolean; json: boolean }
export interface Requires { tools?: boolean; vision?: boolean; json?: boolean; min_context?: number }
export interface Row { id: string; api: string; stream: boolean; weight: number; requires: Requires }
export interface Brief { providers: Provider[]; ratings: Rating[]; manifest: Row[] }

export function readBrief(file: string): Brief {
  const brief: Brief = { providers: [], ratings: [], manifest: [] };
  if (file === '') return brief;
  let text = '';
  try { text = readFileSync(file, 'utf8'); } catch { return brief; }
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line.startsWith('{')) continue;
    let doc: any;
    try { doc = JSON.parse(line); } catch { continue; }
    const schema = typeof doc.schema === 'string' ? doc.schema : '';
    if (schema.endsWith('/providers/v1') && Array.isArray(doc.providers)) brief.providers = doc.providers;
    if (schema.endsWith('/ratings/v1') && Array.isArray(doc.ratings)) brief.ratings = doc.ratings.filter((r: any) => r.free === true);
    if (schema.endsWith('/workload/v1') && Array.isArray(doc.manifest)) brief.manifest = doc.manifest;
  }
  return brief;
}

export interface CatalogModel { id: string; free: boolean; context_length: number | null }
export interface CatalogEntry { id: string; base_url: string; key_env: string; wire: string; models: CatalogModel[] }

// One provider's model list from its own catalog route answer, in the provider's order.
export function modelsFromListing(wire: string, body: any): CatalogModel[] | null {
  const items = wire === 'gemini' ? body?.models : body?.data;
  if (!Array.isArray(items)) return null;
  const out: CatalogModel[] = [];
  for (const item of items) {
    if (item === null || typeof item !== 'object') continue;
    let id = wire === 'gemini' ? String(item.name ?? '') : String(item.id ?? '');
    if (wire === 'gemini' && id.startsWith('models/')) id = id.slice('models/'.length);
    if (id === '') continue;
    const ctx = typeof item.context_length === 'number' ? item.context_length
      : typeof item.inputTokenLimit === 'number' ? item.inputTokenLimit : null;
    out.push({ id, free: item.free === true, context_length: ctx });
  }
  return out;
}

// The fallback when a listing cannot be read: the quality table's free models for that provider.
export function modelsFromRatings(brief: Brief, providerId: string): CatalogModel[] {
  return brief.ratings.filter((r) => r.provider === providerId)
    .map((r) => ({ id: r.model, free: true, context_length: r.context_length ?? null }));
}

export function catalogText(entries: CatalogEntry[]): string {
  return JSON.stringify({ providers: entries });
}

export function freeIds(entries: CatalogEntry[]): string[] {
  const seen = new Set<string>();
  const ids: string[] = [];
  for (const e of entries) for (const m of e.models) if (m.free && !seen.has(m.id)) { seen.add(m.id); ids.push(m.id); }
  return ids;
}

export function meets(r: Rating, req: Requires): boolean {
  if (req.tools && !r.tools) return false;
  if (req.vision && !r.vision) return false;
  if (req.json && !r.json) return false;
  if (typeof req.min_context === 'number' && (r.context_length ?? 0) < req.min_context) return false;
  return true;
}

// Every legal (provider, model) for one request, best first. A provider appears once, with its best model,
// since a refusal is the provider's and its other models would meet the same budget.
export function candidates(brief: Brief, req: Requires, clientApi: string, jsonMode: boolean, free: Set<string>): Rating[] {
  const wireOf = new Map(brief.providers.map((p) => [p.id, p.wire]));
  const order = new Map(brief.providers.map((p, i) => [p.id, i]));
  const legal = brief.ratings.filter((r) => {
    if (!meets(r, req)) return false;
    if (!free.has(`${r.provider}\u0000${r.model}`)) return false;
    const wire = wireOf.get(r.provider);
    if (wire === undefined) return false;
    if (jsonMode && wire === 'anthropic') return false;
    return true;
  });
  legal.sort((a, b) => {
    if (b.rating_milli !== a.rating_milli) return b.rating_milli - a.rating_milli;
    const sa = wireOf.get(a.provider) === clientApi ? 0 : 1;
    const sb = wireOf.get(b.provider) === clientApi ? 0 : 1;
    if (sa !== sb) return sa - sb;
    return (order.get(a.provider) ?? 0) - (order.get(b.provider) ?? 0);
  });
  const seen = new Set<string>();
  return legal.filter((r) => (seen.has(r.provider) ? false : (seen.add(r.provider), true)));
}
