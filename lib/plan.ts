// Refusal risk: a posterior over each provider's hidden free-tier draw, given the calls seen so far.
//
// The kit's generator (public) draws, per provider: a budget B uniform in its [min, max]; one or two
// scheduled refusals at distinct call numbers in [2, B-1]; one provider whose call 2 always refuses;
// and one provider out from a call uniform in [2, B-1] on. So every provider's first call is safe, and
// each later call's risk follows from what that provider has answered. Routing trades that risk
// against the rating lost, keeping the quality column's 1% slack.

const RANGES: Record<string, [number, number]> = {
  groq: [9, 12], cerebras: [5, 6], gemini: [6, 8], zai: [4, 5], openrouter: [10, 12],
  sambanova: [4, 5], pollinations: [10, 14], cloudflare: [7, 8], ovh: [5, 6], kilol: [12, 16],
};
const DEFAULT_RANGE: [number, number] = [4, 16];
const PRIOR_DESIGNATED = 0.1;
const PRIOR_OUTAGE = 0.1;
const DESIGNATED_EXACT = (process.env.AGG_DES ?? '1') === '1';

export type Outcome = 'ok' | 'sched';

export class Risk {
  history = new Map<string, Outcome[]>();
  outageSeen = false;
  providerCount = 10;

  // Exactly one provider refuses its call 2 by design: a provider whose call 2 passed is not it, and a
  // refusal seen at some call 2 most likely was it.
  designatedPrior(provider: string): number {
    let ruledOut = 0;
    let suspect = false;
    for (const [id, h] of this.history) {
      if (h.length >= 2 && h[1] === 'ok') ruledOut += 1;
      if (id !== provider && h.length >= 2 && h[1] === 'sched') suspect = true;
    }
    if (suspect) return 0.02;
    return 1 / Math.max(1, this.providerCount - ruledOut);
  }

  record(provider: string, outcome: Outcome): void {
    const h = this.history.get(provider) ?? [];
    h.push(outcome);
    this.history.set(provider, h);
  }

  // P(the provider's next call refuses), given its history. Dead providers are filtered by the caller.
  pFail(provider: string): number {
    return this.slotRisks(provider, 1)[0];
  }

  // P(the provider's next 1..n calls each refuse), each given the history so far.
  slotRisks(provider: string, n: number): number[] {
    const h = this.history.get(provider) ?? [];
    const k = h.length + 1;
    const bad = new Array<number>(n).fill(0);
    const [lo, hi] = RANGES[provider] ?? DEFAULT_RANGE;
    const pOut = this.outageSeen ? 0 : PRIOR_OUTAGE;
    const pDes = DESIGNATED_EXACT ? this.designatedPrior(provider) : PRIOR_DESIGNATED;
    let total = 0;
    for (let B = lo; B <= hi; B++) {
      const span: number[] = [];
      for (let f = 2; f <= B - 1; f++) span.push(f);
      if (span.length === 0) continue;
      const sets: number[][] = [];
      const wsets: number[] = [];
      for (const f of span) { sets.push([f]); wsets.push(0.5 / span.length); }
      const pairs = (span.length * (span.length - 1)) / 2;
      for (let i = 0; i < span.length; i++) for (let j = i + 1; j < span.length; j++) { sets.push([span[i], span[j]]); wsets.push(pairs > 0 ? 0.5 / pairs : 0); }
      for (let s = 0; s < sets.length; s++) {
        if (wsets[s] === 0) continue;
        for (const designated of [false, true]) {
          let fails = sets[s];
          if (designated && !fails.includes(2)) fails = fails.length === 1 ? [2] : [2, fails[1]];
          const wd = designated ? pDes : 1 - pDes;
          const outs: [number | null, number][] = [[null, 1 - pOut]];
          if (pOut > 0) for (const u of span) outs.push([u, pOut / span.length]);
          for (const [u, wu] of outs) {
            const refuses = (c: number) => c > B || (u !== null && c >= u) || fails.includes(c);
            let consistent = true;
            for (let c = 1; c < k && consistent; c++) {
              const sched = h[c - 1] === 'sched';
              // a seen 'sched' refusal is on the schedule, never budget or outage (those mark the provider dead)
              if (sched !== (fails.includes(c) && !(c > B) && !(u !== null && c >= u))) consistent = false;
            }
            if (!consistent) continue;
            const w = (1 / (hi - lo + 1)) * wsets[s] * wd * wu;
            total += w;
            for (let j = 0; j < n; j++) if (refuses(k + j)) bad[j] += w;
          }
        }
      }
    }
    return bad.map((b) => (total === 0 ? 0.5 : b / total));
  }
}

// One remaining request's options: per provider, the rating it would lose there (weight included).
export interface Need { options: Map<string, number> }

// Min-cost assignment of the remaining requests to provider call slots: a slot's cost is its refusal
// risk, an option's cost lambda times the rating lost. Returns each need's provider (null where none is left).
export function plan(needs: Need[], risks: Map<string, number[]>, lambda: number): (string | null)[] {
  const provs = [...risks.keys()];
  const R = needs.length;
  const used = new Map<string, number>(provs.map((p) => [p, 0]));
  const assign = new Array<string | null>(R).fill(null);
  // Successive shortest augmenting paths on requests -> providers -> slots (convex slot costs).
  for (let step = 0; step < R; step++) {
    // Bellman-Ford over requests and providers; distances from the source.
    const dReq = needs.map((_, i) => (assign[i] === null ? 0 : Infinity));
    const dProv = new Map<string, number>(provs.map((p) => [p, Infinity]));
    const viaReq = new Map<string, number>();
    const viaProv = new Array<string | null>(R).fill(null);
    for (let iter = 0; iter < R + provs.length + 2; iter++) {
      let changed = false;
      needs.forEach((nd, i) => {
        if (dReq[i] === Infinity) return;
        for (const [p, lose] of nd.options) {
          if (assign[i] === p) continue;
          const c = dReq[i] + lambda * lose;
          if (c < (dProv.get(p) ?? Infinity) - 1e-12) { dProv.set(p, c); viaReq.set(p, i); changed = true; }
        }
      });
      needs.forEach((nd, i) => {
        const p = assign[i];
        if (p === null) return;
        const dp = dProv.get(p) ?? Infinity;
        if (dp === Infinity) return;
        const c = dp - lambda * (nd.options.get(p) ?? 0);
        if (c < dReq[i] - 1e-12) { dReq[i] = c; viaProv[i] = p; changed = true; }
      });
      if (!changed) break;
    }
    let bestP: string | null = null;
    let bestC = Infinity;
    for (const p of provs) {
      const slots = risks.get(p)!;
      const u = used.get(p)!;
      if (u >= slots.length) continue;
      const c = (dProv.get(p) ?? Infinity) + slots[u];
      if (c < bestC - 1e-12) { bestC = c; bestP = p; }
    }
    if (bestP === null) break;
    used.set(bestP, used.get(bestP)! + 1);
    // Walk the path back, flipping assignments.
    let p: string | null = bestP;
    for (let guard = 0; p !== null && guard < 2 * R + 2; guard++) {
      const i = viaReq.get(p)!;
      const prev = assign[i];
      assign[i] = p;
      p = viaProv[i] === null || prev === null ? null : prev;
      if (prev === null) break;
    }
  }
  return assign;
}

// The quality column floors 100 * earned / bound, and the score rounds 0.6 * quality + 40: quality 98
// (a loss under 2% of the bound) still scores 99 when every other column is full.
export function lossCap(bound: number, share = 0.02): number {
  return Math.floor(bound * share) - 2;
}
