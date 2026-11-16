// Wire translation between the OpenAI, Anthropic and Gemini APIs, requests and answers, JSON and SSE.

export interface Part { type: 'text' | 'image'; text?: string; mime?: string; data?: string; url?: string }
export interface Msg { role: 'user' | 'assistant'; parts: Part[] }
export interface Tool { name: string; description?: string; schema: any }
export interface Neutral { system: string; messages: Msg[]; tools: Tool[]; maxTokens: number | null; stops: string[]; json: boolean; temperature: number | null }
export interface ToolCall { id: string; name: string; args: any }
export interface Answer { text: string; calls: ToolCall[]; stop: string; input: number; output: number }

const num = (v: any): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const strs = (v: any): string[] => (typeof v === 'string' ? [v] : Array.isArray(v) ? v.filter((x) => typeof x === 'string') : []);

function imagePart(url: string): Part {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  return m ? { type: 'image', mime: m[1] || 'image/png', data: m[2] } : { type: 'image', url };
}

function partsOf(content: any, api: string): Part[] {
  if (typeof content === 'string') return [{ type: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const out: Part[] = [];
  for (const b of content) {
    if (b === null || typeof b !== 'object') continue;
    if (b.type === 'text' && typeof b.text === 'string') out.push({ type: 'text', text: b.text });
    else if (api === 'openai' && b.type === 'image_url' && b.image_url && typeof b.image_url.url === 'string') out.push(imagePart(b.image_url.url));
    else if (api === 'anthropic' && b.type === 'image' && b.source) {
      if (b.source.type === 'base64') out.push({ type: 'image', mime: b.source.media_type ?? 'image/png', data: b.source.data });
      else if (typeof b.source.url === 'string') out.push({ type: 'image', url: b.source.url });
    }
  }
  return out;
}

const textJoin = (parts: Part[]) => parts.filter((p) => p.type === 'text').map((p) => p.text ?? '').join('\n');

export function neutralOf(api: string, body: any): Neutral {
  const n: Neutral = { system: '', messages: [], tools: [], maxTokens: null, stops: [], json: false, temperature: num(body.temperature) };
  const msgs = Array.isArray(body.messages) ? body.messages : [];
  if (api === 'openai') {
    const sys: string[] = [];
    for (const m of msgs) {
      if (m === null || typeof m !== 'object') continue;
      if (m.role === 'system') { const t = textJoin(partsOf(m.content, api)); if (t !== '') sys.push(t); }
      else if (m.role === 'user' || m.role === 'assistant') n.messages.push({ role: m.role, parts: partsOf(m.content, api) });
    }
    n.system = sys.join('\n');
    for (const t of Array.isArray(body.tools) ? body.tools : []) {
      if (t && t.function && typeof t.function.name === 'string') n.tools.push({ name: t.function.name, description: t.function.description, schema: t.function.parameters ?? { type: 'object' } });
    }
    n.maxTokens = num(body.max_tokens) ?? num(body.max_completion_tokens);
    n.stops = strs(body.stop);
    const f = body.response_format && body.response_format.type;
    n.json = f === 'json_object' || f === 'json_schema';
  } else {
    n.system = typeof body.system === 'string' ? body.system : textJoin(partsOf(body.system, api));
    for (const m of msgs) if (m && (m.role === 'user' || m.role === 'assistant')) n.messages.push({ role: m.role, parts: partsOf(m.content, api) });
    for (const t of Array.isArray(body.tools) ? body.tools : []) {
      if (t && typeof t.name === 'string') n.tools.push({ name: t.name, description: t.description, schema: t.input_schema ?? { type: 'object' } });
    }
    n.maxTokens = num(body.max_tokens);
    n.stops = strs(body.stop_sequences);
  }
  return n;
}

// The body sent to a provider. Same wire: the client's own body, model chosen, unstreamed.
export function providerBody(wire: string, api: string, client: any, n: Neutral, model: string): any {
  if (wire === api) {
    const b = { ...client, model, stream: false };
    delete b.stream_options;
    return b;
  }
  if (wire === 'openai') {
    const messages: any[] = [];
    if (n.system !== '') messages.push({ role: 'system', content: n.system });
    for (const m of n.messages) {
      const blocks = m.parts.map((p) => (p.type === 'text' ? { type: 'text', text: p.text }
        : { type: 'image_url', image_url: { url: p.data !== undefined ? `data:${p.mime};base64,${p.data}` : p.url } }));
      messages.push({ role: m.role, content: m.role === 'assistant' ? textJoin(m.parts) : blocks });
    }
    const b: any = { model, messages };
    if (n.maxTokens !== null) b.max_tokens = n.maxTokens;
    if (n.temperature !== null) b.temperature = n.temperature;
    if (n.stops.length > 0) b.stop = n.stops;
    if (n.tools.length > 0) b.tools = n.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.schema } }));
    if (n.json) b.response_format = { type: 'json_object' };
    return b;
  }
  if (wire === 'anthropic') {
    const messages = n.messages.map((m) => ({ role: m.role, content: m.parts.map((p) => (p.type === 'text' ? { type: 'text', text: p.text }
      : { type: 'image', source: p.data !== undefined ? { type: 'base64', media_type: p.mime, data: p.data } : { type: 'url', url: p.url } })) }));
    const b: any = { model, messages };
    if (n.maxTokens !== null) b.max_tokens = n.maxTokens;
    if (n.system !== '') b.system = n.system;
    if (n.temperature !== null) b.temperature = n.temperature;
    if (n.stops.length > 0) b.stop_sequences = n.stops;
    if (n.tools.length > 0) b.tools = n.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema }));
    return b;
  }
  const contents = n.messages.map((m) => ({ role: m.role === 'assistant' ? 'model' : 'user', parts: m.parts.map((p) => (p.type === 'text' ? { text: p.text }
    : p.data !== undefined ? { inlineData: { mimeType: p.mime, data: p.data } } : { fileData: { fileUri: p.url } })) }));
  const b: any = { contents };
  if (n.system !== '') b.systemInstruction = { parts: [{ text: n.system }] };
  if (n.tools.length > 0) b.tools = [{ functionDeclarations: n.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.schema })) }];
  const g: any = {};
  if (n.maxTokens !== null) g.maxOutputTokens = n.maxTokens;
  if (n.stops.length > 0) g.stopSequences = n.stops;
  if (n.temperature !== null) g.temperature = n.temperature;
  if (n.json) g.responseMimeType = 'application/json';
  if (Object.keys(g).length > 0) b.generationConfig = g;
  return b;
}

// Stop reasons, normalised to the OpenAI family names.
const FAMILY: Record<string, string> = { stop: 'stop', end_turn: 'stop', stop_sequence: 'stop', STOP: 'stop', length: 'length', max_tokens: 'length', MAX_TOKENS: 'length',
  tool_calls: 'tool_calls', function_call: 'tool_calls', tool_use: 'tool_calls', content_filter: 'content_filter', refusal: 'content_filter', SAFETY: 'content_filter' };
const TO_ANTHROPIC: Record<string, string> = { stop: 'end_turn', length: 'max_tokens', tool_calls: 'tool_use', content_filter: 'refusal' };

export function answerOf(wire: string, v: any): Answer | null {
  if (v === null || typeof v !== 'object') return null;
  const a: Answer = { text: '', calls: [], stop: 'stop', input: 0, output: 0 };
  if (wire === 'openai') {
    const c = Array.isArray(v.choices) ? v.choices[0] : null;
    if (!c || !c.message) return null;
    a.text = typeof c.message.content === 'string' ? c.message.content : '';
    for (const t of Array.isArray(c.message.tool_calls) ? c.message.tool_calls : []) {
      let args: any = {};
      try { args = JSON.parse(t.function?.arguments ?? '{}'); } catch { args = {}; }
      a.calls.push({ id: String(t.id ?? `call_${a.calls.length}`), name: String(t.function?.name ?? ''), args });
    }
    a.stop = FAMILY[c.finish_reason] ?? 'stop';
    a.input = num(v.usage?.prompt_tokens) ?? 0; a.output = num(v.usage?.completion_tokens) ?? 0;
  } else if (wire === 'anthropic') {
    if (!Array.isArray(v.content)) return null;
    for (const b of v.content) {
      if (b?.type === 'text') a.text += b.text ?? '';
      else if (b?.type === 'tool_use') a.calls.push({ id: String(b.id), name: String(b.name), args: b.input ?? {} });
    }
    a.stop = FAMILY[v.stop_reason] ?? 'stop';
    a.input = num(v.usage?.input_tokens) ?? 0; a.output = num(v.usage?.output_tokens) ?? 0;
  } else {
    const c = Array.isArray(v.candidates) ? v.candidates[0] : null;
    if (!c) return null;
    for (const p of Array.isArray(c.content?.parts) ? c.content.parts : []) {
      if (typeof p.text === 'string') a.text += p.text;
      else if (p.functionCall) a.calls.push({ id: `call_${a.calls.length}`, name: String(p.functionCall.name), args: p.functionCall.args ?? {} });
    }
    a.stop = a.calls.length > 0 ? 'tool_calls' : FAMILY[c.finishReason] ?? 'stop';
    a.input = num(v.usageMetadata?.promptTokenCount) ?? 0; a.output = num(v.usageMetadata?.candidatesTokenCount) ?? 0;
  }
  return a;
}

const sse = (event: string | null, data: any) => `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;

// The client's answer: content type and body text.
export function clientAnswer(api: string, stream: boolean, a: Answer, model: string, n: number): { type: string; body: string } {
  if (api === 'openai') {
    const id = `chatcmpl-agg-${n}`;
    const usage = { prompt_tokens: a.input, completion_tokens: a.output, total_tokens: a.input + a.output };
    const calls = a.calls.map((c, i) => ({ index: i, id: c.id, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.args) } }));
    if (!stream) {
      const message: any = { role: 'assistant', content: a.text };
      if (calls.length > 0) message.tool_calls = calls.map(({ index, ...rest }) => rest);
      return { type: 'application/json', body: JSON.stringify({ id, object: 'chat.completion', created: 0, model, choices: [{ index: 0, message, finish_reason: a.stop, logprobs: null }], usage }) };
    }
    const chunk = (delta: any, finish: string | null) => ({ id, object: 'chat.completion.chunk', created: 0, model, choices: [{ index: 0, delta, finish_reason: finish }] });
    let out = sse(null, chunk({ role: 'assistant', content: '' }, null));
    if (a.text !== '') out += sse(null, chunk({ content: a.text }, null));
    if (calls.length > 0) out += sse(null, chunk({ tool_calls: calls }, null));
    out += sse(null, chunk({}, a.stop));
    out += sse(null, { id, object: 'chat.completion.chunk', created: 0, model, choices: [], usage });
    return { type: 'text/event-stream', body: out + 'data: [DONE]\n\n' };
  }
  const id = `msg_agg_${n}`;
  const stop = TO_ANTHROPIC[a.stop] ?? 'end_turn';
  const blocks: any[] = [];
  if (a.text !== '' || a.calls.length === 0) blocks.push({ type: 'text', text: a.text });
  for (const c of a.calls) blocks.push({ type: 'tool_use', id: c.id, name: c.name, input: c.args });
  if (!stream) {
    return { type: 'application/json', body: JSON.stringify({ id, type: 'message', role: 'assistant', model, content: blocks, stop_reason: stop, stop_sequence: null, usage: { input_tokens: a.input, output_tokens: a.output } }) };
  }
  let out = sse('message_start', { type: 'message_start', message: { id, type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: a.input, output_tokens: 0 } } });
  blocks.forEach((b, i) => {
    if (b.type === 'text') {
      out += sse('content_block_start', { type: 'content_block_start', index: i, content_block: { type: 'text', text: '' } });
      out += sse('content_block_delta', { type: 'content_block_delta', index: i, delta: { type: 'text_delta', text: b.text } });
    } else {
      out += sse('content_block_start', { type: 'content_block_start', index: i, content_block: { type: 'tool_use', id: b.id, name: b.name, input: {} } });
      out += sse('content_block_delta', { type: 'content_block_delta', index: i, delta: { type: 'input_json_delta', partial_json: JSON.stringify(b.input) } });
    }
    out += sse('content_block_stop', { type: 'content_block_stop', index: i });
  });
  out += sse('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: a.output } });
  out += sse('message_stop', { type: 'message_stop' });
  return { type: 'text/event-stream', body: out };
}

export function clientError(api: string, status: number, message: string): string {
  if (api === 'openai') return JSON.stringify({ error: { message, type: status === 429 ? 'rate_limit_error' : 'api_error', param: null, code: null } });
  return JSON.stringify({ type: 'error', error: { type: status === 429 ? 'rate_limit_error' : 'api_error', message } });
}
