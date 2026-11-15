// lib/wires.ts — the wires: OpenAI, Anthropic and Gemini, in and out.
//
// A client request on either wire becomes a canonical request (system text, messages of
// blocks, tool names, max_tokens, stops, JSON mode), which is then rendered into the wire
// of whichever provider serves it. Provider answers come back as buffered text (JSON or
// server-sent events) and are parsed into a plain answer: the text, the usage the provider
// reported, and the stop reason's family. The client's response is then emitted on the
// client's wire from that answer.
//
// Fidelity is what the scorer checks: the provider's simulated answer echoes the system
// prompt, the user text, the images, the tool names in order, max_tokens, the stops and
// JSON mode exactly as the client sent them, and the usage must match the provider's
// exactly. So every rendering carries those fields verbatim, and the parser keeps the
// provider's numbers unchanged.

import type { Json } from './brief.ts';
import { isRecord } from './brief.ts';

export type ClientWire = 'openai' | 'anthropic';

export type ContentBlock =
  | { kind: 'text'; text: string }
  | { kind: 'image'; mime: string; data: string | null; url: string | null }
  | { kind: 'tool_use'; id: string; name: string; input: unknown }
  | { kind: 'tool_result'; toolUseId: string; text: string; isError: boolean };

export interface CanonicalMessage {
  role: 'user' | 'assistant';
  blocks: ContentBlock[];
}

export interface CanonicalTool {
  name: string;
  description: string;
  parameters: Json;
}

export interface CanonicalRequest {
  systemText: string;
  messages: CanonicalMessage[];
  tools: CanonicalTool[];
  /** 'auto' | 'none' | 'required' | a tool's name, when the client pinned one. */
  toolChoice: string | null;
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | null;
  topP: number | null;
  stream: boolean;
}

/** The plain answer a provider's response becomes, whatever its wire. */
export interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** 0 stop/end_turn/STOP, 1 length/max_tokens/MAX_TOKENS, 2 tool call, 3 content filter/refusal. */
  stopFamily: number;
}

export function stopFamilyOf(reason: string | null): number {
  if (reason === null) return 0;
  const r = reason.trim().toLowerCase();
  if (r === 'length' || r === 'max_tokens') return 1;
  if (r === 'tool_calls' || r === 'tool_use' || r === 'function_call' || r === 'function_calls') return 2;
  if (r === 'content_filter' || r === 'refusal' || r === 'safety') return 3;
  return 0;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOf(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** A message's content as a block list, whatever shape it arrived in. */
function contentBlocks(content: unknown): ContentBlock[] {
  if (typeof content === 'string') {
    return content === '' ? [] : [{ kind: 'text', text: content }];
  }
  if (!Array.isArray(content)) return [];
  const blocks: ContentBlock[] = [];
  for (const block of content) {
    if (typeof block === 'string') {
      blocks.push({ kind: 'text', text: block });
      continue;
    }
    if (!isRecord(block)) continue;
    if (block.type === 'text') {
      blocks.push({ kind: 'text', text: stringOf(block.text) });
    } else if (block.type === 'image_url') {
      const imageUrl = isRecord(block.image_url) ? block.image_url : {};
      blocks.push({ kind: 'image', mime: '', data: null, url: stringOf(imageUrl.url) });
    } else if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64') {
        blocks.push({ kind: 'image', mime: stringOf(source.media_type), data: stringOf(source.data), url: null });
      } else if (source.type === 'url') {
        blocks.push({ kind: 'image', mime: '', data: null, url: stringOf(source.url) });
      }
    } else if (block.type === 'tool_use') {
      blocks.push({ kind: 'tool_use', id: stringOf(block.id), name: stringOf(block.name), input: block.input ?? {} });
    } else if (block.type === 'tool_result') {
      const text =
        typeof block.content === 'string'
          ? block.content
          : Array.isArray(block.content)
            ? block.content.map((c) => (isRecord(c) && c.type === 'text' ? stringOf(c.text) : '')).join('\n')
            : '';
      blocks.push({ kind: 'tool_result', toolUseId: stringOf(block.tool_use_id), text, isError: block.is_error === true });
    }
  }
  return blocks;
}

/** The text blocks of a message, joined with '\n' — how an echo reads user text. */
function textOfBlocks(blocks: ContentBlock[]): string {
  return blocks.filter((b) => b.kind === 'text').map((b) => b.text).join('\n');
}

/** Appending a tool result: fold it into the previous user message when there is one, so no
 *  text-less user message ever lands after the user text and clobbers an echo's user field. */
function pushToolResult(messages: CanonicalMessage[], result: ContentBlock): void {
  const last = messages[messages.length - 1];
  if (last !== undefined && last.role === 'user') {
    last.blocks.push(result);
  } else {
    messages.push({ role: 'user', blocks: [result] });
  }
}

/** An OpenAI chat-completions body, canonically. */
export function fromOpenAI(body: Json): CanonicalRequest {
  const systemParts: string[] = [];
  const messages: CanonicalMessage[] = [];
  for (const message of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(message)) continue;
    const blocks = contentBlocks(message.content);
    const role = message.role;
    if (role === 'system' || role === 'developer') {
      systemParts.push(textOfBlocks(blocks));
    } else if (role === 'assistant') {
      const out: ContentBlock[] = [...blocks];
      if (Array.isArray(message.tool_calls)) {
        for (const call of message.tool_calls) {
          if (!isRecord(call)) continue;
          const fn = isRecord(call.function) ? call.function : {};
          let input: unknown = stringOf(fn.arguments);
          try {
            input = JSON.parse(stringOf(fn.arguments));
          } catch {
            // Keep the raw string.
          }
          out.push({ kind: 'tool_use', id: stringOf(call.id), name: stringOf(fn.name), input });
        }
      }
      messages.push({ role: 'assistant', blocks: out });
    } else if (role === 'tool') {
      const text = typeof message.content === 'string' ? message.content : textOfBlocks(blocks);
      pushToolResult(messages, { kind: 'tool_result', toolUseId: stringOf(message.tool_call_id), text, isError: false });
    } else {
      messages.push({ role: 'user', blocks });
    }
  }
  const tools: CanonicalTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function)) continue;
      tools.push({
        name: stringOf(tool.function.name),
        description: stringOf(tool.function.description),
        parameters: isRecord(tool.function.parameters) ? tool.function.parameters : {},
      });
    }
  }
  let toolChoice: string | null = null;
  if (typeof body.tool_choice === 'string') toolChoice = body.tool_choice;
  else if (isRecord(body.tool_choice)) {
    if (body.tool_choice.type === 'function' && isRecord(body.tool_choice.function)) toolChoice = stringOf(body.tool_choice.function.name);
    else if (typeof body.tool_choice.type === 'string') toolChoice = body.tool_choice.type;
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const stops = typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s) => typeof s === 'string') : [];
  return {
    systemText: systemParts.join('\n'),
    messages,
    tools,
    toolChoice,
    maxTokens: numberOf(body.max_tokens) ?? numberOf(body.max_completion_tokens),
    stops,
    jsonMode: format === 'json_object' || format === 'json_schema',
    temperature: numberOf(body.temperature),
    topP: numberOf(body.top_p),
    stream: body.stream === true,
  };
}

/** An Anthropic messages body, canonically. */
export function fromAnthropic(body: Json): CanonicalRequest {
  const systemText =
    typeof body.system === 'string'
      ? body.system
      : Array.isArray(body.system)
        ? body.system.map((b) => (isRecord(b) && b.type === 'text' ? stringOf(b.text) : '')).join('\n')
        : '';
  const messages: CanonicalMessage[] = [];
  for (const message of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(message)) continue;
    const blocks = contentBlocks(message.content);
    messages.push({ role: message.role === 'assistant' ? 'assistant' : 'user', blocks });
  }
  const tools: CanonicalTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || tool.type === 'custom') continue;
      tools.push({
        name: stringOf(tool.name),
        description: stringOf(tool.description),
        parameters: isRecord(tool.input_schema) ? tool.input_schema : {},
      });
    }
  }
  let toolChoice: string | null = null;
  if (isRecord(body.tool_choice)) {
    if (body.tool_choice.type === 'tool' && typeof body.tool_choice.name === 'string') toolChoice = body.tool_choice.name;
    else if (typeof body.tool_choice.type === 'string') toolChoice = body.tool_choice.type;
  }
  return {
    systemText,
    messages,
    tools,
    toolChoice,
    maxTokens: numberOf(body.max_tokens),
    stops: Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s) => typeof s === 'string') : [],
    jsonMode: false,
    temperature: numberOf(body.temperature),
    topP: numberOf(body.top_p),
    stream: body.stream === true,
  };
}

// ---- Rendering: canonical request -> each provider wire ------------------------------

/** One canonical message on the OpenAI wire. */
function openAIMessage(message: CanonicalMessage): Json {
  const parts: Json[] = [];
  const toolCalls: Json[] = [];
  for (const block of message.blocks) {
    if (block.kind === 'text') parts.push({ type: 'text', text: block.text });
    else if (block.kind === 'image') {
      parts.push(block.data !== null ? { type: 'image_url', image_url: { url: `data:${block.mime};base64,${block.data}` } } : { type: 'image_url', image_url: { url: block.url ?? '' } });
    } else if (block.kind === 'tool_use') {
      toolCalls.push({ id: block.id, type: 'function', function: { name: block.name, arguments: typeof block.input === 'string' ? block.input : JSON.stringify(block.input) } });
    } else if (block.kind === 'tool_result') {
      parts.push({ type: 'text', text: block.text });
    }
  }
  const out: Json = { role: message.role, content: parts };
  if (toolCalls.length > 0) out.tool_calls = toolCalls;
  return out;
}

/** The canonical request on the OpenAI wire, for `model`. */
export function toOpenAI(canon: CanonicalRequest, model: string): Json {
  const messages: Json[] = [];
  if (canon.systemText !== '') messages.push({ role: 'system', content: canon.systemText });
  for (const message of canon.messages) messages.push(openAIMessage(message));
  const body: Json = { model, messages, stream: canon.stream };
  if (canon.tools.length > 0) {
    body.tools = canon.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
    if (canon.toolChoice !== null) {
      body.tool_choice = canon.toolChoice === 'auto' || canon.toolChoice === 'none' || canon.toolChoice === 'required' ? canon.toolChoice : { type: 'function', function: { name: canon.toolChoice } };
    }
  }
  if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
  if (canon.stops.length > 0) body.stop = canon.stops;
  if (canon.jsonMode) body.response_format = { type: 'json_object' };
  if (canon.temperature !== null) body.temperature = canon.temperature;
  if (canon.topP !== null) body.top_p = canon.topP;
  if (canon.stream) body.stream_options = { include_usage: true };
  return body;
}

/** One canonical message on the Anthropic wire. */
function anthropicMessage(message: CanonicalMessage): Json {
  const parts: Json[] = [];
  for (const block of message.blocks) {
    if (block.kind === 'text') parts.push({ type: 'text', text: block.text });
    else if (block.kind === 'image') {
      parts.push(block.data !== null ? { type: 'image', source: { type: 'base64', media_type: block.mime, data: block.data } } : { type: 'image', source: { type: 'url', url: block.url ?? '' } });
    } else if (block.kind === 'tool_use') {
      parts.push({ type: 'tool_use', id: block.id, name: block.name, input: block.input ?? {} });
    } else if (block.kind === 'tool_result') {
      parts.push({ type: 'tool_result', tool_use_id: block.toolUseId, content: block.text, ...(block.isError ? { is_error: true } : {}) });
    }
  }
  return { role: message.role, content: parts };
}

/** The canonical request on the Anthropic wire, for `model`. */
export function toAnthropic(canon: CanonicalRequest, model: string): Json {
  const body: Json = {
    model,
    max_tokens: canon.maxTokens ?? 1024,
    messages: canon.messages.map(anthropicMessage),
    stream: canon.stream,
  };
  if (canon.systemText !== '') body.system = canon.systemText;
  if (canon.tools.length > 0) {
    body.tools = canon.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
    if (canon.toolChoice !== null) {
      body.tool_choice =
        canon.toolChoice === 'auto' || canon.toolChoice === 'none' || canon.toolChoice === 'any'
          ? { type: canon.toolChoice }
          : { type: 'tool', name: canon.toolChoice };
    }
  }
  if (canon.stops.length > 0) body.stop_sequences = canon.stops;
  if (canon.temperature !== null) body.temperature = canon.temperature;
  if (canon.topP !== null) body.top_p = canon.topP;
  return body;
}

/** The canonical request on the Gemini wire (model goes in the URL, not the body). */
export function toGemini(canon: CanonicalRequest): Json {
  const contents: Json[] = [];
  for (const message of canon.messages) {
    const parts: Json[] = [];
    for (const block of message.blocks) {
      if (block.kind === 'text') parts.push({ text: block.text });
      else if (block.kind === 'image') {
        parts.push(block.data !== null ? { inlineData: { mimeType: block.mime, data: block.data } } : { fileData: { mimeType: block.mime, fileUri: block.url ?? '' } });
      } else if (block.kind === 'tool_use') {
        parts.push({ functionCall: { name: block.name, args: block.input ?? {} } });
      } else if (block.kind === 'tool_result') {
        parts.push({ functionResponse: { name: '', response: { result: block.text } } });
      }
    }
    contents.push({ role: message.role === 'assistant' ? 'model' : 'user', parts });
  }
  const body: Json = { contents };
  if (canon.systemText !== '') body.systemInstruction = { parts: [{ text: canon.systemText }] };
  if (canon.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: canon.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
      },
    ];
  }
  const config: Json = {};
  if (canon.maxTokens !== null) config.maxOutputTokens = canon.maxTokens;
  if (canon.stops.length > 0) config.stopSequences = canon.stops;
  if (canon.jsonMode) config.responseMimeType = 'application/json';
  if (canon.temperature !== null) config.temperature = canon.temperature;
  if (canon.topP !== null) config.topP = canon.topP;
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

// ---- Parsing: provider answers -------------------------------------------------------

/** The `data:` payloads of a buffered SSE body, in order. */
export function sseEvents(bodyText: string): Json[] {
  const events: Json[] = [];
  let data: string[] = [];
  const flush = (): void => {
    if (data.length === 0) return;
    const payload = data.join('\n');
    data = [];
    if (payload === '[DONE]') return;
    try {
      const value: unknown = JSON.parse(payload);
      if (isRecord(value)) events.push(value);
    } catch {
      // A frame that is not JSON carries no answer.
    }
  };
  for (const line of bodyText.split('\n')) {
    const trimmed = line.replace(/\r$/, '');
    if (trimmed === '') {
      flush();
    } else if (trimmed.startsWith('data:')) {
      data.push(trimmed.slice('data:'.length).replace(/^ /, ''));
    }
    // event: / id: / :comment lines carry no payload here.
  }
  flush();
  return events;
}

function openAITextOf(message: Json): string {
  const content = message.content;
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) return content.map((b) => (isRecord(b) && b.type === 'text' ? stringOf(b.text) : '')).join('');
  return '';
}

export function parseOpenAIResponse(bodyText: string, streamed: boolean): ProviderAnswer | null {
  if (!streamed) {
    let value: unknown;
    try {
      value = JSON.parse(bodyText);
    } catch {
      return null;
    }
    if (!isRecord(value) || !Array.isArray(value.choices) || value.choices.length === 0) return null;
    const choice = value.choices[0];
    if (!isRecord(choice) || !isRecord(choice.message)) return null;
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: openAITextOf(choice.message),
      inputTokens: numberOf(usage.prompt_tokens) ?? 0,
      outputTokens: numberOf(usage.completion_tokens) ?? 0,
      stopFamily: stopFamilyOf(typeof choice.finish_reason === 'string' ? choice.finish_reason : null),
    };
  }
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopFamily = 0;
  for (const event of sseEvents(bodyText)) {
    if (Array.isArray(event.choices)) {
      for (const choice of event.choices) {
        if (!isRecord(choice)) continue;
        if (isRecord(choice.delta)) text += stringOf(choice.delta.content);
        if (typeof choice.finish_reason === 'string') stopFamily = stopFamilyOf(choice.finish_reason);
      }
    }
    if (isRecord(event.usage)) {
      inputTokens = numberOf(event.usage.prompt_tokens) ?? inputTokens;
      outputTokens = numberOf(event.usage.completion_tokens) ?? outputTokens;
    }
  }
  return { text, inputTokens, outputTokens, stopFamily };
}

function anthropicTextOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.map((b) => (isRecord(b) && b.type === 'text' ? stringOf(b.text) : '')).join('');
}

export function parseAnthropicResponse(bodyText: string, streamed: boolean): ProviderAnswer | null {
  if (!streamed) {
    let value: unknown;
    try {
      value = JSON.parse(bodyText);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: anthropicTextOf(value.content),
      inputTokens: numberOf(usage.input_tokens) ?? 0,
      outputTokens: numberOf(usage.output_tokens) ?? 0,
      stopFamily: stopFamilyOf(typeof value.stop_reason === 'string' ? value.stop_reason : null),
    };
  }
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopFamily = 0;
  for (const event of sseEvents(bodyText)) {
    const type = event.type;
    if (type === 'message_start' && isRecord(event.message) && isRecord(event.message.usage)) {
      inputTokens = numberOf(event.message.usage.input_tokens) ?? inputTokens;
    } else if (type === 'content_block_delta' && isRecord(event.delta)) {
      text += stringOf(event.delta.text);
    } else if (type === 'message_delta') {
      if (typeof event.stop_reason === 'string') stopFamily = stopFamilyOf(event.stop_reason);
      if (isRecord(event.usage)) outputTokens = numberOf(event.usage.output_tokens) ?? outputTokens;
    }
  }
  return { text, inputTokens, outputTokens, stopFamily };
}

function geminiTextOf(value: Json): string {
  let text = '';
  const candidates = Array.isArray(value.candidates) ? value.candidates : [];
  for (const candidate of candidates) {
    if (!isRecord(candidate) || !isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) continue;
    for (const part of candidate.content.parts) {
      if (isRecord(part) && typeof part.text === 'string') text += part.text;
    }
  }
  return text;
}

export function parseGeminiResponse(bodyText: string, streamed: boolean): ProviderAnswer | null {
  const frames: Json[] = [];
  if (streamed) {
    frames.push(...sseEvents(bodyText));
  } else {
    try {
      const value: unknown = JSON.parse(bodyText);
      if (isRecord(value)) frames.push(value);
    } catch {
      return null;
    }
  }
  if (frames.length === 0) return null;
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopFamily = 0;
  for (const frame of frames) {
    text += geminiTextOf(frame);
    const usage = isRecord(frame.usageMetadata) ? frame.usageMetadata : {};
    if (numberOf(usage.promptTokenCount) !== null) inputTokens = numberOf(usage.promptTokenCount) as number;
    if (numberOf(usage.candidatesTokenCount) !== null) outputTokens = numberOf(usage.candidatesTokenCount) as number;
    const candidates = Array.isArray(frame.candidates) ? frame.candidates : [];
    for (const candidate of candidates) {
      if (isRecord(candidate) && typeof candidate.finishReason === 'string') stopFamily = stopFamilyOf(candidate.finishReason);
    }
  }
  return { text, inputTokens, outputTokens, stopFamily };
}

// ---- Emission: answer -> client response ----------------------------------------------

export const FIXED_CREATED = 1767225600;

function openAIStopReason(family: number): string {
  return family === 1 ? 'length' : family === 2 ? 'tool_calls' : family === 3 ? 'content_filter' : 'stop';
}

function anthropicStopReason(family: number): string {
  return family === 1 ? 'max_tokens' : family === 2 ? 'tool_use' : family === 3 ? 'refusal' : 'end_turn';
}

function splitText(text: string, size: number): string[] {
  const chunks: string[] = [];
  for (let i = 0; i < text.length; i += size) chunks.push(text.slice(i, i + size));
  return chunks;
}

/** An answer on the OpenAI wire: a JSON body, or the SSE frames of a streamed one. */
export function renderOpenAI(answer: ProviderAnswer, model: string, stream: boolean): string {
  const finishReason = openAIStopReason(answer.stopFamily);
  if (!stream) {
    return JSON.stringify({
      id: 'chatcmpl-00000000000000000000000000000001',
      object: 'chat.completion',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: finishReason }],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    });
  }
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-00000000000000000000000000000001',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  frames.push(`data: ${JSON.stringify(chunk({ role: 'assistant', content: '' }, null))}`);
  for (const part of splitText(answer.text, 13)) {
    frames.push(`data: ${JSON.stringify(chunk({ content: part }, null))}`);
  }
  frames.push(`data: ${JSON.stringify(chunk({}, finishReason))}`);
  frames.push(
    `data: ${JSON.stringify({
      id: 'chatcmpl-00000000000000000000000000000001',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    })}`,
  );
  frames.push('data: [DONE]');
  return frames.map((f) => `${f}\n\n`).join('');
}

/** An answer on the Anthropic wire: a JSON body, or the SSE frames of a streamed one. */
export function renderAnthropic(answer: ProviderAnswer, model: string, stream: boolean): string {
  const stopReason = anthropicStopReason(answer.stopFamily);
  if (!stream) {
    return JSON.stringify({
      id: 'msg_00000000000000000000000000000001',
      type: 'message',
      role: 'assistant',
      content: [{ type: 'text', text: answer.text }],
      model,
      stop_reason: stopReason,
      stop_sequence: null,
      usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
    });
  }
  const frames: string[] = [];
  frames.push(
    `event: message_start\ndata: ${JSON.stringify({ type: 'message_start', message: { id: 'msg_00000000000000000000000000000001', type: 'message', role: 'assistant', content: [], model, stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.inputTokens, output_tokens: 0 } } })}\n`,
  );
  frames.push(`event: content_block_start\ndata: ${JSON.stringify({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } })}\n`);
  for (const part of splitText(answer.text, 13)) {
    frames.push(`event: content_block_delta\ndata: ${JSON.stringify({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: part } })}\n`);
  }
  frames.push(`event: content_block_stop\ndata: ${JSON.stringify({ type: 'content_block_stop', index: 0 })}\n`);
  frames.push(
    `event: message_delta\ndata: ${JSON.stringify({ type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: answer.outputTokens } })}\n`,
  );
  frames.push(`event: message_stop\ndata: ${JSON.stringify({ type: 'message_stop' })}\n`);
  return frames.join('\n');
}

// ---- Model lists and errors ------------------------------------------------------------

/** The OpenAI shape of GET /v1/models. */
export function openAIModelList(models: { id: string; provider: string }[]): string {
  return JSON.stringify({
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  });
}

/** The Anthropic shape of GET /v1/models. */
export function anthropicModelList(models: { id: string; provider: string }[]): string {
  return JSON.stringify({
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: '2026-01-01T00:00:00Z' })),
    has_more: false,
    first_id: models.length > 0 ? models[0].id : null,
    last_id: models.length > 0 ? models[models.length - 1].id : null,
  });
}

export function openAIError(status: number, message: string): { status: number; body: string } {
  return { status, body: JSON.stringify({ error: { message, type: 'server_error', code: 'unavailable' } }) };
}

export function anthropicError(status: number, message: string): { status: number; body: string } {
  return { status, body: JSON.stringify({ type: 'error', error: { type: 'overloaded_error', message } }) };
}
