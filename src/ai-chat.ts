/**
 * @wwl/wwl-ai-chat — headless on-device AiChat engine.
 * Source promoted from labs/ai-chat.js (UI omitted; hosts supply their own shell).
 */

import {
  buildChatSystemPrompt,
  DEFAULT_LLM_GUARD_PATTERNS,
  validateLlmInput,
  validateLlmOutput,
  LLM_OUTPUT_BLOCK_MESSAGE,
  validateToolCall,
  parseXmlToolCalls,
  formatXmlToolResult,
} from './guardrails/llm.js';
import {
  flowchartSnapshot,
  parseLooseToolCalls,
  visionImagesForModel,
} from './guardrails/tools.js';
import type { GuardrailProfile } from './guardrails/moderation.js';
import type { GuardrailResult } from './guardrails/core.js';
import { toGuardrailError } from './guardrails/core.js';
import {
  abortError,
  boundedInteger,
  estimateTokens,
  selectContext,
  sourceContext,
  runBoundedTool,
  type GenerateOptions,
  type ChatMessage,
  type GuardrailEvent,
} from './session.js';
import { CheckedDeliveryGate, visibleReply } from './checked-delivery.js';
import { cancelRuntime, disposeRuntime } from './runtime.js';
import type { TransformersRuntime, TransformersEngine } from './transformers-runtime.js';

/**
 * Headless AiChat — Gemma via LiteRT-LM / WebLLM in the browser (WebGPU).
 *
 * @example
 * import { AiChat } from '@wwl/wwl-ai-chat';
 * const chat = new AiChat({ modelId: 'gemma-4-E2B-it-web' });
 * await chat.load();
 */

// ═══════════════════════════════════════════════════════════════════════
// CDN Configuration
// ═══════════════════════════════════════════════════════════════════════

const CDN = {
  webllm: 'https://esm.run/@mlc-ai/web-llm@0.2.86',
  litert: 'https://cdn.jsdelivr.net/npm/@litert-lm/core@0.17.1/+esm',
};

export const WWL_AI_CHAT_VERSION = '0.1.1';

export const TOOLS_UNSUPPORTED_ERROR =
  'Tool calling is only supported on LiteRT Gemma (E2B/E4B) models.';

/** @typedef {{ name: string, description?: string, parameters?: Record<string, unknown> }} AiToolDefinition */

let _webllmModule: any = null;
let _litertModule: any = null;

import { MODEL_OPTIONS } from './model-catalog.js';
export { MODEL_OPTIONS, MODEL_GROUPS } from './model-catalog.js';
const GiB = 1024 ** 3;

/**
 * Soft copy for the freeze window during WASM/WebGPU init (unavoidable in-browser).
 * Shown while weights upload / shaders compile after download.
 */
export const LOAD_FREEZE_HINT =
  'Uploading weights to the GPU and compiling shaders — the tab may freeze for several seconds. This is normal for in-browser WebGPU.';

/**
 * Infer whether a progress `text` / URL looks like cache, local `/models`, or network.
 * Prefer an explicit `source` field on progress events when AiChat provides one.
 *
 * @param {unknown} progressText
 * @returns {'cache' | 'local' | 'network' | 'unknown'}
 */
export function inferLoadSource(progressText: unknown): 'cache' | 'local' | 'network' | 'unknown' {
  const text = String(progressText || '').toLowerCase();
  if (!text) return 'unknown';
  if (/\/models\//.test(text) || /\blocal\b/.test(text)) return 'local';
  if (/(cache|cached|indexeddb)/.test(text)) return 'cache';
  if (/(download|fetch|http|https|network|transfer|bytes|\bkb\b|\bmb\b|\bgb\b)/.test(text)) {
    return 'network';
  }
  return 'unknown';
}

/**
 * Map an AiChat `onProgress` payload into UI-ready load status fields.
 * Shared by Labs WwlAiChatUI / AiChatUI and host apps (e.g. ts-school).
 *
 * @param {Record<string, unknown> | null | undefined} data
 * @param {{ likelyCached?: boolean, freezeHint?: string }} [options]
 * @returns {{
 *   stage: string,
 *   progressPct: number,
 *   progressText: string,
 *   statusText: string,
 *   statusTone: 'muted' | 'warn' | 'ok' | 'danger',
 *   freezeHint: string,
 *   source: 'cache' | 'local' | 'network' | 'unknown',
 * }}
 */
export function describeLoadProgress(
  data: any,
  options: Record<string, any> = {},
): {
  stage: string;
  progressPct: number;
  progressText: any;
  statusText: string;
  statusTone: 'muted' | 'warn' | 'ok' | 'danger';
  freezeHint: any;
  source: 'cache' | 'local' | 'network' | 'unknown';
} {
  const stage = String(data?.stage || '');
  const loaded = typeof data?.loaded === 'number' ? data.loaded : 0;
  const pct = Math.max(0, Math.min(100, Math.round(loaded * 100)));
  const message = String(data?.message || '');
  const text = String(data?.text || '');
  const freezeDefault = options.freezeHint || LOAD_FREEZE_HINT;
  const likelyCached = options.likelyCached === true;
  const explicitSource = data?.source;
  type LoadSource = 'cache' | 'local' | 'network' | 'unknown';
  let inferred: LoadSource;
  if (
    explicitSource === 'cache' ||
    explicitSource === 'local' ||
    explicitSource === 'network' ||
    explicitSource === 'unknown'
  ) {
    inferred = explicitSource;
  } else {
    inferred = inferLoadSource(text);
    if (inferred === 'unknown') inferred = inferLoadSource(message);
  }

  if (stage === 'init') {
    return {
      stage,
      progressPct: 0,
      progressText: message || 'Initializing…',
      statusText: 'Loading…',
      statusTone: 'warn',
      freezeHint: '',
      source: 'unknown',
    };
  }

  if (stage === 'downloading') {
    let source: LoadSource = inferred;
    if (source === 'unknown' && likelyCached) source = 'cache';

    let prefix = message || 'Preparing model…';
    if (source === 'local') {
      prefix = message || 'Using local LiteRT model…';
    } else if (source === 'network') {
      prefix = message || 'Downloading model from web (first load may take a while).';
    } else if (source === 'cache') {
      prefix = message || 'Loading model from browser cache (no full re-download).';
    } else if (likelyCached) {
      prefix = message || 'Loading model from browser cache…';
    }

    return {
      stage,
      progressPct: pct,
      progressText: text ? `${prefix} ${text}` : prefix,
      statusText: `Loading ${pct}%`,
      statusTone: 'warn',
      freezeHint: '',
      source,
    };
  }

  if (stage === 'compiling') {
    const hint = message || freezeDefault;
    return {
      stage,
      progressPct: 100,
      progressText: hint,
      statusText: 'Compiling…',
      statusTone: 'warn',
      freezeHint: hint,
      source: inferred === 'unknown' ? 'unknown' : inferred,
    };
  }

  if (stage === 'ready') {
    return {
      stage,
      progressPct: 100,
      progressText: message || 'Ready',
      statusText: 'Ready',
      statusTone: 'ok',
      freezeHint: '',
      source: 'unknown',
    };
  }

  if (stage === 'error') {
    const errText = message || text || 'Failed to load model.';
    return {
      stage,
      progressPct: 0,
      progressText: errText,
      statusText: errText.length > 80 ? 'Error' : errText,
      statusTone: 'danger',
      freezeHint: '',
      source: 'unknown',
    };
  }

  return {
    stage: stage || 'unknown',
    progressPct: pct,
    progressText: message || text || '',
    statusText: message || 'Loading…',
    statusTone: 'warn',
    freezeHint: '',
    source: inferred,
  };
}

/**
 * PrefillDecode `.litertlm` spikes cannot load in the current
 * `@litert-lm/core` web runtime:
 * - default `Backend.GPU_ARTISAN` always uses streaming ModelAssets →
 *   "Streaming kTfLitePrefillDecode models is not supported yet."
 * - `Backend.GPU` uses non-streaming `createEngine`, but web wasm fails with
 *   "null function" on these artifacts (probed 2026-08-08).
 * Google’s JS docs still list only Gemma `*-it-web` builds.
 */
export const LITERT_PREFILLDECODE_UNSUPPORTED_REASON =
  'Unsupported in the current LiteRT-LM.js runtime: PrefillDecode .litertlm models cannot load (GPU_ARTISAN streaming rejected; Backend.GPU non-stream create fails). Use Gemma 4 LiteRT (web-official).';

/**
 * @param {string | { backend?: string, litertKind?: string, litertRuntime?: string } | null | undefined} modelOrId
 */
export function isLiteRTPrefillDecodeUnsupported(modelOrId) {
  const option =
    typeof modelOrId === 'string' || modelOrId == null ? getModelOption(modelOrId) : modelOrId;
  if (!option || option.backend !== 'litert') return false;
  if (option.litertKind === 'web-official') return false;
  return option.litertRuntime === 'prefilldecode-unsupported';
}

/**
 * @param {string | object | null | undefined} modelOrId
 * @returns {string} empty when load is not blocked for LiteRT runtime reasons
 */
export function getLiteRTRuntimeBlockReason(modelOrId) {
  return isLiteRTPrefillDecodeUnsupported(modelOrId) ? LITERT_PREFILLDECODE_UNSUPPORTED_REASON : '';
}

/**
 * Rewrite raw LiteRT PrefillDecode / streaming errors into the Labs catalog message.
 * @param {unknown} err
 */
export function rewriteLiteRTLoadError(err) {
  // CSP / <script> load failures often surface as Event, not Error.
  if (err && typeof err === 'object' && !(err instanceof Error) && 'type' in err) {
    const target = /** @type {{ target?: { src?: string, href?: string } }} */ err.target;
    const src = target?.src || target?.href || '';
    if (/jsdelivr|unpkg|esm\.run/i.test(src)) {
      return new Error(
        'LiteRT WASM was blocked by Content-Security-Policy (CDN script). Pass AiChat({ liteRtWasmPath }) to a same-origin /wasm/ directory.',
      );
    }
    return new Error(
      src
        ? `Failed to load LiteRT runtime script (${src}).`
        : 'Failed to load LiteRT runtime (script or network error).',
    );
  }
  const msg = String(err?.message || err || '');
  if (/PrefillDecode|Streaming kTfLite/i.test(msg)) {
    return new Error(LITERT_PREFILLDECODE_UNSUPPORTED_REASON);
  }
  if (/Content Security Policy|violates.*script-src/i.test(msg)) {
    return new Error(
      'LiteRT WASM was blocked by Content-Security-Policy. Pass AiChat({ liteRtWasmPath }) to a same-origin /wasm/ directory.',
    );
  }
  // Keep definitive HTTP status errors from loadLiteRTModelBytes as-is.
  if (/Failed to fetch model \(\d+/.test(msg)) {
    return err instanceof Error ? err : new Error(msg);
  }
  if (
    /network error|failed to fetch|load failed|js stream error|err_network|aborted|readableStream|stream error/i.test(
      msg,
    )
  ) {
    return new Error(
      'LiteRT model download or stream failed (network error). Retry load; on headless Chrome prefer E2B or use a headed browser for E4B cold loads. Warm Cache Storage avoids re-download.',
    );
  }
  return err instanceof Error ? err : new Error(msg || 'Failed to load model.');
}

/**
 * Confirm-dialog copy when heuristics flag a constrained device.
 * @param {{ approxGb: number, recommendedLabel?: string }} opts
 */
export function buildWeakDeviceConfirmCopy({
  approxGb,
  recommendedLabel,
}: { approxGb?: number; recommendedLabel?: string } = {}) {
  const size =
    approxGb != null && Number.isFinite(approxGb) ? `~${approxGb.toFixed(1)} GB` : 'a large';
  const prefer = recommendedLabel
    ? `Prefer “${recommendedLabel}” on older or low-RAM machines, close other heavy tabs, then continue.`
    : 'Close other heavy tabs, then continue.';
  return [
    `This device looks constrained for ${size} local model.`,
    '',
    '• The tab may freeze while WebGPU loads weights into GPU memory',
    '• Low-RAM systems can crash (Aw Snap) if the GPU runs out of memory',
    `• ${prefer}`,
    '',
    'Load anyway?',
  ].join('\n');
}

/**
 * Whether the chat composer should receive keyboard focus.
 * Prefer the composer after send/stream/load; avoid stealing from selects, modals, etc.
 *
 * @param {{
 *   force?: boolean,
 *   modalOpen?: boolean,
 *   chatReady?: boolean,
 *   activeIsComposer?: boolean,
 *   activeIsOtherControl?: boolean,
 * }} [opts]
 */
export function shouldFocusChatComposer(opts: Record<string, any> = {}) {
  if (opts.modalOpen) return false;
  if (!opts.chatReady) return false;
  if (opts.force) return true;
  if (opts.activeIsComposer) return true;
  if (opts.activeIsOtherControl) return false;
  return true;
}

/** localStorage flag prefix — set after a successful model load (weights may be in Cache Storage). */
export const MODEL_CACHE_FLAG_PREFIX = 'wwl-ai-chat-model-cached:';

/** Cache Storage bucket for LiteRT `.litertlm` weights (app-owned; not the opaque HTTP disk cache). */
export const LITERT_MODEL_CACHE_NAME = 'wwl-litert-models';

const DEFAULT_GENERATION_CONFIG = {
  max_tokens: 512,
  temperature: 0.7,
  top_p: 0.9,
};

/** Gemma 4 may emit `<|channel>thought…<channel|>` when thinking is on; that burns tokens and yields tiny visible replies. */
const GEMMA4_GENERATION_CONFIG = {
  ...DEFAULT_GENERATION_CONFIG,
  max_tokens: 768,
};

function generationConfigForModel(modelId) {
  const option = getModelOption(modelId);
  const wantsGemmaBudget = option?.family === 'gemma4' || option?.group === 'gemma4';
  const cfg: Record<string, any> = wantsGemmaBudget
    ? { ...GEMMA4_GENERATION_CONFIG }
    : { ...DEFAULT_GENERATION_CONFIG };
  // WebLLM 0.2.86 reads enable_thinking from extra_body when a model sets disableThinking.
  if (option?.disableThinking) cfg.extra_body = { enable_thinking: false };
  if (option?.reasoning === 'required') cfg.max_tokens = 2048;
  return cfg;
}

function modelBackend(modelId) {
  return getModelOption(modelId)?.backend || 'webllm';
}

function isLiteRTModel(modelId) {
  return modelBackend(modelId) === 'litert';
}

function extractLiteRTText(response) {
  const parts = response?.content;
  if (typeof response === 'string') return response;
  if (typeof parts === 'string') return parts;
  if (!Array.isArray(parts)) {
    return normalizeCompletionText(response?.text ?? response?.message?.content ?? '');
  }
  return parts
    .map((part) => {
      if (typeof part === 'string') return part;
      if (part?.type === 'text' || typeof part?.text === 'string') return part.text || '';
      return normalizeCompletionText(part);
    })
    .join('');
}

/**
 * Consume LiteRT `sendMessageStreaming` output across browsers.
 * Official API returns a `ReadableStream`. Chromium often supports
 * `for await...of` on streams; Safari/WebKit frequently does not
 * (`ReadableStream.prototype[Symbol.asyncIterator]` missing), which throws:
 * `undefined is not a function (near '...s of a...')`.
 * Prefer async iteration when present; otherwise use `getReader()`.
 * @param {AsyncIterable|ReadableStream|Promise<AsyncIterable|ReadableStream>|null|undefined} streamLike
 */
export async function* iterateMessageStream(streamLike) {
  let stream = streamLike;
  if (stream != null && typeof stream.then === 'function') {
    stream = await stream;
  }
  if (stream == null) return;

  if (typeof stream[Symbol.asyncIterator] === 'function') {
    yield* stream;
    return;
  }

  if (typeof stream.getReader === 'function') {
    const reader = stream.getReader();
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        yield value;
      }
    } finally {
      try {
        reader.releaseLock();
      } catch {
        // ignore — lock may already be released after close/error
      }
    }
    return;
  }

  if (typeof stream[Symbol.iterator] === 'function') {
    yield* stream;
    return;
  }

  throw new TypeError('LiteRT streaming response is not an async iterable or ReadableStream');
}

/** Strip accidental thinking / turn markers from streamed text (defense in depth). */
export function sanitizeModelReply(text) {
  if (!text) return '';
  let out = String(text);
  // Drop full thought channels if the runtime leaked them into content.
  // Only strip closed blocks — never `$`-to-EOF, or streaming suffixes after an
  // unclosed open tag are wiped (callers use `sanitize(...) || reply` for empties).
  out = out.replace(/<\|channel>thought[\s\S]*?<channel\|>/gi, '');
  out = out.replace(/<\|think\|>[\s\S]*?<\|\/think\|>/gi, '');
  out = out.replace(/<think>[\s\S]*?<\/think>/gi, '');
  // Orphan open/close markers left mid-stream (no closed pair yet).
  out = out.replace(/<\|think\|>/g, '');
  out = out.replace(/<\/?think>/gi, '');
  out = out.replace(/<\/?turn\|>/g, '');
  out = out.replace(/<\|turn>(?:user|model|system)?/g, '');
  return out.trim();
}

/** Hold unfinished thought channels until they close; never display raw fallback text. */
function visibleGenerationText(text: string): string {
  return sanitizeModelReply(
    text
      .replace(/<think>(?![\s\S]*<\/think>)[\s\S]*$/gi, '')
      .replace(/<\|think\|>(?![\s\S]*<\|\/think\|>)[\s\S]*$/gi, '')
      .replace(/<\|channel>thought(?![\s\S]*<channel\|>)[\s\S]*$/gi, ''),
  );
}

/**
 * Replace jailbreak-compliance phrasing with a fixed safe reply.
 * @param {string} text
 * @returns {string}
 */
export function applyOutputGuardrails(text, profile: GuardrailProfile = 'family-friendly') {
  const cleaned = sanitizeModelReply(text) || String(text || '').trim();
  const check = validateLlmOutput({ text: cleaned, profile });
  if (!check.allowed) {
    return check.message || LLM_OUTPUT_BLOCK_MESSAGE;
  }
  return cleaned;
}

export function getModelOption(modelId) {
  return MODEL_OPTIONS.find((m) => m.id === modelId) || null;
}

export function getModelDisplayName(modelId) {
  const option = getModelOption(modelId);
  if (!option) return modelId;
  return option.label
    .split('(~')[0]
    .replace(/\s+-\s+\w+$/, '')
    .trim();
}

/**
 * Yield to the browser so paint/input can run before a long sync stretch.
 * Engine.create still does heavy WASM/WebGPU work we cannot slice ourselves.
 */
export async function yieldToMain() {
  if (typeof globalThis.scheduler?.yield === 'function') {
    await globalThis.scheduler.yield();
    return;
  }
  await new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Collect coarse device signals for load guardrails.
 * Browsers do not expose free VRAM; deviceMemory is fingerprint-capped (often ≤8).
 * @param {GPUAdapter | null} [adapter]
 */
export function collectDeviceSignals(adapter: any = null) {
  const nav = typeof navigator !== 'undefined' ? navigator : null;
  const limits = adapter?.limits;
  return {
    deviceMemory: typeof nav?.deviceMemory === 'number' ? nav.deviceMemory : null,
    hardwareConcurrency:
      typeof nav?.hardwareConcurrency === 'number' ? nav.hardwareConcurrency : null,
    maxStorageBufferBindingSize:
      typeof limits?.maxStorageBufferBindingSize === 'number'
        ? limits.maxStorageBufferBindingSize
        : null,
    maxBufferSize: typeof limits?.maxBufferSize === 'number' ? limits.maxBufferSize : null,
  };
}

/**
 * LM Studio–style pre-load heuristic (browser-limited).
 * Returns level: 'ok' | 'caution' | 'high' plus human reasons. No smaller catalog model is recommended.
 *
 * @param {{ modelId?: string, systemInfo?: Record<string, unknown> }} [opts]
 */
export function assessLoadCapacity(opts: Record<string, any> = {}) {
  const modelId = opts.modelId || MODEL_OPTIONS[0]?.id;
  const option = getModelOption(modelId);
  const systemInfo = opts.systemInfo || {};
  const approxBytes = option?.approxBytes || 2 * GiB;
  const approxGb = approxBytes / GiB;

  const deviceMemory =
    typeof systemInfo.deviceMemory === 'number'
      ? systemInfo.deviceMemory
      : collectDeviceSignals().deviceMemory;
  const cores =
    typeof systemInfo.hardwareConcurrency === 'number'
      ? systemInfo.hardwareConcurrency
      : collectDeviceSignals().hardwareConcurrency;
  const maxStorage =
    typeof systemInfo.maxStorageBufferBindingSize === 'number'
      ? systemInfo.maxStorageBufferBindingSize
      : null;

  /** @type {'ok' | 'caution' | 'high'} */
  let level: 'ok' | 'caution' | 'high' = 'ok';
  const reasons: string[] = [];

  // WebLLM / Android pattern: ≤128 MiB storage binding ≈ very constrained GPU.
  const LOW_STORAGE_CAP = 128 * 1024 * 1024;
  if (maxStorage != null && maxStorage <= LOW_STORAGE_CAP && approxBytes > 0.5 * GiB) {
    level = 'high';
    reasons.push(
      'GPU maxStorageBufferBindingSize is very low (typical of constrained mobile GPUs). Large models often crash the tab.',
    );
  }

  // deviceMemory is intentionally coarse and capped; ≤4 is a strong weak-device signal.
  if (deviceMemory != null && deviceMemory <= 4 && approxBytes >= 1 * GiB) {
    level = 'high';
    reasons.push(
      `Browser reports ~${deviceMemory} GB RAM (approximate / capped). A ~${approxGb.toFixed(1)} GB model may freeze or OOM.`,
    );
  } else if (deviceMemory != null && deviceMemory <= 4) {
    if (level === 'ok') level = 'caution';
    reasons.push(
      `Browser reports ~${deviceMemory} GB RAM — expect longer freezes during WebGPU init.`,
    );
  }

  // Largest catalog models on ambiguous 8 GB deviceMemory bucket.
  if (deviceMemory != null && deviceMemory <= 8 && approxBytes >= 3.5 * GiB && level === 'ok') {
    level = 'caution';
    reasons.push(
      `~${approxGb.toFixed(1)} GB model on a device reporting ≤8 GB RAM — close other tabs before loading.`,
    );
  }

  if (cores != null && cores <= 2 && approxBytes >= 1 * GiB) {
    if (level === 'ok') level = 'caution';
    reasons.push(
      `Only ${cores} logical CPU cores reported — download/compile may stall the UI longer.`,
    );
  } else if (cores != null && cores <= 4 && approxBytes >= 2.5 * GiB && level === 'ok') {
    level = 'caution';
    reasons.push(`Modest CPU (${cores} cores) with a ~${approxGb.toFixed(1)} GB model.`);
  }

  return {
    level,
    reasons,
    approxBytes,
    approxGb,
    deviceMemory,
    hardwareConcurrency: cores,
    maxStorageBufferBindingSize: maxStorage,
    recommendedModelId: null,
    recommendedLabel: null,
    freezeHint: LOAD_FREEZE_HINT,
  };
}

/**
 * @param {string} modelId
 * @returns {string}
 */
export function modelCacheFlagKey(modelId) {
  return `${MODEL_CACHE_FLAG_PREFIX}${modelId}`;
}

/**
 * Whether localStorage records a prior successful load for this model id.
 * @param {string} modelId
 * @returns {boolean}
 */
export function isModelMarkedCached(modelId) {
  if (!modelId) return false;
  try {
    return localStorage.getItem(modelCacheFlagKey(modelId)) === '1';
  } catch {
    return false;
  }
}

/**
 * Record that weights for this model id were loaded successfully (Cache Storage and/or HTTP).
 * @param {string} modelId
 */
export function markModelCached(modelId) {
  if (!modelId) return;
  try {
    localStorage.setItem(modelCacheFlagKey(modelId), '1');
  } catch {
    // private mode / disabled storage
  }
}

/**
 * @returns {Promise<Cache | null>}
 */
export async function openLiteRTModelCache() {
  if (typeof caches === 'undefined' || typeof caches.open !== 'function') return null;
  try {
    return await caches.open(LITERT_MODEL_CACHE_NAME);
  } catch {
    return null;
  }
}

/**
 * @param {string} url
 * @returns {Promise<Response | null>}
 */
export async function matchCachedModel(url) {
  const cache = await openLiteRTModelCache();
  if (!cache || !url) return null;
  try {
    const hit = await cache.match(url);
    return hit || null;
  } catch {
    return null;
  }
}

/**
 * Persist model bytes under the LiteRT Cache Storage bucket.
 * @param {string} url
 * @param {Blob} blob
 * @returns {Promise<boolean>}
 */
export async function putCachedModel(url, blob) {
  const cache = await openLiteRTModelCache();
  if (!cache || !url || !blob) return false;
  try {
    const headers = new Headers({
      'Content-Type': 'application/octet-stream',
      'Content-Length': String(blob.size),
    });
    await cache.put(url, new Response(blob, { status: 200, headers }));
    return true;
  } catch {
    // QuotaExceededError or Cache API unavailable for this origin size
    return false;
  }
}

/**
 * Delete the LiteRT model Cache Storage bucket (best-effort).
 * @returns {Promise<boolean>}
 */
export async function deleteLiteRTModelCache() {
  if (typeof caches === 'undefined' || typeof caches.delete !== 'function') return false;
  try {
    return await caches.delete(LITERT_MODEL_CACHE_NAME);
  } catch {
    return false;
  }
}

/**
 * Report progress while reading a Response/Blob body into a Blob.
 * @param {Response | Blob} source
 * @param {(p: { loaded: number, received: number, totalBytes: number }) => void} [onProgress]
 * @param {number} [knownTotal]
 * @returns {Promise<Blob>}
 */
async function readBodyToBlobWithProgress(source, onProgress, knownTotal = 0) {
  if (source instanceof Blob) {
    onProgress?.({
      loaded: 1,
      received: source.size,
      totalBytes: source.size,
    });
    return source;
  }

  const totalBytes = knownTotal || Number(source.headers?.get?.('content-length')) || 0;
  let received = 0;

  if (!source.body || typeof source.body.getReader !== 'function') {
    const blob = await source.blob();
    onProgress?.({
      loaded: 1,
      received: blob.size,
      totalBytes: totalBytes || blob.size,
    });
    return blob;
  }

  const reader = source.body.getReader();
  const chunks: BlobPart[] = [];
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.byteLength;
      const loaded = totalBytes > 0 ? Math.min(1, received / totalBytes) : 0;
      onProgress?.({ loaded, received, totalBytes });
    }
  } catch (err) {
    try {
      reader.cancel?.();
    } catch {
      // ignore cancel after read failure
    }
    throw err;
  }
  const blob = new Blob(chunks, { type: 'application/octet-stream' });
  onProgress?.({
    loaded: 1,
    received: blob.size,
    totalBytes: totalBytes || blob.size,
  });
  return blob;
}

const FETCH_RETRY_STATUS = new Set([429, 502, 503, 504]);
const FETCH_RETRY_ATTEMPTS = 3;
const FETCH_RETRY_BASE_MS = 400;

function sleepMs(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isTransientFetchFailure(err: unknown, status?: number): boolean {
  if (typeof status === 'number') {
    return FETCH_RETRY_STATUS.has(status);
  }
  const msg = String((err as Error)?.message || err || '');
  // Definitive HTTP status failures from this helper are not transient.
  if (/Failed to fetch model \(\d+/.test(msg)) return false;
  return /network error|failed to fetch|load failed|js stream error|err_network|aborted|stream error/i.test(
    msg,
  );
}

function createProgressStream(
  body: ReadableStream<Uint8Array>,
  onProgress?: (p: { loaded: number; received: number; totalBytes: number }) => void,
  totalBytes = 0,
): ReadableStream<Uint8Array> {
  const reader = body.getReader();
  let received = 0;
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      const { done, value } = await reader.read();
      if (done) {
        onProgress?.({ loaded: 1, received, totalBytes: totalBytes || received });
        controller.close();
        return;
      }
      received += value.byteLength;
      const loaded = totalBytes > 0 ? Math.min(1, received / totalBytes) : 0;
      onProgress?.({ loaded, received, totalBytes });
      controller.enqueue(value);
    },
    cancel(reason) {
      return reader.cancel(reason);
    },
  });
}

/**
 * Load LiteRT model bytes from Cache Storage or network, then optionally stream.
 * Always buffers once so we can persist a durable Cache Storage entry (unless >2 GiB, where V8 Blob buffering fails in Chromium).
 * Network fetch + body read retries transient failures (BC: same return shape).
 *
 * @param {string} url
 * @param {{
 *   asStream?: boolean,
 *   urlIsLocal?: boolean,
 *   onProgress?: (p: { loaded: number, received: number, totalBytes: number }) => void,
 *   fetchRetries?: number,
 * }} [options]
 * @returns {Promise<{ modelSource: Blob | ReadableStream, source: 'cache' | 'local' | 'network' }>}
 */
export async function loadLiteRTModelBytes(url: string, options: Record<string, any> = {}) {
  const asStream = options.asStream === true;
  const urlIsLocal = options.urlIsLocal === true;
  const onProgress = options.onProgress;
  const maxAttempts =
    typeof options.fetchRetries === 'number' && options.fetchRetries >= 0
      ? Math.floor(options.fetchRetries) + 1
      : FETCH_RETRY_ATTEMPTS;

  const cached = await matchCachedModel(url);
  if (cached) {
    const blob = await readBodyToBlobWithProgress(cached, onProgress);
    const modelSource = asStream && typeof blob.stream === 'function' ? blob.stream() : blob;
    return { modelSource, source: 'cache' };
  }

  let lastErr: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const res = await fetch(url);
      if (!res.ok) {
        const statusErr = new Error(
          `Failed to fetch model (${res.status} ${res.statusText || ''}).`.trim(),
        );
        if (attempt < maxAttempts && isTransientFetchFailure(statusErr, res.status)) {
          lastErr = statusErr;
          await sleepMs(FETCH_RETRY_BASE_MS * attempt);
          continue;
        }
        throw statusErr;
      }
      const totalBytes = Number(res.headers.get('content-length')) || 0;
      const networkSource = urlIsLocal ? 'local' : 'network';
      if (totalBytes >= 2 * GiB && res.body && typeof res.body.getReader === 'function') {
        return {
          modelSource: createProgressStream(res.body, onProgress, totalBytes),
          source: networkSource,
        };
      }
      const blob = await readBodyToBlobWithProgress(res, onProgress, totalBytes);
      await putCachedModel(url, blob);

      const modelSource = asStream && typeof blob.stream === 'function' ? blob.stream() : blob;
      return { modelSource, source: networkSource };
    } catch (err) {
      lastErr = err;
      if (attempt < maxAttempts && isTransientFetchFailure(err)) {
        await sleepMs(FETCH_RETRY_BASE_MS * attempt);
        continue;
      }
      break;
    }
  }

  const finalMsg = String((lastErr as Error)?.message || lastErr || 'Failed to fetch model.');
  if (/Failed to fetch model \(\d+/i.test(finalMsg)) {
    throw lastErr instanceof Error ? lastErr : new Error(finalMsg);
  }
  throw rewriteLiteRTLoadError(lastErr instanceof Error ? lastErr : new Error(finalMsg));
}

const localModelProbeCache = new Map();

/** Validate an app-owned mirror as a complete, revision-pinned set before loading it. */
async function hasValidatedLocalModel(option) {
  if (!option?.id) return false;
  if (localModelProbeCache.has(option.id)) return localModelProbeCache.get(option.id);

  const localRoot = `/models/${option.id}`;
  try {
    const response = await fetch(`${localRoot}/.labs-model.json`, {
      method: 'GET',
      cache: 'no-store',
    });
    const contentType = response.headers?.get?.('content-type') || '';
    if (!response.ok || /text\/html/i.test(contentType)) {
      localModelProbeCache.set(option.id, false);
      return false;
    }

    const marker = await response.json();
    if (marker?.modelId !== option.id || (option.revision && marker.revision !== option.revision)) {
      localModelProbeCache.set(option.id, false);
      return false;
    }

    for (const artifact of option.artifacts || []) {
      const path = artifact.path
        .split('/')
        .map((part) => encodeURIComponent(part))
        .join('/');
      const file = await fetch(`${localRoot}/resolve/main/${path}`, {
        method: 'HEAD',
        cache: 'no-store',
      });
      const fileType = file.headers?.get?.('content-type') || '';
      const bytes = Number(file.headers?.get?.('content-length'));
      if (
        !file.ok ||
        /text\/html/i.test(fileType) ||
        (Number.isFinite(artifact.bytes) && artifact.bytes > 0 && bytes !== artifact.bytes)
      ) {
        localModelProbeCache.set(option.id, false);
        return false;
      }
    }
    localModelProbeCache.set(option.id, true);
    return true;
  } catch {
    localModelProbeCache.set(option.id, false);
    return false;
  }
}

function absoluteUrl(pathname) {
  if (typeof location === 'undefined' || !location?.origin) return pathname;
  return new URL(pathname, location.origin).href;
}

/**
 * Prefer a local Vite mirror at `/models/<id>/` (from `pnpm models:fetch`) when present.
 * Falls back to the Hugging Face URLs on MODEL_OPTIONS.
 *
 * WebLLM treats model URLs like HF repos and requests `{model}/resolve/main/<file>`,
 * so the local model URL includes that suffix and Vite strips it when serving.
 */
async function resolveModelSource(option) {
  if (!option?.modelUrl || !option?.modelLibUrl) {
    return { modelUrl: option?.modelUrl, modelLibUrl: option?.modelLibUrl, local: false };
  }

  const localRoot = `/models/${option.id}`;
  const localModelUrl = absoluteUrl(`${localRoot}/resolve/main/`);
  const localLibUrl = absoluteUrl(`/webllm-wasm/${option.id}.wasm`);
  if (await hasValidatedLocalModel(option))
    return { modelUrl: localModelUrl, modelLibUrl: localLibUrl, local: true };

  return { modelUrl: option.modelUrl, modelLibUrl: option.modelLibUrl, local: false };
}

async function resolveLiteRTModelUrl(option) {
  if (!option?.modelUrl) return option?.modelUrl;
  const fileName = option.modelFile || pathBasename(option.modelUrl);
  const localPath = `/models/${option.id}/${fileName}`;
  if (await hasValidatedLocalModel(option)) {
    console.warn(`[AiChat] Using local LiteRT model for ${option.id}: ${localPath}`);
    return absoluteUrl(localPath);
  }

  return option.modelUrl;
}

function pathBasename(urlOrPath) {
  const cleaned = String(urlOrPath || '').split('?')[0];
  const parts = cleaned.split('/').filter(Boolean);
  return parts[parts.length - 1] || cleaned;
}

function buildChatPayload(modelId, historyMessages, systemPrompt = buildChatSystemPrompt()) {
  // Always copy — WebLLM/request holders must not share our mutable history array.
  const history = historyMessages.map((message) => ({
    role: message.role,
    content: message.content,
  }));
  return [{ role: 'system', content: systemPrompt }, ...history];
}

/** Stop waiting for a model load promptly, and release engines that arrive late. */
async function awaitAbortableLoad<T>(
  pending: Promise<T>,
  signal: AbortSignal,
  releaseLate: (value: T) => void,
): Promise<T> {
  let onAbort!: () => void;
  const aborted = new Promise<never>((_, reject) => {
    onAbort = () => reject(abortError());
    if (signal.aborted) onAbort();
    else signal.addEventListener('abort', onAbort, { once: true });
  });
  try {
    return await Promise.race([pending, aborted]);
  } catch (error) {
    if (signal.aborted) void pending.then(releaseLate, () => {});
    throw error;
  } finally {
    signal.removeEventListener('abort', onAbort);
  }
}

async function buildModelAppConfig(modelId, runtimeAppConfig) {
  const option = getModelOption(modelId);
  if (!option?.modelUrl || !option?.modelLibUrl) {
    return { appConfig: null, source: { local: false } };
  }
  const source = await resolveModelSource(option);
  if (source.local) {
    console.warn(`[AiChat] Using local model mirror for ${option.id}: ${source.modelUrl}`);
  }
  const runtimeRecord =
    runtimeAppConfig?.model_list?.find((record) => record.model_id === option.id) || {};
  return {
    source,
    appConfig: {
      model_list: [
        {
          ...runtimeRecord,
          model: source.modelUrl,
          model_id: option.id,
          model_lib: source.modelLibUrl,
          required_features: option.requires || runtimeRecord.required_features || [],
          overrides: { ...runtimeRecord.overrides, ...option.overrides },
        },
      ],
    },
  };
}

function normalizeCompletionText(value) {
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) {
    return value.map((part) => normalizeCompletionText(part)).join('');
  }
  if (value && typeof value === 'object') {
    return normalizeCompletionText(value.text ?? value.content ?? value.value ?? '');
  }
  return '';
}

function extractCompletionChoiceText(choice) {
  if (!choice) return '';
  return (
    normalizeCompletionText(choice.delta?.content) ||
    normalizeCompletionText(choice.delta?.text) ||
    normalizeCompletionText(choice.message?.content) ||
    normalizeCompletionText(choice.text)
  );
}

function extractCompletionResponseText(response) {
  const choice = response?.choices?.[0];
  return extractCompletionChoiceText(choice);
}

async function loadWebLLM(customLoader: any = null) {
  if (customLoader) return customLoader();
  if (_webllmModule) return _webllmModule;
  if (typeof window !== 'undefined' && window.__vdlWebLLMModule) {
    _webllmModule = window.__vdlWebLLMModule;
    return _webllmModule;
  }
  try {
    if (typeof customLoader === 'function') {
      _webllmModule = await customLoader();
    } else {
      _webllmModule = await import(/* @vite-ignore */ /* webpackIgnore: true */ CDN.webllm as any);
    }
    if (typeof window !== 'undefined') {
      if (window.__vdlWebLLMModule && window.__vdlWebLLMModule !== _webllmModule) {
        console.warn(
          '[AiChat] Multiple WebLLM module instances detected; Tokenizer bindings may fail. Hard-refresh the tab.',
        );
      }
      window.__vdlWebLLMModule = _webllmModule;
    }
    return _webllmModule;
  } catch (err) {
    console.error('[AiChat] Failed to load WebLLM:', err);
    throw err;
  }
}

async function loadLiteRT(customLoader: any = null) {
  if (customLoader) return customLoader();
  if (_litertModule) return _litertModule;
  if (typeof window !== 'undefined' && window.__vdlLiteRTModule) {
    _litertModule = window.__vdlLiteRTModule;
    return _litertModule;
  }
  try {
    if (typeof customLoader === 'function') {
      _litertModule = await customLoader();
    } else {
      _litertModule = await import(/* @vite-ignore */ /* webpackIgnore: true */ CDN.litert as any);
    }
    if (typeof window !== 'undefined') {
      window.__vdlLiteRTModule = _litertModule;
    }
    return _litertModule;
  } catch (err) {
    console.error('[AiChat] Failed to load LiteRT-LM:', err);
    throw err;
  }
}

// ═══════════════════════════════════════════════════════════════════════
// FOSS Guardrails (Deterministic Scanner & System Prompt)
// ═══════════════════════════════════════════════════════════════════════

export const InputGuardrail = {
  patterns: DEFAULT_LLM_GUARD_PATTERNS.map((pattern) => pattern.regex),

  validate(text) {
    const result = validateLlmInput({ text });
    if (!result.allowed) {
      return {
        isValid: false,
        reason: result.message,
      };
    }
    return { isValid: true };
  },
};

// ═══════════════════════════════════════════════════════════════════════
// AiChat — Headless API
// ═══════════════════════════════════════════════════════════════════════

export type AiChatOptions = {
  guardrailProfile?: GuardrailProfile;
  onGuardrail?: (event: GuardrailEvent) => void;
  modelId?: string;
  systemPromptOptions?: Record<string, unknown>;
  toolProtocol?: 'auto' | 'native' | 'xml';
  loadLiteRT?: () => Promise<unknown>;
  loadWebLLM?: () => Promise<unknown>;
  liteRtWasmPath?: string;
  loadTransformers?: () => Promise<TransformersRuntime>;
};

export class AiChat {
  private _previewGate: CheckedDeliveryGate | null = null;
  static VERSION = WWL_AI_CHAT_VERSION;

  modelId: string;
  engine: any;
  _conversation: any;
  messages: Array<{ role: string; content: string; [key: string]: unknown }>;
  _progressSubscribers: Array<(data: any) => void>;
  _isLoaded: boolean;
  _isLoading: boolean;
  _needsEngineReload: boolean;
  _tools: Array<{ name: string; description: string; parameters: Record<string, unknown> }>;
  _systemPromptOptions: Record<string, unknown>;
  _toolProtocol: 'auto' | 'native' | 'xml';
  _nativeToolsSupported: boolean | null;
  _customLoadLiteRT: ((...args: any[]) => any) | null;
  _customLoadWebLLM: ((...args: any[]) => any) | null;
  _liteRtWasmPath: string | null;

  private _guardrailProfile: GuardrailProfile;
  private _onGuardrail?: (event: GuardrailEvent) => void;
  private _loadTransformers?: () => Promise<TransformersRuntime>;
  private _loadAbort = new AbortController();
  private _active: AbortController | null = null;
  private _settled: Promise<void> = Promise.resolve();
  private _loadSettled: Promise<void> = Promise.resolve();
  private _loadEpoch = 0;
  private _finishOperation: (() => void) | null = null;
  private _replay: ChatMessage[] = [];
  private _outputTokens = 1024;
  private _lastContextStart = 0;
  private _hadSources = false;

  constructor(options: AiChatOptions = {}) {
    this._guardrailProfile = options.guardrailProfile || 'family-friendly';
    this._onGuardrail = options.onGuardrail;
    this._loadTransformers = options.loadTransformers;
    this.modelId = options.modelId || MODEL_OPTIONS[0].id;
    this.engine = null;
    this._conversation = null;
    this.messages = [];
    this._progressSubscribers = [];
    this._isLoaded = false;
    this._isLoading = false;
    this._needsEngineReload = false;
    /** @type {AiToolDefinition[]} */
    this._tools = [];
    this._systemPromptOptions = options.systemPromptOptions || {};
    /** @type {'auto' | 'native' | 'xml'} */
    this._toolProtocol = options.toolProtocol || 'auto';
    this._nativeToolsSupported = null;
    /** Optional host-provided loaders (bundle LiteRT/WebLLM for strict CSP).
     * Named distinctly from `_loadLiteRT()` / `_loadWebLLM()` methods — an own
     * property with the same name would shadow the prototype method and make
     * `load()` import the package without ever calling Engine.create. */
    this._customLoadLiteRT = typeof options.loadLiteRT === 'function' ? options.loadLiteRT : null;
    this._customLoadWebLLM = typeof options.loadWebLLM === 'function' ? options.loadWebLLM : null;
    /**
     * Same-origin directory (or .js URL) for LiteRT WASM glue.
     * Required under CSP `script-src 'self'` — the package default is jsDelivr.
     * @type {string | null}
     */
    this._liteRtWasmPath =
      typeof options.liteRtWasmPath === 'string' && options.liteRtWasmPath.trim()
        ? options.liteRtWasmPath.trim()
        : null;
  }

  /**
   * @param {AiToolDefinition[]} defs
   */
  registerTools(defs) {
    if (this._active) throw new Error('Stop generation before changing tools.');
    const list = Array.isArray(defs) ? defs : [];
    this._tools = list
      .map((d) => ({
        name: String(d?.name || '').trim(),
        description: String(d?.description || ''),
        parameters:
          d?.parameters && typeof d.parameters === 'object'
            ? d.parameters
            : { type: 'object', properties: {} },
      }))
      .filter((d) => d.name);
    // Conversation preface may include tools — force refresh on next turn.
    this._needsEngineReload = true;
    this._nativeToolsSupported = null;
  }

  /**
   * @param {Record<string, unknown>} options
   */
  setSystemPromptOptions(options = {}) {
    if (this._active) throw new Error('Stop generation before changing instructions.');
    this._systemPromptOptions = options && typeof options === 'object' ? { ...options } : {};
    this._needsEngineReload = true;
  }

  _composeSystemPrompt() {
    const useNativeProtocol =
      isLiteRTModel(this.modelId) && this._toolProtocol !== 'xml' && this._toolsSupportedForModel();
    return buildChatSystemPrompt({
      ...this._systemPromptOptions,
      profile: this._guardrailProfile,
      toolsEnabled: this._tools.length > 0,
      toolNames: this._tools.map((t) => t.name),
      tools: this._tools,
      toolProtocol: useNativeProtocol ? 'native' : 'xml',
    });
  }

  _toolsSupportedForModel() {
    return !!getModelOption(this.modelId)?.capabilities?.tools || this._toolProtocol === 'xml';
  }

  async setModelId(modelId: string, options: Record<string, any> = {}) {
    if (!getModelOption(modelId))
      throw new Error(`Model ${modelId} is not in the supported catalog. Choose a retained model.`);
    const { resetMessages = false, force = false } = options;
    this.cancel();
    await this._settled;
    if (this._isLoading) {
      throw new Error('Cannot change model ID while loading.');
    }

    const modelChanged = this.modelId !== modelId;
    // Always await teardown before pointing at a new backend — LiteRT/WebLLM
    // share the WebGPU adapter and racing dispose breaks subsequent loads.
    // `force` also tears down when reloading the same catalog id (host Reload).
    if ((modelChanged || force) && (this._isLoaded || this.engine || this._conversation)) {
      await this._disposeEngine();
      this.engine = null;
      this._conversation = null;
      this._isLoaded = false;
      this._needsEngineReload = false;
    }

    this.modelId = modelId;
    if (resetMessages) {
      this.reset();
    }
  }

  onProgress(callback) {
    this._progressSubscribers.push(callback);
    return () => {
      this._progressSubscribers = this._progressSubscribers.filter((cb) => cb !== callback);
    };
  }

  _emitProgress(data) {
    for (const cb of this._progressSubscribers) cb(data);
  }

  async _disposeEngine() {
    try {
      await disposeRuntime(this._conversation, this.engine);
    } catch {
      /* lost device */
    }
    this._conversation = null;
    this.engine = null;
  }

  cancel() {
    this._active?.abort();
    if (this._isLoading) this._loadAbort.abort();
    cancelRuntime(this._conversation, this.engine);
    this._needsEngineReload = true;
  }

  getHistory(): ChatMessage[] {
    return this.messages.map((m) => ({ role: m.role, content: m.content }));
  }

  async setHistory(history: ChatMessage[]) {
    this.cancel();
    await this._settled;
    this.messages = history.map((m) => ({ role: m.role, content: String(m.content) }));
    this._lastContextStart = 0;
    this._needsEngineReload = true;
  }

  private _begin(options: GenerateOptions) {
    if (this._active)
      throw new Error('Generation is already running. Stop it before starting another turn.');
    if (options.signal?.aborted) throw abortError();
    const controller = new AbortController();
    const stop = () => this.cancel();
    options.signal?.addEventListener('abort', stop, { once: true });
    this._active = controller;
    this._settled = new Promise((resolve) => {
      this._finishOperation = resolve;
    });
    return {
      controller,
      check: () => {
        if (controller.signal.aborted) throw abortError();
      },
      finish: () => {
        options.signal?.removeEventListener('abort', stop);
        this._active = null;
        this._finishOperation?.();
        this._finishOperation = null;
      },
    };
  }

  private _reportGuardrail(
    stage: GuardrailEvent['stage'],
    result: GuardrailResult,
    options: GenerateOptions = {},
    identity: Partial<GuardrailEvent> = {},
  ) {
    (options.onGuardrail || this._onGuardrail)?.({
      ...identity,
      stage,
      code: result.code || 'guardrail.blocked',
      ruleIds: result.matchedPatternIds || [],
    });
  }

  private _checkInput(text: string, options: GenerateOptions) {
    const guard = validateLlmInput({ text, profile: this._guardrailProfile });
    if (!guard.allowed) {
      this._reportGuardrail('input', guard, options);
      throw toGuardrailError(guard);
    }
  }

  private _checkedReply(reply: string, options: GenerateOptions): string {
    const cleaned = sanitizeModelReply(visibleReply(reply));
    const guard = validateLlmOutput({ text: cleaned, profile: this._guardrailProfile });
    if (!guard.allowed) {
      this._reportGuardrail('output', guard, options);
      this._needsEngineReload = true;
      return LLM_OUTPUT_BLOCK_MESSAGE;
    }
    return cleaned;
  }

  private async _prepareContext(text: string, options: GenerateOptions) {
    const windowTokens = boundedInteger(
      options.contextTokenBudget,
      getModelOption(this.modelId)?.maxNumTokens || 4096,
      getModelOption(this.modelId)?.maxNumTokens || 4096,
    );
    const priorOutputTokens = this._outputTokens;
    this._outputTokens = boundedInteger(
      options.maxOutputTokens,
      isLiteRTModel(this.modelId)
        ? this._tools.length > 0
          ? 1536
          : 1024
        : generationConfigForModel(this.modelId).max_tokens,
      Math.min(2048, Math.floor(windowTokens / 2)),
    );
    if (isLiteRTModel(this.modelId) && priorOutputTokens !== this._outputTokens)
      this._needsEngineReload = true;
    let rejectedSources = 0,
      rejectedHistoryTurns = 0;
    const sources = (options.sources || [])
      .slice(0, 4)
      .map((source) => ({
        id: String(source.id).slice(0, 120),
        title: String(source.title).slice(0, 200),
        text: String(source.text).slice(0, 1600),
      }))
      .filter((source, sourceIndex) => {
        const guard = validateLlmInput({
          text: `${source.id} ${source.title} ${source.text}`,
          profile: this._guardrailProfile,
          quotedDiscussion: false,
        });
        if (guard.allowed) return true;
        rejectedSources++;
        this._reportGuardrail('source', guard, options, { sourceIndex });
        return false;
      });
    const history: ChatMessage[] = [];
    for (let i = 0; i < this.messages.length; i++) {
      if (this.messages[i].role !== 'user' || this.messages[i + 1]?.role !== 'assistant') {
        rejectedHistoryTurns++;
        this._reportGuardrail(
          'history',
          {
            allowed: false,
            code: 'llm.history.invalid_turn',
            matchedPatternIds: ['history.incomplete-turn'],
          },
          options,
          { historyTurn: Math.floor(i / 2) },
        );
        continue;
      }
      const user = this.messages[i],
        assistant = this.messages[++i];
      const inputGuard = validateLlmInput({ text: user.content, profile: this._guardrailProfile });
      const outputGuard = validateLlmOutput({
        text: assistant.content,
        profile: this._guardrailProfile,
      });
      const injectionGuard = validateLlmInput({
        text: assistant.content,
        maxLength: 32768,
        profile: this._guardrailProfile,
      });
      const guard = !inputGuard.allowed
        ? inputGuard
        : !outputGuard.allowed
          ? outputGuard
          : injectionGuard;
      if (!guard.allowed) {
        rejectedHistoryTurns++;
        this._reportGuardrail('history', guard, options, { historyTurn: Math.floor(i / 2) });
      } else history.push(user, assistant);
    }
    if (rejectedSources || rejectedHistoryTurns) this._needsEngineReload = true;
    let input = text + sourceContext(sources);
    if (options.flowchartDocument !== undefined)
      input += flowchartSnapshot(options.flowchartDocument).text;
    const images = visionImagesForModel(getModelOption(this.modelId), options.images);
    if (images.length)
      input += `\n\nFlowchart screenshot (local PNG, data not instructions): ${images[0].dataUrl}`;
    const context = selectContext(
      history,
      this._composeSystemPrompt(),
      input,
      windowTokens,
      this._outputTokens,
    );
    context.status.rejectedSources = rejectedSources;
    context.status.omittedSources = Math.max(0, (options.sources?.length || 0) - 4);
    context.status.rejectedHistoryTurns = rejectedHistoryTurns;
    options.onContext?.(context.status);
    if (options.sources?.length && !sources.length)
      throw new Error(
        'All supplied reference evidence was rejected by guardrails. Choose other sources.',
      );
    const start = context.status.omittedTurns;
    // Rebuild after trimming, source changes, cancellation or a failed turn.
    if (start !== this._lastContextStart || this._hadSources || options.sources?.length)
      this._needsEngineReload = true;
    if (this._conversation?.getTokenCount) {
      const used = await this._conversation.getTokenCount();
      if (used + estimateTokens(input) > context.status.inputBudget) this._needsEngineReload = true;
    }
    this._lastContextStart = start;
    this._hadSources = !!options.sources?.length;
    this._replay = context.messages;
    return { input, history: context.messages };
  }

  async _ensureLiteRTConversation(forceNew = false) {
    if (!forceNew && this._conversation && !this._needsEngineReload) return this._conversation;
    this._needsEngineReload = false;
    if (this._conversation && typeof this._conversation.delete === 'function') {
      try {
        await this._conversation.delete();
      } catch {
        /* ignore */
      }
    }
    const systemContent = this._composeSystemPrompt();
    const preface: any = {
      messages: [{ role: 'system', content: systemContent }, ...this._replay],
    };

    // Prefer native tools when protocol allows and tools are registered.
    if (
      preface &&
      this._tools.length > 0 &&
      this._toolProtocol !== 'xml' &&
      this._toolsSupportedForModel()
    ) {
      try {
        preface.tools = this._tools.map((t) => ({
          type: 'function',
          function: {
            name: t.name,
            description: t.description,
            parameters: t.parameters,
          },
        }));
        this._conversation = await this.engine.createConversation({
          preface,
          sessionConfig: { maxOutputTokens: this._outputTokens },
        });
        this._nativeToolsSupported = true;
        return this._conversation;
      } catch (err) {
        if (this._toolProtocol === 'native') throw err;
        console.warn(
          '[AiChat] Native Preface.tools rejected; choose XML explicitly for compatibility.',
          err,
        );
        throw err;
      }
    }

    this._conversation = await this.engine.createConversation({
      preface,
      sessionConfig: { maxOutputTokens: this._outputTokens },
    });
    if (this._tools.length > 0 && this._nativeToolsSupported !== true) {
      this._nativeToolsSupported = false;
    }
    return this._conversation;
  }

  /**
   * Extract structured tool calls from a LiteRT message if present.
   * @param {unknown} message
   * @returns {Array<{ name: string, args: Record<string, unknown> }>}
   */
  _extractNativeToolCalls(message) {
    const raw = message?.tool_calls || message?.toolCalls || [];
    if (!Array.isArray(raw) || raw.length === 0) return [];
    return raw
      .map((call) => {
        const name = call?.function?.name || call?.name || '';
        let args = call?.function?.arguments ?? call?.arguments ?? {};
        if (typeof args === 'string') {
          try {
            args = JSON.parse(args);
          } catch {
            args = { _parseError: true, raw: args };
          }
        }
        if (!args || typeof args !== 'object' || Array.isArray(args)) args = { _parseError: true };
        return { name: String(name), args };
      })
      .filter((c) => c.name);
  }

  /**
   * @param {string} userText
   * @param {{
   *   execute: (name: string, args: Record<string, unknown>) => unknown | Promise<unknown>,
   *   maxRounds?: number,
   *   onUpdate?: (text: string) => void,
   *   onFinish?: (usage: unknown) => void,
   *   onTool?: (info: { name: string, args: Record<string, unknown>, result: unknown }) => void,
   * }} options
   */
  async generateWithTools(
    userText: string,
    options: GenerateOptions & {
      execute: (
        name: string,
        args: Record<string, unknown>,
        context?: { signal: AbortSignal },
      ) => unknown | Promise<unknown>;
      maxRounds?: number;
      maxCalls?: number;
      toolTimeoutMs?: number;
      maxResultBytes?: number;
      onTool?: (info: { name: string; args: Record<string, unknown>; result: unknown }) => void;
    },
  ) {
    if (typeof options.execute !== 'function')
      throw new Error('generateWithTools requires an execute(name, args) callback.');
    if (!this._toolsSupportedForModel()) throw new Error(TOOLS_UNSUPPORTED_ERROR);
    const webllmTools = modelBackend(this.modelId) === 'webllm';
    if (!isLiteRTModel(this.modelId) && !webllmTools) throw new Error(TOOLS_UNSUPPORTED_ERROR);
    if (!this._tools.length) throw new Error('No tools registered. Call registerTools() first.');
    this._checkInput(userText, options);
    if (!this.isLoaded()) throw new Error('Model not loaded. Call load() first.');
    const operation = this._begin(options);
    const rounds = boundedInteger(options.maxRounds, 4, 8);
    const maxCalls = boundedInteger(options.maxCalls, 8, 16);
    const resultLimit = boundedInteger(options.maxResultBytes, 8192, 32768);
    let callCount = 0;
    let anyToolSucceeded = false;
    try {
      const { input, history } = await this._prepareContext(userText, options);
      operation.check();
      const transcript = [...history, { role: 'user', content: input }];
      let pending: any = input;
      for (let round = 0; round < rounds; round++) {
        let reply = '';
        let rawMessage: any = null;
        let parseFailure = false;
        if (webllmTools) {
          if (this._needsEngineReload) await this._reloadEngine('reset');
          operation.check();
          const payload = buildChatPayload(this.modelId, transcript, this._composeSystemPrompt());
          const config = generationConfigForModel(this.modelId);
          const completed = await this._completeOnce(
            payload,
            { ...config, max_tokens: this._outputTokens },
            null,
          );
          reply = completed.reply;
        } else {
          try {
            const detailed = await this._completeOnceLiteRTDetailed(pending, null);
            reply = detailed.reply;
            rawMessage = detailed.rawMessage;
          } catch (err: any) {
            const message = String(err?.message || err);
            if (!parseLooseToolCalls(message).length) throw err;
            reply = message;
            parseFailure = true;
          }
        }
        operation.check();
        const xml = webllmTools || this._toolProtocol === 'xml';
        const nativeCalls = xml
          ? parseXmlToolCalls(reply).calls
          : this._extractNativeToolCalls(rawMessage);
        const looseCalls =
          !xml && !nativeCalls.length
            ? parseLooseToolCalls(`${reply}\n${rawMessage?.content || ''}`)
            : [];
        const calls = nativeCalls.length ? nativeCalls : looseCalls;
        if (looseCalls.length && /Failed to parse|<\|\s*tool_call/i.test(reply))
          parseFailure = true;
        if (xml && /<\/?tool_call\b/i.test(parseXmlToolCalls(reply).remainder))
          throw new Error('Malformed tool protocol.');
        if (!calls.length) {
          if (/<tool_call\b/i.test(reply))
            throw new Error('Malformed or unsupported tool protocol.');
          const candidate =
            !reply.trim() && round > 0 && anyToolSucceeded ? 'Updated the flowchart.' : reply;
          const final = this._checkedReply(candidate, options);
          if (!final)
            throw new Error(`Model ${this.modelId} returned an empty response during tool loop.`);
          this.messages.push(
            { role: 'user', content: userText },
            { role: 'assistant', content: final },
          );
          options.onUpdate?.(final);
          options.onFinish?.(null);
          this._needsEngineReload = true;
          return final;
        }
        const results: Array<{ name: string; result: unknown }> = [];
        for (const call of calls) {
          operation.check();
          if (++callCount > maxCalls) throw new Error('Tool call limit exceeded.');
          let validation = validateToolCall({
            name: call.name,
            args: call.args,
            allowlist: this._tools,
          });
          if (validation.allowed)
            validation = validateLlmInput({
              text: JSON.stringify(call.args),
              maxLength: 16384,
              profile: this._guardrailProfile,
              quotedDiscussion: false,
            });
          const knownTool = this._tools.some((tool) => tool.name === call.name);
          if (!validation.allowed)
            this._reportGuardrail(
              'tool-arguments',
              validation,
              options,
              knownTool ? { toolName: call.name } : {},
            );
          let result: unknown;
          if (!validation.allowed) result = { error: validation.code, message: validation.message };
          else {
            try {
              result = await runBoundedTool(
                (signal) => options.execute(call.name, call.args, { signal }),
                operation.controller.signal,
                boundedInteger(options.toolTimeoutMs, 15000, 60000),
              );
              const encoded = JSON.stringify(result ?? null);
              if (new TextEncoder().encode(encoded).length > resultLimit)
                result = { error: 'tool.result.too_large' };
              else result = JSON.parse(encoded);
            } catch (err: any) {
              operation.check();
              result = {
                error: 'tool.execute_failed',
                message: String(err?.message || err).slice(0, 240),
              };
            }
          }
          operation.check();
          const resultGuard = validateLlmInput({
            text: JSON.stringify(result ?? null),
            maxLength: 32768,
            profile: this._guardrailProfile,
            quotedDiscussion: false,
          });
          if (!resultGuard.allowed) {
            this._reportGuardrail(
              'tool-result',
              resultGuard,
              options,
              knownTool ? { toolName: call.name } : {},
            );
            result = { error: 'tool.result.blocked' };
          }
          // Unknown names and rejected arguments must never escape via callbacks or protocol.
          if (knownTool) {
            if (result && typeof result === 'object' && (result as { ok?: boolean }).ok === true) {
              anyToolSucceeded = true;
            }
            results.push({ name: call.name, result });
            options.onTool?.({
              name: call.name,
              args: validation.allowed ? call.args : {},
              result,
            });
          } else {
            const rejected = { error: 'tool.name.not_allowed' };
            results.push({ name: 'unknown', result: rejected });
            options.onTool?.({ name: 'unknown', args: {}, result: rejected });
          }
        }
        pending = xml
          ? results.map((r) => formatXmlToolResult(r.name, r.result)).join('\n')
          : {
              role: 'tool',
              content: results.map((r) => ({
                type: 'tool_response',
                name: r.name,
                response: r.result,
              })),
            };
        if (webllmTools) {
          transcript.push({ role: 'assistant', content: reply });
          transcript.push({
            role: 'user',
            content: typeof pending === 'string' ? pending : JSON.stringify(pending),
          });
        }
        if (parseFailure) {
          const applied = results.some(
            (item) =>
              !!item.result &&
              typeof item.result === 'object' &&
              (item.result as { ok?: boolean }).ok === true,
          );
          const failure = results.find(
            (item) =>
              !!item.result &&
              typeof item.result === 'object' &&
              typeof (item.result as { message?: unknown }).message === 'string',
          );
          const final = applied
            ? 'Updated the flowchart.'
            : String(
                (failure?.result as { message?: string } | undefined)?.message ||
                  'The flowchart update was rejected.',
              );
          const checked = this._checkedReply(final, options);
          if (!checked)
            throw new Error(`Model ${this.modelId} returned an empty response during tool loop.`);
          this.messages.push(
            { role: 'user', content: userText },
            { role: 'assistant', content: checked },
          );
          options.onUpdate?.(checked);
          options.onFinish?.(null);
          this._needsEngineReload = true;
          return checked;
        }
      }
      throw new Error(`Tool loop exceeded maxRounds (${rounds}) without a final assistant reply.`);
    } catch (err) {
      this._needsEngineReload = true;
      if (operation.controller.signal.aborted) throw abortError();
      throw err;
    } finally {
      operation.finish();
    }
  }

  /**
   * Like _completeOnceLiteRT but also returns the last raw message for tool_calls.
   */
  async _completeOnceLiteRTDetailed(userText, onUpdate) {
    const conversation = await this._ensureLiteRTConversation(false);
    let reply = '';
    let rawMessage: any = null;
    const calls: any[] = [];
    const accept = (chunk) => {
      if (this._active?.signal.aborted) throw abortError();
      rawMessage = chunk;
      if (chunk?.tool_calls) calls.push(...chunk.tool_calls);
      const delta = extractLiteRTText(chunk);
      if (delta)
        reply = Array.isArray(chunk?.content)
          ? reply + delta
          : delta.startsWith(reply)
            ? delta
            : reply + delta;
      if (reply.length > 131072) throw new Error('Model reply exceeded the processing limit.');
      const visible = visibleGenerationText(reply);
      if (onUpdate && visible) onUpdate(visible);
    };
    if (typeof conversation.sendMessageStreaming === 'function') {
      for await (const chunk of iterateMessageStream(conversation.sendMessageStreaming(userText)))
        accept(chunk);
    } else accept(await conversation.sendMessage(userText));
    if (calls.length) rawMessage = { ...rawMessage, tool_calls: calls };
    return { reply: visibleGenerationText(reply), usage: null, rawMessage };
  }

  async load() {
    if (!getModelOption(this.modelId))
      throw new Error(
        `Model ${this.modelId} is not in the supported catalog. Choose Gemma 4 E2B or another retained model.`,
      );
    // Heal inconsistent state: disposed engine with stale loaded flag would
    // no-op forever and leave hosts showing Ready without a usable runtime.
    if (this._isLoaded && this.engine) return;
    if (this._isLoaded && !this.engine) {
      this._isLoaded = false;
    }
    if (this._isLoading) throw new Error('Model is already loading.');

    const runtimeBlock = getLiteRTRuntimeBlockReason(this.modelId);
    if (runtimeBlock) {
      const err = new Error(runtimeBlock);
      this._emitProgress({ stage: 'error', message: err.message });
      throw err;
    }

    this._loadAbort = new AbortController();
    this._isLoading = true;
    const epoch = ++this._loadEpoch;
    let finishLoad!: () => void;
    this._loadSettled = new Promise((resolve) => {
      finishLoad = resolve;
    });
    try {
      // Clear any partial engine left by a previous failed load attempt.
      if (this.engine || this._conversation) {
        await this._disposeEngine();
        this.engine = null;
        this._conversation = null;
      }
      if (isLiteRTModel(this.modelId)) {
        await this._loadLiteRT();
      } else if (modelBackend(this.modelId) === 'transformers') {
        if (!this._loadTransformers)
          throw new Error('This model requires a host-provided loadTransformers worker adapter.');
        const runtime = await this._loadTransformers();
        const signal = this._loadAbort.signal;
        const pending = runtime.createEngine(getModelOption(this.modelId)!, {
          signal,
          onProgress: (p) => this._emitProgress(p),
        });
        this.engine = await awaitAbortableLoad(pending, signal, (engine) => {
          void disposeRuntime(null, engine).catch(() => {});
        });
      } else {
        await this._loadWebLLM();
      }
      if (!this.engine) {
        throw new Error('Model engine failed to initialize.');
      }
      if (this._loadAbort.signal.aborted || epoch !== this._loadEpoch) throw abortError();
      this._isLoaded = true;
      this._needsEngineReload = false;
      markModelCached(this.modelId);
      this._emitProgress({ stage: 'ready', message: 'Model loaded and ready!' });
    } catch (err: any) {
      const normalized: any = isLiteRTModel(this.modelId) ? rewriteLiteRTLoadError(err) : err;
      try {
        await this._disposeEngine();
      } catch {
        // ignore teardown after failed load
      }
      this.engine = null;
      this._conversation = null;
      this._isLoaded = false;
      this._emitProgress({
        stage: 'error',
        message: normalized?.message || 'Failed to load model.',
      });
      throw normalized;
    } finally {
      this._isLoading = false;
      finishLoad();
    }
  }

  async _loadLiteRT() {
    const option = getModelOption(this.modelId);
    const runtimeBlock = getLiteRTRuntimeBlockReason(option);
    if (runtimeBlock) {
      throw new Error(runtimeBlock);
    }

    const litert = await loadLiteRT(this._customLoadLiteRT);
    const { Engine, loadLiteRtLm, hasGlobalLiteRtLm, hasGlobalLiteRtLmPromise } = litert;
    if (typeof Engine?.create !== 'function') {
      throw new Error('LiteRT module loaded without Engine.create — check loadLiteRT().');
    }

    // Default package path is jsDelivr; CSP hosts must pass liteRtWasmPath (same-origin).
    if (
      this._liteRtWasmPath &&
      typeof loadLiteRtLm === 'function' &&
      !hasGlobalLiteRtLmPromise?.() &&
      !hasGlobalLiteRtLm?.()
    ) {
      this._emitProgress({
        stage: 'init',
        message: 'Loading LiteRT WASM runtime (same-origin)…',
      });
      await yieldToMain();
      try {
        await loadLiteRtLm(this._liteRtWasmPath);
      } catch (err) {
        throw rewriteLiteRTLoadError(err);
      }
    }

    this._emitProgress({ stage: 'init', message: 'Initializing LiteRT WebGPU engine…' });
    await yieldToMain();

    const modelUrl = await resolveLiteRTModelUrl(option);
    const displayName = getModelDisplayName(this.modelId);
    const isLocal = /\/models\//.test(String(modelUrl || ''));
    const alreadyCached = Boolean(await matchCachedModel(modelUrl));
    let loadSource = alreadyCached ? 'cache' : isLocal ? 'local' : 'network';
    this._emitProgress({
      stage: 'downloading',
      message: alreadyCached
        ? `Loading ${displayName} from browser cache…`
        : isLocal
          ? `Using local LiteRT model for ${displayName}…`
          : `Downloading / reading ${displayName} (LiteRT)…`,
      text: modelUrl,
      loaded: 0,
      source: loadSource,
    });

    // Engine accepts URL | ReadableStream | Blob.
    // App-fetched weights are always fully buffered (Cache Storage). Passing
    // Blob (not blob.stream()) avoids a second ReadableStream pass that can
    // fail mid-load in headless Chrome for large E4B models.
    if (option?.litertKind !== 'web-official') {
      console.warn(`[AiChat] Buffering portable LiteRT model as Blob: ${option?.id}`);
    }
    const onFetchProgress = ({ loaded, received, totalBytes }) => {
      const pct =
        totalBytes > 0
          ? `${Math.round(loaded * 100)}%`
          : `${(received / (1024 * 1024)).toFixed(1)} MB`;
      this._emitProgress({
        stage: 'downloading',
        text:
          totalBytes > 0
            ? `${pct} · ${(received / (1024 * 1024)).toFixed(0)} / ${(totalBytes / (1024 * 1024)).toFixed(0)} MB`
            : pct,
        loaded: totalBytes > 0 ? loaded : Math.min(0.95, received / (2 * GiB)),
        message:
          loadSource === 'cache'
            ? 'Reading cached model weights…'
            : isLocal
              ? 'Reading local model weights…'
              : 'Fetching model weights…',
        source: loadSource,
      });
    };
    const loadedBytes = await loadLiteRTModelBytes(modelUrl, {
      asStream: false,
      urlIsLocal: isLocal,
      onProgress: onFetchProgress,
    });
    loadSource = loadedBytes.source;
    const modelSource = loadedBytes.modelSource;

    this._emitProgress({
      stage: 'compiling',
      message: LOAD_FREEZE_HINT,
      text: 'GPU upload / shader compile',
      loaded: 1,
    });
    await yieldToMain();

    try {
      this.engine = await Engine.create({
        model: modelSource,
        mainExecutorSettings: {
          maxNumTokens: option?.maxNumTokens || 4096,
        },
      });
    } catch (err) {
      throw rewriteLiteRTLoadError(err);
    }
    if (!this.engine || typeof this.engine.createConversation !== 'function') {
      throw new Error('LiteRT Engine.create failed to initialize an engine.');
    }
    await this._ensureLiteRTConversation(true);
  }

  async _loadWebLLM() {
    const webllm = await loadWebLLM(this._customLoadWebLLM);
    const { CreateMLCEngine } = webllm;
    this._emitProgress({ stage: 'init', message: 'Initializing WebGPU engine...' });
    await yieldToMain();

    const { appConfig, source } = await buildModelAppConfig(this.modelId, webllm.prebuiltAppConfig);
    this._emitProgress({
      stage: 'init',
      source: source.local ? 'local' : 'network',
    });
    const engineConfig: Record<string, any> = {
      // Host adapters can use this signal to terminate an initializing worker.
      signal: this._loadAbort.signal,
      initProgressCallback: (progress: any) => {
        const loaded = typeof progress.progress === 'number' ? progress.progress : 0;
        const text = String(progress.text || '');
        const compiling =
          loaded >= 0.98 || /compil|shader|finish loading|loading model to gpu/i.test(text);
        this._emitProgress({
          stage: compiling ? 'compiling' : 'downloading',
          text: progress.text,
          loaded: progress.progress,
          message: compiling ? LOAD_FREEZE_HINT : undefined,
        });
      },
    };
    if (appConfig) {
      engineConfig.appConfig = appConfig;
    }

    const signal = this._loadAbort.signal;
    const pending = Promise.resolve(CreateMLCEngine(this.modelId, engineConfig));
    this.engine = await awaitAbortableLoad(pending, signal, (engine) => {
      void disposeRuntime(null, engine).catch(() => {});
    });
  }

  isLoaded() {
    return this._isLoaded;
  }

  isLoading() {
    return this._isLoading;
  }

  async _reloadEngine(reason = 'reset') {
    if (modelBackend(this.modelId) === 'transformers') {
      await this.engine.reset();
      this._needsEngineReload = false;
      return;
    }
    if (isLiteRTModel(this.modelId)) {
      await this._ensureLiteRTConversation(true);
      this._needsEngineReload = false;
      return;
    }
    // Reset the conversation without evicting model weights.
    if (typeof this.engine?.resetChat === 'function') {
      await this.engine.resetChat();
      this._needsEngineReload = false;
      return;
    }
    if (!this.engine || typeof this.engine.reload !== 'function') {
      this._needsEngineReload = false;
      return;
    }
    this._emitProgress({
      stage: 'init',
      message: reason === 'reset' ? 'Resetting model state…' : 'Refreshing model state…',
    });
    await this.engine.reload(this.modelId);
    this._needsEngineReload = false;
    this._emitProgress({ stage: 'ready', message: 'Model ready.' });
  }

  async _completeOnceLiteRT(userText, onUpdate) {
    return this._completeOnceLiteRTDetailed(userText, onUpdate);
  }

  async _completeOnce(payload, genConfig, onUpdate) {
    const chunks = await this.engine.chat.completions.create({
      messages: payload,
      ...genConfig,
      stream: true,
      stream_options: { include_usage: true },
    });

    let reply = '';
    let usage = null;
    let oversized = false;

    for await (const chunk of chunks) {
      // Drain the worker iterator after interrupt so WebLLM releases its generation lock.
      if (this._active?.signal.aborted || oversized || this._previewGate?.blocked) {
        this.engine?.interruptGenerate?.();
        continue;
      }
      if (chunk.usage) usage = chunk.usage;
      const delta = extractCompletionResponseText(chunk);
      if (!delta) continue;
      if (reply.length + delta.length > 131072) {
        oversized = true;
        this.engine?.interruptGenerate?.();
        continue;
      }
      reply += delta;
      const cleanedPartial = visibleGenerationText(reply);
      if (onUpdate && cleanedPartial) onUpdate(cleanedPartial);
    }

    reply = visibleGenerationText(reply);

    if (this._active?.signal.aborted) throw abortError();
    if (oversized) throw new Error('Model reply exceeded the processing limit.');
    if (this._previewGate?.blocked) throw new Error('Output rejected by guardrails.');
    if (!reply.trim()) {
      await this._reloadEngine('empty');
      const completion = await this.engine.chat.completions.create({
        messages: payload,
        ...genConfig,
        stream: false,
      });
      reply = visibleGenerationText(extractCompletionResponseText(completion));
      usage = completion?.usage || usage;
      if (reply && onUpdate) onUpdate(reply);
    }

    return { reply, usage };
  }

  async generate(
    userText: string,
    optionsOrUpdate?: GenerateOptions | ((partial: string) => void) | null,
    onFinish?: ((usage: unknown) => void) | null,
  ) {
    const options: GenerateOptions =
      typeof optionsOrUpdate === 'function' || optionsOrUpdate == null
        ? { onUpdate: optionsOrUpdate || undefined, onFinish: onFinish || undefined }
        : optionsOrUpdate;
    this._checkInput(userText, options);
    if (!this.isLoaded()) throw new Error('Model not loaded. Call load() first.');
    const operation = this._begin(options);
    const gate =
      options.delivery === 'checked-stream'
        ? new CheckedDeliveryGate(this._guardrailProfile, options.onPreview, (guard) => {
            this._reportGuardrail('output', guard, options);
            this._needsEngineReload = true;
            cancelRuntime(this._conversation, this.engine);
          })
        : null;
    this._previewGate = gate;
    const update = (partial: string) => {
      operation.check();
      gate?.accept(partial);
    };
    try {
      const { input, history } = await this._prepareContext(userText, options);
      operation.check();
      let result;
      if (isLiteRTModel(this.modelId)) {
        result = await this._completeOnceLiteRT(input, update);
      } else if (modelBackend(this.modelId) === 'transformers') {
        const engine = this.engine as TransformersEngine;
        const payload = [
          { role: 'system', content: this._composeSystemPrompt() },
          ...history,
          { role: 'user', content: input },
        ];
        const inputTokens = await engine.countTokens(payload);
        operation.check();
        const budget = Math.min(
          options.contextTokenBudget || 4096,
          getModelOption(this.modelId)?.maxNumTokens || 4096,
        );
        if (inputTokens + this._outputTokens > budget)
          throw new Error('This message exceeds the model context budget. Shorten the message.');
        await engine.reset();
        result = await engine.generate(payload, {
          maxOutputTokens: this._outputTokens,
          signal: operation.controller.signal,
          onUpdate: (text) => {
            const visible = visibleGenerationText(text);
            if (visible) update(visible);
          },
        });
        result.reply = visibleGenerationText(result.reply);
        if (!result.reply.trim())
          throw new Error(
            'The model reached its output limit before producing a visible answer. Try a shorter question.',
          );
        this._needsEngineReload = false;
      } else {
        if (this._needsEngineReload) await this._reloadEngine('reset');
        operation.check();
        const payload = buildChatPayload(
          this.modelId,
          [...history, { role: 'user', content: input }],
          this._composeSystemPrompt(),
        );
        const config = generationConfigForModel(this.modelId);
        result = await this._completeOnce(
          payload,
          { ...config, max_tokens: this._outputTokens },
          update,
        );
      }
      operation.check();
      if (!result.reply.trim() && !gate?.blocked)
        throw new Error(
          `Model ${this.modelId} returned an empty response. Reload the model and try again.`,
        );
      const safe = gate?.blocked
        ? LLM_OUTPUT_BLOCK_MESSAGE
        : this._checkedReply(result.reply, options);
      this.messages.push({ role: 'user', content: userText }, { role: 'assistant', content: safe });
      if (safe !== result.reply) this._needsEngineReload = true;
      gate?.clear();
      options.onUpdate?.(safe);
      options.onFinish?.(result.usage);
      return safe;
    } catch (err) {
      this._needsEngineReload = true;
      if (operation.controller.signal.aborted) throw abortError();
      if (gate?.blocked) {
        this.messages.push(
          { role: 'user', content: userText },
          { role: 'assistant', content: LLM_OUTPUT_BLOCK_MESSAGE },
        );
        gate.clear();
        options.onUpdate?.(LLM_OUTPUT_BLOCK_MESSAGE);
        options.onFinish?.(null);
        return LLM_OUTPUT_BLOCK_MESSAGE;
      }
      throw err;
    } finally {
      gate?.clear();
      this._previewGate = null;
      operation.finish();
    }
  }

  reset() {
    this.cancel();
    this._replay = [];
    this._lastContextStart = 0;
    this.messages = [];
    // Rebuild model context before the next turn.
    this._needsEngineReload = true;
  }

  /** Release WebGPU/WASM engine resources (eval harness / model switch). */
  async dispose() {
    this._loadEpoch++;
    this.cancel();
    await this._loadSettled;
    await this._settled;
    await this._disposeEngine();
    this._isLoaded = false;
    this._isLoading = false;
    this.messages = [];
    this._needsEngineReload = false;
  }
}

// ═══════════════════════════════════════════════════════════════════════
// AiChatUI — legacy imperative DOM component (compat / tests)
// Labs site uses Vue `WwlAiChatUI` instead.
// ═══════════════════════════════════════════════════════════════════════
