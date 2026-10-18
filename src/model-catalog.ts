import manifests from './model-artifacts.json';
export type ModelArtifact = { path: string; bytes: number; sha256?: string; gitSha1?: string };
export type ModelOption = {
  id: string;
  label: string;
  tier: string;
  group: string;
  family: string;
  backend: 'litert' | 'webllm' | 'transformers';
  requires: string[];
  approxBytes: number;
  maxNumTokens?: number;
  experimental?: boolean;
  litertKind?: string;
  litertRuntime?: string;
  disableThinking?: boolean;
  fallbackId?: string;
  modelFile?: string;
  modelUrl?: string;
  modelLibUrl?: string;
  overrides?: Record<string, number>;
  repo?: string;
  revision?: string;
  artifacts?: ModelArtifact[];
  precision?: string;
  license?: string;
  purpose?: string;
  estimatedWorkingBytes?: number;
  reasoning?: 'none' | 'optional' | 'required';
  variantOf?: string;
  catalogRole?: 'primary' | 'variant';
  capabilities?: { chat: boolean; docs: boolean; tools: boolean; vision?: boolean };
  runtimeVersion?: string;
  externalDataFiles?: number;
  toolCalling?: {
    modelSupport: 'documented';
    integration: 'native' | 'xml' | 'pending';
    source: string;
  };
};
export const MODEL_GROUPS = [
  { id: 'gemma4', label: 'Gemma 4' },
  { id: 'liquid', label: 'Liquid LFM 2.5' },
  { id: 'ministral3', label: 'Ministral 3' },
  { id: 'mistral7b', label: 'Mistral 7B' },
];
const WEBLLM_LIB_PREFIX =
  'https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_86/base/';

const models: ModelOption[] = [
  ...(['E2B', 'E4B'] as const).map((size): ModelOption => ({
    id: `gemma-4-${size}-it-web`,
    label: `Gemma 4 ${size}${size === 'E2B' ? ' · Default' : ' · Quality'}`,
    tier: size === 'E2B' ? 'Fast' : 'Quality',
    group: 'gemma4',
    family: 'gemma4',
    backend: 'litert',
    litertKind: 'web-official',
    requires: ['shader-f16'],
    approxBytes: 0,
    maxNumTokens: 8192,
    modelFile: `gemma-4-${size}-it-web.litertlm`,
    catalogRole: 'primary',
    capabilities: { chat: true, docs: true, tools: true },
    toolCalling: {
      modelSupport: 'documented',
      integration: 'native',
      source: 'https://deepmind.google/models/gemma/gemma-4/',
    },
    purpose:
      size === 'E2B'
        ? 'Recommended general chat and documentation assistant.'
        : 'Larger general chat and documentation assistant.',
    runtimeVersion: '0.17.1',
    reasoning: 'none',
  })),
  ...(['q4f16', 'q4'] as const).map((precision): ModelOption => ({
    id: `LFM2.5-2.6B-${precision}-ONNX`,
    label: 'LFM2.5 2.6B · Candidate',
    tier: 'Reasoning candidate',
    group: 'liquid',
    family: 'lfm25',
    backend: 'transformers',
    requires: precision === 'q4f16' ? ['shader-f16'] : [],
    approxBytes: 0,
    maxNumTokens: 4096,
    experimental: true,
    precision,
    repo: 'LiquidAI/LFM2.5-2.6B-ONNX',
    catalogRole: precision === 'q4' ? 'variant' : 'primary',
    ...(precision === 'q4' ? { variantOf: 'LFM2.5-2.6B-q4f16-ONNX' } : {}),
    license: 'LFM-1.0',
    purpose: 'Reasoning and grounded answers; candidate.',
    reasoning: 'required',
    runtimeVersion: '4.3.0',
    externalDataFiles: precision === 'q4f16' ? 2 : 1,
    capabilities: { chat: true, docs: true, tools: false },
    toolCalling: {
      modelSupport: 'documented',
      integration: 'pending',
      source: 'https://huggingface.co/LiquidAI/LFM2.5-2.6B#tool-use',
    },
  })),
  ...[
    {
      id: 'Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC',
      label: 'Ministral 3 3B Instruct · Experimental',
      tier: 'Experimental',
      wasm: 'Ministral-3-3B-Instruct-2512-BF16-q4f16_1_cs1k-webgpu.wasm',
      reasoning: 'none' as const,
      purpose: 'Experimental general chat candidate.',
      source: 'https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512',
      estimatedWorkingBytes: Math.round(2863.69 * 1e6),
      overrides: { context_window_size: 4096 },
    },
    {
      id: 'Ministral-3-3B-Reasoning-2512-q4f16_1-MLC',
      label: 'Ministral 3 3B Reasoning · Experimental',
      tier: 'Reasoning candidate',
      wasm: 'Ministral-3-3B-Reasoning-2512-q4f16_1_cs1k-webgpu.wasm',
      reasoning: 'required' as const,
      purpose: 'Experimental reasoning peer beside LFM 2.5. It does not replace LFM.',
      source: 'https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512',
      estimatedWorkingBytes: Math.round(2863.69 * 1e6),
      overrides: { context_window_size: 4096 },
    },
  ].map((model): ModelOption => ({
    id: model.id,
    label: model.label,
    tier: model.tier,
    group: 'ministral3',
    family: 'ministral3',
    backend: 'webllm',
    requires: ['shader-f16'],
    approxBytes: 0,
    maxNumTokens: 4096,
    experimental: true,
    precision: 'q4f16',
    catalogRole: 'primary',
    license: 'apache-2.0',
    purpose: model.purpose,
    reasoning: model.reasoning,
    runtimeVersion: '0.2.86',
    estimatedWorkingBytes: model.estimatedWorkingBytes,
    overrides: model.overrides,
    modelLibUrl: `${WEBLLM_LIB_PREFIX}${model.wasm}`,
    capabilities: { chat: true, docs: true, tools: false },
    toolCalling: {
      modelSupport: 'documented',
      integration: 'pending',
      source: model.source,
    },
  })),
  {
    id: 'Mistral-7B-Instruct-v0.3-q4f16_1-MLC',
    label: 'Mistral 7B Instruct · Experimental',
    tier: 'Experimental',
    group: 'mistral7b',
    family: 'mistral7b',
    backend: 'webllm',
    requires: ['shader-f16'],
    approxBytes: 0,
    maxNumTokens: 4096,
    experimental: true,
    precision: 'q4f16',
    catalogRole: 'primary',
    license: 'apache-2.0',
    purpose: 'Larger experimental chat option.',
    reasoning: 'none',
    runtimeVersion: '0.2.86',
    estimatedWorkingBytes: Math.round(4573.39 * 1e6),
    overrides: { context_window_size: 4096, sliding_window_size: -1 },
    modelLibUrl: `${WEBLLM_LIB_PREFIX}Mistral-7B-Instruct-v0.3-q4f16_1_cs1k-webgpu.wasm`,
    capabilities: { chat: true, docs: true, tools: false },
    toolCalling: {
      modelSupport: 'documented',
      integration: 'pending',
      source: 'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3',
    },
  },
];
export const MODEL_OPTIONS: ModelOption[] = models.map((model) => ({
  ...model,
  ...(manifests as Record<string, Partial<ModelOption>>)[model.id],
}));
export const PRIMARY_MODEL_OPTIONS = MODEL_OPTIONS.filter(
  (model) => model.catalogRole === 'primary',
);
export function getModelVariants(id: string): ModelOption[] {
  const primary = MODEL_OPTIONS.find((model) => model.id === id)?.variantOf || id;
  return MODEL_OPTIONS.filter((model) => model.id === primary || model.variantOf === primary);
}
export function getModelChoiceLabel(model: ModelOption): string {
  const engine = { litert: 'LiteRT', webllm: 'WebLLM', transformers: 'Transformers.js / ONNX' }[
    model.backend
  ];
  const tools = model.capabilities?.tools
    ? 'Tools available'
    : 'Tool-capable · integration pending';
  return `${model.label} · ${engine} · ${Math.round(model.approxBytes / 1e6)} MB · ${tools}`;
}
/** Compatibility export: retained LFM evaluation candidates. */
export const NEW_MODEL_IDS = PRIMARY_MODEL_OPTIONS.filter((model) => model.family === 'lfm25').map(
  (model) => model.id,
);
export const BASELINE_MODEL_IDS = ['gemma-4-E2B-it-web', 'gemma-4-E4B-it-web'];
