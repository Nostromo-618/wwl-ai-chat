export {
  AiChat,
  WWL_AI_CHAT_VERSION,
  TOOLS_UNSUPPORTED_ERROR,
  MODEL_GROUPS,
  MODEL_OPTIONS,
  LOAD_FREEZE_HINT,
  inferLoadSource,
  describeLoadProgress,
  LITERT_PREFILLDECODE_UNSUPPORTED_REASON,
  isLiteRTPrefillDecodeUnsupported,
  getLiteRTRuntimeBlockReason,
  rewriteLiteRTLoadError,
  buildWeakDeviceConfirmCopy,
  shouldFocusChatComposer,
  MODEL_CACHE_FLAG_PREFIX,
  LITERT_MODEL_CACHE_NAME,
  iterateMessageStream,
  sanitizeModelReply,
  applyOutputGuardrails,
  getModelOption,
  getModelDisplayName,
  yieldToMain,
  collectDeviceSignals,
  assessLoadCapacity,
  modelCacheFlagKey,
  isModelMarkedCached,
  markModelCached,
  openLiteRTModelCache,
  matchCachedModel,
  putCachedModel,
  deleteLiteRTModelCache,
  loadLiteRTModelBytes,
  InputGuardrail,
} from './ai-chat.js';

export {
  validateLlmInput,
  validateLlmOutput,
  buildChatSystemPrompt,
  FLOWCHART_CANVAS_OFF_CLAUSE,
  LLM_BLOCK_MESSAGE,
  LLM_OUTPUT_BLOCK_MESSAGE,
  normalizeJailbreakScanText,
  chatGuardrails,
  DEFAULT_LLM_GUARD_PATTERNS,
  DEFAULT_LLM_OUTPUT_GUARD_PATTERNS,
  FOSS_ROLE_LOCK_RULES,
  BASE_FOSS_GUARDRAILS_SYSTEM_PROMPT,
  FOSS_SYSTEM_PROMPT_TRAILER,
  validateToolCall,
  parseXmlToolCalls,
  formatXmlToolResult,
} from './guardrails/llm.js';
export {
  SET_FLOWCHART_DOCUMENT,
  FLOWCHART_DRAW_FOLLOW_UP,
  FLOWCHART_TOOL_DESCRIPTION,
  FLOWCHART_TOOL_PARAMETERS,
  prepareFlowchartDocument,
  flowchartDrawNeedsFollowUp,
  validateFlowchartDocument,
  flowchartSnapshot,
  visionImagesForModel,
  parseLooseToolCalls,
} from './guardrails/tools.js';

export { labsMarkdownToHtml, markdownToHtml } from './markdown.js';

export {
  VD_GUARDRAILS_VERSION,
  allow,
  block,
  normalizeText,
  toGuardrailError,
} from './guardrails/core.js';
export type { GuardrailResult, GuardrailError } from './guardrails/core.js';
export type { ToolDefinition } from './guardrails/tools.js';

export type {
  GenerateOptions,
  ContextSource,
  ContextStatus,
  ChatMessage,
  GuardrailEvent,
} from './session.js';
export type { AiChatOptions } from './ai-chat.js';

export {
  NEW_MODEL_IDS,
  BASELINE_MODEL_IDS,
  PRIMARY_MODEL_OPTIONS,
  getModelVariants,
  getModelChoiceLabel,
} from './model-catalog.js';
export type { ModelOption, ModelArtifact } from './model-catalog.js';
export type { TransformersEngine, TransformersRuntime } from './transformers-runtime.js';
export { onnxTextLoadPlan } from './transformers-runtime.js';
export type { OnnxTextLoadPlan } from './transformers-runtime.js';

export type { GuardrailProfile } from './guardrails/moderation.js';
