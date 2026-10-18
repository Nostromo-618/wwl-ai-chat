import type { ModelOption } from './model-catalog.js';
import type { ChatMessage } from './session.js';

/** Host-owned worker adapter; this package never imports a heavy ONNX dependency. */
export type TransformersEngine = {
  generate(
    messages: ChatMessage[],
    options: {
      maxOutputTokens: number;
      signal: AbortSignal;
      onUpdate: (text: string) => void;
    },
  ): Promise<{ reply: string; usage: unknown }>;
  countTokens(messages: ChatMessage[]): Promise<number>;
  cancel(): void;
  reset(): Promise<void>;
  dispose(): Promise<void>;
};
export type TransformersRuntime = {
  createEngine(
    model: ModelOption,
    options: {
      signal: AbortSignal;
      onProgress: (progress: Record<string, unknown>) => void;
    },
  ): Promise<TransformersEngine>;
};

/** Which Transformers.js class and files a catalog ONNX model may load. */
export type OnnxTextLoadPlan = {
  modelClass: 'AutoModelForCausalLM';
  dtype?: string;
  useExternalDataFormat?: number;
};

/** LFM keeps the single `model_*.onnx` decoder. */
export function onnxTextLoadPlan(
  model: Pick<ModelOption, 'precision' | 'externalDataFiles'>,
): OnnxTextLoadPlan {
  return {
    modelClass: 'AutoModelForCausalLM',
    dtype: model.precision,
    useExternalDataFormat: model.externalDataFiles,
  };
}
