# wwl-ai-chat

Headless on-device chat for Weekend Web Labs. LiteRT, WebLLM, and Transformers.js, with local guardrails and CSP-safe markdown.

Not published. [Labs](https://github.com/nostromo-618/weekend-web-labs) consumes it as a sibling (`link:../wwl-ai-chat`).

Default model: Gemma 4 E2B LiteRT (`gemma-4-E2B-it-web`). Needs WebGPU.

Gemma 4 (`gemma-4-E2B-it-web` and `gemma-4-E4B-it-web`) are the priority models for full native tool calling and flowchart canvas interaction (`set_flowchart_document` + canvas snapshot explanation). Experimental WebLLM Mistral and LFM models are supported for Chat and Compare (`toolCalling.integration: 'pending'`, omitted from flowchart-interaction selection while the Flowchart panel is open).

```bash
pnpm install && pnpm build
```

Spec: [openspec/](./openspec/).

MIT — see [LICENSE](./LICENSE).

