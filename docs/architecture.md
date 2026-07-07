# StyleReel Architecture

```
                          ┌──────────────────────────────────────────┐
   /input/tasks.json ────►│  orchestrator (asyncio, 4 clips parallel) │
                          │  deadline watchdog · skeleton-first output │
                          └───────────────────┬──────────────────────┘
                                              │  per clip
                        ┌─────────────────────┼─────────────────────┐
                        ▼                                            ▼
              ┌───────────────────┐                       ┌───────────────────┐
              │  DOWNLOAD (stream, │                       │  AUDIO             │
              │  size/time caps)   │                       │  silero-VAD gate → │
              └─────────┬─────────┘                        │  faster-whisper    │
                        ▼                                   │  (timestamped)     │
              ┌───────────────────┐                        └─────────┬─────────┘
              │  SMART FRAMES      │                                  │
              │  ffmpeg sample →   │                                  │
              │  HSV-diversity +   │                                  │
              │  sharpness → 16    │                                  │
              └─────────┬─────────┘                                  │
                        └───────────────┬──────────────────────────── ┘
                                        ▼  frames + transcript
                    ┌────────────────────────────────────────────┐
                    │  STAGE 1 · DESCRIBE   (vision)              │
                    │  ┌──────────────────────────────────────┐  │
                    │  │ PRIMARY: Gemma 3 27B @ AMD MI300X     │  │  ◄── $3k Gemma prize
                    │  │ (vLLM/ROCm) via GEMMA_ENDPOINT        │  │
                    │  │ FAILOVER: Fireworks Kimi K2.6 vision  │  │
                    │  └──────────────────────────────────────┘  │
                    │  draft facts → VERIFY pass (re-check vs     │
                    │  frames) → grounded description             │
                    └───────────────────┬────────────────────────┘
                                        ▼  facts only (no frames leak downstream)
                    ┌────────────────────────────────────────────┐
                    │  STAGE 2 · STYLIZE ×4  (text, parallel)     │
                    │  persona + gold exemplars + Witscript       │
                    │  angle-selection · best-of-5 (humor)        │
                    │  formal│sarcastic│humorous_tech│h_non_tech  │
                    └───────────────────┬────────────────────────┘
                                        ▼  candidates
                    ┌────────────────────────────────────────────┐
                    │  STAGE 3 · SELF-JUDGE  (text, independent   │
                    │  model family, DeepSeek V4)                │
                    │  accuracy hard-gate → revise; tone → pick   │
                    └───────────────────┬────────────────────────┘
                                        ▼
                          /output/results.json  (every style, valid JSON, exit 0)

  DEGRADED LADDER (per clip, deadline-aware):
    full → skip judge → skip best-of-N → single-shot from 8 frames → grounded fallback
    A style key is NEVER missing; JSON is NEVER malformed; exit is ALWAYS 0.
```

## Model roster (Fireworks serverless, verified live)
| Stage | Primary | Failover chain |
|---|---|---|
| Describe (vision) | Gemma 3 27B @ MI300X → Kimi K2.6 | Kimi K2.6 → Kimi K2.5 |
| Stylize (text) | Kimi K2.6 | DeepSeek V4 Pro → GLM 5.2 |
| Judge (text) | DeepSeek V4 Pro | GLM 5.2 |

Judge is a different model family than the generator → no self-preference bias.
`reasoning_effort=none` + `<think>` stripping keeps captions clean (these are reasoning models).
