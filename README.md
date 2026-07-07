# StyleReel, Multi-Style Video Captioning Agent

**AMD Developer Hackathon: ACT II, Track 2**

StyleReel watches a short video clip and writes a caption in four distinct voices, `formal`, `sarcastic`, `humorous_tech`, and `humorous_non_tech`, that stay *factually
grounded in what actually happens on screen*. It ships as a single Docker image that reads
`/input/tasks.json` and writes `/output/results.json`.

## Quick start

```bash
docker run --rm \
  -v "$PWD/input:/input:ro" -v "$PWD/output:/output" \
  ghcr.io/<user>/stylereel:latest
```

`input/tasks.json`:
```json
[
  {"task_id": "v1",
   "video_url": "https://.../clip.mp4",
   "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]}
]
```

`output/results.json`:
```json
[
  {"task_id": "v1",
   "captions": {"formal": "...", "sarcastic": "...",
                "humorous_tech": "...", "humorous_non_tech": "..."}}
]
```

## How it works

```
video ─┬─ ffmpeg smart-frame sampling ── 16 diverse, sharp, timestamped frames
       └─ faster-whisper ASR (VAD-gated) ── timestamped transcript (dialogue = humor fuel)
                          │
          ┌───────────────┴───────────────┐
          ▼                                
   STAGE 1 · DESCRIBE                       one dense, structured, factual pass
   (vision VLM)                             (subjects, actions+timestamps, setting,
                                             on-screen text, audio, incongruity)
          │
          ▼  facts only (no frames) → prevents per-style hallucination
   STAGE 2 · STYLIZE ×4                      persona + gold exemplars + Witscript
   (best-of-3 per style)                     angle-selection; facts inside the jokes
          │
          ▼
   STAGE 3 · SELF-JUDGE                      rubric judge (different model family):
                                             accuracy hard-gate → revise; tone breaks ties
          │
          ▼
   results.json                              every requested style, always valid JSON
```

**Why this wins points**
- **Two-stage describe→stylize**: all four styles derive from one verified fact sheet, so
  they never contradict the video or each other. Accuracy is the dimension you *lose*, not
  win, one hallucinated detail is a direct deduction, so every claim is grounded.
- **Facts inside the funny styles**: the jokes are *about* the real subject, action, and
  setting, not generic "when Monday hits" filler. This is the largest scoring gap in the
  field on the humorous/sarcastic captions.
- **Smart frames, not uniform**: ffmpeg samples candidates, then greedy HSV-diversity +
  Laplacian-sharpness picks 16 that cover every scene, no wasted frames on static clips,
  nothing missed on action clips.
- **Audio matters**: Fireworks deprecated its ASR, so most entrants have no dialogue.
  We transcribe locally, dialogue is what makes sarcasm and humor land.

## Resilience (the score-protection layer)
- **Skeleton-first output**: a complete, valid `results.json` exists from t=0 and is
  atomically refreshed after every clip, a crash or timeout never yields *nothing*.
- **Degraded-mode ladder** per clip: full pipeline → skip judge → skip best-of-N →
  single-shot caption → grounded fallback. A style key is never missing.
- **Deadline watchdog + hard exit**: results are written inside the event loop and the
  process hard-exits, so an abandoned CPU thread can't push the write past the 10-min kill.
- **Model fallback chains + retry/backoff**: survives 5xx, 429, empty responses, non-JSON.
- **Verified**: `make chaos` runs dead URLs, odd/empty/case-variant styles, wrapped input,
  and malformed entries, all produce complete valid JSON and exit 0.

## Backends
Fireworks AI serverless (Qwen 3.7 Plus vision · Kimi K2.6 stylize · DeepSeek V4 Flash judge),
with an optional **self-hosted Gemma 3 12B on AMD Radeon PRO W7900 (vLLM/ROCm)** as the preferred
describe backend (`GEMMA_ENDPOINT`), automatically failing over to Fireworks. See
[`docs/GEMMA.md`](docs/GEMMA.md).

## Development
```bash
uv sync                       # install
make test                     # 39 unit tests (mocked APIs, no key needed)
make chaos                    # fault-injection contract suite
make build                    # linux/amd64 image
make harness                  # dev-set run + LLM-judge scoring (needs FIREWORKS_API_KEY)
```
Put your key in `.fireworks_key` at the repo root (gitignored) or `export FIREWORKS_API_KEY=...`.

## Evaluation
We ship a local eval harness (`harness/`) that runs the container against a 15-clip dev set
and scores every caption with an independent LLM-judge on accuracy + style, plus a
blind-style-ID distinctness metric. Current dev-set standing:

| metric | score |
|---|---|
| combined (accuracy + style) | ~0.90 |
| style match | ~0.93 |
| blind style-ID (can a judge tell the 4 apart?) | **1.00** |
| 2-min multi-scene clip, end-to-end | 51 s |
| chaos suite (dead URL, odd styles, malformed input) | 5/5 pass |

See `docs/architecture.md` (pipeline), `docs/GEMMA.md` (the AMD Radeon PRO W7900 + Gemma story),
`docs/SUBMISSION.md` (demo script), and the ablation table in `harness/out/ablation/`.

## License
MIT (see `LICENSE`). Bundled: faster-whisper (MIT), PySceneDetect (BSD-3), OpenCV (Apache-2.0).
Gemma usage is governed by the Gemma Terms of Use.
