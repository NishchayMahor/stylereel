# How Gemma Powers StyleReel (and the $3,000 prize plan)

**Claim:** StyleReel's `describe` stage, the vision model that *watches each video* and
writes the grounded description all four style captions derive from, runs
**`google/gemma-3-12b-it` (multimodal), self-hosted on an AMD Radeon PRO W7900 via
vLLM/ROCm**. Fireworks serves Gemma text-only, so running the *vision* Gemma ourselves is
the differentiator, and it simultaneously nails "best use of AMD platforms."

## Architecture
```
frames (ffmpeg) ──► Gemma 3 12B multimodal @ AMD Radeon PRO W7900 (vLLM/ROCm)  ──► dense description
                     via GEMMA_ENDPOINT (OpenAI-compatible)                │
                     └─ health-check fail ─► Fireworks Kimi vision failover ┘
                                                                           ▼
                                                        stylize ×4 ──► judge ──► results.json
```
`GEMMA_ENDPOINT` is already wired in `fw.py` as the preferred backend with automatic
Fireworks failover, so the 10-minute leaderboard run never depends on endpoint uptime,
while the demo/logs credit Gemma for the real work.

## Deployment (vendor-blessed, ~2 days)
Base image: newest `rocm/vllm-dev:<tag>` (mid-2026 transformers ≥4.53 supports Gemma 3
vision natively). Gemma is a **gated HF model**, accept the license + set `HF_TOKEN` on day 0.

```bash
docker run -it --ipc=host --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --group-add video \
  -e PYTORCH_ROCM_ARCH=gfx942 -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  rocm/vllm-dev:<latest-tag>

vllm serve google/gemma-3-12b-it \
  --dtype bfloat16 \            # bf16, NOT fp8 (fp8 dtype instability reports; Radeon PRO W7900 has HBM headroom)
  --max-model-len 32768 \
  --limit-mm-per-prompt image=8 \   # REQUIRED, default rejects multi-frame requests
  --gpu-memory-utilization 0.9 --port 8000
```
- Use **Gemma 3**, never **Gemma 3n** (3n is text-only in vLLM).
- 12B fits one Radeon PRO W7900 (48GB) at TP=1, no tensor-parallel/NCCL tuning.
- Sanity-test the vision path with a `curl` image_url request before trusting it.
- 12B is the pragmatic failover tier if 12B prefill exceeds the 45s/clip budget.

## Evidence pack (this is a human-judged prize, artifacts win it)
- [ ] **Ablation table** on the dev set: (i) full pipeline w/ Gemma describe, (ii) describe
      removed / blind stylize, (iii) Gemma swapped for Kimi vision → show accuracy delta.
      **This likely wins the prize alone.**
- [ ] `rocm-smi` screenshot during a live Gemma describe run (next to vLLM logs)
- [ ] vLLM startup log: model loaded + vision tower + `--limit-mm-per-prompt image=8`
- [ ] `curl` image request → description (proves vision path)
- [ ] Architecture diagram with GEMMA_ENDPOINT + Fireworks failover
- [ ] Demo video segment: clip → Gemma's description → 4 styled captions
- [ ] README "How Gemma is used" paragraph + tags (Gemma, AMD, Radeon PRO W7900, vLLM)
- [ ] Per-call backend log proving Gemma served the reported results
- [ ] Honest failover disclosure + Gemma Terms of Use note

## Fine-tuning verdict
Do **not** fine-tune the vision side (overfit trap; hidden 8-category eval; harness
penalizes hardcoding). Optional stretch AFTER core is solid: a rank-16 Gemma-3 **text**
LoRA for the stylize stage on synthetic (description→styled-caption) pairs, creates a
clean "Gemma sees, Gemma styles" story. Drop it if it doesn't beat prompt-only in one
afternoon.

## Blocker
AMD Developer Cloud access + $100 credits are pending approval, the critical path.
Escalate now. Evidence is captured the moment the endpoint is healthy (survives instance
reclamation). Insurance: rent an Radeon PRO W7900 elsewhere, or Gemma-4 on a Fireworks dedicated
deployment (weaker "self-host on AMD" story).

Sources: AMD ROCm blog "Deploying Gemma 3 with vLLM on Radeon PRO W7900"
(rocm.blogs.amd.com/artificial-intelligence/deployingGemma-vllm), vLLM Supported Models
(docs.vllm.ai/en/latest/models/supported_models), rocm/vllm-dev on Docker Hub.
