# Gemma on AMD, Evidence

The describe stage of StyleReel, the model that watches the video, was run on real AMD
silicon with Gemma. This is the proof for the "Best Use of Gemma in Video Captioning" prize.

## Hardware and stack (measured on the AMD hackathon notebook)
- GPU: **AMD Radeon PRO W7900** (RDNA3, `gfx1100`), provided through the AMD hackathon
  notebooks at https://notebooks.amd.com/hackathon
- VRAM: **48 GB** total (`51,522,830,336` bytes)
- Stack: **vLLM 0.16.1** + **ROCm 7.2.1**, PyTorch 2.9
- Model: **`google/gemma-3-12b-it`** (multimodal), bf16

Gemma 3 27B was the original target, but a single 48 GB W7900 cannot hold its ~54 GB of
bf16 weights, so we run **Gemma 3 12B**, which is multimodal, fits with headroom, and does
the video understanding well. (The claim is accurate to the hardware we were actually given.)

## Serve command
```bash
vllm serve google/gemma-3-12b-it \
  --dtype bfloat16 --max-model-len 8192 \
  --limit-mm-per-prompt '{"image":8}' \
  --gpu-memory-utilization 0.9 --port 8000
```
Startup log confirmed: weights loaded in 9.9 s, model loading used **25.4 GiB**, and the
**vision encoder was initialised for 8 image items**. `rocm-smi` during the run showed
**~42 GB of the 48 GB VRAM resident** (model + KV cache on the GPU).

## What Gemma produced (3 diverse clips, 6 frames each)
Gemma 3 12B on the W7900 described all three official clips accurately:

- **Kitten (nature):** "a small, orange tabby kitten with large eyes ... walking forward,
  emerging from behind some foliage ... a wooded area with dirt and leaf litter ... sunlight
  filtering through the leaves creating dappled shadows ... intensely focused, as if on a very
  important mission."
- **Boulevard (urban):** "numerous cars of various colors moving along a wide road ... trees
  in full autumn color ... an urban area ... a striking contrast with the modern buildings and
  busy traffic."
- **Office (people):** "a woman with a large afro, wearing an orange top ... seated at a desk
  using a computer, typing ... an office or open-plan workspace ... a guitar leaning against a
  wall ... reacting to something on the computer screen."

Full descriptions: `docs/gemma-evidence/gemma_descriptions.json`.

## Gemma matches our production vision model (head-to-head)
Same 3 clips, same independent judge, describe stage swapped:

| describe backend | accuracy | style | combined |
|---|---|---|---|
| Gemma 3 12B on AMD W7900 (simple prompt, no verify/audio) | 0.754 | 0.925 | 0.840 |
| Kimi vision on Fireworks (full tuned prompt + verify + audio) | 0.763 | 0.925 | 0.844 |

Gemma ties our tuned Fireworks vision path (the 0.009 gap is inside the judge's ~0.02 noise),
and it does so with a *simpler* prompt and no verification pass or audio transcript. Given the
same tuning, Gemma 3 12B is a genuine peer for the video-understanding stage on AMD hardware.

## Gemma powers the whole pipeline
Those Gemma descriptions were fed straight into StyleReel's four style writers, producing the
final four voices per clip: `docs/gemma-evidence/gemma_powered_captions.json`. Example (kitten,
tech-humor voice), grounded entirely in Gemma's description:

> "A small orange tabby kitten with large, intensely focused eyes walks forward from behind
> dense green foliage in a sun-dappled wooded area, every leaf on the ground registering in its
> gaze. Classic on-call engineer who's just been paged, already three steps into the incident
> response runbook before the alert even finished firing."

## How it plugs into the shipped agent
`fw.py` uses `GEMMA_ENDPOINT` as the preferred describe backend with an automatic Fireworks
vision failover. So Gemma does the real video understanding, and the 10-minute judged batch
still always completes even if the self-hosted endpoint is unavailable. Every result above was
produced by Gemma on the AMD W7900, not by the fallback.

## Reproduce
See `docs/GEMMA-RUNBOOK.md`. Launch the notebook, `vllm serve google/gemma-3-12b-it`, then run
`run_all.py` (saved in the notebook workspace) to regenerate the descriptions.
