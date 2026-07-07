# Gemma-on-AMD Runbook (execute inside the AMD notebook, one 4-hour block)

GPU access: https://notebooks.amd.com/hackathon (team-3594). Environment:
ROCm 7.2 + vLLM 0.16.0 + PyTorch 2.9. **Quota: 4 hours per 24 hours**, so run this as one
focused block. This produces the $3,000 Gemma-prize evidence; the submitted image keeps the
Fireworks backend (Gemma is evidence, not a live judging dependency).

## Prerequisite (do before launching, saves quota)
Gemma 3 is a **gated** Hugging Face model. Before the session:
1. Sign in at huggingface.co, open https://huggingface.co/google/gemma-3-27b-it and click
   "Agree and access repository" (accept the Gemma license).
2. Create a read token at https://huggingface.co/settings/tokens. Have it ready as HF_TOKEN.

## In the Jupyter terminal (File > New > Terminal)
```bash
# 0. Confirm the GPU is an MI300X and see memory
rocm-smi                      # screenshot this (evidence)

# 1. Gemma is gated: authenticate
export HF_TOKEN=<paste your HF token>
pip install -q -U huggingface_hub
hf auth login --token "$HF_TOKEN" || huggingface-cli login --token "$HF_TOKEN"

# 2. Serve Gemma 3 27B multimodal (vLLM is preinstalled)
#    bf16 (not fp8), cap context to fit, allow up to 8 frames per request.
vllm serve google/gemma-3-27b-it \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --limit-mm-per-prompt image=8 \
  --gpu-memory-utilization 0.9 \
  --port 8000 > /tmp/vllm.log 2>&1 &
#    watch startup (model download ~54GB, then load). Screenshot the log line that shows
#    the model loaded + the vision tower initialised.
tail -f /tmp/vllm.log          # wait for "Application startup complete" / server on :8000

# 3. Prove the vision path works (evidence)
python - <<'PY'
import base64, requests, glob, os
# grab one example frame if present, else make a tiny test image
img = sorted(glob.glob("frame*.jpg")) or None
if not img:
    import urllib.request
    urllib.request.urlretrieve("https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4","clip.mp4")
    os.system("ffmpeg -y -ss 1 -i clip.mp4 -frames:v 1 -vf scale=768:-2 frame.jpg -loglevel error")
    img=["frame.jpg"]
b64=base64.b64encode(open(img[0],'rb').read()).decode()
r=requests.post("http://localhost:8000/v1/chat/completions", json={
  "model":"google/gemma-3-27b-it","max_tokens":200,"temperature":0.2,
  "messages":[{"role":"user","content":[
    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}},
    {"type":"text","text":"Describe what is happening in this video frame in 2 sentences."}]}]})
print(r.json()["choices"][0]["message"]["content"])
PY
#    screenshot this description -> proof Gemma 3 does the video understanding on MI300X.
```

## Generate the ablation evidence (Gemma vs Fireworks)
Two options depending on whether the notebook can be reached from your Mac:

**Option A (simplest): run the ablation on your Mac against the notebook's Gemma**, if the
Jupyter server proxies port 8000 (URL like `https://notebooks.amd.com/.../proxy/8000/v1`).
On the Mac:
```bash
cd stylereel
GEMMA_ENDPOINT="<proxied vLLM /v1 url>" GEMMA_MODEL="google/gemma-3-27b-it" \
  uv run python harness/ablation.py --n 6 --arms full,blind
# compare 'full' (Gemma describe) against the existing Fireworks/Kimi run in docs/ABLATION.md
```

**Option B (self-contained): clone the repo into the notebook and run there.**
```bash
pip install -q uv && git clone <repo-with-a-temp-token> && cd stylereel
export FIREWORKS_API_KEY=<key>   # for the stylize/judge stages
export GEMMA_ENDPOINT=http://localhost:8000/v1 GEMMA_MODEL=google/gemma-3-27b-it
uv run python harness/ablation.py --n 6 --arms full,blind
```

## Evidence to capture (screenshot each)
- [ ] `rocm-smi` showing the MI300X and utilisation during a describe call
- [ ] vLLM startup log: `google/gemma-3-27b-it` loaded + vision tower + `--limit-mm-per-prompt image=8`
- [ ] The curl/python image description (Gemma captioning a real frame)
- [ ] Ablation numbers: Gemma-describe vs blind (and vs the Kimi run already in docs/ABLATION.md)
- [ ] 2 or 3 example clips captioned with Gemma as the describe backend

## Notes
- Use Gemma **3** (has vision in vLLM), never Gemma 3n (text-only in vLLM).
- If 27B download/serve is too slow for the window, fall back to `google/gemma-3-12b-it`
  (still multimodal, faster) and note it.
- Stop the notebook when done to preserve the 4h/24h quota.
