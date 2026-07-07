# StyleReel, Submission Package Checklist & Demo Script

Deadline: **2026-07-11 09:00 PDT**. Submit via lablab.ai platform.

## Required submission fields (lablab)
- [ ] Project title: **StyleReel, Multi-Style Video Captioning on AMD + Gemma**
- [ ] Short description (put "Gemma" + "AMD Radeon PRO W7900" in the first line for prize discoverability)
- [ ] Long description (problem → pipeline → resilience → Gemma/AMD → eval results)
- [ ] Tags: `video`, `AMD`, `Radeon PRO W7900`, `Gemma`, `Fireworks`, `agents`, `vLLM`, `ROCm`
- [ ] Cover image (original)
- [ ] **Video presentation** (≤5 min MP4), see storyboard below
- [ ] **Slide deck** (PDF, ≤10 slides, mirrors video)
- [ ] **Public GitHub repo** (MIT) with reproduce-in-one-command README
- [ ] Demo / hosted URL (static page showing example outputs)
- [ ] Docker image on GHCR, **public**, linux/amd64, ≤10GB

## Demo video storyboard (≤5:00, demo-first)
1. **0:00–0:30, Problem.** "Four voices, one video, and a 10-minute clock. Miss a style or
   crash and you score zero." Show the 4 style names.
2. **0:30–2:30, Live run.** Terminal: `docker run ... ghcr.io/<user>/stylereel` on the 3
   official clips. Show it pulling, then the results.json appearing. Cut to a clip playing
   with its 4 captions overlaid (use example-01 kitten + example-03 2-min). Emphasize: facts
   inside the jokes; the 2-min clip covers all scenes + audio.
3. **2:30–3:30, Under the hood.** Architecture diagram (docs/architecture.md). Smart frames,
   local Whisper (Fireworks ASR is dead, we transcribe), two-stage describe→stylize, verify
   pass, self-judge, degraded ladder.
4. **3:30–4:30, Gemma on AMD.** "The describe stage, the model that *watches* the video, is Gemma 3 12B multimodal, self-hosted on an AMD Radeon PRO W7900 via vLLM/ROCm." Show rocm-smi +
   vLLM logs + the ablation table (with-Gemma vs without). This is the $3k prize segment.
5. **4:30–5:00, Results & resilience.** Dev-set score table; chaos suite green; "never zero,
   never malformed, always exit 0." Close on the repo + image URL.

## Eval results to show (from harness)
- Dev-set (15 clips × 4 styles) combined ~0.90, style ~0.93, **blind-style-ID 1.0**
- Ablation table (harness/out/ablation/table.json), proves each stage contributes
- Chaos suite: 5/5 scenarios → valid JSON, exit 0
- 2-min multi-scene clip: 51s end-to-end, all scenes + audio covered

## Final freeze ritual (do NOT skip)
1. Rotate Fireworks key → `python scripts/obfuscate_key.py <new_key>` (bakes into local keybox.py)
2. `docker buildx build --platform linux/amd64 -t ghcr.io/<user>/stylereel:latest --push .`
3. `git checkout src/stylereel/keybox.py` (revert baked key, never commit it)
4. Make GHCR package public; `docker logout ghcr.io && docker pull ghcr.io/<user>/stylereel:latest`
5. Run smoke on the pulled digest; confirm valid results.json
6. Submit that exact tag/digest on lablab ≥12h before deadline; keep 3 submission attempts in reserve
