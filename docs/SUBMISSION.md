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

## Key handling (Track 2)
The guide is explicit for Track 2: "No API key or model restriction is injected. Use your own
credentials inside the container." So baking the Fireworks key into the image is the intended
design, not a violation. The only caveat is that the image must be publicly pullable, so the
baked key is technically extractable. It is a capped hackathon key (~$50), so blast radius is
small. You do NOT need to rotate before pushing. Just REVOKE the key after judging.

## Final freeze ritual (do NOT skip)
1. Push the private build now to de-risk mechanics:
   `PUBLISH=0 FIREWORKS_API_KEY=$(cat .fireworks_key) bash scripts/release.sh`
2. At submission, flip public with the same current (or a fresh) key:
   `PUBLISH=1 FIREWORKS_API_KEY=$(cat .fireworks_key) bash scripts/release.sh`
   (release.sh bakes the key, builds linux/amd64, pushes, reverts keybox, then pull-tests + smokes)
3. Flip the GitHub repo and Pages public ONLY at the last moment before submitting (the guide
   requires a public repo). Minimise exposure: the public Docker image already contains our
   source and prompts, so a public repo adds little; keep it private until submission and set it
   back to private after judging ends.
4. Submit that exact tag/digest on lablab >=12h before deadline; keep 3 submission attempts in reserve.

## AFTER judging (do NOT forget)
- [ ] **Revoke the baked Fireworks key** at https://app.fireworks.ai/settings/users/api-keys
      (the image is public, so the key is exposed until revoked). This fully closes the exposure.
- [ ] Optionally make the GHCR package private again and stop any AMD notebook still running.
