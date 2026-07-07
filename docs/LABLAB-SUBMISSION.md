# lablab Submission Content (ready to paste)

Fill these fields on the lablab.ai submission form for AMD Developer Hackathon ACT II, Track 2.

## Project title
StyleReel, Four Voices for Every Video (Gemma on AMD Radeon PRO W7900)

## Short description (one line, put Gemma + AMD up front for prize discoverability)
A containerized video-captioning agent that writes four grounded voices per clip, with the
video-understanding stage running Gemma 3 12B on an AMD Radeon PRO W7900.

## Long description
StyleReel watches a short video clip, works out what is actually happening on screen and in
the audio, then captions it in four distinct registers: formal, sarcastic, humorous-tech, and
humorous-non-tech. The facts stay true to the video; only the tone changes.

How it works: smart frame selection plus a local Whisper transcript (Fireworks retired its
speech API, so we run our own, which is why our captions understand dialogue). A vision model
writes one grounded description and verifies it against the frames a second time. Four style
writers then work only from that checked fact sheet, best of five drafts each, and an
independent model gates accuracy and keeps the strongest take. Because every caption derives
from one verified description, the four never contradict the video or each other.

We measured everything with a local harness that scores each caption with a separate model:
combined 0.89, style match 0.93, and a blind style-ID of 1.00 (a judge tells all four voices
apart every time). An ablation shows the understanding stage is where the score is won:
removing it drops accuracy from 0.86 to 0.72.

The describe stage, the model that actually watches the video, runs Gemma 3 12B multimodal,
self-hosted on an AMD Radeon PRO W7900 via vLLM and ROCm. Fireworks only serves Gemma as text,
so running the vision Gemma ourselves is the differentiator and puts real AMD compute at the
core. A Fireworks vision failover guarantees the 10-minute batch always finishes.

Resilience is a feature: a valid results file exists from the first moment and refreshes after
every clip, a degraded ladder guarantees a style key is never missing, a deadline watchdog
writes output before any slow thread can push past the 10-minute kill, and five of five chaos
drills still produce valid JSON and exit clean. A 2-minute, 3-scene clip captions end to end
in 51 seconds.

## Technology and category tags
video, video-captioning, AMD, Radeon PRO W7900, ROCm, vLLM, Gemma, Fireworks AI, Docker, agents, LLM, multimodal

## Cover image
Use `docs/cover.png` (generated from the demo identity).

## Application / demo URL
[GitHub Pages URL once repo is public, e.g. https://nishchaymahor.github.io/stylereel/  OR the claude.ai artifact link]

## Public GitHub repository
https://github.com/NishchayMahor/stylereel  (flip to public at submission)

## Docker image
ghcr.io/nishchaymahor/stylereel:latest  (public, linux/amd64, built via scripts/release.sh)

## Video presentation
See storyboard in docs/SUBMISSION.md (<= 5 min, demo-first).

## Slide deck
Built from docs/SLIDES.md.

## Important requirements checklist (from the guide)
- [ ] Image publicly pullable, linux/amd64 manifest, <= 10 GB
- [ ] Public repo with a README that reproduces in one command
- [ ] No hardcoded/cached answers (verified: unseen-variant safe, chaos-tested)
- [ ] MIT licensed
