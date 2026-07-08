# lablab Submission Content (ready to paste)

The submission is a **3-step wizard** and it **saves** ("Save and return to the team page"),
so you can fill it in over several sittings and finish before the July 11 09:00 PDT deadline.
Field limits below are the real ones from the form. All copy is written to fit them and to
read like a person wrote it.

## Step 1: Basic Information

### Submission Title (5–50 characters)
Pick one:
- `StyleReel: four voices for every video`  (38 chars, clean)
- `StyleReel: Gemma video captions, four voices`  (44 chars, puts "Gemma" in the title for prize discoverability, recommended)

### Short Description (50–255 characters)
> One clip, four captions, four completely different moods: formal, sarcastic, tech-humor, and everyday-humor. The part that actually watches the video is Gemma 3, self-hosted on an AMD Radeon PRO W7900. The facts stay true, only the tone changes.

(233 characters.)

### Long Description (600–2000 characters, 100+ words)
> StyleReel takes a short video and writes four captions for it, each in a different voice: formal, sarcastic, tech-humor, and everyday-humor. The catch is that all four stay true to what is actually on screen. Only the tone changes.
>
> Here is how it works. It pulls a handful of smart frames and transcribes the audio itself (Fireworks retired its speech API, so we run Whisper locally, which is why our captions actually get the dialogue). Then one model watches those frames, writes down what is happening, and checks its notes against the frames a second time so nothing gets invented. Four separate writers each take that fact sheet and find their voice, best of five drafts. A different model grades every draft for accuracy and keeps the strongest one. Because all four captions come from the same checked description, they never argue with the video or with each other.
>
> The model that watches the video is Gemma 3, self-hosted on an AMD Radeon PRO W7900 with vLLM and ROCm. We did not just call Gemma over an API. We ran its multimodal vision ourselves on AMD hardware, and head to head it matched our best cloud vision model on accuracy and style.
>
> We also cared about never failing. A valid results file exists from the first second and refreshes after every clip, so a crash or a timeout never leaves you with nothing. Every style is always there, the JSON is always valid, and it exits clean. We threw dead URLs, odd inputs and malformed data at it and it kept working. On our own test set it scores about 0.93 on style, and a blind judge tells all four voices apart every time.

(~1,650 characters. Trim the last paragraph if a field counter complains.)

### Categories / Event Tracks
Track 2: Video Captioning Agent

### Technologies Used
Gemma 3, AMD Radeon PRO W7900, ROCm, vLLM, Fireworks AI, Docker, Whisper, Python

## Step 2: Cover image and presentation
- **Cover image:** `docs/cover.png` (make one from the demo identity, dark aurora + "One video. Four voices.")
- **Video presentation:** the demo-first cut, see storyboard in `docs/SUBMISSION.md`
- **Slide presentation:** built from `docs/SLIDES.md`

## Step 3: App hosting and code
- **Public GitHub repository:** https://github.com/NishchayMahor/stylereel  (flip to public at submission)
- **Docker image / application URL:** ghcr.io/nishchaymahor/stylereel:latest  (public, linux/amd64, via `scripts/release.sh`)
- **Demo URL:** the GitHub Pages page (re-enable at submission) or the claude.ai artifact link

## Requirements checklist (from the guide)
- [ ] Image publicly pullable, linux/amd64 manifest, <= 10 GB
- [ ] Public repo with a README that reproduces in one command
- [ ] No hardcoded or cached answers (verified: unseen-variant safe, chaos-tested)
- [ ] MIT licensed
- [ ] Gemma discoverable in title, short description, and tags
