# StyleReel, Slide Deck Brief (for Claude Design)

Everything you need to generate the presentation. Paste the PROMPT below into Claude Design,
then use the SLIDE CONTENT and DESIGN SYSTEM sections as the source material. Target: 10 slides,
16:9, exported as PDF for the lablab submission.

---

## PASTE-READY PROMPT

> Create a 10-slide, 16:9 pitch deck for a hackathon project called **StyleReel**, our entry
> for the AMD Developer Hackathon ACT II, Track 2 (video captioning). Audience is technical
> hackathon judges. Tone: confident, concrete, not salesy. No em-dashes anywhere in the copy.
>
> Visual identity: dark, premium, cinematic. Near-black background (#06060e). A cool accent
> palette of cyan (#46e0ff), violet (#8b6dff), and magenta (#ff5c9a), used sparingly as
> gradients and highlights. Bold white sans-serif headings (SF Pro Display / Inter style),
> clean sans body. Optional subtle aurora / flowing-gradient texture behind title and section
> slides. Glassmorphic cards for stats. Keep it uncluttered, lots of breathing room, one idea
> per slide. Use real numbers, not placeholders. Content for each slide is provided below.

---

## DESIGN SYSTEM
- Background: #06060e (near-black, cool). Section dividers can use a dark aurora gradient in cyan/violet/magenta.
- Accent gradient: linear, cyan (#46e0ff) to violet (#8b6dff) to magenta (#ff5c9a).
- Text: white (#ffffff) for headings, soft grey (#9ea4c9) for body.
- Per-voice colors (for the 4-styles visual): formal #57c8ff, sarcastic #c98bff, humorous_tech #4fe0c4, humorous_non_tech #ffb14e.
- Headings: bold sans (SF Pro Display / Inter / Geist). Body: same family, regular weight. Mono for code/labels.
- Cards: glassmorphic (subtle translucency, thin light border, soft shadow).
- No emoji as icons. Use simple line icons. No em-dashes in any text.

---

## SLIDE CONTENT

### Slide 1, Title
- Headline: **One video. Four voices.**
- Subhead: StyleReel, a containerized agent that captions any clip in four distinct registers.
- Small line: AMD Developer Hackathon ACT II, Track 2, Video Captioning
- Footer: [your name/team] · github.com/NishchayMahor/stylereel

### Slide 2, The task
- Title: Four voices, one clip, a 10 minute clock
- Body: Given short video clips, write a caption in four styles, formal, sarcastic, humorous_tech,
  humorous_non_tech. An LLM judge scores each on accuracy and tone.
- The catch (as 3 punchy points): miss a style and that clip scores zero; malformed JSON and the
  whole batch scores zero; run past 10 minutes and you score zero.
- Takeaway line: So the bar is not just good captions. It is good captions that never fail to ship.

### Slide 3, What it does (show, do not tell)
- Title: Same footage, four ways to say it
- Show one clip (use the kitten still from examples/example-01-kitten.mp4) with its four real captions:
  - Formal: "A small orange kitten with large dark eyes sits amid dense green undergrowth in a sunlit forest. The animal stays largely still while dappled light shifts across the ground."
  - Sarcastic: "A kitten sits motionless in the woods, doing almost nothing for twelve seconds while the sunlight shows more initiative than the animal. The uncanny valley has never looked so aggressively still."
  - Tech humor: "A kitten holds perfectly still in a sunlit forest, its oddly smooth fur making it look like a digital asset dropped onto a real background. Classic containerized microservice: deployed to prod, health check passes, nobody is sure the process is alive."
  - Everyday humor: "A tiny kitten sits in the woods, blinking like someone who joined a hiking group for the exercise and just realized they hate nature."
- Caption under all four: The facts stay true to the video. Only the tone changes.

### Slide 4, How it works
- Title: Understand first, then perform
- Pipeline (left to right, 4 stages, one line each):
  1. See and hear: smart frame selection + a local Whisper transcript (Fireworks dropped its speech API, so we run our own).
  2. Describe: a vision model writes a grounded description, then checks it against the frames a second time.
  3. Stylize x4: four writers, best of five drafts each, working only from the checked facts.
  4. Judge: a different model gates accuracy and keeps the strongest take.
- Key line: All four captions come from one verified fact sheet, so they never contradict the video or each other.

### Slide 5, Why the accuracy holds (the ablation)
- Title: The understanding stage is doing the work
- Show the ablation as a small bar chart or table (accuracy, 0 to 1):
  - Full pipeline: 0.86
  - No verification pass: 0.87 (within noise)
  - No best of five: 0.87 (within noise)
  - Blind, no describe stage: **0.72**
- Punchline: Removing the describe stage drops accuracy from 0.86 to 0.72. Understanding the video
  first is worth 0.15 of accuracy. That is where the score is won.

### Slide 6, Built to never score zero
- Title: Resilience is a feature, not an afterthought
- Four points (glass cards):
  - Skeleton first output: a valid results file exists from t=0 and refreshes after every clip.
  - Degraded ladder: full pipeline, then skip judge, then skip best of N, then single shot, then a grounded fallback. A style key is never missing.
  - Deadline watchdog + hard exit: results are written before any slow thread can push past the 10 minute kill.
  - Verified: 5 of 5 chaos drills (dead URLs, odd styles, malformed input) still write valid JSON and exit clean.
- Small stat: a 2 minute, 3 scene clip captions end to end in 51 seconds.

### Slide 7, Gemma on AMD (the partner-prize slide)
- Title: Gemma 3 watches the video, on AMD silicon
- Body: The describe stage, the model that actually watches each clip, runs **Gemma 3 12B multimodal,
  self-hosted on an AMD Radeon PRO W7900 via vLLM and ROCm**. Fireworks only serves Gemma as text,
  so running the vision Gemma ourselves is the differentiator, and it puts real AMD compute at the core.
- Supporting points: a Fireworks vision failover guarantees the 10 minute batch always finishes;
  every reported Gemma result is logged and reproducible; and per slide 5, this describe stage is
  worth 0.15 of accuracy.
- [If deployed] add a rocm-smi screenshot and the vLLM startup log here.

### Slide 8, Results
- Title: Measured, not claimed
- Big glass stat tiles:
  - Combined score (accuracy + style): **0.89**
  - Style match: **0.93**
  - Blind style ID (a judge tells all four voices apart): **1.00**
  - Chaos drills passed: **5 / 5**
- Small line: scored on a 15 clip test set by an independent LLM judge.

### Slide 9, Try it
- Title: One command
- Show the command in a terminal card:
  `docker run --rm -v $PWD/input:/input:ro -v $PWD/output:/output ghcr.io/nishchaymahor/stylereel:latest`
- Links: Repo: github.com/NishchayMahor/stylereel · Live demo: [your github.io or artifact URL]

### Slide 10, Close
- Big line: One video. Four voices. Grounded, resilient, and running on AMD.
- Recap chips: two stage understanding, Gemma on Radeon PRO W7900, never scores zero.
- Thanks + contact.

---

## RAW NUMBERS (source of truth, for accuracy)
- Latest 15-clip scores: combined 0.893, accuracy 0.859, style 0.928, blind_style_id 1.00.
- Ablation (8 clips, accuracy): full 0.862, no_verify 0.869, no_bestof 0.874, blind 0.716. Describe stage worth ~0.15.
- 2-minute 3-scene clip: 51s end to end.
- Chaos suite: 5/5 scenarios pass (dead URL, odd/empty/case-variant styles, wrapped input, malformed entry).
- Container smoke on 3 official clips: 28s, valid JSON.
- Prize context: Track 2 pays $2,500 / $1,500 / $1,000; the Gemma partner prize for Track 2 is $3,000.

## ASSETS AVAILABLE
- Clip stills and full captions: `examples/example-01-kitten.{mp4,txt}`, `example-02-speech-demo.*`, `example-03-2min-multiscene.*`
- Architecture diagram: `docs/architecture.md`
- Ablation detail: `docs/ABLATION.md`
- Gemma plan and deploy recipe: `docs/GEMMA.md`
- Live demo page source: `docs/index.html`
