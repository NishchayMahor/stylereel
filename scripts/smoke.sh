#!/usr/bin/env bash
# End-to-end smoke test: build linux/amd64 image, run against the 3 official
# example clips, assert valid complete output within the time budget.
# Requires FIREWORKS_API_KEY in env (or a baked key in the image).
set -euo pipefail
cd "$(dirname "$0")/.."

IMG="${IMG:-stylereel:smoke}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/input" "$WORK/output"
cat > "$WORK/input/tasks.json" <<'EOF'
[
  {"task_id": "v1",
   "video_url": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4",
   "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]},
  {"task_id": "v2",
   "video_url": "https://storage.googleapis.com/amd-hackathon-clips/13825391-uhd_3840_2160_30fps.mp4",
   "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]},
  {"task_id": "v3",
   "video_url": "https://storage.googleapis.com/amd-hackathon-clips/3044693-uhd_3840_2160_24fps.mp4",
   "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]}
]
EOF

echo "== building $IMG (linux/amd64) =="
docker buildx build --platform linux/amd64 -t "$IMG" --load .

echo "== running =="
START=$(date +%s)
docker run --rm --platform linux/amd64 \
  -v "$WORK/input:/input:ro" -v "$WORK/output:/output" \
  ${FIREWORKS_API_KEY:+-e FIREWORKS_API_KEY} \
  ${GEMMA_ENDPOINT:+-e GEMMA_ENDPOINT} \
  "$IMG"
ELAPSED=$(( $(date +%s) - START ))

echo "== validating (took ${ELAPSED}s) =="
python3 - "$WORK/output/results.json" "$ELAPSED" <<'EOF'
import json, sys
data = json.load(open(sys.argv[1]))
elapsed = int(sys.argv[2])
styles = {"formal", "sarcastic", "humorous_tech", "humorous_non_tech"}
assert {d["task_id"] for d in data} == {"v1", "v2", "v3"}, "missing task ids"
for d in data:
    assert set(d["captions"]) == styles, f"missing styles for {d['task_id']}"
    for s, c in d["captions"].items():
        assert isinstance(c, str) and len(c.strip()) > 20, f"suspicious caption {d['task_id']}/{s}"
assert elapsed < 600, f"took {elapsed}s > 10 min"
print(f"SMOKE PASS in {elapsed}s")
for d in data:
    print(f"\n=== {d['task_id']} ===")
    for s, c in d["captions"].items():
        print(f"[{s}] {c}")
EOF
