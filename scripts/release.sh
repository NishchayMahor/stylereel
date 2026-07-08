#!/usr/bin/env bash
# Final-freeze release: bake the (rotated) Fireworks key, build linux/amd64, push public
# to GHCR, then pull-test from a clean state. Run at submission time.
#
# Usage:  FIREWORKS_API_KEY=fw_ROTATED_KEY bash scripts/release.sh
# Prereqs: gh token has write:packages  ( gh auth refresh -h github.com -s write:packages )
set -euo pipefail
cd "$(dirname "$0")/.."

: "${FIREWORKS_API_KEY:?Set FIREWORKS_API_KEY to the rotated key before releasing}"
OWNER="nishchaymahor"
IMAGE="ghcr.io/${OWNER}/stylereel:latest"

echo "== 1. login to GHCR =="
gh auth token | docker login ghcr.io -u "$OWNER" --password-stdin

echo "== 2. bake obfuscated key into keybox.py (local only, reverted after) =="
python scripts/obfuscate_key.py "$FIREWORKS_API_KEY"
trap 'git checkout -- src/stylereel/keybox.py' EXIT   # never leave the key on disk

echo "== 3. build + push linux/amd64 =="
docker buildx build --platform linux/amd64 -t "$IMAGE" --push .

if [ "${PUBLISH:-0}" = "1" ]; then
  echo "== 4. make the GHCR package PUBLIC (PUBLISH=1) =="
  gh api -X PATCH "user/packages/container/stylereel/visibility" -f visibility=public 2>/dev/null \
    || echo "  (set visibility to Public in the GHCR web UI if this failed)"
else
  echo "== 4. leaving package PRIVATE (set PUBLISH=1 at final submission to make it public) =="
fi

echo "== 5. verify: pull from a clean state and smoke =="
[ "${PUBLISH:-0}" = "1" ] && docker logout ghcr.io   # public: prove no-auth pull works
docker rmi "$IMAGE" 2>/dev/null || true
docker pull "$IMAGE"
docker manifest inspect "$IMAGE" | grep -q '"architecture": "amd64"' && echo "  amd64 manifest OK"
IMG="$IMAGE" bash scripts/smoke.sh

echo "== DONE. Submitted image: $IMAGE =="
echo "Remember: rotate the Fireworks key again AFTER the hackathon (this key is now in a public image)."
