.PHONY: test chaos build smoke harness harness-quick baseline lint

test:
	uv run pytest -q

# Fault-injection: every scenario must yield complete valid JSON and exit 0.
chaos:
	uv run python scripts/chaos.py

build:
	docker buildx build --platform linux/amd64 -t stylereel:local --load .

smoke:
	IMG=stylereel:local bash scripts/smoke.sh

# Full dev-set run + LLM-judge scoring (needs FIREWORKS_API_KEY / .fireworks_key)
harness:
	uv run python harness/run.py --local-clips

harness-quick:
	uv run python harness/run.py --local-clips --n 4

baseline:
	uv run python harness/run.py --local-clips --baseline
