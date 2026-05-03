#!/usr/bin/env bash
set -euo pipefail

HOST_NAME="${HOST_NAME:-127.0.0.1}"
PORT="${PORT:-8765}"
DATABASE="${DATABASE:-${TMPDIR:-/tmp}/aibrain-dev.sqlite3}"
ENV_FILE="${ENV_FILE:-.env}"
MODEL="${MODEL:-gpt-5-nano}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Run scripts/setup.sh after installing uv." >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  "$SCRIPT_DIR/setup.sh" --skip-tests
fi

if [[ ! -f "$ENV_FILE" && -f .env.example ]]; then
  cp .env.example "$ENV_FILE"
  echo "Created $ENV_FILE from .env.example. Add your OPENAI_API_KEY before making live OpenAI calls."
fi

ARGS=(
  run
  aibrain
  serve
  --host "$HOST_NAME"
  --port "$PORT"
  --database "$DATABASE"
  --model "$MODEL"
)

if [[ -f "$ENV_FILE" ]]; then
  ARGS+=(--env-file "$ENV_FILE")
fi

echo "Starting AI Brain Core at http://$HOST_NAME:$PORT"
uv "${ARGS[@]}"
