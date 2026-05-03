#!/usr/bin/env bash
set -euo pipefail

CORE=0
NO_DEV=0
SKIP_TESTS=0
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --core)
      CORE=1
      shift
      ;;
    --no-dev)
      NO_DEV=1
      shift
      ;;
    --skip-tests)
      SKIP_TESTS=1
      shift
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

export UV_PYTHON="$PYTHON_VERSION"
SYNC_ARGS=(sync)
if [[ "$CORE" -eq 0 ]]; then
  SYNC_ARGS+=(--extra all)
fi
if [[ "$NO_DEV" -eq 0 ]]; then
  SYNC_ARGS+=(--extra dev)
fi

echo "Syncing environment with uv..."
uv "${SYNC_ARGS[@]}"

if [[ ! -f .env && -f .env.example ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Add your OPENAI_API_KEY before making live OpenAI calls."
fi

uv run python -c "import aibrain; print('aibrain import ok')"

if [[ "$SKIP_TESTS" -eq 0 && "$NO_DEV" -eq 0 ]]; then
  uv run pytest -q -p no:cacheprovider
fi

echo "Setup complete."
