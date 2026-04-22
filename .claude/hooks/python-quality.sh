#!/usr/bin/env bash
# python-quality.sh — PostToolUse hook (matcher: Edit|Write).
# On .py edits: run ruff check --fix, ruff format, final ruff check, pyright.
# On violation: exit 2 with stderr so Claude sees the diagnostics and fixes.
#
# Ground truth lives in pyproject.toml ([tool.ruff], [tool.pyright]).
# Prefers `uv run` when ./uv.lock exists; falls back to direct invocation.

set -u
INPUT=$(cat)
FILE=$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // empty')

[[ -z "$FILE" || "$FILE" != *.py || ! -f "$FILE" ]] && exit 0

run() {
  if command -v uv &>/dev/null && [[ -f ./uv.lock ]]; then
    uv run --quiet "$@"
  else
    "$@"
  fi
}

ERR=$(mktemp)
PYOUT=$(mktemp)
trap 'rm -f "$ERR" "$PYOUT"' EXIT
FAIL=0

run ruff check --fix --quiet "$FILE" 2>>"$ERR" || true
run ruff format --quiet      "$FILE" 2>>"$ERR" || true
run ruff check --quiet       "$FILE" >>"$ERR" 2>&1 || FAIL=1

run pyright --outputjson "$FILE" >"$PYOUT" 2>/dev/null || true
python3 - "$PYOUT" >>"$ERR" <<'PY' || FAIL=1
import json, sys
try:
    with open(sys.argv[1]) as fh:
        data = json.load(fh)
except Exception:
    sys.exit(0)
errs = [d for d in data.get("generalDiagnostics", []) if d.get("severity") == "error"]
for d in errs:
    r = d["range"]["start"]
    print(f'{d["file"]}:{r["line"]+1}:{r["character"]+1}: {d.get("rule","")}: {d["message"]}')
sys.exit(1 if errs else 0)
PY

if (( FAIL )); then
  {
    echo "=== python-quality hook: fix these before continuing ==="
    cat "$ERR"
  } >&2
  exit 2
fi
exit 0
