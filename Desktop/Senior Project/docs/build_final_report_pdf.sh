#!/usr/bin/env bash
# Build docs/final_report.pdf from final_report.md (Pandoc + XeLaTeX).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f final_report.md ]]; then
  echo "ERROR: final_report.md not found in ${ROOT}" >&2
  exit 1
fi

PANDOC_BIN="${PANDOC:-}"
if [[ -z "$PANDOC_BIN" ]]; then
  if command -v pandoc >/dev/null 2>&1; then
    PANDOC_BIN="$(command -v pandoc)"
  elif [[ -x /opt/homebrew/bin/pandoc ]]; then
    PANDOC_BIN=/opt/homebrew/bin/pandoc
  else
    echo "ERROR: pandoc not found. Install it or set PANDOC=/path/to/pandoc" >&2
    exit 1
  fi
fi

HIGHLIGHT_ARGS=()
if "$PANDOC_BIN" --help 2>&1 | grep -q '[= ]--syntax-highlighting'; then
  HIGHLIGHT_ARGS=(--syntax-highlighting tango)
else
  HIGHLIGHT_ARGS=(--highlight-style tango)
fi

"$PANDOC_BIN" final_report.md \
  --to pdf \
  --from markdown+autolink_bare_uris+tex_math_single_backslash \
  --output final_report.pdf \
  --standalone \
  "${HIGHLIGHT_ARGS[@]}" \
  --variable graphics \
  --variable 'geometry:margin=1in' \
  --pdf-engine=xelatex

echo "OK: ${ROOT}/final_report.pdf"
