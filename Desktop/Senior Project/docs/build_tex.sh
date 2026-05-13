#!/usr/bin/env bash
# Regenerate final_report.tex and presentation_script.tex (Pandoc → LaTeX only).
# Run: ./build_tex.sh  (from docs/) or bash "/path/to/Senior Project/docs/build_tex.sh"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PANDOC_BIN="${PANDOC:-}"
if [[ -z "$PANDOC_BIN" ]]; then
  if command -v pandoc >/dev/null 2>&1; then
    PANDOC_BIN="$(command -v pandoc)"
  elif [[ -x /opt/homebrew/bin/pandoc ]]; then
    PANDOC_BIN=/opt/homebrew/bin/pandoc
  else
    echo "ERROR: pandoc not found" >&2
    exit 1
  fi
fi

HIGHLIGHT_ARGS=()
if "$PANDOC_BIN" --help 2>&1 | grep -q '[= ]--syntax-highlighting'; then
  HIGHLIGHT_ARGS=(--syntax-highlighting tango)
else
  HIGHLIGHT_ARGS=(--highlight-style tango)
fi

FROM=markdown+autolink_bare_uris+tex_math_single_backslash
COMMON=(--standalone "${HIGHLIGHT_ARGS[@]}" --variable graphics --variable 'geometry:margin=1in')

"$PANDOC_BIN" final_report.md --to latex --from "$FROM" --output final_report.tex "${COMMON[@]}"
echo "OK: ${ROOT}/final_report.tex"

"$PANDOC_BIN" presentation_script.md --to latex --from "$FROM" --output presentation_script.tex "${COMMON[@]}"
echo "OK: ${ROOT}/presentation_script.tex"
