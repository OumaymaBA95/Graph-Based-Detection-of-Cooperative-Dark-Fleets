#!/usr/bin/env bash
# Build docs/presentation_script.pdf with Pandoc + XeLaTeX.
# Run from anywhere:  bash "/path/to/Senior Project/docs/build_presentation_pdf.sh"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f presentation_script.md ]]; then
  echo "ERROR: presentation_script.md not found in ${ROOT}" >&2
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

# Avoid R rmarkdown lua filters here: they assume Pandoc APIs that Homebrew pandoc does not provide.

# Older pandoc uses --highlight-style; 3.x+ prefers --syntax-highlighting (warns if you use the old name).
HIGHLIGHT_ARGS=()
if "$PANDOC_BIN" --help 2>&1 | grep -q '[= ]--syntax-highlighting'; then
  HIGHLIGHT_ARGS=(--syntax-highlighting tango)
else
  HIGHLIGHT_ARGS=(--highlight-style tango)
fi

"$PANDOC_BIN" presentation_script.md \
  --to pdf \
  --from markdown+autolink_bare_uris+tex_math_single_backslash \
  --output presentation_script.pdf \
  --standalone \
  "${HIGHLIGHT_ARGS[@]}" \
  --variable graphics \
  --variable 'geometry:margin=1in' \
  --pdf-engine=xelatex

echo "OK: ${ROOT}/presentation_script.pdf"
