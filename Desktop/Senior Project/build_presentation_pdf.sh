#!/usr/bin/env bash
# Wrapper: build docs/presentation_script.pdf from any working directory.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/docs/build_presentation_pdf.sh"
