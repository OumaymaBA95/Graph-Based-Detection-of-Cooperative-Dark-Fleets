#!/usr/bin/env bash
# Wrapper: build docs/final_report.pdf from any working directory.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/docs/build_final_report_pdf.sh"
