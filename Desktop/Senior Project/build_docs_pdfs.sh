#!/usr/bin/env bash
# Build both presentation talk notes and thesis PDFs (Pandoc + XeLaTeX).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$HERE/docs/build_presentation_pdf.sh"
"$HERE/docs/build_final_report_pdf.sh"
echo "Done: presentation_script.pdf and final_report.pdf in docs/"
