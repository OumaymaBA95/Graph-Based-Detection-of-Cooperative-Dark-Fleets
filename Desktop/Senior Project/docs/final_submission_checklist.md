# Final submission checklist (`final_report.md`)

Use this after you **Knit** or **Pandoc** `docs/final_report.md` to Word/PDF.

## Before exporting

- [ ] Read the **blockquote** under §1 (“For readers and committees”)—it is the **elevator pitch** for distance rules and Table 1.
- [ ] Skim **§5.1 Threats to validity**—likely defense Q&A.
- [ ] Confirm **figure paths** resolve (`../artifacts/plots/...`) from your export working directory, or embed images in Word manually.

## In Word / PDF

- [ ] Apply **Heading 1 / 2 / 3** styles so the auto **TOC** (from YAML) looks right.
- [ ] **List of figures** (optional): Insert → Table of Figures, or add manually after TOC.
- [ ] **Figure font size**: zoom to 100% and check legibility on a projector-sized page.
- [ ] **Page breaks** before major sections or large figures if anything splits awkwardly.
- [ ] **Spell-check** names (committee, advisor).
- [ ] **PDF**: embed fonts if your program requires it.

## Optional

- [ ] Print **one** copy and mark figure/table placement.
- [ ] Save **final** PDF with a dated filename (e.g. `FinalReport_2026-02-19.pdf`).

---

*YAML in `final_report.md` sets `toc: true` for `word_document`, `html_document`, and `pdf_document`.*
