# Prompt log 012 — Phase 12: Cover PDF + tag + submission

**Phase:** 12 — cover PDF (exercise 5), tag v1.00, prep submission
**Authors:** Abdallah Khaldi, Tasneem Natour
**Date:** 2026-06-20

## What was done

- **Cover sheet** — `scripts/generate_cover_pdf.py` (ported in Phase 0, exercise 5 /
  ex05 URL / both student IDs / self-score 85 / late = no) + a unit test
  (`tests/unit/test_scripts/test_cover.py`) asserting exercise = "5", the ex05 repo URL,
  group COSMOS77, score 85, and both student IDs. The script filled the template
  correctly (verified: `COSMOS77-ex05.filled.docx` contains 212389712, 323118794,
  COSMOS77, the ex05 URL, both names, 85). The Word→PDF conversion (`docx2pdf`) needs an
  interactive macOS GUI session (it returned `AppleEvent timed out` in the headless
  shell), so the final PDF render is the student's one manual step (command below).
- **Release** — tag `v1.00` + GitHub release from `CHANGELOG.md`. `*.pdf` is gitignored
  (the cover lives at `~/COSMOS77/HW5/`, outside the repo); only the script + test are
  committed.

## The two manual web steps (Hard Stop 2)

1. **Generate + check the PDF** (Word must be allowed to automate):
   `uvx --with python-docx --with docx2pdf python scripts/generate_cover_pdf.py
   --template ~/COSMOS77/HW5/cover_template/uoh-rl07-ex01.docx
   --output ~/COSMOS77/HW5/COSMOS77-ex05.pdf --self-score 85 --exercise-number 5`
   (or open `~/COSMOS77/HW5/COSMOS77-ex05.filled.docx` in Word → Save As PDF). Confirm
   exercise = 5, the ex05 URL, layout untouched.
2. **Visibility / Moodle** — the repo is public (done); optionally add
   `rmisegal@gmail.com` as a collaborator. Both students upload
   `~/COSMOS77/HW5/COSMOS77-ex05.pdf` to Moodle separately (per-student timer).
