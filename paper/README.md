# Paper (arXiv-ready LaTeX)

Self-contained source for the preprint. `PAPER.md` in the repo root is the
Markdown mirror of the same content.

## Files
- `main.tex` — the paper (single file, standard `article` class).
- `study_summary.png` — the results figure (copy of `results/study_summary.png`,
  regenerable via `make study N=100` + `python -m src.experiments.analyze`).

## Build

**Easiest — Overleaf:** create a new project, upload `main.tex` and
`study_summary.png`, compile.

**Locally (needs a TeX distribution, e.g. MacTeX/TeX Live):**
```bash
cd paper
pdflatex main.tex && pdflatex main.tex   # twice, to resolve refs
```

## Submitting to arXiv
Upload `main.tex` and `study_summary.png` together (arXiv compiles the source).
Primary category suggestion: `cs.AI` (cross-list `cs.LG` / `cs.RO` as fits).
Fill in any funding/acknowledgements before submission.
