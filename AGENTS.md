# Repository workflow

Use the installed `rex-goal-copilot` skill when it is available.

- At session start, read `cc_mnt/INDEX.md`, then `CONTINUITY_LEDGER.md` and
  `CODEBASE_INDEX.md` inside that directory before broad exploration or edits.
- If a fresh clone lacks continuity files, create the three files from the current
  repository state. Keep `cc_mnt/` and `ARCHIVE/` ignored by Git.
- Update the ledger and codebase index when project state changes; update the
  continuity index after changes in `cc_mnt/`.
- Use `cc_mnt/` for continuity only. Use `ARCHIVE/` for recycled original files,
  never for code that is imported or used by the pipeline.
- Keep raw data untouched. Never commit datasets, outputs, credentials, or notebook
  outputs containing competition data. Keep durable documentation in `README.md`.
- Keep preprocessing inside each fold's training pipeline to avoid validation leakage.
- Run `python -m ruff check .`, `python -m ruff format --check .`, and
  `python -m pytest` after code changes. Use `python train.py --smoke-test` for
  an end-to-end check without competition data.
- Prefer `python3` when available; on Windows a nonfunctional Store alias may
  require `python` or the virtual environment interpreter.
- Do not commit, push, or replace repository history unless requested.
