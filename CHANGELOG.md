# Changelog

## 2026-10-06

### Fixed

- **Scoring commands that label their inputs.** The aggregator only split
  `--input path:label` when the path started with `/` or `.`. With the
  documented form, `results/x.json:direct`, it looked for a file literally
  named `results/x.json:direct`, so every documented scoring and union command
  failed with "file not found". Labels now parse for any path.
- **Missing ground truth is reported instead of failing silently.** The public
  BIRD-Interact release leaves `sol_sql` and `test_cases` empty.
  - The agentic runner used to score every submission as failed and report a
    reward near zero. It now stops at startup when these fields are missing.
  - The detection-only runner and the encoder replay now warn that the
    encoder prompt's reference-SQL section is empty.
  - `data/README.md` now explains how to request the ground truth from the
    BIRD team and merge it in.
- **The Mcs-10 (`se_ast`) sampling prompt now includes column meanings.** It
  used a minimal prompt with only the schema and knowledge base. It now uses
  the prompt of the paper's SE+AST runs: a PostgreSQL system prompt plus the
  agent's view of the database (schema, column meanings, and the knowledge base
  as JSON). Expect `se_ast` results to change.
- **Encoder failures no longer count as misses without warning.** When an
  encoder call failed, for example because the endpoint was unreachable, its
  question was dropped and the run finished normally with lower recall. Such
  samples are now marked as failed:
  - the run exits with status 1;
  - `--resume` retries them;
  - the aggregator warns about them.
- **`--resume` no longer mixes runs.** Resuming into an output file written
  with different settings (model, method, encoder, seed, or a method knob)
  used to reuse that file's results. It now stops with an error. The README
  quick-start example that reused `results/smoke.json` for a second model now
  writes to its own file.
