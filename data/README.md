# Data

This directory is the canonical mount point for the **BIRD-Interact lite-300**
dataset (and the **AMBROSIA** dataset for the cross-benchmark pipeline). No
files are bundled — fetch them from the upstream releases below.

## BIRD-Interact lite-300

Used by the agentic and detection-only pipelines. The lite-300 release ships
`bird_interact_data.jsonl` (300 question records with their ambiguity
annotations) plus per-database schemas, knowledge bases, and column-meaning
files.

Sources (pick one):

- **Hugging Face** — `https://huggingface.co/datasets/birdsql/bird-interact-lite`
- **GitHub** — `https://github.com/bird-bench/BIRD-Interact`

### Ground truth (reference SQL and test cases)

To prevent leakage through web crawling, the public release leaves the ground
truth empty: every record's `sol_sql`, `test_cases` and `external_knowledge`,
including those under `follow_up`. The pipelines depend on these fields as
follows:

- **Agentic pipeline:** it scores each submission against `sol_sql` and
  `test_cases`, so it cannot run without them. It stops at startup if they
  are missing.
- **Detection-only pipeline:** it runs without them, but the encoder prompt
  shows the reference SQL. Without it, the encoder's judgments differ from
  the paper's setup, and the runner prints a warning.

To get the ground truth:

1. Email `bird.bench25@gmail.com` with `[bird-interact-lite GT&Test Cases]` in
   the subject line. The ground-truth file is sent back automatically.
2. Merge it into the public data with upstream's `combine_public_with_gt.py`,
   from the root of `https://github.com/bird-bench/BIRD-Interact`:

   ```bash
   python combine_public_with_gt.py \
       data/bird-interact-lite/bird_interact_data.jsonl \
       /path/to/the/ground_truth_file.jsonl \
       data/bird-interact-lite/bird_interact_data_with_gt.jsonl
   mv data/bird-interact-lite/bird_interact_data_with_gt.jsonl \
      data/bird-interact-lite/bird_interact_data.jsonl
   ```

The script fills those fields by matching `instance_id`. We checked that the
merged lite-300 file matches the data used for the paper, record for record.

### Layout

Once extracted, the layout should be:

```
data/
└── bird-interact-lite/
    ├── bird_interact_data.jsonl
    ├── alien/
    │   ├── alien_schema.txt
    │   ├── alien_kb.jsonl
    │   └── alien_column_meaning_base.json
    ├── archeology/
    ├── credit/
    └── …  (20 databases total)
```

That is what every runner expects at `--data_path data/bird-interact-lite/bird_interact_data.jsonl`
and `--data_dir data/bird-interact-lite`. Pass different paths if you place
the files elsewhere.

## Postgres state for a-Interact

The agentic pipeline also needs a live Postgres instance loaded with each
database's data. Fetch the pre-built image from upstream:

```bash
./scripts/load_postgres.sh
```

The script pulls `docker.io/shawnxxh/bird-interact-postgresql:latest` and
starts it on `localhost:5432` with `POSTGRES_USER=root POSTGRES_PASSWORD=123123`.

## AMBROSIA

Used by the AMBROSIA cross-benchmark pipeline. Fetch the dataset from the
official release, then point `--ambrosia_dir` (or `AMBROSIA_DIR`) at the
directory containing `ambrosia.csv` plus the per-domain SQLite files
(`attachment/*`, `scope/*`, `vague/*`):

- Saparina & Lapata (NeurIPS 2024) — https://github.com/saparina/AMBROSIA
