# autoresearch-gnn

This is an experiment to have the LLM do its own research on **Graph-WaveNet + METR-LA**.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main/master commit.
3. **Read the in-scope files** (full context first):
   - `README.md` — repository context.
   - `program.md` — this operating protocol.
   - `train.py` — training/eval entry and metric printing.
   - `model.py` — Graph-WaveNet architecture.
   - `engine.py` — trainer implementation.
   - `orchestrator.py` — automated trial runner and decision loop.
4. **Verify data exists**:
   - `data/METR-LA/{train,val,test}.npz`
   - `data/sensor_graph/adj_mx.pkl`
   If missing, run data prep first.
5. **Initialize logging**:
   - `experiments/results.tsv` (human-readable)
   - `experiments/ledger.jsonl` (machine-readable)
6. **Confirm and go**: setup should be reproducible before starting autonomous trials.

---

## Experimentation

Each trial must run under a **fixed wall-clock budget** so results are comparable.

### Budget rule (critical)

- Use `time_budget_sec` as the fixed trial budget (default: `1200` seconds).
- Training must stop near budget and still produce **valid validation/test metrics**.
- `eval_reserve_sec` is reserved for validation/testing tail work.

### Primary metric

- **Goal: minimize `Valid Loss (MAE)`**.
- Auxiliary metrics for reporting: `Test MAE`, `Test MAPE`, `Test RMSE`.

### What you CAN do

- In `hyper` mode: tune allowed training flags (nhid/dropout/lr/weight_decay/batch_size/graph flags).
- In `code` mode: mutate only allowlisted files, then commit and test.

### What you CANNOT do

- Do not modify dataset files or data paths.
- Do not modify files outside allowlist in autonomous code mode.
- Do not disable logging, rollback, or metric parsing.

### Allowlist (code mode)

- `train.py`
- `model.py`

(If you need broader edits, require explicit human approval.)

---

## Output format (from train logs)

Training logs should include these parseable lines:

- `Valid Loss: ...`
- `On average over 12 horizons, Test MAE: ..., Test MAPE: ..., Test RMSE: ...`

If these lines are missing, mark the trial as failed.

---

## Logging results

### 1) `experiments/results.tsv` (tab-separated)

Header:

```tsv
commit	valid_loss	test_mae	test_mape	test_rmse	duration_sec	status	decision	description
```

- `commit`: git short hash after keep/discard handling
- `status`: `ok` / `error` / `timeout`
- `decision`: `accept` / `reject`

### 2) `experiments/ledger.jsonl`

One JSON object per trial with full traceability, including:

- trial id / timestamp
- commit before/run/after
- config
- duration/status
- parsed metrics
- decision/reason
- current best snapshot

---

## Decision policy

- First successful trial => baseline (`accept`).
- Later trial is `accept` only if relative improvement over best is >= threshold (default `0.5%`).
- Else `reject`.
- In `code` mode:
  - `accept` => keep commit and advance branch.
  - `reject` => rollback with `git reset --hard <pre_commit>`.

---

## The experiment loop

For each trial:

1. Record `pre_commit`.
2. (Code mode only) mutate allowlisted files and commit.
3. Run one budgeted trial.
4. Parse metrics from log.
5. Decide `accept/reject` by threshold.
6. If code mode and reject: rollback to `pre_commit`.
7. Append both `results.tsv` and `ledger.jsonl`.
8. Move to next trial.

If a trial crashes/timeouts/has missing metrics, treat as failed (`reject`) and continue.

---

## Recommended commands

### Hyperparameter mode (safe default)

```bash
source .venv/bin/activate
python orchestrator.py --mode hyper --trials 10 --time_budget_sec 1200 --device cpu
```

### Code-evolution mode (Karpathy-style keep/discard)

```bash
source .venv/bin/activate
python orchestrator.py --mode code --trials 10 --time_budget_sec 1200 --device cpu --mutator_cmd "<your mutator command>"
```

---

## Stop conditions

- Stop immediately if human asks to stop.
- Stop if repository/data integrity is at risk.
- Otherwise continue autonomous loop for requested trial count.
