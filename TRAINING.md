# Check-selection SFT

A small supervised fine-tuning pipeline for Meridia's game-side check selection. Given a scene
and the player's action (or the event the game observed), the model decides **whether a check
is required, which skill or attribute, and the difficulty**. Dice and resolution stay in the
game (`DialogueCheck.gd`); the model only chooses the check.

**Scope of this version: single checks only** — one skill or one attribute, or no roll. Pair,
conditional, parameterized, optional-roll and special-skill labels are validated and
*reported*, never flattened into single checks (see [Extending](#extending)).

## Install

```bash
pip install -r requirements.txt         # validation, prepare, tests — no ML stack
pip install -r requirements-train.txt   # + torch, transformers, peft for train / predict / evaluate
```

Run every command from the repository root; config paths are relative to it.

## Layout

| Path | What |
| --- | --- |
| `training_files/dice_rolling_train.json` | The creator-labelled source. Read, hashed, never written. |
| `catalog/meridia_catalog.json` | Snapshot of MeridiaGame's `SkillBank.gd` and `server/check_turn.py` (`ATTRIBUTES`, `DIFFICULTIES`, `KINDS`), with git blob SHAs. |
| `configs/sft_example.yaml` | Editable training config (every key documented inline). |
| `configs/smoke.yaml` | The smoke-test config: 8 named examples, 10 steps. |
| `check_model/adapter.py` | Source records → rows; validation and the per-record report. |
| `check_model/splits.py` | Near-duplicate grouping and the train / val / test split. |
| `check_model/prompt.py` | System prompt, input and output format, strict output parser, game-request mapping. |
| `check_model/train.py` · `infer.py` · `evaluate.py` | LoRA SFT, loading a run, scoring. |

## 1. Validate and prepare the data

```bash
python -m check_model prepare --config configs/sft_example.yaml
```

This writes `build/data/{train,val,test}.jsonl`, `build/data/splits_provenance.json` (the sources
those split files were built from, which `evaluate` checks) and `build/data/report.json`, and prints a
summary. Every source record lands in exactly one bucket:

| Bucket | Meaning | 2026-09-23, dataset at db81744 (50 records; rerun `prepare` for current figures) |
| --- | --- | --- |
| trainable | a creator-labelled single check or confirmed no-roll | 40 |
| eval_only | several accepted answers, none selected (`alternative_examples`) — scored against all of them, never trained on | 1 (000038) |
| unsupported | labelled, in a form this version does not handle; the reason is listed | 6 |
| pending / skipped | no usable label | 1 / 2 |
| **errors** | a missing required field, a label outside the catalog, a duplicate id or input | 0 |

**Errors stop the pipeline**, leave the last clean split files untouched, and name the record (`ERROR dice_train_000001 (…:examples[0]): unknown
skill 'maintenance'`). A label is valid when the game would accept it: a skill must be a
`SkillBank` row with a non-blank `initial` (the rule of `DialogueCheck.is_rollable_skill`), an
attribute one of `check_turn.ATTRIBUTES`, a difficulty one of `success | hard | extreme`, and
the roll system `unidirectional`.

**Splits.** General scenes → train / validation. `game_specific` scenes → the independent test
set. Variations of one scenario share a group and never straddle splits. A group is formed by the
dataset's `related_example_id` links, by `data.extra_groups`, and by TF-IDF similarity (over every
language the record carries) ≥
`near_duplicate_threshold`. The summary prints each similarity pair as `GROUPED a + b`: check
them. A group's split comes from hashing its key (its smallest id), so adding unrelated examples
does not move old ones. It is not frozen: a new row that joins a group and sorts first, or a
similarity merge, can move the group, and `prepare` prints a WARNING for every row whose split
changed since the last run. Scores stay sound either way, since `evaluate` excludes every row a
run trained on.

### When new examples are added

1. Label them in `training_files/dice_rolling_train.json` following its `annotation_policy`.
2. `python -m check_model prepare` → fix any ERROR, review the GROUPED pairs, and check that
   the unsupported / pending lists are what you expect.
3. If the dataset was labelled against a newer `SkillBank` (a catalog WARNING), refresh the
   snapshot and commit it:
   `python -m check_model sync-catalog --meridia ../MeridiaGame`
4. Train a new run. Runs already trained keep the prompt and catalog they were trained with;
   evaluating one of them on the changed data needs `--allow-data-change` (see *Evaluate*).

Meridia-specific test scenes go in a **separate source file** with
`"scenario_scope": "game_specific"` (same schema), listed under `data.sources`. Its rows
become the test split automatically.

## 2. Smoke test (5–10 examples)

```bash
python -m check_model smoke --config configs/smoke.yaml          # downloads Qwen2.5-0.5B-Instruct
python -m check_model smoke --config configs/smoke.yaml --tiny   # offline: a random-weight local model
```

Stages: load + validate → convert → a few training steps → save → reload → inference → score.
It prints `[ok]` per stage. **Smoke mode verifies the pipeline, not the model**: its scores are
on the rows it just trained on and are labelled `held_out: false`. `--tiny` builds a 2-layer
random model with a tokenizer trained on this dataset, so the pipeline can run with no
download. Its answers are noise.

## 3. Train

```bash
python -m check_model train --config configs/sft_example.yaml
```

Each run gets its own directory `runs/<run_name>-<timestamp>-<random>/`, never shared, containing:

- `model/` — the LoRA adapter, tokenizer and chat template
- `catalog.json` — the label catalog the run was trained on
- `train_log.jsonl` — the Trainer's log history (loss, learning rate, final `eval_loss` on validation)
- `run_manifest.json` — the base model (name, pinned revision, resolved hub commit), the full config, the system prompt and its hash, the source files' SHA-256, every train and val example id, token-length stats, library versions and the repo commit

How the training is set up:

- **Chat template:** the base model's own `apply_chat_template`. The rendered prompt must be a
  token prefix of the full conversation, otherwise the run stops, since the loss mask
  depends on it.
- **Loss:** computed on the answer only. Prompt tokens are labelled `-100`; the answer and the
  template's end-of-turn token are trained.
- **Length:** nothing is truncated. A row longer than `max_seq_length` stops the run and names
  the row.

### Suggested starting configuration

| Setting | Value | Why |
| --- | --- | --- |
| base model | `Qwen/Qwen2.5-1.5B-Instruct` | Instruct model with a chat template, good at both English and Chinese (the records are bilingual; the creator labels in Chinese). Small enough for LoRA on one ~16 GB GPU. Move to `Qwen2.5-7B-Instruct` if validation errors look like capacity rather than data. Check the model card's licence before shipping. |
| LoRA | r 16, alpha 32, dropout 0.05, all linear layers | A common starting point. The adapter is small, the base weights stay untouched, and switching base models is a config change. |
| learning rate | 2e-4, cosine, 10% warmup | The usual LoRA range (1e-4 – 3e-4). |
| epochs × batch | 5 × (4 × 2 accumulation) | ~100 examples leave ~80 for training: about 10 steps per epoch, about 50 in total. Small data needs several passes. If `eval_loss` rises while training loss falls, use fewer epochs. |
| max_seq_length | 1024 | System prompt (the 68-skill list) + one scene + a ~30-token answer. Raise it if a run refuses a row; never truncate. |
| language | `en` | Catalog names and the game's request contract are English. `zh` is one config switch away and is worth comparing once validation has enough rows. |
| decoding | greedy, 64 new tokens | The answer is a ~30-token JSON object. |

These values are a starting point. They have not been run against a real base model yet (see
[Verification status](#verification-status)).

## 4. Load a run and predict

```python
from check_model.infer import CheckModel
model = CheckModel("runs/check-sft-20260923-120000")
model.predict({"scene": "…", "player_action": "…"})
```
```bash
python -m check_model predict --run runs/<run> --input query.json   # one object or a list
```

**Input.** These are the fields the creator was shown when labelling
(`task_scope.presented_input_fields`):

```json
{"scene": "…", "player_action": "…"}
{"scene": "…", "observed_event": "…", "runtime_state": {"in_combat": false}}
```

The query needs `scene`, exactly one of `player_action` / `observed_event`, and optionally a
`runtime_state` object. Any other field is rejected.

**Output.** The model replies with one JSON object:

```json
{"roll_required": true, "checks": [{"kind": "skill", "name": "Climbing", "difficulty": "extreme"}]}
```

`predict` wraps that reply for the backend:

```json
{"valid": true, "decision": {…}, "game_request": {"skill": "Climbing", "difficulty": "extreme"},
 "note": null, "errors": [], "raw_output": "…"}
```

- The model's reply is parsed strictly: only that one JSON object is accepted, and only with
  labels from the catalog the run was trained on.
- `game_request` is in the single-check shape that `DialogueCheck.from_request()` accepts. It is
  `null` when no roll is needed.
- A single **attribute** check comes back as a decision with a `note` and no `game_request`,
  because the game has no single-attribute request form yet (see [Open questions](#open-questions)).

## 5. Evaluate

```bash
python -m check_model evaluate --run runs/<run> --split val    # or test / train
```

`evaluate` and `predict` use the config the run was trained with. Pass `--config` to override it;
`evaluate` then refuses prepared data whose sources or prompt language differ from the run's,
since scoring another experiment's data would still print plausible metrics. It also compares each
source's SHA-256 with the one recorded at training: data edited since then is refused unless you
pass `--allow-data-change`, and `metrics.json` records the revision either way (`data_revision`).
Rows the run trained on stay excluded in both cases.

Results are written to `runs/<run>/eval/<split>/`:

- `metrics.json` — one entry per field
- `predictions.jsonl` — per example: its id, the input, the accepted answers, the raw output, the parsed decision, the format errors and the per-field result

What each field measures:

- **`format_valid`** — the reply parsed as a decision with catalog labels.
- **`roll_required`** — whether a check is needed. Scored on every row.
- **`check`** (skill or attribute) and **`difficulty`** — scored only where the reference
  requires a roll. A match against any accepted alternative counts as right. `difficulty` is
  compared with the chosen option's difficulty, so it is measured separately from the skill.
- **`exact`** — the whole decision matches an accepted answer.

An unparseable reply counts as wrong on every field it applies to.

Rows the run trained on are excluded from val and test metrics (their ids are listed), so
re-preparing the data after training cannot leak them in. `held_out` says that no scored row
was trained on. Only the `test` split — Meridia-specific scenes — is marked
`independent_test`; validation is held out but drawn from the same general scenes as training.

## Extending

Each form below is detected and reported as `unsupported` today:

- **Pair checks** (000044, 000029). Accept `check_mode: pair` in `adapter._single_checks` and
  raise `prompt.MAX_CHECKS` to 2. Map to the game's pair request `{"checks": [e1, e2]}` in
  `to_game_request`, and score the pair in `evaluate.score`. The output format already carries
  `checks` as a list.
- **Conditional labels** (000019, 000020, 000030). Once the state that selects a branch is part
  of the input, expand each branch whose `when` matches into a row. The policy forbids
  exporting before that state is known.
- **Alternatives with a selection.** `selected_alternative` has no defined format yet, so a set
  value is reported rather than guessed.
- **Optional rolls** (000040), **special-skill checks**, **other roll systems** — each needs a
  target format decided first.

## Verification status

Run on 2026-09-23 in a cloud container: CPU only, no GPU, `huggingface.co` blocked. Libraries: torch 2.14, transformers 5.17, peft 0.21.

| Step | Status |
| --- | --- |
| Catalog snapshot vs the dataset's pinned blob SHAs | ✅ identical (`53680ef…`, `4db9cc4…`) |
| Validation and prepare on the real dataset | ✅ 0 errors, every record accounted for |
| Unit tests (`pytest`) | ✅ passing; the key checks were confirmed to fail with their logic removed |
| Smoke, `--tiny`: train → save → reload → inference → score | ✅ ran end to end on CPU |
| Loss masking and generation stopping | ✅ the tiny model memorises 4 targets exactly through the pipeline |
| `train` / `evaluate` / `predict` commands | ✅ ran with the tiny model |
| **The configured base models (Qwen2.5-0.5B / 1.5B-Instruct), their chat templates, GPU / bf16 training** | ❌ **not run**: the model download is blocked here |
| **Whether the model learns the check rules** | ❌ not measurable yet: 40 trainable rows, no test set |

## Open questions

- **Single attribute checks have no game request form.** `from_request()` rolls a single
  `{skill, difficulty}` as a skill only (`is_rollable_skill`), so STR in 000021 / 000045 cannot
  be sent as a single check. Pairs do accept attributes (`validate_entry`).
- **No independent test set.** Every scene is `general`; the only `game_specific` label is
  000030's deferred branch.
- **No confirmed no-roll example.** The only `roll_required: false` label (000040) is an optional
  roll, and 000015 was skipped as borderline. The model never sees a "no check" answer, and
  `roll_required` can only be scored on positive rows.
- **`roll_system_source: contextual_default`** (000020, 000021, 000045 — the attribute checks)
  is not defined in `annotation_policy`, whose default covers "one skill and a difficulty". The
  rows are used because the recorded `roll_system` is `unidirectional`.
- **Auto-grouped near-duplicates** (000005 + 000042, 000010 + 000043, 000032 + 000046) should be
  confirmed as variations of one scenario, or kept apart by raising the threshold.
- **Runtime inputs are not captured.** `task_scope.runtime_evaluation_ready` is false: the
  records carry no NPC, dialogue history or player `check_rows`. The input format above is the
  annotation's own, and no game code sends it yet.
