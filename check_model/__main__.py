"""Command line: python -m check_model <command> --help"""

from __future__ import annotations

import argparse
import copy
import json
import secrets
import sys
import time
from pathlib import Path

from .config import load_config
from .prepare import split_hashes


def _prepare(config: dict):
    from .prepare import SPLIT_KEYS, build, check_outputs, checked_catalog, summary, write

    try:
        check_outputs(config)  # before anything is read, and so before anything is written
    except ValueError as exc:
        sys.exit(f"refusing to prepare: {exc}")
    path = config["data"]["catalog"]
    try:
        catalog = checked_catalog(path)
    except (OSError, ValueError) as exc:
        sys.exit(f"data.catalog {path}: {exc}")
    rows, report = build(config, catalog)
    print(summary(report))
    hashes = write(rows, report, config["data"]["prepared_dir"], splits=not report.errors,
                   split_config={k: config["data"][k] for k in SPLIT_KEYS})
    if report.errors:
        print(f"\n{len(report.errors)} error(s): fix the source records above; nothing downstream will run; "
              "the split files from the last clean prepare are left as they were.",
              file=sys.stderr)
        sys.exit(1)
    # The hashes of the bytes just written from these rows -- not a later read of the files,
    # which another prepare could have replaced by then -- and the catalog the rows were
    # validated against, not a later read of data.catalog, which a sync could have replaced.
    return rows, report, hashes, catalog


def _require_training_stack() -> None:
    try:
        import peft, torch, transformers  # noqa: F401
    except ImportError as exc:
        sys.exit(f"{exc.name} is not installed: pip install -r requirements-train.txt")


def _run_dir(config: dict, prefix: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path(config["train"]["output_dir"]) / f"{prefix}-{stamp}-{secrets.token_hex(3)}"


def cmd_sync_catalog(args) -> None:
    from .catalog import CHECK_TURN_PATH, SKILL_BANK_PATH, sync_from_meridia, write_catalog
    from .prepare import overwritten
    from .prompt import system_prompt

    clashes = overwritten([Path(args.out)], [Path(args.meridia) / SKILL_BANK_PATH, Path(args.meridia) / CHECK_TURN_PATH])
    if clashes:
        sys.exit(f"refusing to sync the catalog: --out {args.out} is {clashes}, a file the catalog is read from")
    try:
        catalog = sync_from_meridia(args.meridia)
        system_prompt(catalog)  # a difficulty the prompt cannot describe is refused before writing
    except ValueError as exc:
        sys.exit(f"refusing to sync the catalog: {exc}")
    write_catalog(catalog, args.out)
    print(f"wrote {args.out}: {len(catalog.rollable_skills)} rollable skills, "
          f"{len(catalog.special_skills)} special skills, attributes {list(catalog.attributes)}, "
          f"SkillBank blob {catalog.skill_blob_sha}")


def cmd_prepare(args) -> None:
    _prepare(load_config(args.config))


def cmd_train(args) -> None:
    _require_training_stack()
    from .train import train

    config = load_config(args.config)
    rows, report, split_sha256, catalog = _prepare(config)
    train_rows = [r.to_json() for r in rows if r.split == "train"]
    val_rows = [r.to_json() for r in rows if r.split == "val"]
    if not train_rows:
        sys.exit("no training rows")
    run_dir = _run_dir(config, config["run_name"])
    manifest = train(config, train_rows, val_rows, run_dir=run_dir, source_files=report.sources,
                     split_sha256=split_sha256, catalog=catalog)
    print(json.dumps(manifest["metrics"], indent=2))
    print(f"saved {run_dir}")


def _manifest(run: str) -> dict:
    from .infer import read_manifest

    try:
        return read_manifest(run)
    except (OSError, ValueError) as exc:  # a missing, truncated or ambiguous manifest
        sys.exit(f"--run {run}: {exc}")


def _run_config(args, manifest: dict) -> dict:
    """The run's own recorded config, unless --config names another one."""
    if args.config is not None:
        return load_config(args.config)
    return manifest["config"]


def _read_provenance(prepared: Path) -> tuple[dict | None, str | None]:
    """(provenance, None), or (None, why it cannot be used). An interrupted prepare can leave the
    file truncated; that is a refusal like any other mismatch, not a traceback."""
    from . import strictjson
    from .prepare import PROVENANCE

    path = prepared / PROVENANCE
    if not path.exists():
        return None, f"{prepared} has no {PROVENANCE}; re-run prepare"
    try:
        data = strictjson.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return None, f"{path} is unreadable ({exc}); re-run prepare"
    sources = data.get("sources") if isinstance(data, dict) else None
    if not (isinstance(sources, list) and isinstance(data.get("split_config"), dict)
            and all(isinstance(s, dict) and isinstance(s.get("path"), str) and isinstance(s.get("sha256"), str)
                    for s in sources)):
        return None, f"{path} is not a provenance record (sources with path and sha256, split_config); re-run prepare"
    return data, None


def _data_mismatch(config: dict, manifest: dict, rows: list[dict] | None = None) -> str | None:
    """Why the prepared data is not the data this run was trained on, or None. Scoring a run on
    another experiment's prepared files would still print plausible metrics."""
    from .prepare import PROVENANCE, SPLIT_KEYS

    prepared = Path(config["data"]["prepared_dir"])
    provenance, problem = _read_provenance(prepared)
    if problem:
        return problem
    have = [s["path"] for s in provenance["sources"]]
    want = list(manifest["config"]["data"]["sources"])
    if have != want:
        return f"{prepared} was prepared from {have}, but the run was trained on {want}"
    trained_split = {k: manifest["config"]["data"].get(k) for k in SPLIT_KEYS}
    if provenance.get("split_config") != trained_split:
        return (f"{prepared} was split with {provenance.get('split_config')}, but the run was trained on a "
                f"split made with {trained_split}: the validation rows differ")
    lang = manifest["prompt"]["language"]
    other = sorted({r.get("lang") for r in rows or []} - {lang})
    if other:
        return f"{prepared} holds {other} rows, but the run's prompts are in {lang!r}"
    return None


def _changed_splits(manifest: dict, now: dict) -> list[dict]:
    """Prepared split files whose content (`now`, hashed from the files) differs from the ones the
    run was trained from. The same sources and settings can still give different rows after an
    adapter or splitter change, and a split file can be edited or replaced after `prepare`."""
    trained = manifest["examples"].get("split_sha256") or {}
    return [{"split": s, "trained_sha256": trained.get(s), "prepared_sha256": now.get(s)}
            for s in ("train", "val", "test") if trained.get(s) != now.get(s)]


def _short(sha256: str | None, missing: str = "no recorded hash") -> str:
    """A hash cut for a message; the fallback whole (cutting it too printed "no recorded ")."""
    return sha256[:12] if sha256 else missing


def _changed_sources(config: dict, manifest: dict) -> list[dict]:
    """Sources whose content differs from what the run was trained on (SHA-256, not path: a file
    edited in place keeps its path). Empty when the prepared data is the training revision."""
    provenance, problem = _read_provenance(Path(config["data"]["prepared_dir"]))
    if problem:  # _data_mismatch, run first, has already refused this
        raise ValueError(problem)
    trained = {s["path"]: s.get("sha256") for s in manifest["examples"].get("source_files", [])}
    return [{"path": s["path"], "trained_sha256": trained.get(s["path"]), "prepared_sha256": s["sha256"]}
            for s in provenance["sources"] if trained.get(s["path"]) != s["sha256"]]


def cmd_predict(args) -> None:
    from .infer import check_format

    # Read once and handed to the model: a second read could meet another run moved into place.
    # With --config the run is not needed until a query is known to be well formed.
    manifest = _manifest(args.run) if args.config is None else None
    config = _run_config(args, manifest)
    raw = Path(args.input).read_bytes() if args.input != "-" else sys.stdin.buffer.read()
    from . import strictjson

    try:  # UnicodeDecodeError is a ValueError: invalid UTF-8 is a bad input, not a traceback
        queries = strictjson.loads(raw.decode("utf-8"))  # a repeated key is ambiguous, not "last wins"
    except ValueError as e:
        sys.exit(f"--input {args.input}: not valid JSON ({e})")
    if not isinstance(queries, (dict, list)):
        # A string would otherwise be iterated as one query per character, and null or a number crash.
        sys.exit(f"--input {args.input}: expected one query object or a list of them, "
                 f"got {type(queries).__name__}")
    single = isinstance(queries, dict)
    if manifest is None:
        manifest = _manifest(args.run)
    check_format(manifest)
    # Built on the first runnable query: an empty list or a batch of bad queries loads nothing.
    model = _LazyModel(args.run, manifest=manifest, max_new_tokens=config["inference"]["max_new_tokens"])
    results = model.predict_many([queries] if single else queries, batch_size=config["inference"]["batch_size"])
    print(json.dumps(results[0] if single else results, ensure_ascii=False, indent=2))


class _LazyModel:
    """A CheckModel built on the first prediction with a runnable query. An evaluation that
    scores nothing, or a predict batch that is empty or all malformed, never loads a base model."""

    def __init__(self, run_dir: str, **kwargs):
        self._args, self._model = (run_dir, kwargs), None

    def predict_many(self, queries: list[dict], batch_size: int = 8) -> list[dict]:
        from . import prompt
        from .infer import bad_query_result

        if not any(not prompt.query_errors(q) for q in queries):
            # Nothing could reach generation: answer without a model. (An over-long query needs
            # the tokenizer to tell, so it does load the model.)
            return [bad_query_result(prompt.query_errors(q)) for q in queries]
        if self._model is None:
            _require_training_stack()  # only now: a batch or split with nothing to run needs none
            from .infer import CheckModel

            self._model = CheckModel(self._args[0], **self._args[1])
        return self._model.predict_many(queries, batch_size=batch_size)


def _pinned_inside(out: Path, run: Path, manifest: dict) -> Path | None:
    """The pinned directory `out` lies in, or None: the run's model/ (model_files) or, for a
    LoRA run, a local base directory (local_sha256). A file written into either changes its
    digest, and every later load refuses the run -- every run sharing that base, for a base."""
    from .infer import base_source

    pinned = [run / "model"]
    if manifest["adapter"] == "lora":
        base = Path(base_source(manifest["base_model"], run))
        if base.is_dir():
            pinned.append(base)
    resolved = out.resolve()
    return next((p for p in pinned if resolved.is_relative_to(p.resolve())), None)


def _eval_overwrites(out: Path, run: Path, config: dict) -> list[str]:
    """Inputs an evaluation writing into `out` would overwrite: the sources, the catalog, the
    prepared files, and the run's own files."""
    from .evaluate import EVAL_OUTPUT_FILES
    from .prepare import OUTPUT_FILES, overwritten

    data = config["data"]
    inputs = ([Path(s) for s in data["sources"]] + [Path(data["catalog"])]
              + [Path(data["prepared_dir"]) / name for name in OUTPUT_FILES]
              + [run / "run_manifest.json", run / "catalog.json", run / "train_log.jsonl"])
    return overwritten([out / name for name in EVAL_OUTPUT_FILES], inputs)


def cmd_evaluate(args) -> None:
    from .evaluate import EVAL_OUTPUT_FILES, evaluate_rows, training_relatives
    from .infer import check_format, load_run_catalog
    from .prepare import duplicate_ids, duplicate_inputs, read_split

    # Every refusal below needs only the manifest and file hashes: checking them before the model
    # loads spares a base-model download or a GPU allocation for a run that would be refused, and
    # checking them before any split is parsed turns a damaged split file into a refusal. The
    # manifest is read once: the config, the exclusions and the model all come from that read.
    manifest = _manifest(args.run)
    config = _run_config(args, manifest)
    out = Path(args.out) if args.out else Path(args.run) / "eval" / args.split
    pinned = _pinned_inside(out, Path(args.run), manifest)
    if pinned:
        sys.exit(f"refusing to evaluate: --out {args.out} is inside {pinned}, whose files the run is "
                 "checked against at every load; writing there would make the run refuse to load")
    clashes = _eval_overwrites(out, Path(args.run), config)
    if clashes:
        sys.exit(f"refusing to evaluate: writing {list(EVAL_OUTPUT_FILES)} into {out} would overwrite "
                 f"{clashes}; choose another --out")
    mismatch = _data_mismatch(config, manifest)
    if mismatch:
        sys.exit(f"refusing to evaluate: {mismatch}. Omit --config to use the run's own, or re-prepare with it.")
    changed = _changed_sources(config, manifest)
    split_now = split_hashes(config["data"]["prepared_dir"])
    changed_splits = _changed_splits(manifest, split_now)
    if (changed or changed_splits) and not args.allow_data_change:
        listed = "; ".join(
            [f"{c['path']}: trained on {_short(c['trained_sha256'])}, "
             f"prepared from {_short(c['prepared_sha256'])}" for c in changed]
            + [f"{c['split']} split: trained on {_short(c['trained_sha256'])}, "
               f"now {_short(c['prepared_sha256'], 'absent')}" for c in changed_splits])
        sys.exit(f"refusing to evaluate: the prepared data is not the revision the run was trained on ({listed}). "
                 "Pass --allow-data-change to score it anyway; trained rows stay excluded and the metrics "
                 "record the change.")
    parsed_hashes: dict = {}
    try:
        catalog = load_run_catalog(Path(args.run), manifest)  # the labels this run can answer with
        splits = {s: read_split(config["data"]["prepared_dir"], s, catalog, parsed_hashes)
                  for s in ("train", "val", "test")}
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(f"refusing to evaluate: {exc}")
    if parsed_hashes != split_now:
        # data_revision below describes the bytes hashed above; these are the bytes scored.
        sys.exit("refusing to evaluate: a split file changed while it was being read; run evaluate again")
    repeated = duplicate_ids(splits)
    if repeated:
        sys.exit(f"refusing to evaluate: ids in more than one split file: {repeated}; re-run prepare")
    same = duplicate_inputs(splits)
    if same:
        sys.exit(f"refusing to evaluate: rows with the same input under different ids: {same}; re-run prepare")
    rows = splits[args.split]
    mismatch = _data_mismatch(config, manifest, rows)  # now with the rows' language
    if mismatch:
        sys.exit(f"refusing to evaluate: {mismatch}. Omit --config to use the run's own, or re-prepare with it.")
    check_format(manifest)  # refused even when nothing turns out to need the model
    model = _LazyModel(args.run, manifest=manifest, max_new_tokens=config["inference"]["max_new_tokens"])
    train_ids = set(manifest["examples"]["train_ids"])
    fingerprints = frozenset(manifest["examples"].get("train_fingerprints", []))
    all_rows = [r for s in ("train", "val", "test") for r in splits[s]]
    related = training_relatives(all_rows, train_ids=train_ids, train_fingerprints=fingerprints,
                                 train_texts=manifest["examples"].get("train_texts", []),
                                 train_links=manifest["examples"].get("train_links", {}),
                                 train_scenario_ids=set(manifest["examples"].get("train_scenario_ids", [])),
                                 threshold=manifest["config"]["data"]["near_duplicate_threshold"])
    metrics = evaluate_rows(model, rows, split=args.split, train_ids=train_ids, out_dir=out,
                            train_fingerprints=fingerprints, related_to_training=frozenset(related),
                            include_training_rows=args.split == "train",
                            data_revision={"matches_training": not (changed or changed_splits),
                                           "changed_sources": changed, "changed_splits": changed_splits},
                            inference={**config["inference"],
                                       "trained_max_new_tokens": manifest["config"]["inference"]["max_new_tokens"],
                                       "matches_training": config["inference"]["max_new_tokens"]
                                       == manifest["config"]["inference"]["max_new_tokens"]},
                            batch_size=config["inference"]["batch_size"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"per-example predictions: {Path(out) / 'predictions.jsonl'}")


def cmd_smoke(args) -> None:
    _require_training_stack()
    from . import prompt
    from .evaluate import evaluate_rows
    from .infer import CheckModel
    from .train import train

    config = load_config(args.config)
    stages = []

    def done(stage: str, detail: str) -> None:
        stages.append(stage)
        print(f"[ok] {stage}: {detail}", flush=True)

    rows, report, split_sha256, catalog = _prepare(config)
    done("load + validate", f"{len(report.trainable)} trainable rows, {len(report.errors)} errors")

    smoke = config["smoke"]
    candidates = {r.id: r for r in rows if r.target is not None and r.scope == "general"}
    ids = smoke["example_ids"] or sorted(candidates)[: smoke["num_examples"]]
    missing = [i for i in ids if i not in candidates]
    if missing:
        sys.exit(f"smoke example_ids not trainable general rows: {missing}")
    picked = [candidates[i].to_json() for i in ids]
    if not picked:
        sys.exit("smoke: no trainable general rows to train on; check the sources and the prepare report")
    if not 5 <= len(picked) <= 10:
        print(f"note: smoke mode is meant for 5-10 examples, got {len(picked)}")
    done("convert", f"{len(picked)} rows: {', '.join(ids)}")

    run_dir = _run_dir(config, "smoke")
    if args.tiny:
        from .tiny import make_tiny_model

        # Inside the run: its manifest names this path, a shared path would be overwritten by the
        # next smoke run's fresh random weights, and a path beside the prepared data would go
        # when that regenerable directory is cleaned. The run directory is created here,
        # exclusively, so train() is told it already exists.
        run_dir.mkdir(parents=True, exist_ok=False)
        tiny_dir = run_dir / "tiny-random-model"
        system = prompt.system_prompt(catalog)
        make_tiny_model(tiny_dir, [r.to_json() for r in rows], system)
        config = copy.deepcopy(config)
        config["model"].update(base_model=str(tiny_dir), revision=None)
        done("tiny model", f"random-weight model at {tiny_dir} (plumbing only, not a real base model)")

    manifest = train(config, picked, [], run_dir=run_dir, source_files=report.sources,
                     max_steps=smoke["max_steps"], mode="smoke",
                     split_sha256=split_sha256, run_dir_created=args.tiny, catalog=catalog)
    losses = [e["loss"] for e in _train_log(run_dir) if "loss" in e]
    done("train", f"{manifest['global_steps']} steps, loss {losses[0] if losses else '?'} -> "
                  f"{losses[-1] if losses else '?'}")
    done("save", f"{run_dir}")

    model = CheckModel(run_dir, max_new_tokens=config["inference"]["max_new_tokens"])
    done("reload", f"base {manifest['base_model']['name_or_path']} + adapter from {run_dir / 'model'}")

    metrics = evaluate_rows(model, picked, split="train", train_ids={r["id"] for r in picked},
                            out_dir=run_dir / "eval" / "smoke", include_training_rows=True,
                            batch_size=config["inference"]["batch_size"])
    sample = json.loads((run_dir / "eval" / "smoke" / "predictions.jsonl").read_text().splitlines()[0])
    done("inference", f"{metrics['scored']} predictions; first raw output: {sample['raw_output']!r}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\nSmoke mode checks the pipeline runs end to end. Its scores are on the rows it trained on "
          "and say nothing about whether the model learned the check rules.")


def _train_log(run_dir: Path) -> list[dict]:
    path = run_dir / "train_log.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m check_model")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sync-catalog", help="regenerate the catalog snapshot from a MeridiaGame checkout")
    p.add_argument("--meridia", required=True, help="path to a MeridiaGame checkout")
    p.add_argument("--out", default="catalog/meridia_catalog.json")
    p.set_defaults(func=cmd_sync_catalog)

    for name, func, text in (("prepare", cmd_prepare, "validate the sources and write the split files"),
                             ("train", cmd_train, "prepare, then train a run")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--config", default="configs/sft_example.yaml")
        p.set_defaults(func=func)

    p = sub.add_parser("predict", help="run a trained model on query JSON (one object or a list)")
    p.add_argument("--run", required=True)
    p.add_argument("--input", required=True, help="a JSON file, or - for stdin")
    p.add_argument("--config", default=None, help="default: the config the run was trained with")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("evaluate", help="score a trained run on a prepared split")
    p.add_argument("--run", required=True)
    p.add_argument("--split", choices=("val", "test", "train"), default="val")
    p.add_argument("--config", default=None,
                   help="default: the config the run was trained with; another must prepare the same data")
    p.add_argument("--out", default=None)
    p.add_argument("--allow-data-change", action="store_true",
                   help="score prepared data whose sources changed since training (recorded in the metrics)")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("smoke", help="load -> convert -> a few steps -> save -> reload -> inference")
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument("--tiny", action="store_true",
                   help="use a random-weight local model instead of downloading the base model")
    p.set_defaults(func=cmd_smoke)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
