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
    from .prepare import SPLIT_KEYS, build, summary, write

    rows, report = build(config)
    print(summary(report))
    write(rows, report, config["data"]["prepared_dir"], splits=not report.errors,
          split_config={k: config["data"][k] for k in SPLIT_KEYS})
    if report.errors:
        print(f"\n{len(report.errors)} error(s): fix the source records above; nothing downstream will run; "
              "the split files from the last clean prepare are left as they were.",
              file=sys.stderr)
        sys.exit(1)
    return rows, report


def _require_training_stack() -> None:
    try:
        import peft, torch, transformers  # noqa: F401
    except ImportError as exc:
        sys.exit(f"{exc.name} is not installed: pip install -r requirements-train.txt")


def _run_dir(config: dict, prefix: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path(config["train"]["output_dir"]) / f"{prefix}-{stamp}-{secrets.token_hex(3)}"


def cmd_sync_catalog(args) -> None:
    from .catalog import sync_from_meridia, write_catalog

    catalog = sync_from_meridia(args.meridia)
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
    rows, report = _prepare(config)
    train_rows = [r.to_json() for r in rows if r.split == "train"]
    val_rows = [r.to_json() for r in rows if r.split == "val"]
    if not train_rows:
        sys.exit("no training rows")
    run_dir = _run_dir(config, config["run_name"])
    manifest = train(config, train_rows, val_rows, run_dir=run_dir, source_files=report.sources,
                     split_sha256=split_hashes(config["data"]["prepared_dir"]))
    print(json.dumps(manifest["metrics"], indent=2))
    print(f"saved {run_dir}")


def _run_config(args) -> dict:
    """The run's own recorded config, unless --config names another one."""
    if args.config is not None:
        return load_config(args.config)
    manifest = json.loads((Path(args.run) / "run_manifest.json").read_text(encoding="utf-8"))
    return manifest["config"]


def _data_mismatch(config: dict, manifest: dict, rows: list[dict]) -> str | None:
    """Why the prepared data is not the data this run was trained on, or None. Scoring a run on
    another experiment's prepared files would still print plausible metrics."""
    from .prepare import PROVENANCE, SPLIT_KEYS

    prepared = Path(config["data"]["prepared_dir"])
    if not (prepared / PROVENANCE).exists():
        return f"{prepared} has no {PROVENANCE}; re-run prepare"
    provenance = json.loads((prepared / PROVENANCE).read_text(encoding="utf-8"))
    have = [s["path"] for s in provenance["sources"]]
    want = list(manifest["config"]["data"]["sources"])
    if have != want:
        return f"{prepared} was prepared from {have}, but the run was trained on {want}"
    trained_split = {k: manifest["config"]["data"].get(k) for k in SPLIT_KEYS}
    if provenance.get("split_config") != trained_split:
        return (f"{prepared} was split with {provenance.get('split_config')}, but the run was trained on a "
                f"split made with {trained_split}: the validation rows differ")
    lang = manifest["prompt"]["language"]
    other = sorted({r["lang"] for r in rows} - {lang})
    if other:
        return f"{prepared} holds {other} rows, but the run's prompts are in {lang!r}"
    return None


def _changed_splits(config: dict, manifest: dict) -> list[dict]:
    """Prepared split files whose content differs from the ones the run was trained from. The
    same sources and settings can still give different rows after an adapter or splitter change,
    and a split file can be edited or replaced after `prepare`; so the files are hashed now."""
    trained = manifest["examples"].get("split_sha256") or {}
    now = split_hashes(config["data"]["prepared_dir"])
    return [{"split": s, "trained_sha256": trained.get(s), "prepared_sha256": now.get(s)}
            for s in ("train", "val", "test") if trained.get(s) != now.get(s)]


def _short(sha256: str | None, missing: str = "no recorded hash") -> str:
    """A hash cut for a message; the fallback whole (cutting it too printed "no recorded ")."""
    return sha256[:12] if sha256 else missing


def _changed_sources(config: dict, manifest: dict) -> list[dict]:
    """Sources whose content differs from what the run was trained on (SHA-256, not path: a file
    edited in place keeps its path). Empty when the prepared data is the training revision."""
    from .prepare import PROVENANCE

    provenance = json.loads((Path(config["data"]["prepared_dir"]) / PROVENANCE).read_text(encoding="utf-8"))
    trained = {s["path"]: s.get("sha256") for s in manifest["examples"].get("source_files", [])}
    return [{"path": s["path"], "trained_sha256": trained.get(s["path"]), "prepared_sha256": s["sha256"]}
            for s in provenance["sources"] if trained.get(s["path"]) != s["sha256"]]


def cmd_predict(args) -> None:
    _require_training_stack()
    from .infer import CheckModel

    config = _run_config(args)
    text = Path(args.input).read_text(encoding="utf-8") if args.input != "-" else sys.stdin.read()
    try:
        queries = json.loads(text)
    except json.JSONDecodeError as e:
        sys.exit(f"--input {args.input}: not valid JSON ({e})")
    if not isinstance(queries, (dict, list)):
        # A string would otherwise be iterated as one query per character, and null or a number crash.
        sys.exit(f"--input {args.input}: expected one query object or a list of them, "
                 f"got {type(queries).__name__}")
    single = isinstance(queries, dict)
    model = CheckModel(args.run, max_new_tokens=config["inference"]["max_new_tokens"])
    results = model.predict_many([queries] if single else queries, batch_size=config["inference"]["batch_size"])
    print(json.dumps(results[0] if single else results, ensure_ascii=False, indent=2))


def cmd_evaluate(args) -> None:
    _require_training_stack()
    from .evaluate import evaluate_rows, training_relatives
    from .infer import CheckModel
    from .prepare import read_split

    config = _run_config(args)
    rows = read_split(config["data"]["prepared_dir"], args.split)
    model = CheckModel(args.run, max_new_tokens=config["inference"]["max_new_tokens"])
    mismatch = _data_mismatch(config, model.manifest, rows)
    if mismatch:
        sys.exit(f"refusing to evaluate: {mismatch}. Omit --config to use the run's own, or re-prepare with it.")
    changed = _changed_sources(config, model.manifest)
    changed_splits = _changed_splits(config, model.manifest)
    if (changed or changed_splits) and not args.allow_data_change:
        listed = "; ".join(
            [f"{c['path']}: trained on {_short(c['trained_sha256'])}, "
             f"prepared from {_short(c['prepared_sha256'])}" for c in changed]
            + [f"{c['split']} split: trained on {_short(c['trained_sha256'])}, "
               f"now {_short(c['prepared_sha256'], 'absent')}" for c in changed_splits])
        sys.exit(f"refusing to evaluate: the prepared data is not the revision the run was trained on ({listed}). "
                 "Pass --allow-data-change to score it anyway; trained rows stay excluded and the metrics "
                 "record the change.")
    train_ids = set(model.manifest["examples"]["train_ids"])
    fingerprints = frozenset(model.manifest["examples"].get("train_fingerprints", []))
    all_rows = [r for s in ("train", "val", "test") for r in read_split(config["data"]["prepared_dir"], s)]
    related = training_relatives(all_rows, train_ids=train_ids, train_fingerprints=fingerprints,
                                 train_texts=model.manifest["examples"].get("train_texts", []),
                                 train_links=model.manifest["examples"].get("train_links", {}),
                                 train_scenario_ids=set(model.manifest["examples"].get("train_scenario_ids", [])),
                                 threshold=model.manifest["config"]["data"]["near_duplicate_threshold"])
    out = args.out or Path(args.run) / "eval" / args.split
    metrics = evaluate_rows(model, rows, split=args.split, train_ids=train_ids, out_dir=out,
                            train_fingerprints=fingerprints, related_to_training=frozenset(related),
                            include_training_rows=args.split == "train",
                            data_revision={"matches_training": not (changed or changed_splits),
                                           "changed_sources": changed, "changed_splits": changed_splits},
                            batch_size=config["inference"]["batch_size"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"per-example predictions: {Path(out) / 'predictions.jsonl'}")


def cmd_smoke(args) -> None:
    _require_training_stack()
    from . import prompt
    from .catalog import load_catalog
    from .evaluate import evaluate_rows
    from .infer import CheckModel
    from .train import train

    config = load_config(args.config)
    stages = []

    def done(stage: str, detail: str) -> None:
        stages.append(stage)
        print(f"[ok] {stage}: {detail}", flush=True)

    rows, report = _prepare(config)
    done("load + validate", f"{len(report.trainable)} trainable rows, {len(report.errors)} errors")

    smoke = config["smoke"]
    candidates = {r.id: r for r in rows if r.target is not None and r.scope == "general"}
    ids = smoke["example_ids"] or sorted(candidates)[: smoke["num_examples"]]
    missing = [i for i in ids if i not in candidates]
    if missing:
        sys.exit(f"smoke example_ids not trainable general rows: {missing}")
    picked = [candidates[i].to_json() for i in ids]
    if not 5 <= len(picked) <= 10:
        print(f"note: smoke mode is meant for 5-10 examples, got {len(picked)}")
    done("convert", f"{len(picked)} rows: {', '.join(ids)}")

    if args.tiny:
        from .tiny import make_tiny_model

        # One base per run: the run's manifest names this path, and a shared path would be
        # overwritten by the next smoke run's fresh random weights.
        tiny_dir = Path(config["data"]["prepared_dir"]).parent / f"tiny-random-model-{secrets.token_hex(3)}"
        system = prompt.system_prompt(load_catalog(config["data"]["catalog"]))
        make_tiny_model(tiny_dir, [r.to_json() for r in rows], system)
        config = copy.deepcopy(config)
        config["model"].update(base_model=str(tiny_dir), revision=None)
        done("tiny model", f"random-weight model at {tiny_dir} (plumbing only, not a real base model)")

    run_dir = _run_dir(config, "smoke")
    manifest = train(config, picked, [], run_dir=run_dir, source_files=report.sources,
                     max_steps=smoke["max_steps"], mode="smoke",
                     split_sha256=split_hashes(config["data"]["prepared_dir"]))
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
