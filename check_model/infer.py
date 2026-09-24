"""Load a trained run and turn queries into check decisions.

    model = CheckModel("runs/my-run")
    model.predict({"scene": "...", "player_action": "..."})
    -> {"valid": true, "decision": {...}, "game_request": {"skill": ..., "difficulty": ...},
        "note": null, "errors": [], "raw_output": "..."}

The run directory is self-contained: the system prompt and catalog are the ones saved at
training time, not the current ones, so a later catalog sync cannot change what a trained
model is asked or how its answers are checked.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from . import prompt, strictjson
from .catalog import Catalog, parse_catalog


def _is_str_list(v) -> bool:
    return isinstance(v, list) and all(isinstance(i, str) for i in v)


def _is_str_map(v) -> bool:
    return isinstance(v, dict) and all(isinstance(i, str) for i in v.values())


def _is_opt_str(v) -> bool:
    return v is None or isinstance(v, str)


#: What train records as `precision`: resolve_precision's results, never "auto".
RESOLVED_PRECISIONS = ("bf16", "fp16", "fp32")

#: Every manifest field serving and evaluation read, and what it must hold.
_MANIFEST_FIELDS = {
    "precision": lambda v: v in RESOLVED_PRECISIONS,
    "adapter": lambda v: v in ("lora", None),
    "catalog_sha256": lambda v: isinstance(v, str),
    "model_files": _is_str_map,
    "global_steps": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "metrics": lambda v: isinstance(v, dict),
    "prompt.format_version": lambda v: isinstance(v, str),
    "prompt.system_prompt": lambda v: isinstance(v, str),
    "prompt.system_prompt_sha256": lambda v: isinstance(v, str),
    "prompt.language": lambda v: isinstance(v, str),
    "base_model.name_or_path": lambda v: isinstance(v, str),
    "base_model.revision": _is_opt_str,
    "base_model.resolved_commit": _is_opt_str,
    "base_model.local_sha256": _is_opt_str,
    "base_model.in_run": _is_opt_str,
    "base_model.trust_remote_code": lambda v: isinstance(v, bool),
    "examples.train_ids": _is_str_list,
    "examples.val_ids": _is_str_list,
    "examples.train_fingerprints": _is_str_list,
    "examples.train_texts": _is_str_list,
    "examples.train_scenario_ids": _is_str_list,
    "examples.split_sha256": _is_str_map,
    "examples.train_links": lambda v: isinstance(v, dict) and all(_is_str_list(i) for i in v.values()),
    "examples.source_files": lambda v: isinstance(v, list) and all(
        isinstance(s, dict) and isinstance(s.get("path"), str) and isinstance(s.get("sha256"), str) for s in v),
}


def _manifest_problem(data) -> str | None:
    if not isinstance(data, dict):
        return "not a JSON object"
    for key, ok in _MANIFEST_FIELDS.items():
        value = data
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return f"{key} is missing"
            value = value[part]
        if not ok(value):
            return f"{key} has the wrong type or value: {value!r:.80}"
    from .config import check_recorded_config

    try:
        check_recorded_config(data.get("config"))
    except ValueError as exc:
        return f"config: {exc}"
    return None


def read_manifest(run_dir: str | Path) -> dict:
    """run_manifest.json, decoded strictly -- with two `base_model` or `examples` objects (a merge,
    say) the last would otherwise silently decide which base and which training ids the run has
    -- and checked for every field serving and evaluation read, so a damaged manifest is one
    named refusal instead of a KeyError somewhere later."""
    path = Path(run_dir) / "run_manifest.json"
    data = strictjson.loads(path.read_bytes().decode("utf-8"))
    problem = _manifest_problem(data)
    if problem:
        raise ValueError(f"{path} is not a run manifest ({problem})")
    return data


def base_revision(base: dict) -> str | None:
    """The base weights the run was trained on: the hub commit recorded at training time, else
    the configured revision. Without this, a run trained against a moving branch head would be
    served (and scored) against whatever the head is today."""
    return base.get("resolved_commit") or base.get("revision")


def bad_query_result(errors: list[str]) -> dict:
    """The result for a query that never reaches generation."""
    return {"valid": False, "decision": None, "game_request": None, "note": None,
            "errors": ["bad query: " + e for e in errors], "raw_output": None}


def base_source(base: dict, run_dir: Path) -> str:
    """Where to load the base from: inside the run when it was kept there, else as recorded."""
    return str(Path(run_dir) / base["in_run"]) if base.get("in_run") else base["name_or_path"]


def check_local_base(base: dict, source: str) -> None:
    """A LoRA run trained on a local base directory is served only on that same content, and one
    trained from the hub only from the hub: the run stores the adapter alone, and other weights
    under the same name would give other answers."""
    from .train import local_base_sha256

    pinned = base.get("local_sha256")
    if not pinned:
        # Trained from the hub: from_pretrained would take a local path of the same name first,
        # and nothing here could say whether its weights are the recorded revision's.
        if Path(source).exists():
            raise ValueError(f"this run's base {source!r} was trained from the hub, but a local path "
                             f"{Path(source).resolve()} of that name exists here and would be loaded "
                             "instead; run from another directory or move it")
        return
    if not Path(source).is_dir():
        raise FileNotFoundError(f"base model directory {source} is gone; this LoRA run stores only "
                                "its adapter and needs the base it was trained on")
    now = local_base_sha256(source)
    if now != pinned:
        raise ValueError(f"base model directory {source} changed since training (content "
                         f"{now[:12]}, trained on {pinned[:12]}); the adapter would run on other weights")


def load_run_catalog(run_dir: str | Path, manifest: dict) -> Catalog:
    """The run's own catalog, refused unless it is the one recorded at training."""
    raw = (Path(run_dir) / "catalog.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest.get("catalog_sha256"):
        raise ValueError(f"{Path(run_dir) / 'catalog.json'} does not match the SHA-256 recorded at training; "
                         "the run's label set was edited")
    return parse_catalog(raw)


def check_model_files(run_dir: str | Path, manifest: dict) -> None:
    """The run's model/ directory, refused unless it holds exactly the files training saved: an
    edited adapter, config or tokenizer file would serve answers the recorded run never gave."""
    from .train import file_sha256s

    recorded = manifest.get("model_files")
    model_dir = Path(run_dir) / "model"
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("run_manifest.json records no model_files digest; retrain, or load the run "
                         "with the code version that trained it")
    now = file_sha256s(model_dir)
    changed = sorted(p for p in recorded.keys() | now.keys() if recorded.get(p) != now.get(p))
    if changed:
        raise ValueError(f"{model_dir}: {changed} differ from the files saved at training "
                         "(edited, added or removed); the run's model was changed")


def check_format(manifest: dict) -> None:
    got = manifest["prompt"]["format_version"]
    if got != prompt.PROMPT_FORMAT_VERSION:
        raise ValueError(f"run uses prompt format {got}, this code serves format {prompt.PROMPT_FORMAT_VERSION}; "
                         "retrain, or load it with the code version that trained it")
    # The prompt the run is served with is the recorded text; an edited copy would ask the model
    # something it was never trained on.
    if prompt.prompt_sha256(manifest["prompt"]["system_prompt"]) != manifest["prompt"].get("system_prompt_sha256"):
        raise ValueError("run_manifest.json: the system prompt does not match its recorded SHA-256; "
                         "the run's prompt was edited after training")


def _bf16_supported(device: str) -> bool:
    import torch

    with torch.cuda.device(torch.device(device)):
        return torch.cuda.is_bf16_supported()


def serving_dtype(precision: str, device: str):
    """The dtype a run is served in: its training half precision on a CUDA device (indexed or
    not), fp32 on anything else -- and fp32 for a bf16 run on a GPU without bf16, rather than
    kernels that GPU lacks (fp16 could overflow values a bf16 model produces)."""
    import torch

    if torch.device(device).type != "cuda":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16" and _bf16_supported(device):
        return torch.bfloat16
    return torch.float32


class CheckModel:
    def __init__(self, run_dir: str | Path, *, max_new_tokens: int = 64, device: str | None = None,
                 manifest: dict | None = None):
        """`manifest`: one already returned by read_manifest for this run, which evaluate and
        predict pass so that the model is checked against the manifest their exclusions and
        settings came from; every file loaded below is verified against it."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        self.run_dir = Path(run_dir)
        self.manifest = manifest if manifest is not None else read_manifest(self.run_dir)
        check_format(self.manifest)
        self.system = self.manifest["prompt"]["system_prompt"]
        self.catalog = load_run_catalog(self.run_dir, self.manifest)
        model_dir = self.run_dir / "model"
        base = self.manifest["base_model"]
        source = base_source(base, self.run_dir)

        def verify() -> None:
            check_model_files(self.run_dir, self.manifest)
            if self.manifest["adapter"] == "lora":
                check_local_base(base, source)

        verify()  # before loading: a changed run fails without a download or an allocation
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = serving_dtype(self.manifest["precision"], self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=base["trust_remote_code"])
        self.tokenizer.padding_side = "left"  # decoder-only batch generation
        if self.manifest["adapter"] == "lora":
            from peft import PeftModel

            model = AutoModelForCausalLM.from_pretrained(
                source, revision=base_revision(base), dtype=dtype,
                trust_remote_code=base["trust_remote_code"])
            model = PeftModel.from_pretrained(model, model_dir)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype,
                                                         trust_remote_code=base["trust_remote_code"])
        verify()  # and after: the files were not replaced while they were being read
        self.model = model.to(self.device).eval()
        # Greedy, and nothing else from the base model's generation_config: instruct models
        # ship sampling settings and a repetition penalty, which would bend a JSON answer.
        stop = self.model.generation_config.eos_token_id
        self.generation = GenerationConfig(
            do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=stop if stop is not None else self.tokenizer.eos_token_id)
        from .train import context_limit_of

        self.context_limit = context_limit_of(self.model.config, self.tokenizer)

    def render(self, query: dict) -> str:
        return self.tokenizer.apply_chat_template(prompt.messages(self.system, query), tokenize=False,
                                                  add_generation_prompt=True)

    def length_error(self, query: dict) -> str | None:
        """Why a query cannot be generated for within the model's context, or None."""
        if not self.context_limit:
            return None
        used = len(self.tokenizer(self.render(query), add_special_tokens=False)["input_ids"])
        if used + self.generation.max_new_tokens > self.context_limit:
            return (f"query too long: {used} prompt tokens + {self.generation.max_new_tokens} new tokens "
                    f"exceed the model's context of {self.context_limit}")
        return None

    def generate(self, queries: list[dict]) -> list[str]:
        import torch

        texts = [self.render(q) for q in queries]
        batch = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**batch, generation_config=self.generation)
        width = batch["input_ids"].shape[1]
        return [self.tokenizer.decode(seq[width:], skip_special_tokens=True) for seq in out]

    def predict_many(self, queries: list[dict], batch_size: int = 8) -> list[dict]:
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        results: list[dict | None] = [None] * len(queries)
        runnable = []
        for i, q in enumerate(queries):
            errors = prompt.query_errors(q)
            if not errors:
                too_long = self.length_error(q)
                errors = [too_long] if too_long else []
            if errors:
                results[i] = bad_query_result(errors)
            else:
                runnable.append(i)
        for start in range(0, len(runnable), batch_size):
            chunk = runnable[start:start + batch_size]
            for i, raw in zip(chunk, self.generate([queries[i] for i in chunk])):
                decision, errors = prompt.parse_decision(raw, self.catalog)
                request, note = prompt.to_game_request(decision) if decision else (None, None)
                results[i] = {"valid": decision is not None, "decision": decision, "game_request": request,
                              "note": note, "errors": errors, "raw_output": raw}
        return results

    def predict(self, query: dict) -> dict:
        return self.predict_many([query])[0]
