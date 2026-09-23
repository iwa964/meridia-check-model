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

import json
from pathlib import Path

from . import prompt
from .catalog import load_catalog


def base_revision(base: dict) -> str | None:
    """The base weights the run was trained on: the hub commit recorded at training time, else
    the configured revision. Without this, a run trained against a moving branch head would be
    served (and scored) against whatever the head is today."""
    return base.get("resolved_commit") or base.get("revision")


def check_format(manifest: dict) -> None:
    got = manifest["prompt"]["format_version"]
    if got != prompt.PROMPT_FORMAT_VERSION:
        raise ValueError(f"run uses prompt format {got}, this code serves format {prompt.PROMPT_FORMAT_VERSION}; "
                         "retrain, or load it with the code version that trained it")


def serving_dtype(precision: str, device: str):
    """The dtype a run is served in: its training half precision on any CUDA device (indexed
    or not), fp32 otherwise."""
    import torch

    half = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    return half if half is not None and torch.device(device).type == "cuda" else torch.float32


class CheckModel:
    def __init__(self, run_dir: str | Path, *, max_new_tokens: int = 64, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        self.run_dir = Path(run_dir)
        self.manifest = json.loads((self.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        check_format(self.manifest)
        self.system = self.manifest["prompt"]["system_prompt"]
        self.catalog = load_catalog(self.run_dir / "catalog.json")
        model_dir = self.run_dir / "model"
        base = self.manifest["base_model"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = serving_dtype(self.manifest["precision"], self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tokenizer.padding_side = "left"  # decoder-only batch generation
        if self.manifest["adapter"] == "lora":
            from peft import PeftModel

            model = AutoModelForCausalLM.from_pretrained(
                base["name_or_path"], revision=base_revision(base), dtype=dtype,
                trust_remote_code=base["trust_remote_code"])
            model = PeftModel.from_pretrained(model, model_dir)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype,
                                                         trust_remote_code=base["trust_remote_code"])
        self.model = model.to(self.device).eval()
        # Greedy, and nothing else from the base model's generation_config: instruct models
        # ship sampling settings and a repetition penalty, which would bend a JSON answer.
        stop = self.model.generation_config.eos_token_id
        self.generation = GenerationConfig(
            do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=stop if stop is not None else self.tokenizer.eos_token_id)
        self.context_limit = getattr(self.model.config, "max_position_embeddings", None)

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
        results: list[dict | None] = [None] * len(queries)
        runnable = []
        for i, q in enumerate(queries):
            errors = prompt.query_errors(q)
            if not errors:
                too_long = self.length_error(q)
                errors = [too_long] if too_long else []
            if errors:
                results[i] = {"valid": False, "decision": None, "game_request": None, "note": None,
                              "errors": ["bad query: " + e for e in errors], "raw_output": None}
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
