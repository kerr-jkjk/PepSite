#!/usr/bin/env python3
"""Reproducible two-stage Site+MLM training harness.

This file is deliberately self contained so that a run can be rerun without
importing historical experiment scripts. It implements the published
two-stage parameter contract:

* phase 1: site-only warm-up, last four ESM layers plus site head trainable;
* phase 2: joint MLM+site training, all parameters trainable.

The harness is also usable with a small locally-created ESM configuration for
CI/smoke tests.  It never starts a long job unless the ``phase1``, ``phase2``
or ``run`` command is explicitly invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    EsmConfig,
    EsmForMaskedLM,
    get_cosine_schedule_with_warmup,
    set_seed as hf_set_seed,
)


PHASE1_DEFAULTS = {
    "epochs": 20,
    "unfreeze_last_n_layers": 4,
    "site_head_lr": 3e-4,
    "backbone_lr": 2e-6,
    "site_pos_weight": 4.0,
    "site_loss_weight": 1.0,
    "mlm_loss_weight": 0.0,
    "batch_size": 8,
    "eval_batch_size": 16,
    "grad_accum": 1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "max_length": 552,
}
PHASE2_DEFAULTS = {
    "epochs": 5,
    "learning_rate": 0.0007984276816171436,
    "site_pos_weight": 3.0,
    "site_loss_weight": 1.0,
    "mlm_loss_weight": 1.0,
    "batch_size": 4,
    "eval_batch_size": 16,
    "grad_accum": 2,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "max_length": 552,
}


def non_negative_seed(value: str) -> int:
    """Parse the run randomization value accepted by the public harness."""

    try:
        seed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed must be an integer") from exc
    if seed < 0:
        raise argparse.ArgumentTypeError("seed must be non-negative")
    return seed


def jsonable(value: Any) -> Any:
    """Convert numpy/torch scalars and NaN values to JSON-compatible values."""

    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_manifest(paths: Iterable[str | Path]) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file():
            manifest[str(path)] = {"kind": "file", "size": path.stat().st_size, "sha256": sha256_file(path)}
        elif path.is_dir():
            files = {}
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    files[str(child.relative_to(path))] = {
                        "size": child.stat().st_size,
                        "sha256": sha256_file(child),
                    }
            manifest[str(path)] = {"kind": "directory", "files": files}
        else:
            manifest[str(path)] = {"kind": "missing"}
    return manifest


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def seed_everything(seed: int) -> None:
    """Set every RNG before model construction or DataLoader use."""

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    hf_set_seed(seed)


class ProteinSiteDataset(Dataset):
    """CSV-backed receptor/peptide records with peptide MLM and site labels."""

    def __init__(self, file: str | Path, tokenizer, max_length: int = 552):
        self.file = str(Path(file).expanduser().resolve())
        self.max_length = int(max_length)
        # Site masks are digit strings, not numbers. Without dtype=str,
        # pandas may parse e.g. ``001100`` as integer 1100 and silently drop
        # leading zeroes, invalidating the receptor/mask length check.
        data = pd.read_csv(self.file, keep_default_na=False, dtype=str)
        receptor_col = "protein_sequence" if "protein_sequence" in data.columns else "Receptor Sequence"
        ligand_col = "ligand_sequence" if "ligand_sequence" in data.columns else "Binder"
        site_mask_col = "receptor_binding_site_mask"
        required = [receptor_col, ligand_col, site_mask_col]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns {missing} in {self.file}; found {list(data.columns)}")
        if "binding_site_status" in data.columns:
            data = data[data["binding_site_status"] == "ok"].copy()

        data[receptor_col] = data[receptor_col].replace("", pd.NA)
        data[ligand_col] = data[ligand_col].replace("", pd.NA)
        data[site_mask_col] = data[site_mask_col].replace("", pd.NA)
        data = data.dropna(subset=required).copy()

        self.samples: list[tuple[str, str, str, str]] = []
        self.skipped_rows = 0
        for row_idx, row in data.iterrows():
            receptor = str(row[receptor_col])
            peptide = str(row[ligand_col])
            mask = str(row[site_mask_col]).strip()
            query_id = str(row["query_id"]).strip() if "query_id" in data.columns else str(row_idx)
            if not mask or len(mask) != len(receptor) or any(ch not in "01" for ch in mask):
                self.skipped_rows += 1
                continue
            self.samples.append((query_id, receptor, peptide, mask))

        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        query_id, receptor, peptide, site_mask = self.samples[idx]
        masked_peptide = "<mask>" * len(peptide)
        input_ids = self.tokenizer(
            receptor + masked_peptide,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )["input_ids"].squeeze(0)
        attention_mask = self.tokenizer(
            receptor + masked_peptide,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )["attention_mask"].squeeze(0)
        labels = self.tokenizer(
            receptor + peptide,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )["input_ids"].squeeze(0)
        labels = torch.where(input_ids == self.tokenizer.mask_token_id, labels, -100)

        site_labels = torch.full((self.max_length,), -100.0, dtype=torch.float32)
        receptor_len = min(len(site_mask), self.max_length - 2)
        site_values = torch.tensor([float(int(ch)) for ch in site_mask[:receptor_len]], dtype=torch.float32)
        site_labels[1 : 1 + receptor_len] = site_values
        return {
            "query_id": query_id,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "site_labels": site_labels,
            "receptor_lengths": torch.tensor(receptor_len, dtype=torch.long),
        }


def collate_fn(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"query_id": [item["query_id"] for item in batch]}
    for key in ("input_ids", "attention_mask", "labels", "site_labels", "receptor_lengths"):
        result[key] = torch.stack([item[key] for item in batch], dim=0)
    return result


def build_site_head(config, head_type: str = "linear", hidden_dim: int | None = None, dropout: float = 0.1) -> nn.Module:
    hidden_size = int(config.hidden_size)
    if str(head_type).lower() == "linear":
        return nn.Linear(hidden_size, 1)
    if str(head_type).lower() == "mlp":
        width = int(hidden_dim) if hidden_dim and int(hidden_dim) > 0 else max(hidden_size // 2, 1)
        return nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_size, width),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(width, 1),
        )
    raise ValueError(f"Unsupported site head type: {head_type!r}")


class EsmForMaskedLMWithSiteHead(EsmForMaskedLM):
    """ESM MLM head with an auxiliary residue-level site head."""

    def __init__(
        self,
        config,
        site_pos_weight: float | None = None,
        site_loss_weight: float = 1.0,
        mlm_loss_weight: float = 1.0,
        site_head_type: str | None = None,
        site_head_hidden_dim: int | None = None,
        site_head_dropout: float | None = None,
    ):
        super().__init__(config)
        head_type = site_head_type or getattr(config, "site_head_type", "linear")
        hidden_dim = site_head_hidden_dim if site_head_hidden_dim is not None else getattr(config, "site_head_hidden_dim", None)
        head_dropout = site_head_dropout if site_head_dropout is not None else getattr(config, "site_head_dropout", 0.1)
        configured_pos = getattr(config, "site_pos_weight", None)
        if site_pos_weight is None:
            site_pos_weight = configured_pos
        self.site_head = build_site_head(config, head_type, hidden_dim, float(head_dropout))
        self.site_pos_weight = None if site_pos_weight is None else float(site_pos_weight)
        self.site_loss_weight = float(site_loss_weight)
        self.mlm_loss_weight = float(mlm_loss_weight)
        self.config.site_head_type = str(head_type).lower()
        self.config.site_head_hidden_dim = hidden_dim
        self.config.site_head_dropout = float(head_dropout)
        self.config.site_pos_weight = self.site_pos_weight
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, site_labels=None, **kwargs):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = outputs.last_hidden_state
        logits = self.lm_head(sequence_output)
        site_logits = self.site_head(sequence_output).squeeze(-1)

        mlm_loss = None
        site_loss = None
        total_loss = None
        if labels is not None:
            valid_mlm = labels != -100
            if bool(valid_mlm.any()):
                mlm_loss = F.cross_entropy(logits[valid_mlm], labels[valid_mlm])
                if self.mlm_loss_weight > 0:
                    total_loss = self.mlm_loss_weight * mlm_loss
        if site_labels is not None:
            valid_site = site_labels != -100
            if attention_mask is not None:
                valid_site = valid_site & attention_mask.bool()
            if bool(valid_site.any()):
                pos_weight = None
                if self.site_pos_weight is not None and self.site_pos_weight > 0:
                    pos_weight = torch.tensor(self.site_pos_weight, device=site_logits.device, dtype=site_logits.dtype)
                site_loss = F.binary_cross_entropy_with_logits(
                    site_logits[valid_site], site_labels[valid_site], pos_weight=pos_weight
                )
                if self.site_loss_weight > 0:
                    weighted_site = self.site_loss_weight * site_loss
                    total_loss = weighted_site if total_loss is None else total_loss + weighted_site
        return {
            "loss": total_loss,
            "logits": logits,
            "site_logits": site_logits,
            "mlm_loss": mlm_loss.detach() if mlm_loss is not None else None,
            "site_loss": site_loss.detach() if site_loss is not None else None,
        }


def freeze_for_phase1(model: EsmForMaskedLMWithSiteHead, unfreeze_last_n_layers: int = 4) -> list[int]:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.site_head.parameters():
        param.requires_grad = True
    layers = list(model.esm.encoder.layer)
    n = max(0, min(int(unfreeze_last_n_layers), len(layers)))
    indices = list(range(len(layers) - n, len(layers)))
    for index in indices:
        for param in layers[index].parameters():
            param.requires_grad = True
    # Phase 1 intentionally keeps lm_head frozen; MLM is diagnostic only.
    return indices


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def build_optimizer_phase1(model, site_head_lr: float, backbone_lr: float, weight_decay: float):
    site_params, backbone_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (site_params if name.startswith("site_head.") else backbone_params).append(parameter)
    groups = []
    if site_params:
        groups.append({"params": site_params, "lr": float(site_head_lr), "weight_decay": float(weight_decay), "name": "site_head"})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": float(backbone_lr), "weight_decay": float(weight_decay), "name": "backbone_last_layers"})
    if not groups:
        raise ValueError("No trainable parameters for phase 1")
    # Explicit 0.01 is also torch 2.0.1 AdamW's default; recording it avoids
    # accidental dependence on a future torch default.
    return torch.optim.AdamW(groups)


def build_optimizer_phase2(model, learning_rate: float, weight_decay: float):
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError("No trainable parameters for phase 2")
    # Historical phase-2 code supplied AdamW(model.parameters(), lr=...),
    # omitting weight_decay.  In the pinned environment torch==2.0.1 this
    # resolves to 0.01; set it explicitly while preserving that behavior.
    return torch.optim.AdamW(params, lr=float(learning_rate), weight_decay=float(weight_decay))


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(value)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}


def scalar(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _mean(values: Sequence[float | None]) -> float:
    numbers = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(numbers)) if numbers else math.nan


def site_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    if labels.size == 0:
        return {"site_AUROC": math.nan, "site_AUPRC": math.nan, "site_f1": math.nan, "n_eval_site_tokens": 0}
    try:
        from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

        auroc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else math.nan
        auprc = float(average_precision_score(labels, probs)) if labels.sum() else math.nan
        f1 = float(f1_score(labels, probs >= 0.5, zero_division=0))
    except Exception:
        auroc, auprc = math.nan, math.nan
        f1 = math.nan
    return {"site_AUROC": auroc, "site_AUPRC": auprc, "site_f1": f1, "n_eval_site_tokens": int(labels.size)}


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    fp16: bool,
    grad_accum: int,
    max_grad_norm: float,
    phase: str,
    epoch: int,
    global_step: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_values: list[float] = []
    mlm_values: list[float | None] = []
    site_values: list[float | None] = []
    rows: list[dict[str, Any]] = []
    for batch_index, raw_batch in enumerate(loader, start=1):
        batch = move_batch(raw_batch, device)
        with autocast(enabled=bool(fp16 and device.type == "cuda")):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                site_labels=batch["site_labels"],
            )
            loss = outputs["loss"]
            if loss is None:
                continue
            raw_loss = float(loss.detach().cpu().item())
            (scaler.scale(loss / max(1, int(grad_accum))).backward() if scaler is not None else (loss / max(1, int(grad_accum))).backward())

        should_step = batch_index % max(1, int(grad_accum)) == 0 or batch_index == len(loader)
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(max_grad_norm))
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        mlm = scalar(outputs.get("mlm_loss"))
        site = scalar(outputs.get("site_loss"))
        weighted_mlm = mlm * float(model.mlm_loss_weight) if mlm is not None else math.nan
        weighted_site = site * float(model.site_loss_weight) if site is not None else math.nan
        total_values.append(raw_loss)
        mlm_values.append(mlm)
        site_values.append(site)
        rows.append(
            {
                "record_type": "batch",
                "phase": phase,
                "split": "train",
                "epoch": int(epoch),
                "batch": int(batch_index),
                "global_step": int(global_step),
                "total_loss": raw_loss,
                "mlm_loss": mlm,
                "site_loss": site,
                "weighted_mlm_loss": weighted_mlm,
                "weighted_site_loss": weighted_site,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
    return {
        "total_loss": _mean(total_values),
        "mlm_loss": _mean(mlm_values),
        "site_loss": _mean(site_values),
    }, rows, global_step


@torch.no_grad()
def evaluate_one_epoch(model, loader, device, fp16: bool, phase: str, epoch: int, global_step: int, optimizer):
    model.eval()
    total_values: list[float] = []
    mlm_values: list[float | None] = []
    site_values: list[float | None] = []
    labels_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for batch_index, raw_batch in enumerate(loader, start=1):
        batch = move_batch(raw_batch, device)
        with autocast(enabled=bool(fp16 and device.type == "cuda")):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                site_labels=batch["site_labels"],
            )
        total = scalar(outputs.get("loss"))
        mlm = scalar(outputs.get("mlm_loss"))
        site = scalar(outputs.get("site_loss"))
        total_values.append(total if total is not None else math.nan)
        mlm_values.append(mlm)
        site_values.append(site)
        valid = (batch["site_labels"] != -100) & batch["attention_mask"].bool()
        labels_all.append(batch["site_labels"][valid].detach().cpu().numpy())
        probs_all.append(torch.sigmoid(outputs["site_logits"])[valid].detach().cpu().numpy())
        rows.append(
            {
                "record_type": "batch",
                "phase": phase,
                "split": "val",
                "epoch": int(epoch),
                "batch": int(batch_index),
                "global_step": int(global_step),
                "total_loss": total,
                "mlm_loss": mlm,
                "site_loss": site,
                "weighted_mlm_loss": mlm * float(model.mlm_loss_weight) if mlm is not None else math.nan,
                "weighted_site_loss": site * float(model.site_loss_weight) if site is not None else math.nan,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
    labels = np.concatenate(labels_all) if labels_all else np.asarray([])
    probs = np.concatenate(probs_all) if probs_all else np.asarray([])
    metrics = {
        "total_loss": _mean(total_values),
        "mlm_loss": _mean(mlm_values),
        "site_loss": _mean(site_values),
        **site_metrics(labels, probs),
    }
    return metrics, rows


def _noop_scheduler(optimizer):
    # A constant scheduler makes the phase-1 state resumable while preserving
    # the historical fixed learning rates exactly.
    return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)


def save_checkpoint(
    output_dir: Path,
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler,
    phase: str,
    epoch: int,
    global_step: int,
    config: Mapping[str, Any],
    epoch_rows: Sequence[Mapping[str, Any]],
    batch_rows: Sequence[Mapping[str, Any]],
) -> Path:
    checkpoint = output_dir / "checkpoints" / f"epoch_{int(epoch):03d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint))
    tokenizer.save_pretrained(str(checkpoint))
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    torch.save(scheduler.state_dict() if scheduler is not None else None, checkpoint / "scheduler.pt")
    torch.save(scaler.state_dict() if scaler is not None else None, checkpoint / "scaler.pt")
    torch.save(capture_rng_state(), checkpoint / "rng_state.pt")
    write_json(
        checkpoint / "training_state.json",
        {
            "phase": phase,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "seed": int(config["seed"]),
            "config": dict(config),
            "epoch_rows": list(epoch_rows),
            "batch_rows_count": len(batch_rows),
        },
    )
    latest = output_dir / "latest_training_state"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(checkpoint, target_is_directory=True)
    return checkpoint


def load_checkpoint(checkpoint: Path, model, optimizer, scheduler, scaler, device: torch.device) -> dict[str, Any]:
    state = json.loads((checkpoint / "training_state.json").read_text(encoding="utf-8"))
    optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location=device))
    scheduler_state = torch.load(checkpoint / "scheduler.pt", map_location=device)
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    scaler_state = torch.load(checkpoint / "scaler.pt", map_location=device)
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    restore_rng_state(torch.load(checkpoint / "rng_state.pt", map_location="cpu"))
    return state


def save_model_bundle(model, tokenizer, output_dir: Path, config: Mapping[str, Any], name: str = "final_model") -> Path:
    destination = output_dir / name
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(destination))
    tokenizer.save_pretrained(str(destination))
    write_json(destination / "two_stage_config.json", config)
    return destination


def write_loss_files(output_dir: Path, batch_rows: Sequence[Mapping[str, Any]], epoch_rows: Sequence[Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(batch_rows).to_csv(output_dir / "batch_metrics.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(output_dir / "epoch_metrics.csv", index=False)
    # Keep a single discoverable table as well.  The union of columns is
    # intentional: rows marked record_type=batch are per-batch, while rows
    # marked record_type=epoch contain train_* and eval_* aggregates.
    pd.concat([pd.DataFrame(batch_rows), pd.DataFrame(epoch_rows)], ignore_index=True, sort=False).to_csv(
        output_dir / "loss_history.csv", index=False
    )


def plot_loss_files(output_dir: Path, epoch_rows: Sequence[Mapping[str, Any]]) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if not epoch_rows:
        return
    frame = pd.DataFrame(epoch_rows).sort_values("global_epoch")
    x = frame["global_epoch"]
    phase1_boundary = None
    if "phase" in frame.columns and bool((frame["phase"] == "phase1").any()):
        phase1_boundary = float(frame.loc[frame["phase"] == "phase1", "global_epoch"].max()) + 0.5
    specs = [
        ("total_loss", "Total loss", "total_loss_curve.png"),
        ("mlm_loss", "Raw MLM loss (diagnostic in phase 1)", "mlm_loss_curve.png"),
        ("site_loss", "Raw site loss", "site_loss_curve.png"),
    ]
    for component, ylabel, filename in specs:
        plt.figure(figsize=(8, 5))
        for split, prefix, style in (("train", "train_", "-o"), ("val", "eval_", "--o")):
            column = f"{prefix}{component}"
            if column in frame:
                plt.plot(x, frame[column], style, label=f"{split} {component}")
        if phase1_boundary is not None:
            plt.axvline(phase1_boundary, color="black", linestyle=":", linewidth=1, label="phase boundary")
        plt.xlabel("Global epoch")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / filename, dpi=220)
        plt.close()

    plt.figure(figsize=(9, 6))
    for column, label, style in (
        ("train_total_loss", "train total", "-o"),
        ("eval_total_loss", "val total", "--o"),
        ("train_weighted_mlm_loss", "train weighted MLM", "-s"),
        ("train_weighted_site_loss", "train weighted site", "-^"),
    ):
        if column in frame:
            plt.plot(x, frame[column], style, label=label)
    if phase1_boundary is not None:
        plt.axvline(phase1_boundary, color="black", linestyle=":", linewidth=1, label="phase boundary")
    plt.xlabel("Global epoch (phase boundary retained)")
    plt.ylabel("Loss")
    plt.title("Two-stage Site + MLM loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "two_stage_combined_loss.png", dpi=220)
    plt.close()


def write_combined_two_stage_outputs(root: Path, phase1_dir: Path, phase2_dir: Path) -> None:
    """Create one route-level history and the four requested two-stage plots."""

    phase1_rows = pd.read_csv(phase1_dir / "epoch_metrics.csv")
    phase2_rows = pd.read_csv(phase2_dir / "epoch_metrics.csv")
    phase1_rows["phase"] = "phase1"
    phase2_rows["phase"] = "phase2"
    phase1_epochs = int(phase1_rows["epoch"].max())
    phase1_rows["global_epoch"] = phase1_rows["epoch"].astype(int)
    phase2_rows["global_epoch"] = phase1_epochs + phase2_rows["epoch"].astype(int)
    combined = pd.concat([phase1_rows, phase2_rows], ignore_index=True, sort=False)
    combined.to_csv(root / "two_stage_loss_history.csv", index=False)
    plot_loss_files(root, combined.to_dict("records"))


def _phase_config(phase: str, seed: int, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    if phase == "phase1":
        values = {
            **PHASE1_DEFAULTS,
            "epochs": int(args.num_epochs if args.num_epochs is not None else PHASE1_DEFAULTS["epochs"]),
            "max_length": int(args.max_length),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "grad_accum": int(args.grad_accum),
            "weight_decay": float(args.weight_decay),
        }
    else:
        values = {
            **PHASE2_DEFAULTS,
            "epochs": int(args.num_epochs if args.num_epochs is not None else PHASE2_DEFAULTS["epochs"]),
            "max_length": int(args.max_length),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "grad_accum": int(args.grad_accum),
            "weight_decay": float(args.weight_decay),
        }
    values.update({"phase": phase, "seed": int(seed), "output_dir": str(output_dir.resolve()), "fp16": bool(args.fp16)})
    return values


def run_phase(
    phase: str,
    *,
    seed: int,
    init_model_dir: str | Path,
    train_csv: str | Path,
    val_csv: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    fp16: bool = True,
    num_workers: int = 0,
    resume_from_checkpoint: str | Path | None = None,
    num_epochs: int | None = None,
    max_length: int = 552,
    batch_size: int | None = None,
    eval_batch_size: int | None = None,
    grad_accum: int | None = None,
    weight_decay: float | None = None,
    stop_after_epoch: int | None = None,
) -> dict[str, Any]:
    """Run one phase, or resume it from an epoch checkpoint.

    The function is intentionally callable from tests; the CLI below simply
    maps arguments onto it.  ``num_epochs`` is a target epoch count, so a
    resumed call with ``num_epochs=2`` after epoch 1 executes only epoch 2.
    """

    if phase not in {"phase1", "phase2"}:
        raise ValueError(f"phase must be phase1 or phase2, got {phase!r}")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        num_epochs=num_epochs,
        max_length=max_length,
        batch_size=(batch_size if batch_size is not None else (8 if phase == "phase1" else 4)),
        eval_batch_size=(eval_batch_size if eval_batch_size is not None else 16),
        grad_accum=(grad_accum if grad_accum is not None else (1 if phase == "phase1" else 2)),
        weight_decay=(weight_decay if weight_decay is not None else 0.01),
        fp16=fp16,
    )
    config = _phase_config(phase, seed, args, output)
    seed_everything(seed)
    dev = choose_device(device)

    init_path = Path(init_model_dir).expanduser().resolve()
    train_path = Path(train_csv).expanduser().resolve()
    val_path = Path(val_csv).expanduser().resolve()
    resume_path = Path(resume_from_checkpoint).expanduser().resolve() if resume_from_checkpoint else None
    model_path_for_load = resume_path if resume_path is not None else init_path
    tokenizer = AutoTokenizer.from_pretrained(str(model_path_for_load), local_files_only=True)
    train_dataset = ProteinSiteDataset(train_path, tokenizer, max_length=config["max_length"])
    val_dataset = ProteinSiteDataset(val_path, tokenizer, max_length=config["max_length"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=int(num_workers),
        pin_memory=(dev.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=int(num_workers),
        pin_memory=(dev.type == "cuda"),
    )

    if phase == "phase1":
        model = EsmForMaskedLMWithSiteHead.from_pretrained(
            str(model_path_for_load),
            local_files_only=True,
            site_pos_weight=4.0,
            site_loss_weight=1.0,
            mlm_loss_weight=0.0,
        )
        unfrozen = freeze_for_phase1(model, 4)
        optimizer = build_optimizer_phase1(model, 3e-4, 2e-6, config["weight_decay"])
        target_epochs = config["epochs"]
        total_steps = max(1, math.ceil(len(train_loader) / config["grad_accum"]) * target_epochs)
        scheduler = _noop_scheduler(optimizer)
    else:
        model = EsmForMaskedLMWithSiteHead.from_pretrained(
            str(model_path_for_load),
            local_files_only=True,
            site_pos_weight=3.0,
            site_loss_weight=1.0,
            mlm_loss_weight=1.0,
        )
        unfreeze_all(model)
        optimizer = build_optimizer_phase2(model, config["learning_rate"], config["weight_decay"])
        target_epochs = config["epochs"]
        total_steps = max(1, math.ceil(len(train_loader) / config["grad_accum"]) * target_epochs)
        # Match transformers.TrainingArguments.get_warmup_steps: ratio-based
        # warmup is rounded upward so even a tiny smoke run can exercise it.
        warmup_steps = int(math.ceil(total_steps * float(config["warmup_ratio"])))
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        unfrozen = list(range(len(model.esm.encoder.layer)))

    model.to(dev)
    scaler = GradScaler(enabled=bool(config["fp16"] and dev.type == "cuda"))
    epoch_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    start_epoch = 0
    global_step = 0
    if resume_path is not None:
        state = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, dev)
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        epoch_rows = list(state.get("epoch_rows", []))
        # Existing batch rows are retained for a complete audit trail.
        batch_file = output / "batch_metrics.csv"
        if batch_file.exists():
            batch_rows = pd.read_csv(batch_file).replace({np.nan: None}).to_dict("records")
        if start_epoch >= target_epochs:
            write_loss_files(output, batch_rows, epoch_rows)
            plot_loss_files(output, epoch_rows)
            return {"output_dir": str(output), "final_model": str(output / "final_model"), "epochs": start_epoch, "resumed": True}

    write_json(output / "data_manifest.json", hash_manifest([init_path, train_path, val_path]))
    config.update(
        {
            "init_model_dir": str(init_path),
            "model_load_dir": str(model_path_for_load),
            "train_csv": str(train_path),
            "val_csv": str(val_path),
            "train_rows": len(train_dataset),
            "val_rows": len(val_dataset),
            "train_skipped_rows": train_dataset.skipped_rows,
            "val_skipped_rows": val_dataset.skipped_rows,
            "device": str(dev),
            "num_workers": int(num_workers),
            "unfrozen_layers": unfrozen,
            "optimizer_actual_weight_decay": float(optimizer.defaults["weight_decay"]),
            "total_optimizer_steps": total_steps,
            "warmup_steps": 0 if phase == "phase1" else int(math.ceil(total_steps * float(config["warmup_ratio"]))),
            "resume_from_checkpoint": str(resume_path) if resume_path else None,
        }
    )
    write_json(output / "run_config.json", config)

    # Restore checkpoint RNG after all model/data-loader construction.  This
    # ensures the next shuffled epoch and dropout stream match an uninterrupted
    # run exactly.
    if resume_path is not None:
        restore_rng_state(torch.load(resume_path / "rng_state.pt", map_location="cpu"))

    # ``stop_after_epoch`` is used only by the tiny resume test and by an
    # operator intentionally staging a run.  The scheduler is still built for
    # the full target epoch count, so resuming has exactly the same LR stream
    # as an uninterrupted run.
    end_epoch = target_epochs if stop_after_epoch is None else min(target_epochs, int(stop_after_epoch))
    for epoch in range(start_epoch + 1, end_epoch + 1):
        train_metrics, train_batch_rows, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            dev,
            config["fp16"],
            config["grad_accum"],
            config["max_grad_norm"],
            phase,
            epoch,
            global_step,
        )
        eval_metrics, eval_batch_rows = evaluate_one_epoch(
            model, val_loader, dev, config["fp16"], phase, epoch, global_step, optimizer
        )
        batch_rows.extend(train_batch_rows)
        batch_rows.extend(eval_batch_rows)
        row = {
            "record_type": "epoch",
            "phase": phase,
            "epoch": int(epoch),
            "phase_epoch": int(epoch),
            "global_epoch": int(epoch),
            "global_step": int(global_step),
            "train_total_loss": train_metrics["total_loss"],
            "train_mlm_loss": train_metrics["mlm_loss"],
            "train_site_loss": train_metrics["site_loss"],
            "train_weighted_mlm_loss": train_metrics["mlm_loss"] * float(model.mlm_loss_weight),
            "train_weighted_site_loss": train_metrics["site_loss"] * float(model.site_loss_weight),
            "eval_total_loss": eval_metrics["total_loss"],
            "eval_mlm_loss": eval_metrics["mlm_loss"],
            "eval_site_loss": eval_metrics["site_loss"],
            "eval_weighted_mlm_loss": eval_metrics["mlm_loss"] * float(model.mlm_loss_weight),
            "eval_weighted_site_loss": eval_metrics["site_loss"] * float(model.site_loss_weight),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "site_AUROC": eval_metrics.get("site_AUROC"),
            "site_AUPRC": eval_metrics.get("site_AUPRC"),
            "site_f1": eval_metrics.get("site_f1"),
            "n_eval_site_tokens": eval_metrics.get("n_eval_site_tokens"),
        }
        epoch_rows.append(row)
        write_loss_files(output, batch_rows, epoch_rows)
        plot_loss_files(output, epoch_rows)
        if phase == "phase2":
            previous_best = min(
                (float(item["eval_total_loss"]) for item in epoch_rows[:-1]),
                default=math.inf,
            )
            if float(row["eval_total_loss"]) < previous_best:
                save_model_bundle(
                    model,
                    tokenizer,
                    output,
                    {
                        **config,
                        "selection_metric": "eval_total_loss",
                        "selection_direction": "minimize",
                        "best_epoch": int(epoch),
                        "best_eval_total_loss": float(row["eval_total_loss"]),
                    },
                    name="best_model",
                )
        checkpoint = save_checkpoint(
            output,
            model,
            tokenizer,
            optimizer,
            scheduler,
            scaler,
            phase,
            epoch,
            global_step,
            config,
            epoch_rows,
            batch_rows,
        )
        print(json.dumps({"phase": phase, "epoch": epoch, "global_step": global_step, "checkpoint": str(checkpoint), **row}, ensure_ascii=False))

    completed_epochs = int(epoch_rows[-1]["epoch"]) if epoch_rows else start_epoch
    if phase == "phase2":
        best_row = min(epoch_rows, key=lambda item: float(item["eval_total_loss"]))
        best_dir = output / "best_model"
        if not best_dir.is_dir():
            raise RuntimeError(f"Missing phase-2 best model directory: {best_dir}")
        final_dir = output / "final_model"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.copytree(best_dir, final_dir)
        selection = {
            "metric": "eval_total_loss",
            "direction": "minimize",
            "best_epoch": int(best_row["epoch"]),
            "best_value": float(best_row["eval_total_loss"]),
        }
    else:
        final_dir = save_model_bundle(model, tokenizer, output, {**config, "epoch_rows": epoch_rows})
        selection = {"metric": "last_epoch", "best_epoch": completed_epochs}
    write_json(output / "summary.json", {"phase": phase, "seed": seed, "final_model": str(final_dir), "epochs": completed_epochs, "target_epochs": target_epochs, "selection": selection, "epoch_rows": epoch_rows})
    return {"output_dir": str(output), "final_model": str(final_dir), "epochs": completed_epochs, "target_epochs": target_epochs, "resumed": resume_path is not None}


def run_two_stage(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).expanduser().resolve()
    phase1_dir = root / "phase1"
    phase2_dir = root / "phase2"
    phase1_result = run_phase(
        "phase1",
        seed=args.seed,
        init_model_dir=args.init_model_dir,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=phase1_dir,
        device=args.device,
        fp16=args.fp16,
        num_workers=args.num_workers,
    )
    phase2_result = run_phase(
        "phase2",
        seed=args.seed,
        init_model_dir=phase1_result["final_model"],
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=phase2_dir,
        device=args.device,
        fp16=args.fp16,
        num_workers=args.num_workers,
    )
    write_combined_two_stage_outputs(root, phase1_dir, phase2_dir)
    return {"phase1": phase1_result, "phase2": phase2_result}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--seed", type=non_negative_seed, required=True)
        command_parser.add_argument("--init_model_dir", required=True)
        command_parser.add_argument("--train_csv", required=True)
        command_parser.add_argument("--val_csv", required=True)
        command_parser.add_argument("--output_dir", required=True)
        command_parser.add_argument("--device", default="auto")
        command_parser.add_argument("--fp16", action="store_true", default=True)
        command_parser.add_argument("--no-fp16", dest="fp16", action="store_false")
        command_parser.add_argument("--num_workers", type=int, default=2)
        command_parser.add_argument("--num_epochs", type=int, default=None)
        command_parser.add_argument("--max_length", type=int, default=552)
        command_parser.add_argument("--batch_size", type=int, default=None)
        command_parser.add_argument("--eval_batch_size", type=int, default=None)
        command_parser.add_argument("--grad_accum", type=int, default=None)
        command_parser.add_argument("--weight_decay", type=float, default=0.01)
        command_parser.add_argument("--resume_from_checkpoint", default=None)
        command_parser.add_argument("--stop_after_epoch", type=int, default=None)

    p1 = sub.add_parser("phase1", help="run or resume phase 1")
    common(p1)
    p2 = sub.add_parser("phase2", help="run or resume phase 2")
    common(p2)
    run = sub.add_parser("run", help="run phase 1 followed by phase 2")
    run.add_argument("--seed", type=non_negative_seed, required=True)
    run.add_argument("--init_model_dir", required=True)
    run.add_argument("--train_csv", required=True)
    run.add_argument("--val_csv", required=True)
    run.add_argument("--output_root", required=True)
    run.add_argument("--device", default="auto")
    run.add_argument("--fp16", action="store_true", default=True)
    run.add_argument("--no-fp16", dest="fp16", action="store_false")
    run.add_argument("--num_workers", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        print(json.dumps(run_two_stage(args), indent=2, ensure_ascii=False))
        return
    result = run_phase(
        "phase1" if args.command == "phase1" else "phase2",
        seed=args.seed,
        init_model_dir=args.init_model_dir,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=args.output_dir,
        device=args.device,
        fp16=args.fp16,
        num_workers=args.num_workers,
        resume_from_checkpoint=args.resume_from_checkpoint,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_accum=args.grad_accum,
        weight_decay=args.weight_decay,
        stop_after_epoch=args.stop_after_epoch,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
