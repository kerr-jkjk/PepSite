#!/usr/bin/env python3
"""Run the published PepSite Phase 1 site-head warm-up.

This is a thin, auditable launcher around the release's self-contained
``src/pepsite/training_harness.py`` implementation.  The launcher
deliberately does not reimplement training or silently fall back to an
unrelated script.  ``--harness`` (or ``PEPSITE_TRAINING_HARNESS``) is
available only when an auditor wants to run a separately archived copy.

All paths are interpreted relative to the repository root unless absolute.
The defaults reproduce the primary Phase 1 protocol; changing a
hyperparameter is possible for a deliberate experiment, but the resulting
``run_config.json`` remains the authoritative record.  The randomization
value defaults to configs/reproducibility.yaml and is not a model label.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
REPRO_CONFIG = REPO_ROOT / "configs" / "reproducibility.yaml"
DEFAULT_HARNESS_CANDIDATES = (
    REPO_ROOT / "src" / "pepsite" / "training_harness.py",
    REPO_ROOT / "scripts" / "two_stage_harness.py",
)


def _non_negative_seed(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed must be an integer") from exc
    if seed < 0:
        raise argparse.ArgumentTypeError("seed must be non-negative")
    return seed


def _default_seed() -> int:
    for line in REPRO_CONFIG.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("random_seed:"):
            value = int(line.split(":", 1)[1].strip())
            if value >= 0:
                return value
    raise RuntimeError(f"random_seed is missing from {REPRO_CONFIG}")


def _path(value: str | os.PathLike[str]) -> Path:
    """Resolve a CLI path without embedding machine-specific defaults."""

    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate)


def _find_harness(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(_path(explicit))
    configured = os.environ.get("PEPSITE_TRAINING_HARNESS")
    if configured:
        candidates.append(_path(configured))
    candidates.extend(DEFAULT_HARNESS_CANDIDATES)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n  ".join(str(item) for item in candidates)
    raise FileNotFoundError(
        "The release training harness is missing. Restore "
        "src/pepsite/training_harness.py or pass an audited copy with "
        "--harness PATH (or set PEPSITE_TRAINING_HARNESS). Searched:\n  " + searched
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=_non_negative_seed, default=None, help="non-negative run randomization value; defaults to configs/reproducibility.yaml")
    parser.add_argument(
        "--init-model-dir",
        "--init_model_dir",
        default="models/esm2_t33_650M_UR50D_hf_raw",
        help="Raw ESM-2 checkpoint directory (relative to repository root by default).",
    )
    parser.add_argument(
        "--train-csv",
        "--train_csv",
        default="data/train_site_head_5A.csv",
        help="Prepared training CSV.",
    )
    parser.add_argument(
        "--val-csv",
        "--val_csv",
        default="data/val_site_head_5A.csv",
        help="Prepared validation CSV.",
    )
    parser.add_argument("--output-dir", "--output_dir", default=None)
    parser.add_argument(
        "--harness",
        default=None,
        help="Path to the training harness; can also be set via PEPSITE_TRAINING_HARNESS.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", "--num_workers", type=int, default=2)
    parser.add_argument("--num-epochs", "--num_epochs", type=int, default=20)
    parser.add_argument("--max-length", "--max_length", type=int, default=552)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=8)
    parser.add_argument("--eval-batch-size", "--eval_batch_size", type=int, default=16)
    parser.add_argument("--grad-accum", "--grad_accum", type=int, default=1)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=0.01)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable CUDA AMP (not the frozen protocol).")
    parser.set_defaults(fp16=True)
    parser.add_argument("--resume-from-checkpoint", "--resume_from_checkpoint", default=None)
    parser.add_argument("--stop-after-epoch", "--stop_after_epoch", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seed = _default_seed() if args.seed is None else args.seed
    if args.num_epochs <= 0:
        raise SystemExit("--num-epochs must be positive")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")
    harness = _find_harness(args.harness)
    output = _path(args.output_dir or "runs/pepsite/phase1")

    command = [
        sys.executable,
        str(harness),
        "phase1",
        "--seed",
        str(seed),
        "--init_model_dir",
        str(_path(args.init_model_dir)),
        "--train_csv",
        str(_path(args.train_csv)),
        "--val_csv",
        str(_path(args.val_csv)),
        "--output_dir",
        str(output),
        "--device",
        args.device,
        "--num_workers",
        str(args.num_workers),
        "--num_epochs",
        str(args.num_epochs),
        "--max_length",
        str(args.max_length),
        "--batch_size",
        str(args.batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--grad_accum",
        str(args.grad_accum),
        "--weight_decay",
        str(args.weight_decay),
    ]
    command.append("--fp16" if args.fp16 else "--no-fp16")
    if args.resume_from_checkpoint:
        command.extend(["--resume_from_checkpoint", str(_path(args.resume_from_checkpoint))])
    if args.stop_after_epoch is not None:
        command.extend(["--stop_after_epoch", str(args.stop_after_epoch)])

    # Print a shell-independent command record before execution.  The harness
    # itself writes the complete run_config.json and loss artifacts.
    print("[PepSite] Phase 1 launcher")
    print("[PepSite] harness:", harness)
    print("[PepSite] command:", " ".join(map(str, command)))
    completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
