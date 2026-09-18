#!/usr/bin/env python3
"""Generate PepSite candidate scores and route-specific XR proposals.

This is the public, checkpoint-backed inference entry point. It consumes a
matching Phase-2 checkpoint and a label-free context CSV with columns
``query_id,receptor_sequence,peptide_length``.  It writes
``candidate_scores.csv``, ``xr_proposals.csv``, and gate-approved
``xr_selected.csv`` together with the candidate bank and an immutable
manifest.  It reads no machine-local experiment tree or hidden test label.

Example (CPU smoke run)::

    SEED=<seed>
    python scripts/infer_candidates.py \
      --phase2-model models/PepSite \
      --context data/test_context.csv \
      --output-dir runs/pepsite/inference \
      --device cpu

Use ``--limit`` only for a local smoke test.  The release defaults are 64
proposals, full-vocabulary top-k=3, temperature 1.0, and Site threshold 0.5.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from pepsite.inference import (  # noqa: E402
    DEFAULT_PROPOSAL_COUNT,
    DEFAULT_SITE_THRESHOLD,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    run_inference,
)
from pepsite.constants import COMPOSITION_L1_MAX, PPL_RATIO_MAX  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-model", "--model", required=True, help="local Phase-2 checkpoint directory")
    parser.add_argument("--context", "--test-context", required=True, help="label-free CSV: query_id,receptor_sequence,peptide_length")
    parser.add_argument("--output-dir", required=True, help="new, empty output directory")
    parser.add_argument("--seed", type=int, default=None, help="non-negative run randomization value; defaults to configs/reproducibility.yaml")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--proposal-count", type=int, default=DEFAULT_PROPOSAL_COUNT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--site-threshold", type=float, default=DEFAULT_SITE_THRESHOLD)
    parser.add_argument("--ppl-ratio-max", type=float, default=PPL_RATIO_MAX)
    parser.add_argument("--composition-l1-max", type=float, default=COMPOSITION_L1_MAX)
    parser.add_argument("--limit", type=int, default=None, help="smoke limit; use without this option for the full context")
    parser.add_argument("--expected-rows", type=int, default=None, help="optional exact context row count (for example 202)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seed is None:
        repro = REPOSITORY / "configs" / "reproducibility.yaml"
        for line in repro.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("random_seed:"):
                args.seed = int(line.split(":", 1)[1].strip())
                break
        if args.seed is None:
            parser.error("configs/reproducibility.yaml does not define random_seed")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.proposal_count <= 0:
        parser.error("--proposal-count must be positive")
    if args.top_k != DEFAULT_TOP_K:
        parser.error("the release protocol fixes --top-k=3")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if not 0.0 <= args.site_threshold <= 1.0:
        parser.error("--site-threshold must be in [0, 1]")
    if args.ppl_ratio_max <= 0:
        parser.error("--ppl-ratio-max must be positive")
    if args.composition_l1_max < 0:
        parser.error("--composition-l1-max must be non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    # Restricting the context before handing it to the inference library keeps
    # the smoke path label-free while preserving the same row validation.
    context_path = Path(args.context).expanduser().resolve()
    temporary_context: Path | None = None
    if args.limit is not None:
        import pandas as pd

        context = pd.read_csv(context_path, dtype=str, keep_default_na=False)
        if args.limit > len(context):
            parser.error(f"--limit={args.limit} exceeds context rows={len(context)}")
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", prefix="pepsite_context_", delete=False)
        temporary_context = Path(handle.name)
        context.iloc[: args.limit].to_csv(handle, index=False)
        handle.close()
        context_path = temporary_context
        expected_rows = args.limit
    else:
        expected_rows = args.expected_rows

    try:
        manifest = run_inference(
            model_dir=args.phase2_model,
            context_path=context_path,
            output_dir=args.output_dir,
            seed=args.seed,
            device=args.device,
            dtype=args.dtype,
            proposal_count=args.proposal_count,
            top_k=args.top_k,
            temperature=args.temperature,
            ppl_ratio_max=args.ppl_ratio_max,
            composition_l1_max=args.composition_l1_max,
            site_threshold=args.site_threshold,
            expected_rows=expected_rows,
        )
    finally:
        if temporary_context is not None:
            temporary_context.unlink(missing_ok=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
