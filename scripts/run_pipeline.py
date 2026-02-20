"""CLI entrypoint for running the synthetic data pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline import run_pipeline_from_config


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run synthetic tabular data pipeline with CTGAN and TVAE.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--data-path",
        required=False,
        default=None,
        help="Optional CSV data path override.",
    )
    return parser


def main() -> None:
    """Run the configured pipeline."""
    args = build_parser().parse_args()

    summary = run_pipeline_from_config(
        config_path=Path(args.config),
        data_path_override=Path(args.data_path) if args.data_path else None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
