"""Reproducible retraining entry point and registry hand-off."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.registry import ModelRegistry
from src.train import train


def retrain(
    artifact_dir: str | Path = "artifacts",
    registry_dir: str | Path = "artifacts/registry",
    seed: int = 11,
    promote: bool = False,
) -> dict:
    artifact_dir = Path(artifact_dir)
    metrics = train(output=artifact_dir, seed=seed)
    registry = ModelRegistry(registry_dir)
    record = registry.register_candidate(
        artifact_dir / "credit_risk_model.joblib",
        metrics,
        run_id=f"synthetic-seed-{seed}",
    )
    if promote:
        record = registry.promote_to_production(record["version"])
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and register a candidate model")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--registry", default="artifacts/registry")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    print(json.dumps(retrain(args.output, args.registry, args.seed, args.promote), indent=2))
