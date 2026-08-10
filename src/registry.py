"""Minimal file-backed model registry with MLflow-style lifecycle stages.

The local backend keeps the demo runnable without a tracking server. Its manifest
has the same important governance concepts as a production registry: immutable
artifact digest, metrics, candidate/production stages and an explicit promotion
action. A team can replace this class with an MLflow client behind the same calls.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ModelRegistry:
    def __init__(self, root: str | Path = "artifacts/registry") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "registry.json"

    def _read(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"models": [], "production_version": None}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _write(self, data: dict[str, Any]) -> None:
        self.manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def register_candidate(
        self,
        artifact_path: str | Path,
        metrics: dict[str, Any],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Register a trained artifact as a candidate version."""

        artifact = Path(artifact_path)
        if not artifact.exists():
            raise FileNotFoundError(artifact)
        data = self._read()
        digest = self._sha256(artifact)
        version = f"credit-risk-{digest[:12]}"
        record = {
            "version": version,
            "stage": "candidate",
            "artifact": str(artifact),
            "sha256": digest,
            "metrics": metrics,
            "run_id": run_id or version,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        data["models"] = [
            item for item in data["models"] if item["version"] != version
        ] + [record]
        self._write(data)
        return record

    def promote_to_production(self, version: str) -> dict[str, Any]:
        """Promote a candidate and archive the previous production version."""

        data = self._read()
        selected = next(
            (item for item in data["models"] if item["version"] == version), None
        )
        if selected is None:
            raise KeyError(f"unknown model version: {version}")
        for item in data["models"]:
            if item["stage"] == "production":
                item["stage"] = "archived"
        selected["stage"] = "production"
        data["production_version"] = version
        self._write(data)
        return selected

    def production(self) -> dict[str, Any] | None:
        data = self._read()
        version = data.get("production_version")
        return next(
            (item for item in data["models"] if item["version"] == version), None
        )
