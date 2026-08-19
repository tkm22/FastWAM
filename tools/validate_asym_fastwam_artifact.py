"""Validate a LIBERO Asym-FastWAM projection artifact."""

from __future__ import annotations

import argparse

from asymflow.projection import (
    ProjectionArtifact,
    validate_projection_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    parser.add_argument("--require-task-balanced-100k", action="store_true")
    args = parser.parse_args()
    artifact = ProjectionArtifact.load(args.artifact)
    validate_projection_artifact(
        artifact, require_task_balanced_100k=args.require_task_balanced_100k
    )
    print(f"artifact={args.artifact}")
    print(f"metadata={artifact.metadata}")


if __name__ == "__main__":
    main()
