"""Small, model-agnostic pieces used by Asymmetric Flow pixel models.

This package intentionally does not import FastWAM or Wan.  It can therefore be
used by the offline projection fitting job as well as by the training runtime.
"""

from .color import OklabColorEncoder
from .projection import ProjectionArtifact, fit_orthogonal_procrustes
from .velocity import (
    asymflow_calibration,
    asymflow_velocity,
)

__all__ = [
    "OklabColorEncoder",
    "ProjectionArtifact",
    "fit_orthogonal_procrustes",
    "asymflow_calibration",
    "asymflow_velocity",
]
