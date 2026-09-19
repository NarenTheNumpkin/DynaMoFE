"""DynaMoFE: Dynamic Mixture of Forensic Experts for Deepfake Detection."""

from .degradation import DegradationSignatureExtractor
from .router import DynamicGatingRouter, DynaMoFEDetector

__all__ = ["DegradationSignatureExtractor", "DynamicGatingRouter", "DynaMoFEDetector"]
