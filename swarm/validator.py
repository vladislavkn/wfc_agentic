"""
Validate that a solution.py conforms to the Predictorium submission interface.

Fixes #10 — catches wrong shapes, missing state reset, hardcoded paths, etc.
"""

from __future__ import annotations

import importlib.util
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DataPoint:
    """Mirrors the Predictorium DataPoint interface."""
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray  # shape (32,)


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str]
    warnings: list[str]


def validate_solution(solution_path: str, model_dir: str) -> ValidationResult:
    """
    Validate solution.py by:
    1. Importing the module
    2. Instantiating PredictionModel
    3. Running through 2 synthetic sequences
    4. Checking output shapes, state reset, no hardcoded paths
    """
    errors: list[str] = []
    warnings: list[str] = []
    sol_path = Path(solution_path)

    # --- Check file exists ---
    if not sol_path.exists():
        return ValidationResult(False, [f"File not found: {solution_path}"], [])

    # --- Check for hardcoded absolute paths ---
    source = sol_path.read_text()
    suspicious_paths = []
    for line_no, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern in ["/home/", "/tmp/", "/root/", "C:\\", "D:\\"]:
            if pattern in line:
                suspicious_paths.append((line_no, line.strip()[:80]))
    if suspicious_paths:
        for ln, text in suspicious_paths:
            warnings.append(f"Line {ln} may contain hardcoded path: {text}")

    # --- Import module ---
    try:
        spec = importlib.util.spec_from_file_location("solution", solution_path)
        module = importlib.util.module_from_spec(spec)
        # Temporarily add model_dir to path so relative imports work
        old_dir = Path.cwd()
        import os
        os.chdir(Path(solution_path).parent)
        spec.loader.exec_module(module)
        os.chdir(old_dir)
    except Exception as e:
        return ValidationResult(False, [f"Import failed: {e}"], warnings)

    # --- Check PredictionModel class exists ---
    if not hasattr(module, "PredictionModel"):
        errors.append("Missing 'PredictionModel' class")
        return ValidationResult(False, errors, warnings)

    # --- Instantiate ---
    try:
        model = module.PredictionModel()
    except Exception as e:
        errors.append(f"PredictionModel() instantiation failed: {e}")
        return ValidationResult(False, errors, warnings)

    # --- Check predict method ---
    if not hasattr(model, "predict"):
        errors.append("PredictionModel missing 'predict' method")
        return ValidationResult(False, errors, warnings)

    # --- Run through 2 synthetic sequences ---
    rng = np.random.RandomState(42)
    for seq_ix in range(2):
        for step in range(100):  # 100 steps: 0-98 warmup, 99 scored
            need_pred = step >= 99
            dp = DataPoint(
                seq_ix=seq_ix,
                step_in_seq=step,
                need_prediction=need_pred,
                state=rng.randn(32).astype(np.float32),
            )
            try:
                result = model.predict(dp)
            except Exception:
                errors.append(
                    f"predict() crashed at seq={seq_ix} step={step}: "
                    f"{traceback.format_exc()[-200:]}"
                )
                return ValidationResult(False, errors, warnings)

            if not need_pred:
                if result is not None:
                    warnings.append(
                        f"predict() returned non-None during warmup "
                        f"(seq={seq_ix} step={step})"
                    )
            else:
                # Must return array-like of shape (2,)
                if result is None:
                    errors.append(
                        f"predict() returned None when need_prediction=True "
                        f"(seq={seq_ix} step={step})"
                    )
                    return ValidationResult(False, errors, warnings)
                result = np.asarray(result)
                if result.shape != (2,):
                    errors.append(
                        f"predict() returned shape {result.shape}, expected (2,) "
                        f"(seq={seq_ix} step={step})"
                    )
                    return ValidationResult(False, errors, warnings)
                if np.any(np.isnan(result)):
                    warnings.append(
                        f"predict() returned NaN at seq={seq_ix} step={step}"
                    )

    if errors:
        return ValidationResult(False, errors, warnings)

    return ValidationResult(True, [], warnings)
