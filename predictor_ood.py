"""TMEM120A PLS6 prediction adapter for the Streamlit webtool.

The scientific model is the frozen OpenPLS RDKit+Mordred PLS6 package.
An empirical feature-space OOD guard blocks catastrophic PLS extrapolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rdkit import Chem


@dataclass(frozen=True)
class Prediction:
    score: Optional[float]
    category: str
    canonical_smiles: str
    in_domain: bool
    max_abs_z: float
    ood_threshold: float
    nearest_training_similarity: float
    nearest_training_smiles: str
    nearest_training_docking_score: float
    model_name: str
    model_test_r2: float
    model_test_rmse: float
    model_test_mae: float
    model_test_spearman: float


def _candidate_model_dirs() -> list[Path]:
    here = Path(__file__).resolve().parent
    candidates: list[Path] = []

    env_dir = os.environ.get("PLS6_MODEL_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    candidates.extend(
        [
            here / "model",
            here / "PLS6_FINAL_FOR_WEBTOOLS",
            here / "model_assets" / "PLS6_FINAL_FOR_WEBTOOLS",
        ]
    )
    return candidates


def find_model_dir() -> Path:
    """Locate the frozen PLS6 model package plus OOD guard."""
    required = {
        "pls6_webtool_inference.py",
        "OpenPLS_RDKit_Mordred_PLS6.joblib",
        "preprocessing_parameters.joblib",
        "final_931_feature_manifest.csv",
        "training_reference_607.csv",
        "training_reference_morgan2048.npz",
        "feature_space_OOD_guard.npz",
    }

    for path in _candidate_model_dirs():
        if path.is_dir() and all((path / name).exists() for name in required):
            return path

    searched = "\n".join(f"  - {p}" for p in _candidate_model_dirs())
    raise FileNotFoundError(
        "PLS6 model assets were not found.\n"
        "Set environment variable PLS6_MODEL_DIR to the frozen model folder, "
        "or keep the bundled ./model directory beside the app.\n"
        "The folder must also contain feature_space_OOD_guard.npz.\n"
        f"Searched:\n{searched}"
    )


@lru_cache(maxsize=1)
def _load_predictor() -> tuple[Any, Path, np.ndarray, np.ndarray, float]:
    """Load frozen inference module, model, and empirical OOD guard once."""
    model_dir = find_model_dir()
    module_path = model_dir / "pls6_webtool_inference.py"

    spec = importlib.util.spec_from_file_location("pls6_frozen_inference", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import inference module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    predictor_cls = getattr(module, "PLS6WebPredictor", None)
    if predictor_cls is None:
        raise ImportError("PLS6WebPredictor class not found in frozen inference module.")

    predictor = predictor_cls(str(model_dir))

    guard = np.load(model_dir / "feature_space_OOD_guard.npz")
    train_mean = np.asarray(guard["train_mean"], dtype=float)
    train_std = np.asarray(guard["train_std"], dtype=float)
    threshold = float(np.asarray(guard["threshold"]).reshape(-1)[0])

    if train_mean.shape != (931,) or train_std.shape != (931,):
        raise ValueError("OOD guard feature dimensions do not match the 931-feature PLS6 model.")

    return predictor, model_dir, train_mean, train_std, threshold


def get_model_directory() -> str:
    _, model_dir, _, _, _ = _load_predictor()
    return str(model_dir)


def predict_docking_score(mol: Chem.Mol, target_name: str = "TMEM120A") -> Prediction:
    """Predict TMEM120A score only when the query is inside empirical feature space.

    The guard is not a calibrated confidence estimator. Its sole purpose is to
    prevent pathological linear extrapolation (e.g. enormous PLS scores for
    molecules whose descriptors are far outside the 607-compound training set).
    """
    if target_name.strip().upper() != "TMEM120A":
        raise ValueError("This trained QSAR model is specific to TMEM120A.")

    predictor, _, train_mean, train_std, threshold = _load_predictor()
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    parsed_mol, canonical = predictor._parse_smiles(smiles)
    final_x = predictor._build_final_features(parsed_mol)
    z = np.abs((final_x - train_mean) / train_std)
    max_abs_z = float(np.max(z))
    in_domain = bool(max_abs_z <= threshold)

    # Structural context is still useful even when the QSAR score is suppressed.
    nearest = predictor._nearest_training_analogue(parsed_mol)

    if in_domain:
        raw_score = float(predictor.model.predict(final_x.reshape(1, -1)).ravel()[0])
        # Fixed project category thresholds. Keep these synchronized with
        # the frozen model metadata; no dynamic imports are used here.
        if raw_score < -6.45:
            category = "VERY_GOOD"
        elif raw_score < -5.70:
            category = "GOOD"
        elif raw_score < -4.945:
            category = "MEDIUM"
        else:
            category = "POOR"
        score: Optional[float] = raw_score
    else:
        score = None
        category = "OUT_OF_DOMAIN"

    return Prediction(
        score=score,
        category=category,
        canonical_smiles=str(canonical),
        in_domain=in_domain,
        max_abs_z=max_abs_z,
        ood_threshold=threshold,
        nearest_training_similarity=float(nearest["nearest_training_similarity"]),
        nearest_training_smiles=str(nearest["nearest_training_smiles"]),
        nearest_training_docking_score=float(nearest["nearest_training_docking_score"]),
        model_name="OpenPLS_RDKit_Mordred_PLS6",
        model_test_r2=0.5592,
        model_test_rmse=0.6328,
        model_test_mae=0.5055,
        model_test_spearman=0.6643,
    )
