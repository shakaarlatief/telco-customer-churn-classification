"""Immutable protocol specifications and run fingerprints for final comparison.

This module defines the information that must remain fixed for a resumable
training-only model-comparison run. The central design principle is that a run is
identified by more than a directory name. It is identified by the exact evaluation
protocol, the development data, the split policy, and the declared candidate set.

The module intentionally does not include model-specific search spaces. Those are
added later through candidate-registry code. Keeping the protocol and fingerprint
layer independent from estimator libraries makes it possible to validate resume
safety before a computationally expensive model is fitted.

A resume must be rejected when the protocol or data fingerprint differs. Continuing
a run after such a change would merge results from incomparable experiments and
would make the final selection evidence ambiguous.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import importlib.metadata
import json
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class ProtocolValidationError(ValueError):
    """Raised when a protocol is structurally invalid."""


def _canonical_json(payload: Mapping[str, Any]) -> str:
    """Return a deterministic JSON representation suitable for hashing.

    Sorting keys and using compact separators ensures that equivalent protocol
    dictionaries receive identical byte representations. The function is
    deliberately strict: unsupported objects must be converted to ordinary
    JSON-compatible values before entering a frozen protocol.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _sha256_text(text: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentProtocol:
    """Immutable top-level description of one final-comparison experiment.

    Parameters
    ----------
    protocol_id:
        Human-readable identifier, for example
        ``telco_final_comparison_protocol_v1``.
    version:
        Explicit protocol revision. A substantive change to candidates, splits,
        metrics, or search policy should create a new version rather than mutate
        results from an older version.
    candidate_ids:
        Unique identifiers of candidate procedures included in the run.
    primary_metric:
        Exact implementation-level name of the selection metric. The current
        project protocol uses ``average_precision``.
    outer_n_splits, outer_n_repeats:
        Repeated outer cross-validation structure. The full protocol uses five
        folds and ten repeats; the infrastructure smoke test uses smaller values.
    inner_n_splits:
        Number of inner folds used by the later tuning objective. It is included
        here because it changes the tuned-procedure estimand.
    random_state:
        Root seed from which deterministic split and task seeds are derived.
    metadata:
        JSON-compatible supplementary fields. This field can record the
        feature-policy version, imbalance-policy version, package policy, and
        search-budget version without changing the core class shape.

    Notes
    -----
    The class represents a *protocol*, not a fitted estimator. It deliberately
    excludes learned hyperparameters, validation scores, and test-set information.
    """

    protocol_id: str
    version: str
    candidate_ids: tuple[str, ...]
    primary_metric: str
    outer_n_splits: int
    outer_n_repeats: int
    inner_n_splits: int
    random_state: int = 42
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate immutable protocol invariants."""
        if not self.protocol_id.strip():
            raise ProtocolValidationError("protocol_id must not be empty.")
        if not self.version.strip():
            raise ProtocolValidationError("version must not be empty.")
        if not self.candidate_ids:
            raise ProtocolValidationError("candidate_ids must contain at least one entry.")
        if any(not candidate.strip() for candidate in self.candidate_ids):
            raise ProtocolValidationError("candidate_ids cannot contain empty identifiers.")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ProtocolValidationError("candidate_ids must be unique.")
        if not self.primary_metric.strip():
            raise ProtocolValidationError("primary_metric must not be empty.")
        for field_name, value in (
            ("outer_n_splits", self.outer_n_splits),
            ("inner_n_splits", self.inner_n_splits),
        ):
            if value < 2:
                raise ProtocolValidationError(
                    f"{field_name} must be at least two for stratified cross-validation."
                )
        if self.outer_n_repeats < 1:
            raise ProtocolValidationError(
                "outer_n_repeats must be at least one for repeated cross-validation."
            )
        if not isinstance(self.random_state, int):
            raise ProtocolValidationError("random_state must be an integer.")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible representation of the protocol."""
        payload = asdict(self)
        payload["candidate_ids"] = list(self.candidate_ids)
        payload["metadata"] = dict(self.metadata)
        return payload

    @property
    def fingerprint(self) -> str:
        """Return the immutable SHA-256 identity of the protocol."""
        return _sha256_text(_canonical_json(self.to_dict()))


def make_dataframe_fingerprint(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray | Sequence[Any],
) -> dict[str, Any]:
    """Create a content and schema fingerprint for a development dataset.

    The fingerprint combines:

    * row count and column order;
    * feature and target dtypes;
    * the original index;
    * deterministic pandas row hashes.

    It is intentionally designed for resume safety rather than cryptographic
    provenance in an adversarial environment. Reordering rows, altering target
    labels, changing feature values, changing column names, or changing dtypes
    changes the fingerprint and blocks an unsafe resume.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame for a stable data fingerprint.")

    y_series = pd.Series(y, index=X.index, name="__target__")
    if len(X) != len(y_series):
        raise ValueError("X and y must have equal row counts.")
    if y_series.index.equals(X.index) is False:
        raise ValueError("X and y must share the same row index and ordering.")

    combined = X.copy()
    combined["__target__"] = y_series

    row_hashes = pd.util.hash_pandas_object(
        combined,
        index=True,
        categorize=True,
    ).to_numpy(dtype=np.uint64, copy=False)

    schema = {
        "feature_columns": list(X.columns),
        "feature_dtypes": {column: str(dtype) for column, dtype in X.dtypes.items()},
        "target_name": str(y_series.name),
        "target_dtype": str(y_series.dtype),
        "n_rows": int(len(X)),
        "index_name": str(X.index.name),
    }

    digest = sha256()
    digest.update(_canonical_json(schema).encode("utf-8"))
    digest.update(row_hashes.tobytes())

    return {
        "sha256": digest.hexdigest(),
        "schema": schema,
    }


def make_environment_fingerprint(
    package_names: Sequence[str] = (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "threadpoolctl",
    ),
) -> dict[str, Any]:
    """Record the runtime environment relevant to reproducibility.

    Package versions are not used as a substitute for a lock file. They are
    recorded so a resumed run can detect that a dependency changed between the
    start and continuation of a long experiment.
    """
    packages: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            packages[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            packages[package_name] = None

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "sha256": _sha256_text(
            _canonical_json(
                {
                    "python_version": platform.python_version(),
                    "python_implementation": platform.python_implementation(),
                    "platform": platform.platform(),
                    "packages": packages,
                }
            )
        ),
    }
