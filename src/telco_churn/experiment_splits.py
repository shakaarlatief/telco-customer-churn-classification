"""Deterministic repeated stratified split generation for final comparison.

The full final-comparison protocol uses repeated nested cross-validation. This module
only owns deterministic outer-split generation and persistence metadata. Inner tuning
splits are generated later from each outer-training partition using the same seed
derivation convention.

Persisting split definitions is important: a resumed run must use exactly the same
outer training and validation customers as the initial run. Re-generating splits from
memory or from an implicit default random state risks mixing incomparable results.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Iterable, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class OuterSplit:
    """One deterministic outer training-validation partition."""

    repeat_index: int
    fold_index: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    split_hash: str

    def metadata(self) -> dict[str, int | str]:
        """Return lightweight metadata without serializing full index arrays."""
        return {
            "repeat_index": int(self.repeat_index),
            "fold_index": int(self.fold_index),
            "n_train": int(self.train_indices.size),
            "n_validation": int(self.validation_indices.size),
            "split_hash": self.split_hash,
        }


def derive_seed(root_seed: int, *parts: object) -> int:
    """Derive a stable positive 32-bit seed from a root seed and task labels."""
    material = "::".join([str(root_seed), *(str(part) for part in parts)])
    digest = sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _split_hash(
    repeat_index: int,
    fold_index: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> str:
    """Hash split identity and exact integer index arrays."""
    digest = sha256()
    digest.update(
        json.dumps(
            {
                "repeat_index": int(repeat_index),
                "fold_index": int(fold_index),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(np.asarray(train_indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(validation_indices, dtype=np.int64).tobytes())
    return digest.hexdigest()


def make_repeated_stratified_outer_splits(
    y: Sequence[int] | np.ndarray,
    *,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> list[OuterSplit]:
    """Build reproducible repeated stratified outer cross-validation splits.

    Each repeat receives a seed derived from ``random_state`` and its repeat index.
    Within a repeat, every observation appears in exactly one validation fold.
    The returned index arrays are sorted by the splitter's deterministic output and
    are safe to persist as the experiment's split contract.
    """
    y_array = np.asarray(y)
    if y_array.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    if y_array.size == 0:
        raise ValueError("y must not be empty.")
    if n_splits < 2 or n_repeats < 1:
        raise ValueError("n_splits must be at least two and n_repeats at least one.")

    labels, counts = np.unique(y_array, return_counts=True)
    if labels.size < 2:
        raise ValueError("Stratified cross-validation requires at least two classes.")
    if np.min(counts) < n_splits:
        raise ValueError(
            "The smallest class count must be at least n_splits for stratified CV."
        )

    row_positions = np.arange(y_array.size)
    splits: list[OuterSplit] = []

    for repeat_index in range(n_repeats):
        repeat_seed = derive_seed(random_state, "outer", repeat_index)
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=repeat_seed,
        )

        validation_coverage = np.zeros(y_array.size, dtype=int)

        for fold_index, (train_indices, validation_indices) in enumerate(
            splitter.split(row_positions, y_array)
        ):
            train_indices = np.asarray(train_indices, dtype=np.int64)
            validation_indices = np.asarray(validation_indices, dtype=np.int64)

            if np.intersect1d(train_indices, validation_indices).size:
                raise AssertionError("Outer train and validation indices overlap.")

            validation_coverage[validation_indices] += 1
            splits.append(
                OuterSplit(
                    repeat_index=repeat_index,
                    fold_index=fold_index,
                    train_indices=train_indices,
                    validation_indices=validation_indices,
                    split_hash=_split_hash(
                        repeat_index,
                        fold_index,
                        train_indices,
                        validation_indices,
                    ),
                )
            )

        if not np.all(validation_coverage == 1):
            raise AssertionError(
                "Each observation must appear in exactly one validation fold per repeat."
            )

    return splits


def outer_split_manifest(splits: Iterable[OuterSplit]) -> list[dict[str, int | str]]:
    """Return serializable split metadata for a run manifest."""
    return [split.metadata() for split in splits]
