"""Portable atomic-replacement helpers for durable experiment artifacts.

Windows can temporarily deny replacement of a destination file when a short-lived reader,
indexer, antivirus scanner, or editor handle overlaps an ``os.replace`` call. The helpers
in this module retain atomic replacement semantics while retrying only sharing/access
errors that are plausibly transient. They deliberately do not retry unrelated filesystem
failures indefinitely.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path
import time
from typing import Iterable


# The delay sequence gives a transient Windows handle a little more than six seconds to
# disappear while keeping ordinary successful writes immediate. Tests may temporarily set
# this module constant to zero-delay values; production code should retain the default.
DEFAULT_REPLACE_RETRY_DELAYS_SECONDS: tuple[float, ...] = (
    0.05,
    0.10,
    0.20,
    0.40,
    0.80,
    1.60,
    3.20,
)

_RETRYABLE_ERRNOS = frozenset({errno.EACCES, errno.EPERM, errno.EBUSY})
_RETRYABLE_WINDOWS_ERRORS = frozenset({5, 32, 33})


def is_retryable_atomic_replace_error(error: OSError) -> bool:
    """Return whether an atomic-replacement failure may be a temporary file lock.

    ``PermissionError`` is included even when a caller or test double does not populate
    ``errno`` or ``winerror``. Other retryable cases cover POSIX access/busy errors and
    Windows access-denied, sharing-violation, and lock-violation codes. Disk-full,
    path-not-found, serialization, and programming errors remain immediate failures.
    """
    if isinstance(error, PermissionError):
        return True
    if getattr(error, "errno", None) in _RETRYABLE_ERRNOS:
        return True
    return getattr(error, "winerror", None) in _RETRYABLE_WINDOWS_ERRORS


def _normalise_retry_delays(retry_delays: Iterable[float] | None) -> tuple[float, ...]:
    """Validate and materialize the bounded retry-delay sequence."""
    values = (
        DEFAULT_REPLACE_RETRY_DELAYS_SECONDS
        if retry_delays is None
        else tuple(float(value) for value in retry_delays)
    )
    if any(value < 0 for value in values):
        raise ValueError("retry delays must be non-negative.")
    return tuple(float(value) for value in values)


def replace_file_with_retry(
    source_path: Path,
    destination_path: Path,
    *,
    retry_delays: Iterable[float] | None = None,
) -> None:
    """Atomically replace a destination file, retrying only bounded lock-like failures.

    The caller must create and synchronize ``source_path`` in the destination directory
    before calling this function. On success, ``os.replace`` provides the same atomic
    replacement contract as before. On a persistent retryable failure, the final original
    ``OSError`` is re-raised so essential artifact writers can preserve their durability
    contract and report an actionable root cause.
    """
    source = Path(source_path)
    destination = Path(destination_path)
    delays = _normalise_retry_delays(retry_delays)

    for delay in delays:
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if not is_retryable_atomic_replace_error(error):
                raise
            time.sleep(delay)

    # One final attempt is made after the final delay. Re-raise its original exception
    # unchanged so callers retain the platform-specific errno/winerror diagnostics.
    os.replace(source, destination)