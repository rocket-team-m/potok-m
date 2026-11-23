"""Trainer-facing target contract for POTOKSIM validation data.

``potok_ai.data`` parses v1/v2/v3 and keeps every raw target column. This
module adds the distinction a trainer must not get wrong:

* v3 writes ``g_accel_y`` and ``g_accel_x`` (m/s^2, world frame). The
  authoritative fin command is ``[g_accel_y, -g_accel_x]`` in the trainer's
  ``[roll, pitch]`` seam order (``sim/README.md``).
* ``g_roll`` / ``g_pitch`` (``g_theta``) are the angle-domain diagnostic target.
* v1/v2 have no acceleration column: they expose only the diagnostic. Asking
  them for an authoritative command is a loud error, never a substitution.
  ``training_target`` is the single trainer-facing resolver: v3 returns the
  fin command, and fitting g_theta is possible only through the explicit
  ``allow_diagnostic_target`` escape for angle-domain diagnostic training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import Episode, read_validation


@dataclass
class EpisodeTargets:
    """Target record for one episode, with its provenance kept explicit."""

    version: int
    g_theta: np.ndarray  # (T, 2) diagnostic [g_roll, g_pitch], rad
    command: np.ndarray | None  # (T, 2) authoritative [g_accel_y, -g_accel_x], m/s^2

    @property
    def authoritative(self) -> bool:
        """True only when the source file carried the v3 acceleration command."""
        return self.command is not None

    @property
    def training_target(self) -> np.ndarray:
        """The command a trainer must fit; loud when the file is v1/v2."""
        if self.command is None:
            raise ValueError(
                f"validation version {self.version} has no authoritative g_accel command; "
                "g_theta is diagnostic only, upgrade the data to v3"
            )
        return self.command


def authoritative_command(episode: Episode) -> np.ndarray | None:
    """Authoritative ``[g_accel_y, -g_accel_x]`` when present, else ``None``.

    The transform is the one in ``Episode.command_target``; the guard is the
    point of this module: a v1/v2 episode has no command to return, so it
    returns ``None`` instead of silently promoting the diagnostic angle target.
    """
    if episode.g_accel is None:
        return None
    return episode.command_target()


def training_target(
    episode: Episode, *, allow_diagnostic_target: bool = False
) -> np.ndarray:
    """The target a trainer fits: authoritative command, or explicit diagnostic.

    v3 returns the fin command. For v1/v2 there is no command: the call is loud
    unless ``allow_diagnostic_target`` explicitly requests angle-domain
    diagnostic training, in which case the legacy g_theta columns are fitted.
    """
    return episode.command_target(allow_diagnostic_target=allow_diagnostic_target)


def authoritative_target(episode: Episode) -> np.ndarray:
    """Authoritative fin command ``(T, 2)``; loud error for v1/v2 episodes."""
    return training_target(episode)


def diagnostic_target(episode: Episode) -> np.ndarray:
    """Angle-domain ``g_theta`` ``(T, 2)`` - diagnostic, never a fin command."""
    return episode.g_theta()


def read_targets(path: str | Path) -> list[EpisodeTargets]:
    """Read validation v1/v2/v3, returning one target record per episode.

    The version/target-width consistency check lives in ``read_validation``;
    a v3 file is never reinterpreted as v2 here.
    """
    data = read_validation(path)
    return [
        EpisodeTargets(
            version=data.header.version,
            g_theta=diagnostic_target(episode),
            command=authoritative_command(episode),
        )
        for episode in data.episodes
    ]
