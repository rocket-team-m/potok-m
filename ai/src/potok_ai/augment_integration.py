"""Trainer-facing augmentation bridge for smart windows.

``augment.py`` owns the transforms; this module owns the two wiring rules:

1. every window gets its own seed, a pure function of ``(base_seed, index)`` --
   no global RNG, no draw-order or batching dependence;
2. held-out episodes are never wrapped, so evaluation windows stay byte-identical
   to the source data.

Only element 0 of a dataset item (the window) is transformed: targets,
conditioning and every other field pass through untouched. Build the wrapped
dataset once and reuse it; the seed is keyed to the sample index of that dataset.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import DEFAULT_CONFIG, AugmentConfig, augment
from .data import Episode, SmartWindowDataset

_SEED_MASK = 0x7FFF_FFFF_FFFF_FFFF


def window_seed(base_seed: int, index: int) -> int:
    """Stable 63-bit seed for one window; a pure function of its two inputs.

    blake2b is fixed across processes and Python versions (unlike ``hash``), so
    the same ``(base_seed, index)`` reproduces the same augmentation anywhere.
    """
    seed, slot = int(base_seed), int(index)
    if seed < 0 or slot < 0:
        raise ValueError("base_seed and index must be non-negative")
    digest = hashlib.blake2b(f"{seed}:{slot}".encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & _SEED_MASK


def augment_window(
    window, base_seed: int, index: int, config: AugmentConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """Augment one raw window with its own seed; never touches the input."""
    return augment(window, window_seed(base_seed, index), config).window


class AugmentedDataset(Dataset):
    """Wrap a window dataset, augmenting only the first element of each item."""

    def __init__(
        self, base: Dataset, base_seed: int, config: AugmentConfig = DEFAULT_CONFIG
    ) -> None:
        if int(base_seed) < 0:
            raise ValueError("base_seed must be non-negative")
        self.base = base
        self.base_seed = int(base_seed)
        self.config = config

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        item = self.base[index]
        window, rest = (item[0], item[1:]) if isinstance(item, tuple) else (item, ())
        augmented = augment_window(window, self.base_seed, index, self.config)
        if isinstance(window, torch.Tensor):
            augmented = torch.as_tensor(augmented, dtype=window.dtype, device=window.device)
        return (augmented, *rest) if isinstance(item, tuple) else augmented


def split_window_datasets(
    episodes: list[Episode],
    base_seed: int,
    config: AugmentConfig = DEFAULT_CONFIG,
    *,
    window: int = 50,
    step: int = 1,
    limit: int | None = None,
    eval_limit: int | None = None,
) -> tuple[Dataset, Dataset]:
    """Return ``(train_augmented, held_out_plain)`` smart-window datasets.

    The split is by the authoritative ``Episode.held_out`` flag. Held-out
    episodes go straight into a plain dataset: there is no code path that could
    augment them. A missing side is a loud error, never an empty dataset.
    """
    train = [episode for episode in episodes if not episode.held_out]
    held_out = [episode for episode in episodes if episode.held_out]
    if not train or not held_out:
        raise ValueError("need at least one training and one held-out episode")
    train_ds = SmartWindowDataset(train, window=window, step=step, limit=limit)
    held_out_ds = SmartWindowDataset(held_out, window=window, step=step, limit=eval_limit)
    if not len(train_ds) or not len(held_out_ds):
        raise ValueError("both splits need at least one window")
    return AugmentedDataset(train_ds, base_seed, config), held_out_ds
