"""Extract a small STRATA/SCREAM boundary profile set from the public Zarr store.

Only a few columns and time steps are read; the full 19 TB store is never downloaded.
Run with the dependencies from uv:

    uv run --no-project --with xarray --with zarr --with fsspec \
        --with huggingface_hub --with numpy \
        python data_prep/download_boundary.py --out DIR --steps 12 --column 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

STORE = "hf://datasets/nvidia/STRATA-SCREAM-sdy1"
VARIABLES = ("U", "V", "PotentialTemperature", "ps", "qv", "z_mid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--column", type=int, default=0)
    parser.add_argument("--prefix", default="scream")
    parser.add_argument("--vars", nargs="*", default=None)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ds = xr.open_zarr(STORE)
    wanted = args.vars or VARIABLES
    present = [name for name in wanted if name in ds.data_vars]
    missing = [name for name in VARIABLES if name not in ds.data_vars]
    if missing:
        print(f"missing variables (skipped): {missing}")

    selection = {name: ds[name].isel(time=slice(0, args.steps), ncol=args.column).load() for name in present}
    payload = {"source": np.array(STORE), "column": np.array(args.column)}
    for name, data in selection.items():
        payload[name] = data.values
    if "lev" in ds.coords:
        payload["lev"] = ds.coords["lev"].values

    target = args.out / f"{args.prefix}_boundary.npz"
    np.savez_compressed(target, **payload)
    print(f"saved {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
