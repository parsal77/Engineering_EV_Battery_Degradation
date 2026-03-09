"""Utilities for downloading and parsing NASA PCoE battery `.mat` files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
from scipy.io import loadmat

DEFAULT_BATTERY_IDS: tuple[str, ...] = ("B0005", "B0006", "B0007", "B0018")
MIRROR_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "anirudhkhatry/SOH-prediction-using-NASA-Dataset/master"
)


@dataclass(slots=True)
class ParsedCycle:
    """Container for one parsed MATLAB cycle record.

    Parameters
    ----------
    battery_id : str
        Battery identifier (for example, ``B0005``).
    cycle_index : int
        Original cycle index in file order, 1-based.
    cycle_type : str
        Cycle type (``charge``, ``discharge``, or ``impedance``).
    ambient_temperature : float
        Ambient temperature recorded for the cycle (degC).
    start_time : datetime
        Start timestamp of the cycle.
    data : dict[str, Any]
        Mapping of cycle measurement arrays/scalars.
    """

    battery_id: str
    cycle_index: int
    cycle_type: str
    ambient_temperature: float
    start_time: datetime
    data: dict[str, Any]


def project_root() -> Path:
    """Return the project root path.

    Returns
    -------
    pathlib.Path
        Absolute root path inferred from the ``src`` package location.
    """

    return Path(__file__).resolve().parents[1]


def _flatten_numeric(value: Any) -> np.ndarray | float:
    """Convert MATLAB values to Python numeric types.

    Parameters
    ----------
    value : Any
        MATLAB-loaded value.

    Returns
    -------
    numpy.ndarray or float
        1D float array for vector values, float for scalar values.
    """

    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        arr = np.real(arr)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return np.asarray(arr, dtype=float).reshape(-1)


def _parse_cycle_time(matlab_time_vec: np.ndarray) -> datetime:
    """Parse MATLAB date vector into ``datetime``.

    Parameters
    ----------
    matlab_time_vec : numpy.ndarray
        Array like ``[year, month, day, hour, minute, second]``.

    Returns
    -------
    datetime.datetime
        Parsed timestamp.
    """

    values = np.asarray(matlab_time_vec).astype(float).reshape(-1)
    second = int(round(values[5]))
    second = max(0, min(59, second))
    return datetime(
        int(values[0]),
        int(values[1]),
        int(values[2]),
        int(values[3]),
        int(values[4]),
        second,
    )


def download_battery_mat_files(
    output_dir: Path,
    battery_ids: tuple[str, ...] = DEFAULT_BATTERY_IDS,
    base_url: str = MIRROR_BASE_URL,
    timeout: int = 60,
) -> list[Path]:
    """Download required NASA battery MAT files.

    Parameters
    ----------
    output_dir : pathlib.Path
        Target folder where files will be saved.
    battery_ids : tuple[str, ...], optional
        Battery IDs to download.
    base_url : str, optional
        Base URL for raw MAT files.
    timeout : int, optional
        Request timeout in seconds.

    Returns
    -------
    list[pathlib.Path]
        Paths to downloaded/existing MAT files.

    Raises
    ------
    RuntimeError
        If any requested file cannot be downloaded.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for battery_id in battery_ids:
        target_path = output_dir / f"{battery_id}.mat"
        if target_path.exists() and target_path.stat().st_size > 0:
            saved_paths.append(target_path)
            continue

        url = f"{base_url}/{battery_id}.mat"
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download {battery_id}.mat from {url} (status={response.status_code})."
            )

        target_path.write_bytes(response.content)
        saved_paths.append(target_path)

    return saved_paths


def parse_battery_cycles(
    mat_path: Path, battery_id: str | None = None
) -> list[ParsedCycle]:
    """Parse all cycles from one battery MAT file.

    Parameters
    ----------
    mat_path : pathlib.Path
        Path to battery MAT file.
    battery_id : str | None, optional
        Optional explicit battery ID. Defaults to ``mat_path.stem``.

    Returns
    -------
    list[ParsedCycle]
        Parsed cycle records in original order.
    """

    resolved_battery_id = battery_id or mat_path.stem
    mat = loadmat(mat_path)

    if resolved_battery_id not in mat:
        available_keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(
            f"Battery key '{resolved_battery_id}' not found in {mat_path}. "
            f"Available keys: {available_keys}"
        )

    raw_cycles = mat[resolved_battery_id][0, 0]["cycle"][0]
    parsed_cycles: list[ParsedCycle] = []

    for idx, raw_cycle in enumerate(raw_cycles, start=1):
        cycle_type = str(raw_cycle["type"][0]).strip().lower()
        ambient_temperature = float(raw_cycle["ambient_temperature"][0, 0])
        start_time = _parse_cycle_time(raw_cycle["time"][0])

        data_struct = raw_cycle["data"][0, 0]
        parsed_data: dict[str, Any] = {}
        for field_name in data_struct.dtype.names or ():
            parsed_data[field_name] = _flatten_numeric(data_struct[field_name])

        parsed_cycles.append(
            ParsedCycle(
                battery_id=resolved_battery_id,
                cycle_index=idx,
                cycle_type=cycle_type,
                ambient_temperature=ambient_temperature,
                start_time=start_time,
                data=parsed_data,
            )
        )

    return parsed_cycles


def load_all_cycles(
    raw_dir: Path, battery_ids: tuple[str, ...] = DEFAULT_BATTERY_IDS
) -> dict[str, list[ParsedCycle]]:
    """Load cycles for all specified batteries.

    Parameters
    ----------
    raw_dir : pathlib.Path
        Directory containing battery MAT files.
    battery_ids : tuple[str, ...], optional
        Battery IDs to parse.

    Returns
    -------
    dict[str, list[ParsedCycle]]
        Mapping of battery ID to list of parsed cycles.
    """

    all_cycles: dict[str, list[ParsedCycle]] = {}
    for battery_id in battery_ids:
        mat_path = raw_dir / f"{battery_id}.mat"
        all_cycles[battery_id] = parse_battery_cycles(
            mat_path=mat_path, battery_id=battery_id
        )
    return all_cycles
