from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat


PRIMARY_MANIFOLD_DATA_DIR = Path(
    "/Users/mattlowery/Desktop/code/matlab_master_rbffdcodes/Manifolds/OperatorLearning/manifold_datasets"
)
SECONDARY_MANIFOLD_DATA_DIR = Path("/projects/bfel/mlowery/manifold_datasets")


def _resolve_manifold_data_dir() -> Path:
    override = os.environ.get("GNOT_MANIFOLD_DATA_DIR")
    if override:
        return Path(override)
    if PRIMARY_MANIFOLD_DATA_DIR.exists():
        return PRIMARY_MANIFOLD_DATA_DIR
    return SECONDARY_MANIFOLD_DATA_DIR


MANIFOLD_DATA_DIR = _resolve_manifold_data_dir()

ADRSHEAR_GRID_PATH = MANIFOLD_DATA_DIR / "TimeVaryingADRShear_grids.mat"

STATIONARY_DATASETS = {
    "poisson_sphere_3d": ("poisson", "sphere"),
    "poisson_torus_3d": ("poisson", "torus"),
    "nlpoisson_sphere_3d": ("nlpoisson", "sphere"),
    "nlpoisson_torus_3d": ("nlpoisson", "torus"),
}

ADRSHEAR_DATASETS = {"ADRSHEAR_3d", "adrshear_3d"}


def is_manifold_dataset(name: str) -> bool:
    return name in STATIONARY_DATASETS or name in ADRSHEAR_DATASETS


def build_manifold_datasets(args):
    if args.dataset in STATIONARY_DATASETS:
        problem, surf = STATIONARY_DATASETS[args.dataset]
        return _build_stationary_dataset(problem=problem, surf=surf, npoints=args.npoints, train_num=args.train_num, test_num=args.test_num)
    if args.dataset in ADRSHEAR_DATASETS:
        return _build_adrshear_dataset(train_num=args.train_num, test_num=args.test_num)
    raise ValueError(f"unsupported manifold dataset: {args.dataset}")


def _build_stationary_dataset(problem: str, surf: str, npoints, train_num, test_num):
    if npoints == "all":
        raise ValueError("stationary manifold datasets require a concrete --npoints value")

    path = MANIFOLD_DATA_DIR / f"{problem}_{surf}_{npoints}_10000.mat"
    data = loadmat(path)
    fields = np.asarray(data["fs"], dtype=np.float32).T
    targets = np.asarray(data["us"], dtype=np.float32).T
    grid = np.asarray(data["x"], dtype=np.float32)
    n_train_total = int(np.asarray(data["N_train"]).squeeze())
    n_test_total = int(np.asarray(data["N_test"]).squeeze())

    train_fields = fields[:n_train_total]
    train_targets = targets[:n_train_total]
    test_fields = fields[-n_test_total:]
    test_targets = targets[-n_test_total:]

    train_fields, train_targets = _take_first(train_fields, train_targets, train_num)
    test_fields, test_targets = _take_last(test_fields, test_targets, test_num)

    return (
        _build_samples(source_grid=grid, target_grid=grid, fields=train_fields, targets=train_targets),
        _build_samples(source_grid=grid, target_grid=grid, fields=test_fields, targets=test_targets),
    )


def _build_adrshear_dataset(train_num, test_num):
    data = loadmat(MANIFOLD_DATA_DIR / "TimeVaryingADRShear.mat")
    fields = np.asarray(data["fs_all"], dtype=np.float32).T
    targets = np.asarray(data["us_all"], dtype=np.float32).T

    grid_data = loadmat(ADRSHEAR_GRID_PATH)
    fs_grid = np.asarray(grid_data["fs_grid"], dtype=np.float32)
    us_grid = np.asarray(grid_data["us_grid"], dtype=np.float32)

    train_count, test_count = _resolve_adrshear_counts(total=len(fields), train_num=train_num, test_num=test_num)
    train_fields = fields[:train_count]
    train_targets = targets[:train_count]
    test_fields = fields[-test_count:]
    test_targets = targets[-test_count:]

    return (
        _build_samples(source_grid=fs_grid, target_grid=us_grid, fields=train_fields, targets=train_targets),
        _build_samples(source_grid=fs_grid, target_grid=us_grid, fields=test_fields, targets=test_targets),
    )


def _resolve_adrshear_counts(total: int, train_num, test_num):
    if train_num not in {"all", "none"} and test_num not in {"all", "none"}:
        train_count = int(train_num)
        test_count = int(test_num)
    elif train_num not in {"all", "none"}:
        requested_total = int(train_num)
        test_count = max(1, int(0.2 * requested_total))
        train_count = requested_total - test_count
    elif test_num not in {"all", "none"}:
        test_count = int(test_num)
        train_count = total - test_count
    else:
        test_count = max(1, int(0.2 * total))
        train_count = total - test_count

    if train_count <= 0 or test_count <= 0:
        raise ValueError(f"invalid ADRSHEAR split: train={train_count}, test={test_count}")
    if train_count + test_count > total:
        raise ValueError(f"requested ADRSHEAR split train={train_count}, test={test_count}, total={total}")
    return train_count, test_count


def _build_samples(source_grid: np.ndarray, target_grid: np.ndarray, fields: np.ndarray, targets: np.ndarray):
    theta = np.zeros((1,), dtype=np.float32)
    samples = []
    for field, target in zip(fields, targets):
        input_f = np.concatenate((source_grid, field[:, None]), axis=-1).astype(np.float32, copy=False)
        output = target[:, None].astype(np.float32, copy=False)
        samples.append([target_grid, output, theta, input_f])
    return samples


def _take_first(fields: np.ndarray, targets: np.ndarray, count):
    if count in {"all", "none"}:
        if count == "none":
            return fields[:0], targets[:0]
        return fields, targets
    count = min(int(count), len(fields))
    return fields[:count], targets[:count]


def _take_last(fields: np.ndarray, targets: np.ndarray, count):
    if count in {"all", "none"}:
        if count == "none":
            return fields[:0], targets[:0]
        return fields, targets
    count = min(int(count), len(fields))
    return fields[-count:], targets[-count:]
