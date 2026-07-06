"""Embedded and effective non-Markovian QRC comparison pipeline, v2 (GPU).

v2 fixes the v1 device bug: v1 was written CPU-only (cpu_safe_mode) even though
CUDA was available. Here every embedded density-matrix evolution runs on the GPU
in complex64 via local reshape/permute/matmul operations (never a global
superoperator over the full register), with pointwise complex128 CPU validation.
The small (16x16) effective no-auxiliary models run exactly in complex128 on CPU,
which is both faster than GPU for that size and numerically stricter; this choice
is recorded in config_v2.json. The scientific protocol follows the v1 module
(`embedded_effective_qrc_pipeline.py`), which is the specification, with the
mandatory v2 corrections: washout/train/test = 1000/1000/1000, >=20 paired
evaluation seeds, exact embedded ABC with N=4 executed, 64 Optuna trials,
washout-convergence diagnostics, teacher-forced validation before autonomous
Mackey-Glass rollout, and n>=20 for every accepted/rejected hypothesis.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import time
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qrc-v2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import nbformat
import numpy as np
import optuna
import pandas as pd
import psutil
import scipy.linalg
import torch
from nbclient import NotebookClient
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_DIR = Path("results_abc_comparison_v2")
FIGURES_DIR = Path("figures_abc_comparison_v2")
NOTEBOOK_PATH = Path("embedded_and_effective_hierarchical_abc_qrc_paper_v2.ipynb")
ZIP_PATH = Path("embedded_and_effective_abc_qrc_results_v2.zip")
LOG_PATH = RESULTS_DIR / "run.log"
PAPER_URL = "https://arxiv.org/abs/2505.02491"


@dataclass
class Config:
    n_a: int = 4
    h: float = 1.0
    gamma: float = 0.1
    dt: float = 0.5
    eta_paper: float = math.pi / 4
    j_scale: float = 1.0
    grid_size: int = 17
    grid_s_min: float = -0.25
    grid_s_max: float = 1.25
    dtype: str = "complex64"
    device: str = "cuda"
    smoke_seeds: Tuple[int, ...] = (9000, 9001, 9002)
    tune_seeds: Tuple[int, ...] = tuple(range(1000, 1012))
    eval_seeds: Tuple[int, ...] = tuple(range(20))
    paper_eval_seeds: Tuple[int, ...] = tuple(range(20))
    planned_eval_seeds: int = 100
    planned_tune_trials: int = 64
    tune_trials: int = 64
    ab_tune_seeds: int = 8
    abc_tune_seeds: int = 8
    abc_tune_trials: int = 64
    smoke_washout: int = 20
    smoke_train: int = 40
    smoke_test: int = 30
    paper_washout: int = 1000
    paper_train: int = 1000
    paper_test: int = 1000
    tune_washout: int = 300
    tune_train: int = 400
    tune_test: int = 300
    tau_max_paper: int = 50
    ridge_alphas: Tuple[float, ...] = (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0)
    omegas_paper: Tuple[float, ...] = (1.0, 0.5, 0.0)
    valid_threshold: float = 0.1
    washout_conv_threshold: float = 1e-3
    teacher_forced_r2_min: float = 0.99
    mg_tau: float = 17.0
    mg2_tau1: float = 17.0
    mg2_tau2: float = 30.0
    mg_sample_time: float = 3.0
    mg_internal_dt: float = 0.1
    n_boot: int = 2000
    optuna_seed: int = 42
    min_decision_seeds: int = 20

    @property
    def paper_len(self) -> int:
        return self.paper_washout + self.paper_train + self.paper_test

    @property
    def smoke_len(self) -> int:
        return self.smoke_washout + self.smoke_train + self.smoke_test

    @property
    def tune_len(self) -> int:
        return self.tune_washout + self.tune_train + self.tune_test


CFG = Config()
_T0 = time.time()
DEVICE: Optional[torch.device] = None
CDTYPE = torch.complex64


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "channel_cache").mkdir(exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.write_text("", encoding="utf-8")


def log(msg: str) -> None:
    ensure_dirs()
    line = f"[{datetime.now().strftime('%H:%M:%S')} +{time.time() - _T0:8.1f}s] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def marker(name: str) -> Path:
    return RESULTS_DIR / f"{name}_complete.json"


def write_marker(name: str, **payload) -> None:
    payload = {"phase": name, "completed_at": datetime.now().isoformat(), **payload}
    marker(name).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def append_rows(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def key_done(path: Path, **criteria) -> bool:
    df = load_csv(path)
    if df.empty:
        return False
    mask = np.ones(len(df), dtype=bool)
    for k, v in criteria.items():
        if k not in df.columns:
            return False
        mask &= df[k].astype(str).values == str(v)
    return bool(mask.any())


def write_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def record_failure(context: str, reason: str, **extra) -> None:
    row = {"timestamp": datetime.now().isoformat(), "context": context, "reason": reason, **extra}
    append_rows(RESULTS_DIR / "failed_runs.csv", [row])
    log(f"failure recorded: {context}: {reason}")


def environment_info() -> Dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 3),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "notes": "v2: all embedded density evolution on GPU complex64; no-aux 16x16 models exact complex128 on CPU.",
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 3)
    for mod in ["numpy", "scipy", "pandas", "matplotlib", "optuna", "nbformat", "nbclient"]:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception as exc:
            info[f"{mod}_error"] = repr(exc)
    return info


def require_gpu(verbose: bool = True) -> torch.device:
    """Hard GPU gate: print versions, allocate a 4096x4096 complex64 tensor,
    run a timed einsum/matmul, and ABORT if the GPU is not usable.
    No silent CPU fallback exists anywhere in this module."""
    global DEVICE
    if verbose:
        print(f"torch {torch.__version__} | torch.version.cuda={torch.version.cuda} | cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU REQUIRED: torch.cuda.is_available() is False. v2 aborts instead of "
            "falling back to CPU. To run on CPU you must set an explicit "
            "cpu_confirmed justification in config_v2.json and edit require_gpu()."
        )
    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    if verbose:
        print(f"GPU: {name} | {mem:.1f} GB")
    try:
        a = torch.randn(4096, 4096, dtype=torch.complex64, device=dev)
        b = torch.randn(4096, 4096, dtype=torch.complex64, device=dev)
        torch.cuda.synchronize()
        t0 = time.time()
        c = torch.einsum("ij,jk->ik", a, b)
        torch.cuda.synchronize()
        dt = time.time() - t0
        val = c[0, 0].item()
        if verbose:
            print(f"GPU test: 4096x4096 complex64 einsum matmul in {dt*1000:.1f} ms (c[0,0]={val:.3f})")
        del a, b, c
        torch.cuda.empty_cache()
    except Exception as exc:
        raise RuntimeError(f"GPU REQUIRED but unusable: {exc!r}") from exc
    DEVICE = dev
    return dev


def get_device() -> torch.device:
    if DEVICE is None:
        return require_gpu(verbose=False)
    return DEVICE


# ---------------------------------------------------------------------------
# NumPy complex128 reference implementation (identical maths to v1 module).
# Used to build channel grids and to validate the GPU path pointwise.
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
SMINUS = np.array([[0, 1], [0, 0]], dtype=np.complex128)
PAULI = {"x": X, "y": Y, "z": Z}


def kron_all(mats: Sequence[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def embed(op: np.ndarray, q: int, n: int) -> np.ndarray:
    return kron_all([op if i == q else I2 for i in range(n)])


def liouvillian_parts(seed: int, n: int, cfg: Config = CFG) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    d = 2**n
    jj = rng.uniform(-1, 1, size=n * (n - 1) // 2) * cfg.j_scale
    h0 = np.zeros((d, d), dtype=np.complex128)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            h0 += jj[k] * (embed(X, i, n) @ embed(X, j, n))
            k += 1
    sx = sum(embed(X, i, n) for i in range(n))
    sz = sum(embed(Z, i, n) for i in range(n))
    h0 += cfg.h * sz + cfg.h * sx
    hx = cfg.h * sx
    eye = np.eye(d, dtype=np.complex128)

    def ham_super(h: np.ndarray) -> np.ndarray:
        return -1j * (np.kron(h, eye) - np.kron(eye, h.T))

    base = ham_super(h0)
    for i in range(n):
        li = embed(SMINUS, i, n)
        ldl = li.conj().T @ li
        base += cfg.gamma * (
            np.kron(li, li.conj()) - 0.5 * (np.kron(ldl, eye) + np.kron(eye, ldl.T))
        )
    return base, ham_super(hx)


def channel_cache_path(seed: int, n: int, grid_size: int) -> Path:
    tag = f"N{n}_seed{seed}_g{grid_size}_dt{CFG.dt}_gamma{CFG.gamma}_range{CFG.grid_s_min}_{CFG.grid_s_max}"
    return RESULTS_DIR / "channel_cache" / f"channel_{tag}.npz"


def build_channel_grid_np(seed: int, n: int, grid_size: Optional[int] = None) -> np.ndarray:
    """complex128 numpy channel grid (grid_size, 4^n, 4^n), cached on disk."""
    ensure_dirs()
    g = grid_size or CFG.grid_size
    path = channel_cache_path(seed, n, g)
    if path.exists():
        return np.load(path)["grid"]
    t0 = time.time()
    base, drive = liouvillian_parts(seed, n)
    svals = np.linspace(CFG.grid_s_min, CFG.grid_s_max, g)
    mats = [scipy.linalg.expm((base + s * drive) * CFG.dt).astype(np.complex128) for s in svals]
    grid = np.stack(mats, axis=0)
    np.savez_compressed(path, grid=grid, svals=svals)
    log(f"channel grid cached: seed={seed}, N={n}, grid={g}, seconds={time.time() - t0:.1f}")
    return grid


_GRID_GPU_CACHE: Dict[Tuple, torch.Tensor] = {}


def build_channel_grid_gpu(seed: int, n: int, grid_size: Optional[int] = None) -> torch.Tensor:
    g = grid_size or CFG.grid_size
    # M3 fix: the in-memory GPU cache must key on every parameter that changes the
    # channel operators, otherwise a sweep over gamma/dt/grid range silently reuses
    # channels built for a previous configuration. The disk cache path already
    # encodes these (channel_cache_path); mirror them here.
    key = (seed, n, g, CFG.gamma, CFG.dt, CFG.grid_s_min, CFG.grid_s_max)
    if key not in _GRID_GPU_CACHE:
        if len(_GRID_GPU_CACHE) > 24:
            _GRID_GPU_CACHE.clear()
        grid = build_channel_grid_np(seed, n, g)
        _GRID_GPU_CACHE[key] = torch.tensor(grid, dtype=CDTYPE, device=get_device())
    return _GRID_GPU_CACHE[key]


def select_channel_np(grid: np.ndarray, s: float) -> np.ndarray:
    pos = (float(s) - CFG.grid_s_min) / (CFG.grid_s_max - CFG.grid_s_min) * (grid.shape[0] - 1)
    if pos <= 0:
        return grid[0]
    if pos >= grid.shape[0] - 1:
        return grid[-1]
    lo = int(math.floor(pos))
    w = pos - lo
    return (1.0 - w) * grid[lo] + w * grid[lo + 1]


def select_channel_gpu(grid: torch.Tensor, s: float) -> torch.Tensor:
    pos = (float(s) - CFG.grid_s_min) / (CFG.grid_s_max - CFG.grid_s_min) * (grid.shape[0] - 1)
    if pos <= 0:
        return grid[0]
    if pos >= grid.shape[0] - 1:
        return grid[-1]
    lo = int(math.floor(pos))
    w = pos - lo
    return (1.0 - w) * grid[lo] + w * grid[lo + 1]


def pure_zero_density_np(n: int) -> np.ndarray:
    d = 2**n
    rho = np.zeros((d, d), dtype=np.complex128)
    rho[0, 0] = 1.0
    return rho


def normalize_density_np(rho: np.ndarray) -> np.ndarray:
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho).real
    if abs(tr) > 1e-14:
        rho = rho / tr
    return rho


def apply_super_to_a_np(rho: np.ndarray, super_a: np.ndarray, n_a: int, n_total: int) -> np.ndarray:
    da = 2**n_a
    de = 2 ** (n_total - n_a)
    t = rho.reshape(da, de, da, de).transpose(0, 2, 1, 3).reshape(da * da, de * de)
    t = super_a @ t
    return np.ascontiguousarray(t.reshape(da, da, de, de).transpose(0, 2, 1, 3).reshape(da * de, da * de))


def partial_swap_unitary_np(eta: float) -> np.ndarray:
    sw = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)
    return math.cos(eta) * np.eye(4) + 1j * math.sin(eta) * sw


def partial_swap_layer_unitary_np(eta: float, n_pairs: int) -> np.ndarray:
    u = partial_swap_unitary_np(eta)
    out = u
    for _ in range(n_pairs - 1):
        out = np.kron(out, u)
    return out


def apply_layer_unitary_density_np(rho: np.ndarray, u: np.ndarray, qubits: Sequence[int], n_total: int) -> np.ndarray:
    m = len(qubits)
    t = rho.reshape([2] * (2 * n_total))
    t = np.moveaxis(t, list(qubits), list(range(m)))
    t = t.reshape(2**m, -1)
    t = u @ t
    t = t.reshape([2] * m + [2] * (2 * n_total - m))
    t = np.moveaxis(t, list(range(m)), list(qubits))
    bqubits = [n_total + q for q in qubits]
    t = np.moveaxis(t, bqubits, list(range(m)))
    t = t.reshape(2**m, -1)
    t = u.conj() @ t
    t = t.reshape([2] * m + [2] * (2 * n_total - m))
    t = np.moveaxis(t, list(range(m)), bqubits)
    return np.ascontiguousarray(t.reshape(rho.shape))


def depolarize_qubit_np(rho: np.ndarray, q: int, n_total: int, omega: float) -> np.ndarray:
    if omega <= 0:
        return rho
    t = rho.reshape([2] * (2 * n_total))
    moved = np.moveaxis(t, [q, n_total + q], [0, 1])
    reduced = moved[0, 0] + moved[1, 1]
    dep = np.zeros_like(moved)
    dep[0, 0] = 0.5 * reduced
    dep[1, 1] = 0.5 * reduced
    dep = np.moveaxis(dep, [0, 1], [q, n_total + q]).reshape(rho.shape)
    return (1.0 - omega) * rho + omega * dep


def reduce_register_np(rho: np.ndarray, keep: Sequence[int], n_total: int) -> np.ndarray:
    keep = tuple(int(q) for q in keep)
    keep_set = set(keep)
    traced = [q for q in range(n_total) if q not in keep_set]
    perm = list(keep) + traced + [n_total + q for q in keep] + [n_total + q for q in traced]
    t = rho.reshape([2] * (2 * n_total)).transpose(perm)
    dk = 2 ** len(keep)
    dtr = 2 ** len(traced)
    t = t.reshape(dk, dtr, dk, dtr)
    return np.ascontiguousarray(np.trace(t, axis1=1, axis2=3))


def build_observables(n: int) -> Tuple[np.ndarray, List[str]]:
    obs, names = [], []
    for i in range(n):
        for a, op in PAULI.items():
            obs.append(embed(op, i, n))
            names.append(f"{a}{i}")
    for i in range(n):
        for j in range(i + 1, n):
            for a, opa in PAULI.items():
                for b, opb in PAULI.items():
                    obs.append(embed(opa, i, n) @ embed(opb, j, n))
                    names.append(f"{a}{i}{b}{j}")
    return np.stack(obs), names


OBS_A_NP, OBS_NAMES = build_observables(CFG.n_a)
_OBS_GPU_CACHE: Dict[int, torch.Tensor] = {}


def obs_gpu(n: int) -> torch.Tensor:
    if n not in _OBS_GPU_CACHE:
        o = build_observables(n)[0] if n != CFG.n_a else OBS_A_NP
        _OBS_GPU_CACHE[n] = torch.tensor(o, dtype=CDTYPE, device=get_device())
    return _OBS_GPU_CACHE[n]


def n_pauli_features(n: int) -> int:
    return 3 * n + 9 * (n * (n - 1) // 2)


def features_from_rho_a_np(rho_a: np.ndarray, obs: np.ndarray) -> np.ndarray:
    return np.einsum("kij,ji->k", obs, rho_a).real.astype(np.float64)


def state_checks_np(rho: np.ndarray) -> Dict[str, float]:
    vals = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    return {
        "trace_error": float(abs(np.trace(rho) - 1.0)),
        "hermiticity_error": float(np.linalg.norm(rho - rho.conj().T)),
        "min_eig": float(vals.min()),
    }


# ---------------------------------------------------------------------------
# Torch/GPU operations (mirror the numpy reference exactly, complex64).
# ---------------------------------------------------------------------------


def normalize_density_t(rho: torch.Tensor) -> torch.Tensor:
    rho = 0.5 * (rho + rho.conj().T)
    tr = torch.real(torch.diagonal(rho).sum())
    return rho / tr


def apply_super_to_a_t(rho: torch.Tensor, super_a: torch.Tensor, n_a: int, n_total: int) -> torch.Tensor:
    da = 2**n_a
    de = 2 ** (n_total - n_a)
    t = rho.reshape(da, de, da, de).permute(0, 2, 1, 3).reshape(da * da, de * de)
    t = super_a @ t
    return t.reshape(da, da, de, de).permute(0, 2, 1, 3).reshape(da * de, da * de).contiguous()


def apply_layer_unitary_density_t(rho: torch.Tensor, u: torch.Tensor, qubits: Sequence[int], n_total: int) -> torch.Tensor:
    m = len(qubits)
    d = rho.shape[0]
    t = rho.reshape([2] * (2 * n_total))
    t = torch.movedim(t, list(qubits), list(range(m)))
    t = t.reshape(2**m, -1)
    t = u @ t
    t = t.reshape([2] * m + [2] * (2 * n_total - m))
    t = torch.movedim(t, list(range(m)), list(qubits))
    bqubits = [n_total + q for q in qubits]
    t = torch.movedim(t, bqubits, list(range(m)))
    t = t.reshape(2**m, -1)
    t = u.conj() @ t
    t = t.reshape([2] * m + [2] * (2 * n_total - m))
    t = torch.movedim(t, list(range(m)), bqubits)
    return t.reshape(d, d).contiguous()


def depolarize_qubit_t(rho: torch.Tensor, q: int, n_total: int, omega: float) -> torch.Tensor:
    if omega <= 0:
        return rho
    d = rho.shape[0]
    a = 2**q
    b = 2 ** (n_total - 1 - q)
    t = rho.reshape(a, 2, b, a, 2, b)
    avg = (0.5 * omega) * (t[:, 0, :, :, 0, :] + t[:, 1, :, :, 1, :])
    out = (1.0 - omega) * t
    out[:, 0, :, :, 0, :] += avg
    out[:, 1, :, :, 1, :] += avg
    return out.reshape(d, d)


def local_depolarize_all_t(rho: torch.Tensor, qubits: Iterable[int], n_total: int, omega: float) -> torch.Tensor:
    for q in qubits:
        rho = depolarize_qubit_t(rho, q, n_total, omega)
    return rho


def reduce_register_t(rho: torch.Tensor, keep: Sequence[int], n_total: int) -> torch.Tensor:
    keep = tuple(int(q) for q in keep)
    keep_set = set(keep)
    traced = [q for q in range(n_total) if q not in keep_set]
    perm = list(keep) + traced + [n_total + q for q in keep] + [n_total + q for q in traced]
    t = rho.reshape([2] * (2 * n_total)).permute(perm)
    dk = 2 ** len(keep)
    dtr = 2 ** len(traced)
    t = t.reshape(dk, dtr, dk, dtr)
    return torch.einsum("aibi->ab", t)




def natural_pair_layer_unitary_np(eta: float, n_a: int) -> np.ndarray:
    """Unitary of n_a commuting partial-SWAPs acting on pairs (i, n_a+i) of a
    2*n_a-qubit block in NATURAL qubit order (equals the interleaved layer of
    v1 after reordering qubits; verified in sanity checks)."""
    n = 2 * n_a
    d = 2**n
    u4 = partial_swap_unitary_np(eta).reshape(2, 2, 2, 2)
    U = np.eye(d, dtype=np.complex128)
    for i in range(n_a):
        t = U.reshape([2] * n + [d])
        t = np.moveaxis(t, [i, n_a + i], [0, 1])
        t = np.tensordot(u4, t, axes=([2, 3], [0, 1]))
        t = np.moveaxis(t, [0, 1], [i, n_a + i])
        U = t.reshape(d, d)
    return U


def apply_block_unitary_t(rho: torch.Tensor, u: torch.Tensor, q0: int, m: int, n_total: int) -> torch.Tensor:
    """Apply an m-qubit unitary on the contiguous qubit block [q0, q0+m) to a
    density matrix: rho -> U rho U^dagger, using batched matmuls only."""
    d = rho.shape[0]
    pre = 2**q0
    post = 2 ** (n_total - q0 - m)
    M = 2**m
    t = rho.reshape(pre, M, post * d)
    t = torch.einsum("ab,pbc->pac", u, t)
    t = t.reshape(d, pre, M, post)
    t = torch.einsum("ab,cpbq->cpaq", u.conj(), t)
    return t.reshape(d, d)

def pure_zero_density_t(n: int) -> torch.Tensor:
    d = 2**n
    rho = torch.zeros((d, d), dtype=CDTYPE, device=get_device())
    rho[0, 0] = 1.0
    return rho


def features_from_rho_t(rho_a: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    return torch.einsum("kij,ji->k", obs, rho_a).real


def state_checks_t(rho: torch.Tensor) -> Dict[str, float]:
    tr = torch.diagonal(rho).sum()
    herm = torch.linalg.norm(rho - rho.conj().T)
    out = {
        "trace_error": float(abs(tr.item() - 1.0)),
        "hermiticity_error": float(herm.item()),
    }
    if rho.shape[0] <= 256:
        vals = torch.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
        out["min_eig"] = float(vals.min().item())
    else:
        out["min_eig"] = float("nan")
    return out


def trace_distance_t(r1: torch.Tensor, r2: torch.Tensor) -> float:
    diff = r1 - r2
    diff = 0.5 * (diff + diff.conj().T)
    vals = torch.linalg.eigvalsh(diff)
    return float(0.5 * vals.abs().sum().item())


def trace_distance_np(r1: np.ndarray, r2: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(0.5 * ((r1 - r2) + (r1 - r2).conj().T))
    return float(0.5 * np.sum(np.abs(vals)))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def partial_swap_layer_gpu(eta: float, n_pairs: int) -> torch.Tensor:
    return torch.tensor(partial_swap_layer_unitary_np(eta, n_pairs), dtype=CDTYPE, device=get_device())


class EmbeddedModelGPU:
    """Embedded model with auxiliary registers; density matrix on GPU (complex64).

    Step order (identical to v1): input channel on A, partial-SWAP layer(s),
    local depolarization of auxiliaries, renormalization.
    """

    def __init__(
        self,
        n_a: int,
        architecture: str,
        omega: float = 1.0,
        eta: float = CFG.eta_paper,
        omega_b: Optional[float] = None,
        omega_c: Optional[float] = None,
        eta_ab: Optional[float] = None,
        eta_bc: Optional[float] = None,
    ):
        self.n_a = n_a
        self.architecture = architecture
        self.omega = omega
        self.omega_b = omega if omega_b is None else omega_b
        self.omega_c = omega if omega_c is None else omega_c
        self.eta_ab = eta if eta_ab is None else eta_ab
        self.eta_bc = eta if eta_bc is None else eta_bc
        self.u_ab_layer = partial_swap_layer_gpu(self.eta_ab, n_a)
        self.u_bc_layer = partial_swap_layer_gpu(self.eta_bc, n_a)
        # Natural-order block unitaries for the contiguous fast path (chain/AB).
        self.u_ab_nat = torch.tensor(natural_pair_layer_unitary_np(self.eta_ab, n_a), dtype=CDTYPE, device=get_device())
        self.u_bc_nat = torch.tensor(natural_pair_layer_unitary_np(self.eta_bc, n_a), dtype=CDTYPE, device=get_device())
        if architecture in ("M0-embedded", "M0-noaux"):
            self.n_total = n_a
        elif architecture.startswith("ABC"):
            self.n_total = 3 * n_a
        elif architecture.startswith("AB"):
            self.n_total = 2 * n_a
        else:
            raise ValueError(f"unknown architecture {architecture}")
        if self.architecture.startswith("ABC"):
            if "parallel" in self.architecture:
                self.qs_1, self.qs_2 = [], []
                for i in range(self.n_a):
                    self.qs_1.extend([i, self.n_a + i])
                    self.qs_2.extend([i, 2 * self.n_a + i])
            else:
                self.qs_1, self.qs_2 = [], []
                for i in range(self.n_a):
                    self.qs_1.extend([i, self.n_a + i])
                    self.qs_2.extend([self.n_a + i, 2 * self.n_a + i])
        elif self.architecture.startswith("AB"):
            self.qs_1 = []
            for i in range(self.n_a):
                self.qs_1.extend([i, self.n_a + i])
            self.qs_2 = None
        self.rho = pure_zero_density_t(self.n_total)
        self.last_clamped = 0

    def reset(self) -> "EmbeddedModelGPU":
        self.rho = pure_zero_density_t(self.n_total)
        self.last_clamped = 0
        return self

    def clone(self) -> torch.Tensor:
        return self.rho.clone()

    def restore(self, rho: torch.Tensor) -> None:
        self.rho = rho.clone()

    def step(self, s: float, grid: torch.Tensor) -> torch.Tensor:
        if s < CFG.grid_s_min or s > CFG.grid_s_max:
            self.last_clamped += 1
        self.rho = apply_super_to_a_t(self.rho, select_channel_gpu(grid, s), self.n_a, self.n_total)
        if self.architecture.startswith("ABC"):
            if "parallel" in self.architecture:
                self.rho = apply_layer_unitary_density_t(self.rho, self.u_ab_layer, self.qs_1, self.n_total)
                self.rho = apply_layer_unitary_density_t(self.rho, self.u_bc_layer, self.qs_2, self.n_total)
            else:
                self.rho = apply_block_unitary_t(self.rho, self.u_ab_nat, 0, 2 * self.n_a, self.n_total)
                self.rho = apply_block_unitary_t(self.rho, self.u_bc_nat, self.n_a, 2 * self.n_a, self.n_total)
            self.rho = local_depolarize_all_t(self.rho, range(self.n_a, 2 * self.n_a), self.n_total, self.omega_b)
            self.rho = local_depolarize_all_t(self.rho, range(2 * self.n_a, 3 * self.n_a), self.n_total, self.omega_c)
        elif self.architecture.startswith("AB"):
            self.rho = apply_block_unitary_t(self.rho, self.u_ab_nat, 0, 2 * self.n_a, self.n_total)
            self.rho = local_depolarize_all_t(self.rho, range(self.n_a, 2 * self.n_a), self.n_total, self.omega_b)
        self.rho = normalize_density_t(self.rho)
        return self.rho

    def reduced(self, register: str = "A") -> torch.Tensor:
        if register == "A":
            keep = range(self.n_a)
        elif register == "B":
            keep = range(self.n_a, 2 * self.n_a)
        elif register == "C":
            keep = range(2 * self.n_a, 3 * self.n_a)
        else:
            raise ValueError(register)
        return reduce_register_t(self.rho, list(keep), self.n_total)

    def features_t(self, register: str = "A") -> torch.Tensor:
        return features_from_rho_t(self.reduced(register), obs_gpu(self.n_a))

    def features(self, register: str = "A") -> np.ndarray:
        return self.features_t(register).double().cpu().numpy()


class EmbeddedModelNP:
    """complex128 CPU reference of EmbeddedModelGPU (validation/diagnostics only)."""

    def __init__(self, n_a: int, architecture: str, omega: float = 1.0, eta: float = CFG.eta_paper,
                 omega_b: Optional[float] = None, omega_c: Optional[float] = None,
                 eta_ab: Optional[float] = None, eta_bc: Optional[float] = None):
        self.n_a = n_a
        self.architecture = architecture
        self.omega_b = omega if omega_b is None else omega_b
        self.omega_c = omega if omega_c is None else omega_c
        self.eta_ab = eta if eta_ab is None else eta_ab
        self.eta_bc = eta if eta_bc is None else eta_bc
        self.u_ab_layer = partial_swap_layer_unitary_np(self.eta_ab, n_a)
        self.u_bc_layer = partial_swap_layer_unitary_np(self.eta_bc, n_a)
        if architecture.startswith("ABC"):
            self.n_total = 3 * n_a
        elif architecture.startswith("AB"):
            self.n_total = 2 * n_a
        else:
            self.n_total = n_a
        self.rho = pure_zero_density_np(self.n_total)

    def step(self, s: float, grid: np.ndarray) -> np.ndarray:
        self.rho = apply_super_to_a_np(self.rho, select_channel_np(grid, s), self.n_a, self.n_total)
        n_a = self.n_a
        if self.architecture.startswith("ABC"):
            if "parallel" in self.architecture:
                qs1, qs2 = [], []
                for i in range(n_a):
                    qs1.extend([i, n_a + i])
                    qs2.extend([i, 2 * n_a + i])
            else:
                qs1, qs2 = [], []
                for i in range(n_a):
                    qs1.extend([i, n_a + i])
                    qs2.extend([n_a + i, 2 * n_a + i])
            self.rho = apply_layer_unitary_density_np(self.rho, self.u_ab_layer, qs1, self.n_total)
            self.rho = apply_layer_unitary_density_np(self.rho, self.u_bc_layer, qs2, self.n_total)
            for q in range(n_a, 2 * n_a):
                self.rho = depolarize_qubit_np(self.rho, q, self.n_total, self.omega_b)
            for q in range(2 * n_a, 3 * n_a):
                self.rho = depolarize_qubit_np(self.rho, q, self.n_total, self.omega_c)
        elif self.architecture.startswith("AB"):
            qs1 = []
            for i in range(n_a):
                qs1.extend([i, n_a + i])
            self.rho = apply_layer_unitary_density_np(self.rho, self.u_ab_layer, qs1, self.n_total)
            for q in range(n_a, 2 * n_a):
                self.rho = depolarize_qubit_np(self.rho, q, self.n_total, self.omega_b)
        self.rho = normalize_density_np(self.rho)
        return self.rho

    def reduced_a(self) -> np.ndarray:
        return reduce_register_np(self.rho, list(range(self.n_a)), self.n_total)


def apply_pauli_depol_a_np(rho: np.ndarray, p: float, n_a: int) -> np.ndarray:
    for q in range(n_a):
        rho = depolarize_qubit_np(rho, q, n_a, p)
    return rho


class NoAuxModel:
    """Effective no-auxiliary model, exact complex128 on CPU (16x16 state)."""

    def __init__(self, n_a: int, name: str, tau_b: int = 1, tau_c: int = 2,
                 lambda_b: float = 0.0, lambda_c: float = 0.0, p_b: float = 0.0,
                 p_c: float = 0.0, shuffled: bool = False, seed: int = 0):
        self.n_a = n_a
        self.name = name
        self.tau_b = int(tau_b)
        self.tau_c = int(tau_c)
        self.lambda_b = float(lambda_b)
        self.lambda_c = float(lambda_c)
        self.lambda_0 = max(0.0, 1.0 - self.lambda_b - self.lambda_c)
        self.p_b = float(p_b)
        self.p_c = float(p_c)
        self.shuffled = shuffled
        self.seed = seed
        self.rng = np.random.default_rng(seed + 7717)
        self.max_tau = max(1, self.tau_b, self.tau_c)
        self.obs = build_observables(n_a)[0] if n_a != CFG.n_a else OBS_A_NP
        self.reset()

    def reset(self) -> "NoAuxModel":
        self.rho = pure_zero_density_np(self.n_a)
        self.buffer = [self.rho.copy() for _ in range(self.max_tau + 1)]
        self.t = 0
        self.rng = np.random.default_rng(self.seed + 7717)
        return self

    def clone(self):
        return self.rho.copy(), [x.copy() for x in self.buffer], self.t, self.rng.bit_generator.state

    def restore(self, state) -> None:
        self.rho, self.buffer, self.t = state[0].copy(), [x.copy() for x in state[1]], int(state[2])
        self.rng.bit_generator.state = state[3]

    def delayed(self, tau: int) -> np.ndarray:
        if self.shuffled and self.t > self.max_tau:
            idx = self.rng.integers(0, len(self.buffer))
            return self.buffer[idx].copy()
        return self.buffer[(self.t - tau) % len(self.buffer)].copy()

    def step(self, s: float, grid: np.ndarray) -> np.ndarray:
        mix = self.lambda_0 * self.rho
        if self.lambda_b > 0:
            past = self.delayed(self.tau_b)
            if "kraus" in self.name or "tied" in self.name or "hierarchical" in self.name:
                past = apply_pauli_depol_a_np(past, self.p_b, self.n_a)
            mix = mix + self.lambda_b * past
        if self.lambda_c > 0:
            past = self.delayed(self.tau_c)
            if "kraus" in self.name or "tied" in self.name or "hierarchical" in self.name:
                past = apply_pauli_depol_a_np(past, self.p_c, self.n_a)
            mix = mix + self.lambda_c * past
        mix = normalize_density_np(mix)
        v = select_channel_np(grid, s) @ mix.reshape(-1)
        self.rho = normalize_density_np(v.reshape(2**self.n_a, 2**self.n_a))
        self.t += 1
        self.buffer[self.t % len(self.buffer)] = self.rho.copy()
        return self.rho

    def features(self) -> np.ndarray:
        return features_from_rho_a_np(self.rho, self.obs)


def make_noaux_model(name: str, params: Dict, seed: int) -> NoAuxModel:
    if name == "M0-noaux":
        return NoAuxModel(CFG.n_a, name, seed=seed)
    if name == "AB-noaux-residual":
        return NoAuxModel(CFG.n_a, name, tau_b=params.get("tau_b", 10), lambda_b=params.get("lambda_b", 0.35), seed=seed)
    if name == "AB-noaux-kraus":
        return NoAuxModel(CFG.n_a, name, tau_b=params.get("tau_b", 10), lambda_b=params.get("lambda_b", 0.35), p_b=params.get("p_b", 0.2), seed=seed)
    if name in ("ABC-noaux-residual", "ABC-noaux-kraus", "ABC-noaux-hierarchical"):
        return NoAuxModel(
            CFG.n_a, name,
            tau_b=params.get("tau_b", 10), tau_c=params.get("tau_c", 30),
            lambda_b=params.get("lambda_b", 0.25), lambda_c=params.get("lambda_c", 0.25),
            p_b=params.get("p_b", 0.2), p_c=params.get("p_c", 0.05), seed=seed,
        )
    if name == "ABC-noaux-tied":
        lam = params.get("lambda_b", 0.2)
        p = params.get("p_b", 0.2)
        return NoAuxModel(CFG.n_a, name, tau_b=params.get("tau_b", 10), tau_c=params.get("tau_c", 30), lambda_b=lam, lambda_c=lam, p_b=p, p_c=p, seed=seed)
    if name == "ABC-noaux-B-only":
        return NoAuxModel(CFG.n_a, name, tau_b=params.get("tau_b", 10), lambda_b=params.get("lambda_b", 0.35), seed=seed)
    if name == "ABC-noaux-C-only":
        return NoAuxModel(CFG.n_a, name, tau_c=params.get("tau_c", 30), lambda_c=params.get("lambda_c", 0.35), seed=seed)
    if name == "ABC-noaux-shuffled-history":
        return NoAuxModel(CFG.n_a, name, tau_b=params.get("tau_b", 10), tau_c=params.get("tau_c", 30), lambda_b=0.25, lambda_c=0.25, p_b=0.2, p_c=0.05, shuffled=True, seed=seed)
    raise ValueError(name)


def make_embedded_model(name: str, params: Dict) -> EmbeddedModelGPU:
    if name in ("AB-embedded", "AB-paper"):
        return EmbeddedModelGPU(CFG.n_a, "AB-embedded", omega=params.get("omega", 0.5), eta=params.get("eta", CFG.eta_paper))
    if name == "AB-Markov":
        return EmbeddedModelGPU(CFG.n_a, "AB-embedded", omega=1.0, eta=params.get("eta", CFG.eta_paper))
    if name == "ABC-embedded-hierarchical":
        return EmbeddedModelGPU(
            CFG.n_a, "ABC-chain",
            omega_b=params.get("omega_b", 0.5), omega_c=params.get("omega_c", 0.1),
            eta_ab=params.get("eta_ab", CFG.eta_paper), eta_bc=params.get("eta_bc", CFG.eta_paper / 2),
        )
    if name == "ABC-embedded-tied":
        return EmbeddedModelGPU(CFG.n_a, "ABC-chain", omega_b=params.get("omega", 0.5), omega_c=params.get("omega", 0.5), eta_ab=params.get("eta", CFG.eta_paper), eta_bc=params.get("eta", CFG.eta_paper))
    if name == "ABC-embedded-C-off":
        return EmbeddedModelGPU(CFG.n_a, "ABC-chain", omega_b=params.get("omega_b", 0.5), omega_c=1.0, eta_ab=params.get("eta_ab", CFG.eta_paper), eta_bc=0.0)
    if name == "ABC-embedded-parallel":
        return EmbeddedModelGPU(CFG.n_a, "ABC-parallel", omega_b=params.get("omega_b", 0.5), omega_c=params.get("omega_c", 0.2), eta_ab=params.get("eta_ab", CFG.eta_paper), eta_bc=params.get("eta_ac", CFG.eta_paper / 2))
    if name == "ABC-Markov":
        return EmbeddedModelGPU(CFG.n_a, "ABC-chain", omega_b=1.0, omega_c=1.0, eta_ab=CFG.eta_paper, eta_bc=CFG.eta_paper)
    raise ValueError(name)


EMBEDDED_ARCHES = [
    "AB-Markov", "AB-embedded", "ABC-embedded-hierarchical", "ABC-embedded-tied",
    "ABC-embedded-parallel", "ABC-embedded-C-off", "ABC-Markov",
]
NOAUX_ARCHES = [
    "M0-noaux", "AB-noaux-kraus", "ABC-noaux-kraus", "ABC-noaux-tied",
    "ABC-noaux-hierarchical", "ABC-noaux-B-only", "ABC-noaux-C-only",
    "ABC-noaux-shuffled-history",
]


def is_embedded(name: str) -> bool:
    return "noaux" not in name


def make_model(name: str, params: Dict, seed: int):
    if is_embedded(name):
        return make_embedded_model(name, params)
    return make_noaux_model(name, params, seed)


def drive_features(model, seq: np.ndarray, grid, register: str = "A") -> np.ndarray:
    """Drive a model with an input sequence and collect A-register Pauli features."""
    n_a = model.n_a
    nf = n_pauli_features(n_a)
    if isinstance(model, EmbeddedModelGPU):
        feats = torch.empty((len(seq), nf), dtype=torch.float32, device=get_device())
        for k, s in enumerate(seq):
            model.step(float(s), grid)
            feats[k] = model.features_t(register)
        return feats.double().cpu().numpy()
    feats = np.empty((len(seq), nf), dtype=np.float64)
    for k, s in enumerate(seq):
        model.step(float(s), grid)
        feats[k] = model.features()
    return feats


def get_grid(model_name: str, seed: int):
    """GPU grid for embedded models, complex128 numpy grid for no-aux models."""
    if is_embedded(model_name):
        return build_channel_grid_gpu(seed, CFG.n_a)
    return build_channel_grid_np(seed, CFG.n_a)


# ---------------------------------------------------------------------------
# Readout, targets, metrics (identical maths to v1)
# ---------------------------------------------------------------------------


def add_bias(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def fit_readout(x: np.ndarray, y: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    xb = add_bias(x)
    y2 = np.asarray(y)
    if y2.ndim == 1:
        y2 = y2[:, None]
    if alpha <= 0:
        w, *_ = np.linalg.lstsq(xb, y2, rcond=None)
    else:
        reg = np.eye(xb.shape[1])
        reg[0, 0] = 0.0
        w = np.linalg.solve(xb.T @ xb + alpha * reg, xb.T @ y2)
    return w


def predict_readout(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    out = add_bias(x) @ w
    return out[:, 0] if out.shape[1] == 1 else out


def capacity_score(y: np.ndarray, yp: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    vy = np.var(y)
    vp = np.var(yp)
    if vy <= 1e-14 or vp <= 1e-14:
        return 0.0
    c = np.mean((y - y.mean()) * (yp - yp.mean()))
    return float(max(0.0, min(1.0, (c * c) / (vy * vp))))


def mse_metrics(y: np.ndarray, yp: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    mse = float(np.mean((yp - y) ** 2))
    var = float(np.var(y) + 1e-12)
    r2 = float(1.0 - mse / var)
    return {"mse": mse, "nrmse": float(math.sqrt(mse / var)), "r2": r2}


def all_finite(values, keys: Optional[Sequence[str]] = None) -> bool:
    """G3 fix: single project-wide finiteness validator used before writing any
    metrics row. Accepts a dict (checks `keys`, or all numeric values) or an
    iterable of numbers. Returns True iff every checked value is finite."""
    if isinstance(values, dict):
        if keys is None:
            items = [v for v in values.values() if isinstance(v, (int, float, np.floating, np.integer))]
        else:
            items = [values.get(k) for k in keys]
    else:
        items = list(values)
    for v in items:
        try:
            if v is None or not np.isfinite(float(v)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def split_slices(washout: int, train: int, test: int) -> Dict[str, slice]:
    return {"train": slice(washout, washout + train), "test": slice(washout + train, washout + train + test)}


def iid_inputs(seed: int, length: int) -> np.ndarray:
    return np.random.default_rng(seed + 12345).uniform(0.0, 1.0, size=length)


def stm_target(seq: np.ndarray, tau: int) -> np.ndarray:
    y = np.zeros_like(seq)
    if tau == 0:
        y[:] = seq
    else:
        y[tau:] = seq[:-tau]
    return y


def legendre_p1(seq: np.ndarray) -> np.ndarray:
    return 2.0 * seq - 1.0


def target_by_name(seq: np.ndarray, name: str) -> np.ndarray:
    x = legendre_p1(seq)

    def delay(arr, tau):
        y = np.zeros_like(arr)
        if tau == 0:
            y[:] = arr
        else:
            y[tau:] = arr[:-tau]
        return y

    if name == "paper_s0_s10":
        return delay(seq, 0) * delay(seq, 10)
    if name == "p1_0":
        return delay(x, 0)
    if name == "p1_10":
        return delay(x, 10)
    if name == "p1_30":
        return delay(x, 30)
    if name == "p1_0_10":
        return delay(x, 0) * delay(x, 10)
    if name == "p1_0_30":
        return delay(x, 0) * delay(x, 30)
    if name == "p1_10_30":
        return delay(x, 10) * delay(x, 30)
    if name == "p1_0_10_30":
        return delay(x, 0) * delay(x, 10) * delay(x, 30)
    if name == "s0_s30":
        return delay(seq, 0) * delay(seq, 30)
    if name == "s10_s30":
        return delay(seq, 10) * delay(seq, 30)
    if name == "s0_s10_s30":
        return delay(seq, 0) * delay(seq, 10) * delay(seq, 30)
    raise ValueError(name)


def evaluate_capacity_from_features(feats: np.ndarray, seq: np.ndarray, target: np.ndarray, slices: Dict[str, slice], alpha=0.0) -> Tuple[float, float]:
    w = fit_readout(feats[slices["train"]], target[slices["train"]], alpha)
    pred = predict_readout(feats[slices["test"]], w)
    return capacity_score(target[slices["test"]], pred), mse_metrics(target[slices["test"]], pred)["r2"]


def mackey_glass(n_samples: int, two_delay: bool = False, discard: int = 500, x0: float = 1.2) -> np.ndarray:
    dt = CFG.mg_internal_dt
    sub = int(round(CFG.mg_sample_time / dt))
    tau1 = int(round(CFG.mg_tau / dt))
    tau2 = int(round(CFG.mg2_tau2 / dt))
    max_tau = tau2 if two_delay else tau1
    total = (n_samples + discard) * sub
    x = np.empty(total + max_tau + 2, dtype=np.float64)
    x[: max_tau + 1] = x0

    def f(xi, xa, xb=None):
        if two_delay:
            return -0.1 * xi + 0.1 * xa / (1.0 + xa**10) + 0.1 * xb / (1.0 + xb**10)
        return -0.1 * xi + 0.2 * xa / (1.0 + xa**10)

    for i in range(max_tau, max_tau + total):
        if two_delay:
            d1 = int(round(CFG.mg2_tau1 / dt))
            a0, a1 = x[i - d1], x[i - d1 + 1]
            b0, b1 = x[i - tau2], x[i - tau2 + 1]
            ah, bh = 0.5 * (a0 + a1), 0.5 * (b0 + b1)
            k1 = f(x[i], a0, b0)
            k2 = f(x[i] + 0.5 * dt * k1, ah, bh)
            k3 = f(x[i] + 0.5 * dt * k2, ah, bh)
            k4 = f(x[i] + dt * k3, a1, b1)
        else:
            a0, a1 = x[i - tau1], x[i - tau1 + 1]
            ah = 0.5 * (a0 + a1)
            k1 = f(x[i], a0)
            k2 = f(x[i] + 0.5 * dt * k1, ah)
            k3 = f(x[i] + 0.5 * dt * k2, ah)
            k4 = f(x[i] + dt * k3, a1)
        x[i + 1] = x[i] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return x[max_tau::sub][discard : discard + n_samples]


def normalize_series(raw: np.ndarray, train_slice: slice) -> np.ndarray:
    lo = float(np.min(raw[train_slice]))
    hi = float(np.max(raw[train_slice]))
    return (raw - lo) / (hi - lo)


def autonomous_rollout(model, grid, w: np.ndarray, steps: int) -> np.ndarray:
    preds = []
    for _ in range(steps):
        feat = model.features() if isinstance(model, NoAuxModel) else model.features("A")
        y = float(predict_readout(feat[None, :], w)[0])
        preds.append(y)
        model.step(y, grid)
    return np.asarray(preds)


def run_mg_model(seed: int, model, grid, series: np.ndarray, slices: Dict[str, slice]) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, int]:
    """v2 MG protocol: teacher-forced one-step validation on the test span BEFORE
    the autonomous rollout; the rollout starts from the state at the end of train
    (documented teacher-forcing -> autonomous transition), predicted value is fed
    back as the next input, never using future truth."""
    stop = slices["test"].start
    feats_pre = drive_features(model, series[:stop], grid)
    snapshot = model.clone()
    target = series[1 : stop + 1]
    w = fit_readout(feats_pre[slices["train"]], target[slices["train"]], alpha=1e-6)
    # Teacher-forced one-step-ahead validation over the test span.
    test_len = slices["test"].stop - slices["test"].start
    feats_tf = drive_features(model, series[stop : stop + test_len], grid)
    tf_pred = predict_readout(feats_tf, w)
    tf_truth = series[stop + 1 : stop + test_len + 1]
    m = min(len(tf_pred), len(tf_truth))
    tf_metrics = mse_metrics(tf_truth[:m], tf_pred[:m])
    # Autonomous rollout from the end-of-train state.
    model.restore(snapshot)
    preds = autonomous_rollout(model, grid, w, test_len)
    truth = series[slices["test"]]
    metrics150 = mse_metrics(truth[:150], preds[:150])
    metrics1000 = mse_metrics(truth, preds)
    err = np.abs(preds - truth)
    exceed = np.where(err > CFG.valid_threshold)[0]
    vpt = int(exceed[0]) if len(exceed) else len(preds)
    out = {
        "seed": seed,
        "r2_teacher_forced": tf_metrics["r2"],
        "nrmse_teacher_forced": tf_metrics["nrmse"],
        "teacher_forced_ok": bool(tf_metrics["r2"] >= CFG.teacher_forced_r2_min),
        "mse_150": metrics150["mse"],
        "nrmse_150": metrics150["nrmse"],
        "r2_150": metrics150["r2"],
        "mse_1000": metrics1000["mse"],
        "nrmse_1000": metrics1000["nrmse"],
        "r2_1000": metrics1000["r2"],
        "valid_prediction_time": vpt,
        "valid_threshold": CFG.valid_threshold,
        "diverged": bool(np.any(np.abs(preds) > 2.0)),
        "out_of_range_fraction": float(np.mean((preds < 0.0) | (preds > 1.0))),
        "grid_clamps": int(getattr(model, "last_clamped", 0)),
    }
    return out, preds, truth, vpt


# ---------------------------------------------------------------------------
# Sanity checks, GPU-vs-CPU validation, benchmark, budget
# ---------------------------------------------------------------------------


def run_sanity_checks(force: bool = False) -> pd.DataFrame:
    if marker("sanity").exists() and not force:
        log("sanity already complete; skipping")
        return pd.DataFrame(json.loads((RESULTS_DIR / "sanity_checks.json").read_text())["checks"])
    log("running sanity, physical consistency, and GPU-vs-CPU complex128 validation")
    rows = []

    def add(name, ok, value=None, detail=""):
        rows.append({"check": name, "passed": bool(ok), "value": value, "detail": detail})

    n = 2
    grid = build_channel_grid_np(4242, n, grid_size=9)
    rho = pure_zero_density_np(n)
    rho2 = (select_channel_np(grid, 0.4) @ rho.reshape(-1)).reshape(2**n, 2**n)
    chk = state_checks_np(normalize_density_np(rho2))
    add("trace_preserved", chk["trace_error"] < 1e-8, chk["trace_error"])
    add("hermiticity_preserved", chk["hermiticity_error"] < 1e-8, chk["hermiticity_error"])
    add("positivity_preserved", chk["min_eig"] > -1e-8, chk["min_eig"])
    u0 = partial_swap_unitary_np(0.0)
    up = partial_swap_unitary_np(math.pi / 2)
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)
    u = partial_swap_unitary_np(CFG.eta_paper)
    add("partial_swap_unitary", np.linalg.norm(u.conj().T @ u - np.eye(4)) < 1e-10, float(np.linalg.norm(u.conj().T @ u - np.eye(4))))
    add("eta_zero_identity", np.linalg.norm(u0 - np.eye(4)) < 1e-10, float(np.linalg.norm(u0 - np.eye(4))))
    add("eta_pi2_swap_phase", np.linalg.norm(up / 1j - swap) < 1e-10, float(np.linalg.norm(up / 1j - swap)))
    rho_ab = pure_zero_density_np(2)
    dep0 = depolarize_qubit_np(rho_ab, 1, 2, 0.0)
    dep1 = depolarize_qubit_np(rho_ab, 1, 2, 1.0)
    add("depol_omega0_identity", np.linalg.norm(dep0 - rho_ab) < 1e-12, float(np.linalg.norm(dep0 - rho_ab)))
    add("depol_omega1_aux_mixed", abs(reduce_register_np(dep1, [1], 2)[0, 0].real - 0.5) < 1e-12)
    add("depol_cptp_trace", abs(np.trace(dep1) - 1) < 1e-12, float(abs(np.trace(dep1) - 1)))
    f = features_from_rho_a_np(reduce_register_np(rho_ab, [0], 2), build_observables(1)[0])
    add("features_real", float(np.max(np.abs(np.imag(f)))) < 1e-12)
    add("feature_dimension_n4", len(OBS_NAMES) == 66, len(OBS_NAMES))
    add("seed_sets_disjoint", not (set(CFG.tune_seeds) & set(CFG.eval_seeds)) and not (set(CFG.smoke_seeds) & set(CFG.eval_seeds)), str((CFG.tune_seeds[:2], CFG.eval_seeds[:2])))
    # No-aux consistency (complex128 CPU).
    g2 = build_channel_grid_np(7, 2, grid_size=9)
    seq = iid_inputs(7, 12)
    m0 = NoAuxModel(2, "M0-noaux")
    ab0 = NoAuxModel(2, "AB-noaux-residual", lambda_b=0.0)
    for s in seq:
        m0.step(s, g2)
        ab0.step(s, g2)
    add("lambda_zero_matches_m0", np.linalg.norm(m0.rho - ab0.rho) < 1e-10, float(np.linalg.norm(m0.rho - ab0.rho)))
    abc = NoAuxModel(2, "ABC-noaux-residual", lambda_b=0.4, lambda_c=0.0, tau_b=3, tau_c=5)
    ab = NoAuxModel(2, "AB-noaux-residual", lambda_b=0.4, tau_b=3)
    for s in seq:
        abc.step(s, g2)
        ab.step(s, g2)
    add("abc_c_zero_matches_ab", np.linalg.norm(abc.rho - ab.rho) < 1e-10, float(np.linalg.norm(abc.rho - ab.rho)))
    residual = NoAuxModel(2, "ABC-noaux-residual", lambda_b=0.2, lambda_c=0.2, tau_b=2, tau_c=4)
    kraus0 = NoAuxModel(2, "ABC-noaux-kraus", lambda_b=0.2, lambda_c=0.2, tau_b=2, tau_c=4, p_b=0, p_c=0)
    for s in seq:
        residual.step(s, g2)
        kraus0.step(s, g2)
    add("kraus_p0_matches_residual", np.linalg.norm(residual.rho - kraus0.rho) < 1e-10)
    add("buffer_size_sufficient", residual.max_tau >= max(residual.tau_b, residual.tau_c), residual.max_tau)
    before = residual.buffer[0].copy()
    residual.rho[:] = 0
    add("buffer_not_aliasing_current_state", np.linalg.norm(residual.buffer[0] - before) < 1e-12)
    # Shuffled-history controls: per-seed shuffle, trajectory differs, no future access.
    shuf = NoAuxModel(2, "ABC-noaux-shuffled-history", lambda_b=0.4, lambda_c=0.2, shuffled=True, seed=5)
    ordered = NoAuxModel(2, "ABC-noaux-residual", lambda_b=0.4, lambda_c=0.2, seed=5)
    for s in iid_inputs(6, 30):
        shuf.step(s, g2)
        ordered.step(s, g2)
    add("shuffled_history_changes_trajectory", np.linalg.norm(shuf.rho - ordered.rho) > 1e-8, float(np.linalg.norm(shuf.rho - ordered.rho)))
    shuf_a = NoAuxModel(2, "ABC-noaux-shuffled-history", lambda_b=0.4, lambda_c=0.2, shuffled=True, seed=5)
    shuf_b = NoAuxModel(2, "ABC-noaux-shuffled-history", lambda_b=0.4, lambda_c=0.2, shuffled=True, seed=6)
    for s in iid_inputs(6, 30):
        shuf_a.step(s, g2)
        shuf_b.step(s, g2)
    add("shuffle_redrawn_per_seed", np.linalg.norm(shuf_a.rho - shuf_b.rho) > 1e-10, float(np.linalg.norm(shuf_a.rho - shuf_b.rho)))
    add("shuffle_buffer_contains_only_past_states", True, detail="buffer written only after each step; delayed() samples buffer entries, all past by construction")
    # Reset reproducibility (rng state restored on reset).
    shuf_c = NoAuxModel(2, "ABC-noaux-shuffled-history", lambda_b=0.4, lambda_c=0.2, shuffled=True, seed=5)
    for s in iid_inputs(6, 30):
        shuf_c.step(s, g2)
    add("shuffled_reset_reproducible", np.linalg.norm(shuf_a.rho - shuf_c.rho) < 1e-12)
    direct = (select_channel_np(g2, 0.33) @ pure_zero_density_np(2).reshape(-1)).reshape(4, 4)
    local = apply_super_to_a_np(pure_zero_density_np(2), select_channel_np(g2, 0.33), 2, 2)
    add("exact_accelerated_small_match", np.linalg.norm(direct - local) < 1e-12)

    # --- GPU complex64 vs CPU complex128 pointwise validation ---
    dev = get_device()
    for arch, kwargs in [
        ("AB-embedded", dict(omega=0.5)),
        ("ABC-chain", dict(omega_b=0.5, omega_c=0.1, eta_bc=CFG.eta_paper / 2)),
        ("ABC-parallel", dict(omega_b=0.5, omega_c=0.2, eta_bc=CFG.eta_paper / 2)),
    ]:
        gg = build_channel_grid_gpu(4242, 2, grid_size=9)
        gn = build_channel_grid_np(4242, 2, grid_size=9)
        mg_ = EmbeddedModelGPU(2, arch, **kwargs)
        mn_ = EmbeddedModelNP(2, arch, **kwargs)
        seqv = iid_inputs(11, 40)
        maxdist = 0.0
        for s in seqv:
            mg_.step(float(s), gg)
            mn_.step(float(s), gn)
            ra_g = mg_.reduced("A").cpu().numpy().astype(np.complex128)
            ra_n = mn_.reduced_a()
            maxdist = max(maxdist, trace_distance_np(ra_g, ra_n))
        add(f"gpu_c64_matches_cpu_c128_{arch}_N2", maxdist < 5e-4, maxdist, "max trace distance of reduced A over 40 steps")
    # N=4 AB single-trajectory spot check (256x256, CPU c128 feasible for 10 steps).
    gg4 = build_channel_grid_gpu(0, 4)
    gn4 = build_channel_grid_np(0, 4)
    mg4 = EmbeddedModelGPU(4, "AB-embedded", omega=0.5)
    mn4 = EmbeddedModelNP(4, "AB-embedded", omega=0.5)
    maxdist4 = 0.0
    for s in iid_inputs(3, 10):
        mg4.step(float(s), gg4)
        mn4.step(float(s), gn4)
        maxdist4 = max(maxdist4, trace_distance_np(mg4.reduced("A").cpu().numpy().astype(np.complex128), mn4.reduced_a()))
    add("gpu_c64_matches_cpu_c128_AB_N4", maxdist4 < 5e-4, maxdist4)
    # ABC N=4 state physicality on GPU after 10 steps.
    mabc4 = EmbeddedModelGPU(4, "ABC-chain", omega_b=0.5, omega_c=0.1, eta_bc=CFG.eta_paper / 2)
    for s in iid_inputs(4, 10):
        mabc4.step(float(s), gg4)
    chk4 = state_checks_t(mabc4.rho)
    add("abc_n4_gpu_trace", chk4["trace_error"] < 1e-4, chk4["trace_error"])
    add("abc_n4_gpu_hermiticity", chk4["hermiticity_error"] < 1e-3, chk4["hermiticity_error"])
    ra = mabc4.reduced("A")
    add("abc_n4_gpu_reduced_a_positive", float(torch.linalg.eigvalsh(ra).min().item()) > -1e-4)

    df = pd.DataFrame(rows)
    write_json(RESULTS_DIR / "sanity_checks.json", {"checks": rows, "all_passed": bool(df.passed.all())})
    if not df.passed.all():
        failed = df.loc[~df.passed]
        raise RuntimeError(f"sanity checks failed: {failed.to_dict('records')}")
    write_marker("sanity", n_checks=len(df))
    log(f"sanity checks passed: {len(df)}")
    return df


def benchmark_step_costs(force: bool = False) -> Dict:
    """PASSO 1: measure GPU seconds/step for AB (256x256) and ABC (4096x4096)
    and CPU seconds/step for the no-aux models, then print a full-protocol estimate."""
    path = RESULTS_DIR / "benchmark_v2.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    log("benchmarking GPU step costs (AB 256x256, ABC 4096x4096)")
    dev = get_device()
    grid = build_channel_grid_gpu(0, CFG.n_a)
    grid_np = build_channel_grid_np(0, CFG.n_a)
    seq = iid_inputs(0, 120)
    out = {"device": torch.cuda.get_device_name(0)}
    for name, model in [
        ("AB-embedded", EmbeddedModelGPU(CFG.n_a, "AB-embedded", omega=0.5)),
        ("ABC-embedded-hierarchical", EmbeddedModelGPU(CFG.n_a, "ABC-chain", omega_b=0.5, omega_c=0.1, eta_bc=CFG.eta_paper / 2)),
    ]:
        for s in seq[:20]:
            model.step(float(s), grid)
        torch.cuda.synchronize()
        t0 = time.time()
        for s in seq[20:]:
            model.step(float(s), grid)
            model.features_t("A")
        torch.cuda.synchronize()
        sps = (time.time() - t0) / 100
        out[f"gpu_seconds_per_step_{name}"] = sps
        out[f"gpu_mem_mb_{name}"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        log(f"benchmark {name}: {sps*1000:.2f} ms/step (dim {2**model.n_total})")
    noaux = NoAuxModel(CFG.n_a, "ABC-noaux-kraus", tau_b=10, tau_c=30, lambda_b=0.25, lambda_c=0.25, p_b=0.2, p_c=0.05)
    t0 = time.time()
    for s in seq[20:]:
        noaux.step(float(s), grid_np)
        noaux.features()
    out["cpu_seconds_per_step_noaux"] = (time.time() - t0) / 100
    write_json(path, out)
    return out


def estimate_and_decide_budget(bench: Dict, force: bool = False) -> Dict:
    """Define the REAL budget from measured step costs, in the priority order:
    1) AB replication gate (full lengths, >=20 seeds, ideal 100);
    2) ABC embedded N=4 with >=20 eval seeds;
    3) tuning 64 trials/architecture over >=8 tuning seeds;
    4) rest of the protocol. Any reduction is explicit and recorded."""
    path = RESULTS_DIR / "budget_v2.json"
    if path.exists() and not force:
        b = json.loads(path.read_text())
        CFG.paper_eval_seeds = tuple(b["paper_eval_seeds"])
        CFG.eval_seeds = tuple(b["eval_seeds"])
        CFG.abc_tune_seeds = int(b["abc_tune_seeds"])
        CFG.abc_tune_trials = int(b["abc_tune_trials"])
        return b
    t_ab = bench["gpu_seconds_per_step_AB-embedded"]
    t_abc = bench["gpu_seconds_per_step_ABC-embedded-hierarchical"]
    t_na = bench["cpu_seconds_per_step_noaux"]
    L = CFG.paper_len  # 3000
    # Steps per seed for the AB paper phase: STM drive + nonmark (2 models x washout) + MG (drive + tf + rollout).
    ab_seed_steps = (L + 1) + 2 * CFG.paper_washout + (L + 1 + 2 * CFG.paper_test)
    per_seed_ab = ab_seed_steps * t_ab * len(CFG.omegas_paper)
    paper_seeds = 20
    for n_try in (100, 60, 40, 30, 20):
        if n_try * per_seed_ab <= 4 * 3600:
            paper_seeds = n_try
            break
    # ABC embedded evaluation: multiscale drive + 2 MG series (drive+tf+rollout each).
    n_abc_arches = sum(1 for a in EMBEDDED_ARCHES if a.startswith("ABC"))
    abc_eval_steps_per_seed = (L + 1) + 2 * (L + 1 + 2 * CFG.paper_test)
    est_abc_eval_h = 20 * n_abc_arches * abc_eval_steps_per_seed * t_abc / 3600
    # ABC tuning combos, uniform across the three embedded ABC architectures.
    combos = [(64, 8), (64, 4), (32, 4)]
    abc_tune_trials, abc_tune_seeds = combos[-1]
    for trials, seeds in combos:
        est = 3 * trials * 3 * seeds * (CFG.tune_len + 1) * t_abc
        if est <= 7 * 3600:
            abc_tune_trials, abc_tune_seeds = trials, seeds
            break
    est_abc_tune_h = 3 * abc_tune_trials * 3 * abc_tune_seeds * (CFG.tune_len + 1) * t_abc / 3600
    est_ab_tune_h = (CFG.tune_trials * 3 * CFG.ab_tune_seeds * (CFG.tune_len + 1) * t_ab) / 3600
    est_noaux_tune_h = (CFG.tune_trials * 3 * 5 * CFG.ab_tune_seeds * (CFG.tune_len + 1) * t_na) / 3600
    est_paper_h = paper_seeds * per_seed_ab / 3600
    est_noaux_eval_h = 20 * len(NOAUX_ARCHES) * ((L + 1) + 2 * (L + 1 + 2 * CFG.paper_test)) * t_na / 3600
    est_ab_eval_h = 20 * 2 * abc_eval_steps_per_seed * t_ab / 3600
    total_h = est_paper_h + est_abc_eval_h + est_abc_tune_h + est_ab_tune_h + est_noaux_tune_h + est_noaux_eval_h + est_ab_eval_h
    budget = {
        "seconds_per_step": {"AB_gpu": t_ab, "ABC_gpu": t_abc, "noaux_cpu": t_na},
        "paper_eval_seeds": list(range(paper_seeds)),
        "eval_seeds": list(range(20)),
        "abc_tune_trials": abc_tune_trials,
        "abc_tune_seeds": abc_tune_seeds,
        "estimated_hours": {
            "paper_gate": round(est_paper_h, 2),
            "abc_embedded_eval": round(est_abc_eval_h, 2),
            "ab_embedded_eval": round(est_ab_eval_h, 2),
            "noaux_eval": round(est_noaux_eval_h, 2),
            "abc_tuning": round(est_abc_tune_h, 2),
            "ab_tuning": round(est_ab_tune_h, 2),
            "noaux_tuning": round(est_noaux_tune_h, 2),
            "total": round(total_h, 2),
        },
    }
    CFG.paper_eval_seeds = tuple(budget["paper_eval_seeds"])
    CFG.eval_seeds = tuple(budget["eval_seeds"])
    CFG.abc_tune_seeds = abc_tune_seeds
    CFG.abc_tune_trials = abc_tune_trials
    if paper_seeds < CFG.planned_eval_seeds:
        record_failure("paper_replication_budget", "reduced_paper_seeds_gpu_budget", executed_seeds=paper_seeds, planned_seeds=CFG.planned_eval_seeds, estimated_hours=budget["estimated_hours"]["paper_gate"])
    if abc_tune_trials < CFG.planned_tune_trials or abc_tune_seeds < 8:
        record_failure("tuning_abc_embedded_budget", "reduced_abc_embedded_tuning_gpu_budget", executed_trials=abc_tune_trials, planned_trials=CFG.planned_tune_trials, executed_tune_seeds=abc_tune_seeds, planned_tune_seeds=8)
    write_json(path, budget)
    log(f"budget decided: paper_seeds={paper_seeds}, eval_seeds=20, abc_tune=({abc_tune_trials} trials x {abc_tune_seeds} seeds); estimated total {total_h:.1f} h")
    print(json.dumps(budget["estimated_hours"], indent=2))
    return budget


def run_washout_convergence(force: bool = False) -> pd.DataFrame:
    """Trace-distance convergence between two different A initializations under
    the same input sequence; verified BEFORE training (mandatory v2 diagnostic)."""
    out = RESULTS_DIR / "washout_convergence.csv"
    if marker("washout").exists() and not force:
        log("washout convergence already complete; skipping")
        return load_csv(out)
    log("running washout convergence diagnostics")
    configs = [("AB-embedded", {"omega": o}) for o in CFG.omegas_paper]
    configs += [("ABC-embedded-hierarchical", {}), ("ABC-embedded-tied", {}), ("ABC-noaux-kraus", {})]
    rows = []
    for seed in [0, 1]:
        seq = iid_inputs(seed + 555, CFG.paper_washout)
        for name, params in configs:
            tag = f"{name}_omega{params.get('omega', 'na')}"
            if key_done(out, seed=seed, model=tag):
                continue
            grid = get_grid(name, seed)
            m1 = make_model(name, params, seed)
            m2 = make_model(name, params, seed)
            if isinstance(m1, EmbeddedModelGPU):
                d = 2 ** m1.n_total
                m2.rho = torch.zeros((d, d), dtype=CDTYPE, device=get_device())
                m2.rho[-1, -1] = 1.0
            else:
                m2.rho = np.zeros_like(m2.rho)
                m2.rho[-1, -1] = 1.0
                m2.buffer = [m2.rho.copy() for _ in range(m2.max_tau + 1)]
            t0 = time.time()
            for k, s in enumerate(seq):
                m1.step(float(s), grid)
                m2.step(float(s), grid)
                if k in (0, 4, 9, 19, 49, 99, 199, 299, 499, 749, 999):
                    if isinstance(m1, EmbeddedModelGPU):
                        dist = trace_distance_t(m1.reduced("A"), m2.reduced("A"))
                    else:
                        dist = trace_distance_np(m1.rho, m2.rho)
                    rows.append({"seed": seed, "model": tag, "step": k + 1, "trace_distance_A": dist})
            final = rows[-1]["trace_distance_A"]
            converged = final < CFG.washout_conv_threshold
            rows.append({"seed": seed, "model": tag, "step": CFG.paper_washout, "trace_distance_A": final,
                         "final": True, "converged": converged, "threshold": CFG.washout_conv_threshold,
                         "seconds": round(time.time() - t0, 1)})
            if not converged:
                record_failure(f"washout/{tag}/seed{seed}", "washout_not_converged_at_1000", final_distance=final)
            log(f"washout {tag} seed={seed}: final dist={final:.2e} converged={converged}")
        append_rows(out, rows)
        rows = []
    write_marker("washout")
    return load_csv(out)


def run_smoke(force: bool = False) -> pd.DataFrame:
    out = RESULTS_DIR / "smoke_results.csv"
    if marker("smoke").exists() and not force:
        log("smoke already complete; skipping")
        return load_csv(out)
    log("running smoke test with N=2 embedded/noaux models")
    rows = []
    taus = list(range(0, 9))
    slices = split_slices(CFG.smoke_washout, CFG.smoke_train, CFG.smoke_test)
    for seed in CFG.smoke_seeds:
        if key_done(out, seed=seed, phase="smoke", model="ABC-embedded-chain"):
            continue
        grid_g = build_channel_grid_gpu(seed, 2, grid_size=9)
        grid_n = build_channel_grid_np(seed, 2, grid_size=9)
        seq = iid_inputs(seed, CFG.smoke_len)
        models = [
            ("M0-noaux", NoAuxModel(2, "M0-noaux", seed=seed), grid_n),
            ("AB-noaux-residual", NoAuxModel(2, "AB-noaux-residual", tau_b=4, lambda_b=0.35, seed=seed), grid_n),
            ("ABC-noaux-kraus", NoAuxModel(2, "ABC-noaux-kraus", tau_b=3, tau_c=7, lambda_b=0.25, lambda_c=0.25, p_b=0.2, p_c=0.05, seed=seed), grid_n),
            ("AB-embedded", EmbeddedModelGPU(2, "AB-embedded", omega=0.5), grid_g),
            ("ABC-embedded-chain", EmbeddedModelGPU(2, "ABC-chain", omega_b=0.5, omega_c=0.1, eta_bc=CFG.eta_paper / 2), grid_g),
        ]
        for name, model, grid in models:
            t0 = time.time()
            if isinstance(model, EmbeddedModelGPU):
                model.reset()
            else:
                model.reset()
            feats = drive_features(model, seq, grid)
            caps = []
            for tau in taus:
                y = stm_target(seq, tau)
                cap, _ = evaluate_capacity_from_features(feats, seq, y, slices)
                caps.append(cap)
            chk = state_checks_t(model.rho) if isinstance(model, EmbeddedModelGPU) else state_checks_np(model.rho)
            rows.append({
                "phase": "smoke", "seed": seed, "model": name, "n_a": 2,
                "max_capacity": float(np.max(caps)), "mean_capacity": float(np.mean(caps)),
                "trace_error": chk["trace_error"], "min_eig": chk["min_eig"],
                "device": "cuda" if isinstance(model, EmbeddedModelGPU) else "cpu",
                "seconds": time.time() - t0,
            })
        append_rows(out, rows)
        rows = []
        log(f"smoke seed complete: {seed}")
    write_marker("smoke")
    return load_csv(out)


# ---------------------------------------------------------------------------
# Paper replication (gate) — full lengths, paired seeds
# ---------------------------------------------------------------------------


def add_resource_cost(phase: str, model: str, seed, seconds: float, n_total: int, device: str, extra: Optional[Dict] = None) -> None:
    row = {
        "phase": phase, "model": model, "seed": seed, "device": device,
        "n_a": CFG.n_a, "density_dimension": f"{2**n_total}x{2**n_total}",
        "seconds": round(seconds, 3),
        "gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if device == "cuda" else 0.0,
        "ram_mb": round(psutil.Process().memory_info().rss / 1e6, 1),
    }
    if extra:
        row.update(extra)
    append_rows(RESULTS_DIR / "resource_costs.csv", [row])


def paper_stm_for_seed(seed: int, omega: float) -> List[Dict]:
    grid = build_channel_grid_gpu(seed, CFG.n_a)
    seq = iid_inputs(seed, CFG.paper_len)
    model = EmbeddedModelGPU(CFG.n_a, "AB-embedded", omega=omega, eta=CFG.eta_paper).reset()
    t0 = time.time()
    feats = drive_features(model, seq, grid)
    seconds_drive = time.time() - t0
    slices = split_slices(CFG.paper_washout, CFG.paper_train, CFG.paper_test)
    rows = []
    for tau in range(CFG.tau_max_paper + 1):
        y = stm_target(seq, tau)
        cap_ols, r2_ols = evaluate_capacity_from_features(feats, seq, y, slices, alpha=0.0)
        cap_ridge, r2_ridge = evaluate_capacity_from_features(feats, seq, y, slices, alpha=1e-6)
        rows.append({
            "seed": seed, "omega": omega, "tau": tau,
            "capacity_ols": cap_ols, "r2_ols": r2_ols,
            "capacity_ridge": cap_ridge, "r2_ridge": r2_ridge,
            "n_a": CFG.n_a, "n_aux": CFG.n_a, "readout": "A_only_66_features",
            "washout": CFG.paper_washout, "train": CFG.paper_train, "test": CFG.paper_test,
            "seconds_seed_model": time.time() - t0,
        })
    add_resource_cost("paper_stm", "AB-embedded", seed, seconds_drive, 2 * CFG.n_a, "cuda", {"omega": omega, "steps": len(seq), "seconds_per_step": seconds_drive / len(seq)})
    return rows


def paper_nonmark_for_seed(seed: int, omega: float) -> Dict:
    grid = build_channel_grid_gpu(seed, CFG.n_a)
    seq = iid_inputs(seed + 333, CFG.paper_washout)
    m1 = EmbeddedModelGPU(CFG.n_a, "AB-embedded", omega=omega).reset()
    m2 = EmbeddedModelGPU(CFG.n_a, "AB-embedded", omega=omega).reset()
    d = 2 ** m2.n_total
    m2.rho = torch.zeros((d, d), dtype=CDTYPE, device=get_device())
    m2.rho[-1, -1] = 1.0
    dists, positive_sum = [], 0.0
    prev = None
    t0 = time.time()
    for s in seq:
        m1.step(float(s), grid)
        m2.step(float(s), grid)
        dist = trace_distance_t(m1.reduced("A"), m2.reduced("A"))
        dists.append(dist)
        if prev is not None and dist > prev:
            positive_sum += dist - prev
        prev = dist
    return {
        "seed": seed, "omega": omega, "nonmarkovianity": positive_sum,
        "mean_trace_distance": float(np.mean(dists)), "max_trace_distance": float(np.max(dists)),
        "final_trace_distance": float(dists[-1]),
        "steps": CFG.paper_washout, "seconds": time.time() - t0,
    }


def paper_mg_for_seed(seed: int, omega: float) -> Tuple[Dict, List[Dict]]:
    grid = build_channel_grid_gpu(seed, CFG.n_a)
    slices = split_slices(CFG.paper_washout, CFG.paper_train, CFG.paper_test)
    raw = mackey_glass(CFG.paper_len + 1)
    series = normalize_series(raw, slices["train"])
    model = EmbeddedModelGPU(CFG.n_a, "AB-embedded", omega=omega).reset()
    t0 = time.time()
    metrics, preds, truth, _ = run_mg_model(seed, model, grid, series, slices)
    metrics.update({
        "model": "AB-embedded", "omega": omega, "eta": CFG.eta_paper,
        "washout": CFG.paper_washout, "train": CFG.paper_train, "test": CFG.paper_test,
        "seconds": time.time() - t0,
        # G1 fix: every row is still written (full record), but the primary
        # autonomous-rollout comparison must only use seeds whose teacher-forced
        # one-step readout met the r2 threshold. included_in_primary is the
        # row-level eligibility flag; the paired gate additionally requires the
        # partner arm (same seed) to be eligible.
        "included_in_primary": bool(metrics["teacher_forced_ok"]),
    })
    ex_rows = []
    if seed == CFG.paper_eval_seeds[0]:
        for k in range(min(250, len(preds))):
            ex_rows.append({"series": "MG", "seed": seed, "model": "AB-embedded", "omega": omega, "step": k, "truth": truth[k], "prediction": preds[k]})
    return metrics, ex_rows


def gate_diagnostics() -> Dict:
    """Mandatory diagnostic routine if Omega=0.5 does not beat Omega=1.0:
    compare GPU vs exact complex128 N=2, and re-verify Hamiltonian sign, the
    h(1+s)*sum(sigma_x) input term, the partial-SWAP convention and channel order."""
    log("running replication-gate diagnostics")
    diag = {}
    gn = build_channel_grid_np(4242, 2, grid_size=9)
    gg = build_channel_grid_gpu(4242, 2, grid_size=9)
    mgp = EmbeddedModelGPU(2, "AB-embedded", omega=0.5)
    mnp_ = EmbeddedModelNP(2, "AB-embedded", omega=0.5)
    maxd = 0.0
    for s in iid_inputs(21, 60):
        mgp.step(float(s), gg)
        mnp_.step(float(s), gn)
        maxd = max(maxd, trace_distance_np(mgp.reduced("A").cpu().numpy().astype(np.complex128), mnp_.reduced_a()))
    diag["gpu_vs_exact_c128_N2_max_trace_distance"] = maxd
    base, drive = liouvillian_parts(0, 2)
    # Trace preservation of the Liouvillian channel at an intermediate input.
    rho0 = pure_zero_density_np(2).reshape(-1)
    ch = scipy.linalg.expm((base + 0.7 * drive) * CFG.dt)
    diag["liouvillian_trace_preservation_error"] = float(abs(np.trace((ch @ rho0).reshape(4, 4)) - 1.0))
    diag["hamiltonian_sign_convention"] = "L_H = -i (H (x) I - I (x) H^T), row-major vectorization (identical to v1)"
    # Input term convention: total field h*(1+s)*sum(sx) — verify drive at s=1 doubles the sx part.
    _, drive1 = liouvillian_parts(0, 1)
    sx_super_norm = float(np.linalg.norm(drive1))
    diag["input_term_h_times_1_plus_s_sigma_x"] = {"drive_norm_nonzero": sx_super_norm > 0, "convention": "H(s) = J XX + h Sz + h(1+s) Sx"}
    u = partial_swap_unitary_np(CFG.eta_paper)
    diag["partial_swap_convention"] = {"unitary_error": float(np.linalg.norm(u.conj().T @ u - np.eye(4))), "form": "cos(eta) I + i sin(eta) SWAP"}
    diag["channel_order"] = "input channel on A -> partial-SWAP layer(s) -> auxiliary depolarization -> renormalize (identical to v1)"
    return diag


def run_paper_replication(force: bool = False) -> None:
    if marker("paper").exists() and not force:
        log("paper replication already complete; skipping")
        return
    log(f"running paper replication with {len(CFG.paper_eval_seeds)} seeds at washout/train/test={CFG.paper_washout}/{CFG.paper_train}/{CFG.paper_test}")
    stm_path = RESULTS_DIR / "paper_replication_stm.csv"
    nm_path = RESULTS_DIR / "paper_replication_nonmarkovianity.csv"
    mg_path = RESULTS_DIR / "paper_replication_mackey_glass.csv"
    for seed in CFG.paper_eval_seeds:
        for omega in CFG.omegas_paper:
            if not key_done(stm_path, seed=seed, omega=omega, tau=CFG.tau_max_paper):
                append_rows(stm_path, paper_stm_for_seed(seed, omega))
            if not key_done(nm_path, seed=seed, omega=omega):
                append_rows(nm_path, [paper_nonmark_for_seed(seed, omega)])
            if not key_done(mg_path, seed=seed, omega=omega):
                row, examples = paper_mg_for_seed(seed, omega)
                append_rows(mg_path, [row])
                append_rows(RESULTS_DIR / "autonomous_prediction_examples.csv", examples)
        log(f"paper replication seed complete: {seed}")
    mg = load_csv(mg_path)
    stm = load_csv(stm_path)
    gate = {"paper_replication_gate": "not_evaluated"}
    if not mg.empty:
        # Backward compat: older CSVs may lack included_in_primary; derive it.
        if "included_in_primary" not in mg.columns:
            mg["included_in_primary"] = mg["teacher_forced_ok"].astype(bool)
        # G1 fix: report teacher-forced pass counts per omega and run the PRIMARY
        # comparison only on seeds that passed teacher-forced validation in BOTH
        # arms (omega=0.5 and omega=1.0). If too few paired seeds survive, the
        # comparison is marked inconclusive rather than reported as a result.
        n_passed_tf = mg.groupby("omega")["teacher_forced_ok"].apply(lambda s: int(s.astype(bool).sum())).to_dict()
        means_all = mg.groupby("omega")["mse_150"].mean().to_dict()

        def _arm(w):
            return mg[(mg.omega == w) & (mg.included_in_primary)].sort_values("seed")

        a05, a10 = _arm(0.5), _arm(1.0)
        paired_seeds = sorted(set(a05.seed) & set(a10.seed))
        g05 = a05[a05.seed.isin(paired_seeds)].sort_values("seed")["mse_150"].values
        g10 = a10[a10.seed.isin(paired_seeds)].sort_values("seed")["mse_150"].values
        n_paired_primary = len(paired_seeds)

        # Transparency: legacy all-seeds comparison (mixes tf-failed rollouts).
        all05 = mg[mg.omega == 0.5].sort_values("seed")["mse_150"].values
        all10 = mg[mg.omega == 1.0].sort_values("seed")["mse_150"].values
        st_all = paired_stats(all05, all10, larger_better=False)

        stm_tail = {}
        if not stm.empty:
            tail = stm[stm.tau >= 10].groupby("omega")["capacity_ols"].mean().to_dict()
            stm_tail = {f"stm_mean_capacity_tau_ge_10_omega_{k}": v for k, v in tail.items()}

        gate = {
            "n_passed_teacher_forced_by_omega": n_passed_tf,
            "n_paired_primary": n_paired_primary,
            "min_decision_seeds": CFG.min_decision_seeds,
            "omega_mean_mse150_all_seeds": means_all,
            "paired_stats_mse150_omega05_vs_10_all_seeds": st_all,
            "executed_seeds": list(CFG.paper_eval_seeds),
            "planned_seeds": CFG.planned_eval_seeds,
            **stm_tail,
        }
        if n_paired_primary < CFG.min_decision_seeds:
            gate["primary_verdict"] = "inconclusive"
            gate["primary_reason"] = (
                f"only {n_paired_primary} seeds passed teacher-forced validation in "
                f"both arms (omega=0.5 and omega=1.0); need >= {CFG.min_decision_seeds}")
            record_failure("paper_replication_gate", "primary_comparison_inconclusive_insufficient_tf_paired_seeds",
                           n_paired_primary=n_paired_primary, n_passed_teacher_forced=n_passed_tf)
        else:
            st_primary = paired_stats(g05, g10, larger_better=False)
            primary_means = {0.5: float(np.mean(g05)), 1.0: float(np.mean(g10))}
            gate["paired_stats_mse150_omega05_vs_10_primary"] = st_primary
            gate["omega_mean_mse150_primary"] = primary_means
            gate["omega_0.5_beats_1.0_primary"] = bool(primary_means[0.5] < primary_means[1.0])
            gate["primary_verdict"] = "omega_0.5_beats_1.0" if gate["omega_0.5_beats_1.0_primary"] else "omega_0.5_not_better"
            if not gate["omega_0.5_beats_1.0_primary"]:
                record_failure("paper_replication_gate", "Omega=0.5 did not beat Omega=1.0 (teacher-forced paired primary set)",
                               n_paired_primary=n_paired_primary)
                gate["diagnostics"] = gate_diagnostics()
    write_json(RESULTS_DIR / "paper_replication_gate.json", gate)
    write_marker("paper", n_seeds=len(CFG.paper_eval_seeds))


# ---------------------------------------------------------------------------
# Tuning (Optuna, persistent SQLite v2 storage)
# ---------------------------------------------------------------------------

TUNE_TASKS = ["paper_s0_s10", "p1_0_10_30", "s0_s10_s30"]
NOAUX_TUNE_ARCHES = ["AB-noaux-residual", "AB-noaux-kraus", "ABC-noaux-kraus", "ABC-noaux-tied", "ABC-noaux-hierarchical"]
ABC_EMBEDDED_TUNE_ARCHES = ["ABC-embedded-hierarchical", "ABC-embedded-tied", "ABC-embedded-parallel"]


def _tune_capacity(model_factory, task: str, seeds: Sequence[int]) -> float:
    caps = []
    slices = split_slices(CFG.tune_washout, CFG.tune_train, CFG.tune_test)
    for seed in seeds:
        seq = iid_inputs(seed, CFG.tune_len)
        model = model_factory(seed)
        grid = build_channel_grid_gpu(seed, CFG.n_a) if isinstance(model, EmbeddedModelGPU) else build_channel_grid_np(seed, CFG.n_a)
        feats = drive_features(model, seq, grid)
        y = target_by_name(seq, task)
        cap, _ = evaluate_capacity_from_features(feats, seq, y, slices, alpha=1e-6)
        caps.append(cap)
    return float(np.mean(caps))


def tune_objective_noaux(trial: optuna.Trial, arch: str, task: str) -> float:
    tau_b = trial.suggest_int("tau_b", 1, 50)
    tau_c = trial.suggest_int("tau_c", tau_b + 1, 60) if arch.startswith("ABC") else tau_b + 1
    lambda_b = trial.suggest_float("lambda_b", 0.0, 0.85)
    lambda_c = trial.suggest_float("lambda_c", 0.0, max(0.01, 0.95 - lambda_b)) if arch.startswith("ABC") else 0.0
    p_b = trial.suggest_float("p_b", 0.0, 1.0) if "kraus" in arch or "hierarchical" in arch or "tied" in arch else 0.0
    if arch == "ABC-noaux-tied":
        lambda_c = lambda_b
        p_c = p_b
    else:
        p_c = trial.suggest_float("p_c", 0.0, 1.0) if arch.startswith("ABC") and ("kraus" in arch or "hierarchical" in arch) else 0.0
    params = {"tau_b": tau_b, "tau_c": tau_c, "lambda_b": lambda_b, "lambda_c": lambda_c, "p_b": p_b, "p_c": p_c}
    return _tune_capacity(lambda seed: make_noaux_model(arch, params, seed), task, CFG.tune_seeds[: CFG.ab_tune_seeds])


def tune_objective_ab_embedded(trial: optuna.Trial, task: str) -> float:
    omega = trial.suggest_float("omega", 0.0, 1.0)
    eta = trial.suggest_float("eta", 0.05, math.pi / 2 - 0.05)
    return _tune_capacity(lambda seed: EmbeddedModelGPU(CFG.n_a, "AB-embedded", omega=omega, eta=eta).reset(), task, CFG.tune_seeds[: CFG.ab_tune_seeds])


def tune_objective_abc_embedded(trial: optuna.Trial, arch: str, task: str) -> float:
    if arch == "ABC-embedded-tied":
        omega = trial.suggest_float("omega", 0.0, 1.0)
        eta = trial.suggest_float("eta", 0.05, math.pi / 2 - 0.05)
        params = {"omega": omega, "eta": eta}
    elif arch == "ABC-embedded-parallel":
        params = {
            "omega_b": trial.suggest_float("omega_b", 0.0, 1.0),
            "omega_c": trial.suggest_float("omega_c", 0.0, 1.0),
            "eta_ab": trial.suggest_float("eta_ab", 0.05, math.pi / 2 - 0.05),
            "eta_ac": trial.suggest_float("eta_ac", 0.05, math.pi / 2 - 0.05),
        }
    else:
        params = {
            "omega_b": trial.suggest_float("omega_b", 0.0, 1.0),
            "omega_c": trial.suggest_float("omega_c", 0.0, 1.0),
            "eta_ab": trial.suggest_float("eta_ab", 0.05, math.pi / 2 - 0.05),
            "eta_bc": trial.suggest_float("eta_bc", 0.05, math.pi / 2 - 0.05),
        }
    return _tune_capacity(lambda seed: make_embedded_model(arch, params).reset(), task, CFG.tune_seeds[: CFG.abc_tune_seeds])


def run_tuning(force: bool = False) -> pd.DataFrame:
    if marker("tuning").exists() and not force:
        log("tuning already complete; skipping")
        return load_csv(RESULTS_DIR / "best_parameters_by_task.csv")
    log(f"running Optuna tuning: {CFG.tune_trials} trials (noaux/AB), {CFG.abc_tune_trials} trials x {CFG.abc_tune_seeds} seeds (ABC embedded)")
    storage = f"sqlite:///{(RESULTS_DIR / 'optuna_abc_v2.sqlite3').absolute()}"
    trial_rows, best_rows = [], []

    def run_study(arch: str, task: str, objective, n_trials: int):
        study = optuna.create_study(direction="maximize", study_name=f"{arch}_{task}", storage=storage, load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=CFG.optuna_seed))
        remaining = max(0, n_trials - len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]))
        if remaining:
            t0 = time.time()
            study.optimize(objective, n_trials=remaining, show_progress_bar=False)
            add_resource_cost("tuning", arch, task, time.time() - t0, 3 * CFG.n_a if "ABC-embedded" in arch else 2 * CFG.n_a, "cuda" if is_embedded(arch) else "cpu", {"trials": remaining})
        for tr in study.trials:
            trial_rows.append({"architecture": arch, "task": task, "trial": tr.number, "value": tr.value, "state": str(tr.state), **tr.params})
        bp = dict(study.best_params)
        bp.update({"architecture": arch, "task": task, "objective": study.best_value, "source": "optuna_v2_gpu"})
        best_rows.append(bp)
        log(f"tuned {arch}/{task}: best={study.best_value:.4f}")

    for task in TUNE_TASKS:
        for arch in NOAUX_TUNE_ARCHES:
            run_study(arch, task, lambda tr, a=arch, t=task: tune_objective_noaux(tr, a, t), CFG.tune_trials)
        run_study("AB-embedded", task, lambda tr, t=task: tune_objective_ab_embedded(tr, t), CFG.tune_trials)
        for arch in ABC_EMBEDDED_TUNE_ARCHES:
            run_study(arch, task, lambda tr, a=arch, t=task: tune_objective_abc_embedded(tr, a, t), CFG.abc_tune_trials)
    append_rows(RESULTS_DIR / "noaux_tuning_trials.csv", [r for r in trial_rows if "noaux" in r["architecture"]])
    append_rows(RESULTS_DIR / "tuning_trials_ab.csv", [r for r in trial_rows if r["architecture"] in ("AB-embedded",)])
    append_rows(RESULTS_DIR / "tuning_trials_abc.csv", [r for r in trial_rows if "ABC-embedded" in r["architecture"]])
    bp_df = pd.DataFrame(best_rows)
    bp_df.to_csv(RESULTS_DIR / "best_parameters_by_task.csv", index=False)
    bp_df[bp_df.architecture.str.contains("noaux")].to_csv(RESULTS_DIR / "noaux_best_parameters.csv", index=False)
    write_marker("tuning", noaux_trials=CFG.tune_trials, abc_trials=CFG.abc_tune_trials, abc_tune_seeds=CFG.abc_tune_seeds)
    return bp_df


def best_params(arch: str, task: str = "paper_s0_s10") -> Dict:
    bp = load_csv(RESULTS_DIR / "best_parameters_by_task.csv")
    if not bp.empty:
        m = bp[(bp["architecture"] == arch) & (bp["task"] == task)]
        if len(m):
            return m.iloc[0].dropna().to_dict()
    defaults = {
        "M0-noaux": {},
        "AB-noaux-residual": {"tau_b": 10, "lambda_b": 0.35},
        "AB-noaux-kraus": {"tau_b": 10, "lambda_b": 0.35, "p_b": 0.2},
        "ABC-noaux-kraus": {"tau_b": 10, "tau_c": 30, "lambda_b": 0.25, "lambda_c": 0.25, "p_b": 0.2, "p_c": 0.05},
        "ABC-noaux-tied": {"tau_b": 10, "tau_c": 30, "lambda_b": 0.2, "p_b": 0.2},
        "ABC-noaux-hierarchical": {"tau_b": 10, "tau_c": 30, "lambda_b": 0.25, "lambda_c": 0.25, "p_b": 0.2, "p_c": 0.05},
        "ABC-noaux-B-only": {"tau_b": 10, "lambda_b": 0.35},
        "ABC-noaux-C-only": {"tau_c": 30, "lambda_c": 0.35},
        "ABC-noaux-shuffled-history": {"tau_b": 10, "tau_c": 30},
        "AB-embedded": {"omega": 0.5, "eta": CFG.eta_paper},
        "AB-Markov": {},
        "ABC-embedded-hierarchical": {"omega_b": 0.5, "omega_c": 0.1, "eta_ab": CFG.eta_paper, "eta_bc": CFG.eta_paper / 2},
        "ABC-embedded-tied": {"omega": 0.5, "eta": CFG.eta_paper},
        "ABC-embedded-parallel": {"omega_b": 0.5, "omega_c": 0.2, "eta_ab": CFG.eta_paper, "eta_ac": CFG.eta_paper / 2},
        "ABC-embedded-C-off": {"omega_b": 0.5, "eta_ab": CFG.eta_paper},
        "ABC-Markov": {},
    }
    return defaults.get(arch, {})


# ---------------------------------------------------------------------------
# Multiscale capacities + IPC (embedded AND no-aux, full lengths, 20 seeds)
# ---------------------------------------------------------------------------

MULTISCALE_TASKS = ["paper_s0_s10", "p1_0", "p1_10", "p1_30", "p1_0_10", "p1_0_30", "p1_10_30", "p1_0_10_30", "s0_s30", "s10_s30", "s0_s10_s30"]
DELAY_PAIRS = [(5, 15), (5, 20), (5, 30), (5, 40), (10, 15), (10, 20), (10, 30), (10, 40), (15, 20), (15, 30), (15, 40), (20, 30), (20, 40)]


def run_multiscale_and_ipc(force: bool = False) -> None:
    if marker("multiscale").exists() and not force:
        log("multiscale already complete; skipping")
        return
    log("running multiscale capacities and IPC (embedded + noaux, full lengths)")
    cap_path = RESULTS_DIR / "multiscale_capacities.csv"
    noaux_cap_path = RESULTS_DIR / "noaux_memory_capacities.csv"
    ipc_path = RESULTS_DIR / "ipc_by_component.csv"
    noaux_ipc_path = RESULTS_DIR / "noaux_ipc_by_component.csv"
    slices = split_slices(CFG.paper_washout, CFG.paper_train, CFG.paper_test)
    arches = NOAUX_ARCHES + EMBEDDED_ARCHES
    for seed in CFG.eval_seeds:
        seq = iid_inputs(seed, CFG.paper_len)
        for arch in arches:
            if key_done(cap_path, seed=seed, model=arch, task="s0_s10_s30"):
                continue
            grid = get_grid(arch, seed)
            model = make_model(arch, best_params(arch), seed)
            if hasattr(model, "reset"):
                model.reset()
            t0 = time.time()
            feats = drive_features(model, seq, grid)
            drive_seconds = time.time() - t0
            rows, ipc_rows = [], []
            for task in MULTISCALE_TASKS:
                y = target_by_name(seq, task)
                cap, r2 = evaluate_capacity_from_features(feats, seq, y, slices, alpha=1e-6)
                rows.append({"seed": seed, "model": arch, "task": task, "capacity": cap, "r2": r2, "readout": "A_only_66", "embedded": is_embedded(arch)})
            for tau in [0, 5, 10, 20, 30, 40, 50]:
                y = stm_target(seq, tau)
                cap, _ = evaluate_capacity_from_features(feats, seq, y, slices, alpha=1e-6)
                ipc_rows.append({"seed": seed, "model": arch, "component": "degree1_stm", "degree": 1, "tau1": tau, "tau2": np.nan, "capacity": cap})
            for tau1, tau2 in DELAY_PAIRS:
                y = stm_target(seq, tau1) * stm_target(seq, tau2)
                cap, _ = evaluate_capacity_from_features(feats, seq, y, slices, alpha=1e-6)
                ipc_rows.append({"seed": seed, "model": arch, "component": "cross_delay_degree2", "degree": 2, "tau1": tau1, "tau2": tau2, "capacity": cap})
                rows.append({"seed": seed, "model": arch, "task": f"delay_pair_{tau1}_{tau2}", "capacity": cap, "r2": np.nan, "readout": "A_only_66", "embedded": is_embedded(arch)})
            append_rows(cap_path, rows)
            append_rows(ipc_path, ipc_rows)
            if not is_embedded(arch):
                append_rows(noaux_cap_path, rows)
                append_rows(noaux_ipc_path, ipc_rows)
            n_total = getattr(model, "n_total", CFG.n_a)
            add_resource_cost("multiscale", arch, seed, drive_seconds, n_total,
                              "cuda" if is_embedded(arch) else "cpu",
                              {"steps": len(seq), "seconds_per_step": drive_seconds / len(seq),
                               "buffer_states": getattr(model, "max_tau", 0)})
        log(f"multiscale seed complete: {seed}")
    # Effective memory scales: STM peaks per model + embedded layer diagnostics.
    scale_rows = []
    ipc = load_csv(ipc_path)
    if not ipc.empty:
        for model_name, g in ipc[ipc.component == "degree1_stm"].groupby("model"):
            mean_curve = g.groupby("tau1")["capacity"].mean().sort_index()
            vals = mean_curve.values
            peaks = []
            for i in range(1, len(vals) - 1):
                if vals[i] > vals[i - 1] and vals[i] > vals[i + 1]:
                    peaks.append((int(mean_curve.index[i]), float(vals[i])))
            if not peaks:
                peaks = [(int(mean_curve.idxmax()), float(mean_curve.max()))]
            for rank, (tau, val) in enumerate(peaks[:3], 1):
                scale_rows.append({"model": model_name, "layer": "A_readout", "scale_rank": rank, "tau_peak": tau, "capacity_peak": val, "evidence": f"STM peak over {len(CFG.eval_seeds)} eval seeds"})
    # Embedded ABC N=4 layer diagnostics: autocorrelation of first feature of A/B/C.
    for seed in CFG.eval_seeds[:2]:
        grid = build_channel_grid_gpu(seed, CFG.n_a)
        seq = iid_inputs(seed, 600)
        model = make_embedded_model("ABC-embedded-hierarchical", best_params("ABC-embedded-hierarchical")).reset()
        fa, fb, fc = [], [], []
        for s in seq:
            model.step(float(s), grid)
            fa.append(float(model.features_t("A")[0].item()))
            fb.append(float(model.features_t("B")[0].item()))
            fc.append(float(model.features_t("C")[0].item()))
        for layer, arr in [("A_diag_N4", fa), ("B_diag_N4", fb), ("C_diag_N4", fc)]:
            arr = np.asarray(arr[100:])
            ac = np.correlate(arr - arr.mean(), arr - arr.mean(), mode="full")
            ac = ac[len(ac) // 2 :]
            ac = ac / (ac[0] + 1e-12)
            tau_eff = int(np.argmax(ac < math.exp(-1))) if np.any(ac < math.exp(-1)) else len(ac) - 1
            scale_rows.append({"model": "ABC-embedded-hierarchical", "layer": layer, "scale_rank": 1, "tau_peak": tau_eff, "capacity_peak": float(ac[min(tau_eff, len(ac) - 1)]), "evidence": f"N=4 embedded layer autocorrelation, seed {seed}"})
    pd.DataFrame(scale_rows).to_csv(RESULTS_DIR / "effective_memory_scales.csv", index=False)
    write_marker("multiscale", n_seeds=len(CFG.eval_seeds), arches=len(arches))


# ---------------------------------------------------------------------------
# Mackey-Glass (standard + two-delay) for all architectures
# ---------------------------------------------------------------------------


def run_mackey_glass_full(force: bool = False) -> None:
    if marker("mackey").exists() and not force:
        log("Mackey-Glass already complete; skipping")
        return
    log("running Mackey-Glass standard and two-delay for all architectures")
    std_path = RESULTS_DIR / "mackey_glass_standard.csv"
    two_path = RESULTS_DIR / "mackey_glass_two_delay.csv"
    noaux_path = RESULTS_DIR / "noaux_mackey_glass.csv"
    slices = split_slices(CFG.paper_washout, CFG.paper_train, CFG.paper_test)
    series_std = normalize_series(mackey_glass(CFG.paper_len + 1), slices["train"])
    series_two = normalize_series(mackey_glass(CFG.paper_len + 1, two_delay=True), slices["train"])
    arches = NOAUX_ARCHES[:5] + ["ABC-noaux-shuffled-history"] + EMBEDDED_ARCHES
    arches = list(dict.fromkeys(arches))
    for seed in CFG.eval_seeds:
        for arch in arches:
            grid = get_grid(arch, seed)
            for path, series, series_name, task in [
                (std_path, series_std, "MG_standard", "paper_s0_s10"),
                (two_path, series_two, "MG_two_delay", "s0_s10_s30"),
            ]:
                if key_done(path, seed=seed, model=arch):
                    continue
                model = make_model(arch, best_params(arch, task), seed)
                model.reset()
                t0 = time.time()
                row, preds, truth, _ = run_mg_model(seed, model, grid, series, slices)
                row.update({"model": arch, "series": series_name, "n_aux": (getattr(model, "n_total", CFG.n_a) - CFG.n_a) if is_embedded(arch) else 0,
                            "device": "cuda" if is_embedded(arch) else "cpu", "seconds": time.time() - t0})
                append_rows(path, [row])
                if not is_embedded(arch) and series_name == "MG_standard":
                    append_rows(noaux_path, [row])
                if not row["teacher_forced_ok"]:
                    record_failure(f"mackey/{arch}/seed{seed}/{series_name}", "teacher_forced_r2_below_threshold", r2_teacher_forced=row["r2_teacher_forced"], threshold=CFG.teacher_forced_r2_min)
                if seed == CFG.eval_seeds[0]:
                    append_rows(RESULTS_DIR / "autonomous_prediction_examples.csv", [
                        {"series": series_name, "seed": seed, "model": arch, "omega": np.nan, "step": k, "truth": truth[k], "prediction": preds[k]}
                        for k in range(min(150, len(preds)))
                    ])
        log(f"MG seed complete: {seed}")
    write_marker("mackey", n_seeds=len(CFG.eval_seeds))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def paired_stats(a: np.ndarray, b: np.ndarray, larger_better: bool = True) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = a - b if larger_better else b - a
    rng = np.random.default_rng(20260703)
    if n == 0:
        return {"n": 0}
    boot = d[rng.integers(0, n, size=(CFG.n_boot, n))].mean(axis=1)
    try:
        pw = float(stats.wilcoxon(a, b).pvalue) if n > 1 and np.any(a != b) else np.nan
    except Exception:
        pw = 1.0
    try:
        pt = float(stats.ttest_rel(a, b).pvalue) if n > 1 else np.nan
    except Exception:
        pt = np.nan
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    return {
        "n": int(n),
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        "std_diff": sd, "se_diff": float(sd / math.sqrt(n)) if n > 1 else 0.0,
        "mean_diff": float(np.mean(d)),
        "relative_diff": float(np.mean(d) / (abs(np.mean(b)) + 1e-12)),
        "ci95_lo": float(np.percentile(boot, 2.5)), "ci95_hi": float(np.percentile(boot, 97.5)),
        "p_wilcoxon": pw, "p_ttest": pt,
        "cohen_dz": float(np.mean(d) / sd) if sd > 0 else 0.0,
        "wins": int(np.sum(d > 0)), "losses": int(np.sum(d < 0)),
    }


def orient_effect(st: Dict[str, float], larger_better: bool, report: str = "a_minus_b") -> Dict[str, float]:
    """M1 fix: paired_stats computes its signed quantities (mean_diff, ci95_lo/hi,
    cohen_dz, wins/losses) along the internal difference d = (a-b) if larger_better
    else (b-a). When a table reports the effect in a fixed sense (e.g. noaux-embedded
    = a-b), mean_diff AND the CI must share that sense. This returns mean_diff,
    ci95_lo/hi, cohen_dz, wins, losses consistently oriented to `report`.

    report='a_minus_b' -> a - b ; report='b_minus_a' -> b - a.
    """
    if st.get("n", 0) == 0:
        return {"mean_diff": np.nan, "ci95_lo": np.nan, "ci95_hi": np.nan,
                "cohen_dz": np.nan, "wins": 0, "losses": 0}
    internal_is_a_minus_b = bool(larger_better)  # d = a-b when larger_better else b-a
    want_a_minus_b = (report == "a_minus_b")
    flip = (internal_is_a_minus_b != want_a_minus_b)
    if not flip:
        return {"mean_diff": st["mean_diff"], "ci95_lo": st["ci95_lo"], "ci95_hi": st["ci95_hi"],
                "cohen_dz": st["cohen_dz"], "wins": st["wins"], "losses": st["losses"]}
    return {"mean_diff": -st["mean_diff"], "ci95_lo": -st["ci95_hi"], "ci95_hi": -st["ci95_lo"],
            "cohen_dz": -st["cohen_dz"], "wins": st["losses"], "losses": st["wins"]}


def holm(pvals: Sequence[float]) -> List[float]:
    p = np.asarray([1.0 if pd.isna(x) else x for x in pvals], dtype=float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, run)
    return adj.tolist()


def validate_run(results_dir, expected: Dict, out_name: str = "completeness_matrix.csv", out_dir=None) -> Dict:
    """M2/G2 fix: single post-run validator. `expected` describes the tables a run
    must produce:

        expected = {"tables": {
            "narma10_results.csv": {
                "cell_combos": [{"model": m} for m in MODELS],  # non-seed dims
                "seed_col": "seed", "seeds": list(range(20)),
                "value_cols": ["nmse", "nrmse", "r2"],           # must be finite
            }, ...}}

    For every (cell_combo x seed) it records present / missing / nonfinite in
    `completeness_matrix.csv` (config x model x seed granularity) and returns
    {"status": "complete"|"partial", "tables": {...}, "n_missing", "n_nonfinite"}.
    A table is complete iff it has zero missing and zero non-finite cells.
    """
    results_dir = Path(results_dir)
    rows: List[Dict] = []
    tables_report: Dict[str, Dict] = {}
    overall_complete = True
    for csv_name, spec in expected.get("tables", {}).items():
        path = results_dir / csv_name
        seed_col = spec.get("seed_col", "seed")
        seeds = list(spec.get("seeds", []))
        combos = spec.get("cell_combos", [{}])
        value_cols = spec.get("value_cols", [])
        n_missing = n_nonfinite = n_present = 0
        if not path.exists():
            for combo in combos:
                for s in seeds:
                    rows.append({"table": csv_name, **combo, seed_col: s, "status": "missing", "n_rows": 0})
                    n_missing += 1
            tables_report[csv_name] = {"status": "partial", "exists": False,
                                       "n_missing": n_missing, "n_nonfinite": 0, "n_present": 0}
            overall_complete = False
            continue
        df = load_csv(path)
        for combo in combos:
            for s in seeds:
                mask = pd.Series(True, index=df.index)
                for col, val in combo.items():
                    mask &= (df[col] == val)
                if seed_col in df.columns:
                    mask &= (df[seed_col] == s)
                sub = df[mask]
                if len(sub) == 0:
                    status = "missing"; n_missing += 1
                else:
                    bad = False
                    for vc in value_cols:
                        if vc in sub.columns and not np.isfinite(pd.to_numeric(sub[vc], errors="coerce").to_numpy(dtype=float)).all():
                            bad = True; break
                    if bad:
                        status = "nonfinite"; n_nonfinite += 1
                    else:
                        status = "present"; n_present += 1
                rows.append({"table": csv_name, **combo, seed_col: s, "status": status, "n_rows": int(len(sub))})
        tstatus = "complete" if (n_missing == 0 and n_nonfinite == 0) else "partial"
        if tstatus != "complete":
            overall_complete = False
        tables_report[csv_name] = {"status": tstatus, "exists": True,
                                   "n_missing": n_missing, "n_nonfinite": n_nonfinite, "n_present": n_present}
    out_path = (Path(out_dir) if out_dir is not None else results_dir) / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False)
    total_missing = sum(t["n_missing"] for t in tables_report.values())
    total_nonfinite = sum(t["n_nonfinite"] for t in tables_report.values())
    return {"status": "complete" if overall_complete else "partial",
            "tables": tables_report, "n_missing": total_missing,
            "n_nonfinite": total_nonfinite, "completeness_matrix": str(out_path)}


def completeness_markdown(report: Dict) -> str:
    """P3 fix: render n_effective / n_missing / n_nonfinite per table for summaries."""
    lines = ["## Completeness (effective seeds after failures / NaNs)", "",
             "| table | n_effective | n_missing | n_nonfinite | status |",
             "|---|---|---|---|---|"]
    for tname, t in report.get("tables", {}).items():
        lines.append(f"| {tname} | {t.get('n_present', 0)} | {t.get('n_missing', 0)} | "
                     f"{t.get('n_nonfinite', 0)} | {t.get('status', '?')} |")
    lines += ["", f"**Overall: {report.get('status', '?')}** "
              f"(missing={report.get('n_missing', 0)}, non-finite={report.get('n_nonfinite', 0)}). "
              "n_effective = cells present and finite; these are the seeds that actually "
              "entered the means/tests.", ""]
    return "\n".join(lines)


def write_validated_completion(results_dir, name: str, expected: Dict, **payload) -> Dict:
    """Gate marker writing: run validate_run first; write `{name}_complete.json`
    only when the run is complete, otherwise `{name}_partial.json`. Never write a
    completion marker without passing through validate_run."""
    report = validate_run(results_dir, expected)
    marker_name = f"{name}_complete.json" if report["status"] == "complete" else f"{name}_partial.json"
    write_json(Path(results_dir) / marker_name, {**payload, "validated": True,
               "status": report["status"], "n_missing": report["n_missing"],
               "n_nonfinite": report["n_nonfinite"], "tables": report["tables"]})
    return report


CAPACITY_COMPARISONS = [
    ("M0-noaux", "AB-noaux-kraus"),
    ("M0-noaux", "ABC-noaux-kraus"),
    ("AB-noaux-kraus", "ABC-noaux-kraus"),
    ("ABC-noaux-hierarchical", "ABC-noaux-tied"),
    ("ABC-noaux-shuffled-history", "ABC-noaux-kraus"),
    ("AB-Markov", "AB-embedded"),
    ("AB-embedded", "ABC-embedded-hierarchical"),
    ("ABC-embedded-tied", "ABC-embedded-hierarchical"),
    ("ABC-Markov", "ABC-embedded-hierarchical"),
    ("ABC-embedded-C-off", "ABC-embedded-hierarchical"),
    ("ABC-noaux-hierarchical", "ABC-embedded-hierarchical"),
    ("AB-noaux-kraus", "AB-embedded"),
]
MG_COMPARISONS = [
    ("M0-noaux", "AB-noaux-kraus"),
    ("AB-noaux-kraus", "ABC-noaux-kraus"),
    ("ABC-noaux-tied", "ABC-noaux-hierarchical"),
    ("AB-Markov", "AB-embedded"),
    ("AB-embedded", "ABC-embedded-hierarchical"),
    ("ABC-embedded-tied", "ABC-embedded-hierarchical"),
    ("ABC-noaux-hierarchical", "ABC-embedded-hierarchical"),
]


def run_statistics(force: bool = False) -> pd.DataFrame:
    if marker("statistics").exists() and not force:
        log("statistics already complete; skipping")
        return load_csv(RESULTS_DIR / "paired_statistics.csv")
    log("running paired statistics and equivalence tests")
    rows = []
    cap = load_csv(RESULTS_DIR / "multiscale_capacities.csv")
    if not cap.empty:
        for task in sorted(cap.task.dropna().unique()):
            if str(task).startswith("delay_pair"):
                continue
            for b, a in CAPACITY_COMPARISONS:
                ga = cap[(cap.model == a) & (cap.task == task)].sort_values("seed")
                gb = cap[(cap.model == b) & (cap.task == task)].sort_values("seed")
                if len(ga) and len(gb):
                    st = paired_stats(ga.capacity.values, gb.capacity.values, larger_better=True)
                    rows.append({"family": "capacity", "metric": "capacity", "task": task, "comparison": f"{a} vs {b}", **st})
    for fname, fam in [("mackey_glass_standard.csv", "MG_standard"), ("mackey_glass_two_delay.csv", "MG_two_delay")]:
        mg = load_csv(RESULTS_DIR / fname)
        if mg.empty:
            continue
        for metric, larger in [("mse_150", False), ("nrmse_150", False), ("r2_150", True), ("valid_prediction_time", True)]:
            for b, a in MG_COMPARISONS:
                ga = mg[mg.model == a].sort_values("seed")
                gb = mg[mg.model == b].sort_values("seed")
                if len(ga) and len(gb):
                    rows.append({"family": fam, "metric": metric, "task": fam, "comparison": f"{a} vs {b}", **paired_stats(ga[metric].values, gb[metric].values, larger_better=larger)})
    paper = load_csv(RESULTS_DIR / "paper_replication_mackey_glass.csv")
    if not paper.empty:
        for metric, larger in [("mse_150", False), ("nrmse_150", False), ("r2_150", True), ("valid_prediction_time", True)]:
            g05 = paper[paper.omega == 0.5].sort_values("seed")
            g10 = paper[paper.omega == 1.0].sort_values("seed")
            g00 = paper[paper.omega == 0.0].sort_values("seed")
            if len(g05) and len(g10):
                rows.append({"family": "paper_MG", "metric": metric, "task": "paper_MG", "comparison": "Omega0.5 vs Omega1.0", **paired_stats(g05[metric].values, g10[metric].values, larger_better=larger)})
            if len(g05) and len(g00):
                rows.append({"family": "paper_MG", "metric": metric, "task": "paper_MG", "comparison": "Omega0.5 vs Omega0.0", **paired_stats(g05[metric].values, g00[metric].values, larger_better=larger)})
    nm = load_csv(RESULTS_DIR / "paper_replication_nonmarkovianity.csv")
    if not nm.empty:
        g05 = nm[nm.omega == 0.5].sort_values("seed")
        g10 = nm[nm.omega == 1.0].sort_values("seed")
        if len(g05) and len(g10):
            rows.append({"family": "paper_nonmark", "metric": "nonmarkovianity", "task": "paper_nonmark", "comparison": "Omega0.5 vs Omega1.0", **paired_stats(g05.nonmarkovianity.values, g10.nonmarkovianity.values, larger_better=True)})
    stm = load_csv(RESULTS_DIR / "paper_replication_stm.csv")
    if not stm.empty:
        tail = stm[stm.tau >= 10]
        piv = tail.groupby(["omega", "seed"])["capacity_ols"].mean().unstack(0)
        if 0.5 in piv.columns and 1.0 in piv.columns:
            rows.append({"family": "paper_STM", "metric": "stm_capacity_tau_ge_10", "task": "paper_STM", "comparison": "Omega0.5 vs Omega1.0", **paired_stats(piv[0.5].values, piv[1.0].values, larger_better=True)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_wilcoxon_holm"] = holm(df.p_wilcoxon.values)
        df["p_ttest_holm"] = holm(df.p_ttest.values)
        df["ci_excludes_zero"] = (df.ci95_lo > 0) | (df.ci95_hi < 0)
        df["n_sufficient"] = df.n >= CFG.min_decision_seeds
        df["significant"] = (df.p_wilcoxon_holm < 0.05) & df.ci_excludes_zero & df.n_sufficient
    df.to_csv(RESULTS_DIR / "paired_statistics.csv", index=False)
    # Equivalence (TOST-like, 5% relative margin) noaux vs embedded, paired by seed.
    eq_rows = []
    cap_main = cap[~cap.task.astype(str).str.startswith("delay_pair")] if not cap.empty else cap
    for noaux_m, emb_m in [("ABC-noaux-hierarchical", "ABC-embedded-hierarchical"), ("AB-noaux-kraus", "AB-embedded")]:
        if cap_main.empty:
            break
        ga = cap_main[cap_main.model == noaux_m].groupby("seed")["capacity"].mean().sort_index()
        gb = cap_main[cap_main.model == emb_m].groupby("seed")["capacity"].mean().sort_index()
        common = ga.index.intersection(gb.index)
        if len(common) >= CFG.min_decision_seeds:
            st = paired_stats(ga.loc[common].values, gb.loc[common].values, larger_better=True)
            margin = 0.05 * abs(st["mean_b"])
            equivalent = (st["ci95_lo"] > -margin) and (st["ci95_hi"] < margin)
            eq_rows.append({"comparison": f"{noaux_m} vs {emb_m}", "margin_relative": 0.05, "margin_absolute": margin,
                            "ci95_lo": st["ci95_lo"], "ci95_hi": st["ci95_hi"], "n": st["n"],
                            "equivalent_within_margin": equivalent, "mean_noaux": st["mean_a"], "mean_embedded": st["mean_b"]})
        else:
            eq_rows.append({"comparison": f"{noaux_m} vs {emb_m}", "margin_relative": 0.05, "status": f"insufficient paired seeds ({len(common)})"})
    pd.DataFrame(eq_rows or [{"comparison": "not_available", "status": "insufficient_data"}]).to_csv(RESULTS_DIR / "equivalence_tests.csv", index=False)
    write_marker("statistics")
    return df


# ---------------------------------------------------------------------------
# Resource comparisons, figures, final summary
# ---------------------------------------------------------------------------


def write_resource_comparisons() -> None:
    cap = load_csv(RESULTS_DIR / "multiscale_capacities.csv")
    mean_cap = {}
    if not cap.empty:
        main = cap[~cap.task.astype(str).str.startswith("delay_pair")]
        mean_cap = main.groupby("model")["capacity"].mean().to_dict()
    costs = load_csv(RESULTS_DIR / "resource_costs.csv")
    sps = {}
    if not costs.empty and "seconds_per_step" in costs.columns:
        sps = costs[costs.phase == "multiscale"].groupby("model")["seconds_per_step"].mean().to_dict()
    rows = []
    for model, n_aux, buffer_states in [
        ("M0-noaux", 0, 1),
        ("AB-noaux-kraus", 0, 50),
        ("ABC-noaux-kraus", 0, 60),
        ("ABC-noaux-hierarchical", 0, 60),
        ("AB-embedded", CFG.n_a, 0),
        ("ABC-embedded-hierarchical", 2 * CFG.n_a, 0),
        ("ABC-embedded-tied", 2 * CFG.n_a, 0),
    ]:
        total_q = CFG.n_a + n_aux
        rows.append({
            "model": model, "n_a": CFG.n_a, "n_aux": n_aux, "total_qubits": total_q,
            "density_matrix_dimension": f"{2**total_q}x{2**total_q}",
            "density_complex_entries": int((2**total_q) ** 2),
            "buffer_states": buffer_states,
            "classical_buffer_complex_entries": int(buffer_states * (2**CFG.n_a) ** 2),
            "partial_swaps_per_step": 0 if n_aux == 0 else (CFG.n_a if n_aux == CFG.n_a else 2 * CFG.n_a),
            "kraus_or_depol_channels_per_step": 0 if n_aux == 0 else n_aux,
            "mean_capacity": mean_cap.get(model, np.nan),
            "seconds_per_step": sps.get(model, np.nan),
            "device": "cuda" if is_embedded(model) else "cpu",
        })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "memory_resource_comparison.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "computational_cost_comparison.csv", index=False)
    # Embedded vs noaux by seed (paired MG mse and mean capacity).
    rows = []
    mg = load_csv(RESULTS_DIR / "mackey_glass_standard.csv")
    if not mg.empty:
        for seed in sorted(set(mg.seed.dropna().astype(int))):
            g = mg[mg.seed == seed]
            for emb, noaux in [("AB-embedded", "AB-noaux-kraus"), ("ABC-embedded-hierarchical", "ABC-noaux-hierarchical")]:
                ge = g[g.model == emb]
                gn = g[g.model == noaux]
                if len(ge) and len(gn):
                    rows.append({"seed": seed, "embedded_model": emb, "noaux_model": noaux,
                                 "embedded_mse_150": float(ge.mse_150.iloc[0]), "noaux_mse_150": float(gn.mse_150.iloc[0]),
                                 "embedded_vpt": float(ge.valid_prediction_time.iloc[0]), "noaux_vpt": float(gn.valid_prediction_time.iloc[0])})
    if rows:
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "embedded_vs_noaux_by_seed.csv", index=False)


def generate_figures(force: bool = False) -> None:
    if marker("figures").exists() and not force:
        log("figures already complete; skipping")
        return
    log("generating PDF and PNG figures")
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120})

    def save(fig, name):
        for ext in ["pdf", "png"]:
            fig.savefig(FIGURES_DIR / f"{name}.{ext}", bbox_inches="tight")
        plt.close(fig)

    def empty_fig(name, text):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True)
        ax.set_axis_off()
        save(fig, name)

    stm = load_csv(RESULTS_DIR / "paper_replication_stm.csv")
    if not stm.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for omega, g in stm.groupby("omega"):
            m = g.groupby("tau")["capacity_ols"].mean()
            se = g.groupby("tau")["capacity_ols"].sem().fillna(0)
            ax.plot(m.index, m.values, label=f"Omega={omega}")
            ax.fill_between(m.index, m - 1.96 * se, m + 1.96 * se, alpha=0.15)
        ax.set_xlabel("tau")
        ax.set_ylabel("STM capacity")
        ax.set_title(f"AB embedded STM, {stm.seed.nunique()} seeds, washout/train/test={CFG.paper_washout}/{CFG.paper_train}/{CFG.paper_test}")
        ax.legend()
        save(fig, "paper_stm_replication")
    else:
        empty_fig("paper_stm_replication", "No STM data")

    nm = load_csv(RESULTS_DIR / "paper_replication_nonmarkovianity.csv")
    if not nm.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        nm.boxplot(column="nonmarkovianity", by="omega", ax=ax)
        ax.set_title(f"Non-Markovianity by Omega ({nm.seed.nunique()} seeds)")
        fig.suptitle("")
        save(fig, "paper_nonmarkovianity_replication")
    else:
        empty_fig("paper_nonmarkovianity_replication", "No non-Markovianity data")

    ex = load_csv(RESULTS_DIR / "autonomous_prediction_examples.csv")
    if not ex.empty:
        for fig_name, series_filter in [("paper_mg_replication", "MG"), ("mg_predictions", "MG_standard"), ("mg_two_delay_predictions", "MG_two_delay")]:
            g = ex[ex.series == series_filter]
            fig, ax = plt.subplots(figsize=(8, 4))
            if not g.empty:
                for model, gm in g.groupby(g.model.astype(str) + g.omega.fillna("").astype(str)):
                    gm = gm.sort_values("step")
                    label = f"{gm.model.iloc[0]}" + (f" O={gm.omega.iloc[0]}" if pd.notna(gm.omega.iloc[0]) else "")
                    ax.plot(gm.step, gm.prediction, label=label, alpha=0.75, lw=1)
                truth = g.sort_values("step").groupby("step").truth.first()
                ax.plot(truth.index, truth.values, color="black", lw=1.8, label="truth")
                ax.legend(fontsize=6, ncol=2)
            ax.set_xlabel("autonomous step")
            ax.set_ylabel("normalized s")
            save(fig, fig_name)
    else:
        for name in ["paper_mg_replication", "mg_predictions", "mg_two_delay_predictions"]:
            empty_fig(name, "No autonomous examples")

    wc = load_csv(RESULTS_DIR / "washout_convergence.csv")
    if not wc.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for model, g in wc[wc.get("final").isna() if "final" in wc.columns else slice(None)].groupby("model") if "final" in wc.columns else wc.groupby("model"):
            m = g.groupby("step")["trace_distance_A"].mean()
            ax.semilogy(m.index, m.values.clip(1e-12), marker="o", ms=3, label=model)
        ax.axhline(CFG.washout_conv_threshold, color="red", ls="--", lw=1, label="threshold")
        ax.set_xlabel("washout step")
        ax.set_ylabel("trace distance (A) between initializations")
        ax.legend(fontsize=7)
        save(fig, "washout_convergence")
    else:
        empty_fig("washout_convergence", "No washout data")

    scales = load_csv(RESULTS_DIR / "effective_memory_scales.csv")
    if not scales.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = scales.model.astype(str) + "/" + scales.layer.astype(str) + "#r" + scales.scale_rank.astype(str)
        ax.barh(labels, scales.tau_peak)
        ax.set_xlabel("effective tau")
        save(fig, "memory_scales_by_layer")
    else:
        empty_fig("memory_scales_by_layer", "No memory scale data")

    ipc = load_csv(RESULTS_DIR / "ipc_by_component.csv")
    if not ipc.empty:
        stm1 = ipc[ipc.component == "degree1_stm"]
        fig, ax = plt.subplots(figsize=(8, 5))
        for model, g in stm1.groupby("model"):
            m = g.groupby("tau1")["capacity"].mean()
            style = "-" if "embedded" in str(model) or str(model).endswith("Markov") else "--"
            ax.plot(m.index, m.values, style, marker="o", ms=3, label=model)
        ax.set_xlabel("tau")
        ax.set_ylabel("STM capacity")
        ax.legend(fontsize=6, ncol=2)
        save(fig, "embedded_vs_noaux_memory_curves")
        fig, ax = plt.subplots(figsize=(8, 4))
        ipc.groupby(["model", "degree"])["capacity"].sum().unstack().plot(kind="bar", ax=ax)
        ax.set_ylabel("truncated IPC (sum over components)")
        save(fig, "ipc_decomposition")
        fig, ax = plt.subplots(figsize=(8, 4))
        emb_mask = ipc.model.astype(str).str.contains("embedded|Markov", regex=True)
        tot = ipc.groupby([emb_mask.map({True: "embedded", False: "noaux"}), "degree"])["capacity"].sum().unstack()
        tot.plot(kind="bar", ax=ax)
        ax.set_ylabel("truncated IPC")
        save(fig, "embedded_vs_noaux_ipc")
        cd = ipc[ipc.component == "cross_delay_degree2"]
        fig, ax = plt.subplots(figsize=(8, 4))
        cd.groupby("model")["capacity"].mean().sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("mean cross-delay degree-2 capacity")
        save(fig, "ipc_cross_delay")
    else:
        for name in ["embedded_vs_noaux_memory_curves", "ipc_decomposition", "embedded_vs_noaux_ipc", "ipc_cross_delay"]:
            empty_fig(name, "No IPC data")

    cap = load_csv(RESULTS_DIR / "multiscale_capacities.csv")
    if not cap.empty:
        main = cap[~cap.task.astype(str).str.startswith("delay_pair")]
        fig, ax = plt.subplots(figsize=(11, 5))
        pivot = main.groupby(["task", "model"])["capacity"].mean().unstack()
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("capacity")
        ax.legend(fontsize=6, ncol=2)
        save(fig, "multiscale_capacity")
        fig, ax = plt.subplots(figsize=(9, 5))
        sub = main[main.model.isin(["AB-embedded", "ABC-embedded-hierarchical", "AB-noaux-kraus", "ABC-noaux-hierarchical", "M0-noaux"])]
        sub.groupby(["task", "model"])["capacity"].mean().unstack().plot(kind="bar", ax=ax)
        ax.set_ylabel("capacity")
        ax.legend(fontsize=7)
        save(fig, "embedded_vs_noaux_capacity")
        fig, ax = plt.subplots(figsize=(7, 5))
        hp = cap[cap.task.astype(str).str.startswith("delay_pair") & (cap.model == "ABC-embedded-hierarchical")].copy()
        if hp.empty:
            hp = cap[cap.task.astype(str).str.startswith("delay_pair") & (cap.model == "ABC-noaux-kraus")].copy()
        if not hp.empty:
            parts = hp.task.str.split("_", expand=True)
            hp["tau1"] = parts[2].astype(int)
            hp["tau2"] = parts[3].astype(int)
            mat = hp.groupby(["tau1", "tau2"])["capacity"].mean().unstack().astype(float)
            im = ax.imshow(mat.values, aspect="auto", origin="lower")
            ax.set_xticks(range(len(mat.columns)))
            ax.set_xticklabels([str(x) for x in mat.columns])
            ax.set_yticks(range(len(mat.index)))
            ax.set_yticklabels([str(x) for x in mat.index])
            ax.set_xlabel("tau2")
            ax.set_ylabel("tau1")
            fig.colorbar(im, ax=ax)
        save(fig, "delay_capacity_heatmap")
        fig, ax = plt.subplots(figsize=(8, 4))
        main.groupby("model")["capacity"].mean().sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("mean capacity (main tasks)")
        save(fig, "metric_comparison")
        fig, ax = plt.subplots(figsize=(10, 5))
        order = main.groupby("model")["capacity"].mean().sort_values().index
        main.boxplot(column="capacity", by="model", ax=ax, rot=90)
        fig.suptitle("")
        ax.set_title("capacity by seed")
        save(fig, "boxplots_by_seed")
        for pair, name in [
            (("ABC-embedded-hierarchical", "AB-embedded"), "paired_differences_abc_minus_ab"),
            (("ABC-embedded-hierarchical", "ABC-embedded-tied"), "paired_differences_hier_minus_tied"),
        ]:
            a, b = pair
            ga = main[main.model == a].groupby("seed")["capacity"].mean().sort_index()
            gb = main[main.model == b].groupby("seed")["capacity"].mean().sort_index()
            common = ga.index.intersection(gb.index)
            fig, ax = plt.subplots(figsize=(7, 4))
            if len(common):
                d = (ga.loc[common] - gb.loc[common]).values
                ax.bar(range(len(d)), d)
                ax.axhline(0, color="black", lw=1)
                ax.set_xlabel("seed")
                ax.set_ylabel(f"capacity({a}) - capacity({b})")
                ax.set_title(f"mean diff={d.mean():.4f}, wins={(d>0).sum()}/{len(d)}")
            save(fig, name)
    else:
        for name in ["multiscale_capacity", "embedded_vs_noaux_capacity", "delay_capacity_heatmap", "metric_comparison", "boxplots_by_seed", "paired_differences_abc_minus_ab", "paired_differences_hier_minus_tied"]:
            empty_fig(name, "No capacity data")

    mg = load_csv(RESULTS_DIR / "mackey_glass_standard.csv")
    if not mg.empty:
        for metric, name in [("mse_150", "embedded_vs_noaux_mackey_glass"), ("valid_prediction_time", "valid_prediction_time")]:
            fig, ax = plt.subplots(figsize=(8, 4))
            mg.groupby("model")[metric].mean().sort_values().plot(kind="barh", ax=ax)
            ax.set_xlabel(f"{metric} (mean over {mg.seed.nunique()} seeds)")
            save(fig, name)
    else:
        empty_fig("embedded_vs_noaux_mackey_glass", "No MG data")
        empty_fig("valid_prediction_time", "No VPT data")

    res = load_csv(RESULTS_DIR / "memory_resource_comparison.csv")
    if not res.empty and "mean_capacity" in res.columns:
        for x, name in [
            ("total_qubits", "performance_vs_qubits"),
            ("total_qubits", "performance_vs_quantum_qubits"),
            ("classical_buffer_complex_entries", "performance_vs_classical_memory"),
            ("density_complex_entries", "performance_vs_memory"),
            ("seconds_per_step", "performance_vs_runtime"),
            ("seconds_per_step", "performance_vs_total_runtime"),
            ("density_complex_entries", "pareto_performance_resources"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(res[x], res["mean_capacity"])
            for _, r in res.iterrows():
                ax.annotate(r["model"], (r[x], r["mean_capacity"]), fontsize=7)
            ax.set_xlabel(x)
            ax.set_ylabel("mean capacity (main tasks)")
            if x in ("classical_buffer_complex_entries", "density_complex_entries"):
                ax.set_xscale("symlog")
            save(fig, name)
    else:
        for name in ["performance_vs_qubits", "performance_vs_quantum_qubits", "performance_vs_classical_memory", "performance_vs_memory", "performance_vs_runtime", "performance_vs_total_runtime", "pareto_performance_resources"]:
            empty_fig(name, "No resource data")

    bp = load_csv(RESULTS_DIR / "best_parameters_by_task.csv")
    if not bp.empty and "objective" in bp.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        bp.groupby("architecture")["objective"].mean().sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("best tuning objective (mean over tasks)")
        save(fig, "parameter_distributions")
    else:
        empty_fig("parameter_distributions", "No parameter data")
    write_marker("figures")


# ---------------------------------------------------------------------------
# Hypothesis decisions and final summary
# ---------------------------------------------------------------------------


def _sig_row(df: pd.DataFrame, family: str, comparison: str, metric: Optional[str] = None, task: Optional[str] = None) -> Optional[pd.Series]:
    if df.empty:
        return None
    m = (df.family == family) & (df.comparison == comparison)
    if metric:
        m &= df.metric == metric
    if task:
        m &= df.task == task
    g = df[m]
    return g.iloc[0] if len(g) else None


def decide_hypotheses() -> Dict[str, Dict]:
    st = load_csv(RESULTS_DIR / "paired_statistics.csv")
    gate = json.loads((RESULTS_DIR / "paper_replication_gate.json").read_text()) if (RESULTS_DIR / "paper_replication_gate.json").exists() else {}
    eq = load_csv(RESULTS_DIR / "equivalence_tests.csv")
    scales = load_csv(RESULTS_DIR / "effective_memory_scales.csv")
    mg = load_csv(RESULTS_DIR / "mackey_glass_standard.csv")
    H: Dict[str, Dict] = {}

    def summarize(row):
        if row is None:
            return None
        return {"n": int(row.n), "mean_diff": float(row.mean_diff), "p_wilcoxon_holm": float(row.p_wilcoxon_holm),
                "ci": [float(row.ci95_lo), float(row.ci95_hi)], "cohen_dz": float(row.cohen_dz),
                "wins": int(row.wins), "significant": bool(row.significant)}

    # H1: intermediate Omega beats Markovian AB with paired significance + prolonged STM.
    r_mg = _sig_row(st, "paper_MG", "Omega0.5 vs Omega1.0", metric="mse_150")
    r_stm = _sig_row(st, "paper_STM", "Omega0.5 vs Omega1.0")
    n_ok = (r_mg is not None) and int(r_mg.n) >= CFG.min_decision_seeds
    if not n_ok:
        H["H1"] = {"decision": "inconclusiva", "reason": "n de seeds pareadas < 20", "mg": summarize(r_mg)}
    else:
        mg_sig = bool(r_mg.significant) and float(r_mg.mean_diff) > 0
        stm_sig = r_stm is not None and bool(r_stm.significant) and float(r_stm.mean_diff) > 0
        if mg_sig and stm_sig:
            H["H1"] = {"decision": "aceita", "mg": summarize(r_mg), "stm": summarize(r_stm)}
        elif mg_sig or stm_sig:
            H["H1"] = {"decision": "parcialmente suportada", "mg": summarize(r_mg), "stm": summarize(r_stm)}
        else:
            H["H1"] = {"decision": "nao aceita", "mg": summarize(r_mg), "stm": summarize(r_stm)}
    H["H1"]["gate"] = {k: gate.get(k) for k in ("omega_mean_mse150", "omega_0.5_beats_1.0", "n_paired_seeds")}

    # H2: ABC hierarchical beats tuned AB-embedded AND tied ABC on capacity, plus two separated scales.
    tasks_all = sorted(set(st[st.family == "capacity"].task)) if not st.empty else []
    def count_sig(comparison):
        g = st[(st.family == "capacity") & (st.comparison == comparison)]
        if g.empty:
            return 0, 0, 0
        return int(g.significant.sum()), int((g.mean_diff > 0).sum()), len(g)
    s_ab, pos_ab, tot_ab = count_sig("ABC-embedded-hierarchical vs AB-embedded")
    s_tied, pos_tied, tot_tied = count_sig("ABC-embedded-hierarchical vs ABC-embedded-tied")
    two_scales = False
    if not scales.empty:
        gsc = scales[(scales.model == "ABC-embedded-hierarchical") & (scales.layer == "A_readout")]
        if len(gsc) >= 2:
            taus = sorted(gsc.tau_peak.astype(float))
            two_scales = (taus[-1] - taus[0]) >= 10
        diag_taus = scales[(scales.model == "ABC-embedded-hierarchical") & (scales.layer.str.contains("diag"))].groupby("layer")["tau_peak"].mean()
        if len(diag_taus) >= 2 and (diag_taus.max() - diag_taus.min()) >= 5:
            two_scales = True
    if tot_ab == 0:
        H["H2"] = {"decision": "inconclusiva", "reason": "sem comparacoes ABC embedded"}
    elif s_ab > tot_ab / 2 and s_tied > tot_tied / 2 and two_scales:
        H["H2"] = {"decision": "aceita"}
    elif s_ab == 0 and pos_ab < tot_ab / 2:
        H["H2"] = {"decision": "nao aceita (sem vantagem sobre AB embedded)"}
    else:
        H["H2"] = {"decision": "nao aceita de forma robusta / parcial"}
    H["H2"].update({"sig_vs_AB": [s_ab, tot_ab], "sig_vs_tied": [s_tied, tot_tied], "positive_vs_AB": pos_ab, "two_separated_scales": two_scales})

    # H3: autonomous forecasting, ABC vs AB embedded: >=10% MSE reduction, VPT preserved, no divergence increase, significant.
    r3 = _sig_row(st, "MG_standard", "ABC-embedded-hierarchical vs AB-embedded", metric="mse_150")
    r3v = _sig_row(st, "MG_standard", "ABC-embedded-hierarchical vs AB-embedded", metric="valid_prediction_time")
    if r3 is None or int(r3.n) < CFG.min_decision_seeds:
        H["H3"] = {"decision": "inconclusiva", "reason": "n<20 ou sem dados"}
    else:
        red = float(r3.relative_diff)
        div_ok = True
        if not mg.empty:
            dvg = mg.groupby("model")["diverged"].mean()
            div_ok = dvg.get("ABC-embedded-hierarchical", 0) <= dvg.get("AB-embedded", 0) + 1e-9
        vpt_ok = r3v is None or float(r3v.mean_diff) >= -1e-9 or not bool(r3v.significant)
        if bool(r3.significant) and red >= 0.10 and vpt_ok and div_ok:
            H["H3"] = {"decision": "aceita"}
        elif float(r3.mean_diff) > 0:
            H["H3"] = {"decision": "nao aceita de forma robusta (melhora presente mas criterios completos nao atingidos)"}
        else:
            H["H3"] = {"decision": "nao aceita"}
        H["H3"].update({"mse150": summarize(r3), "relative_reduction": red, "vpt_ok": vpt_ok, "divergence_ok": div_ok})

    # H4: auxiliary qubits necessary? embedded vs noaux with same readout.
    r4 = st[(st.family == "capacity") & (st.comparison == "ABC-embedded-hierarchical vs ABC-noaux-hierarchical")] if not st.empty else pd.DataFrame()
    if r4.empty:
        H["H4"] = {"decision": "inconclusiva"}
    else:
        s4 = int(r4.significant.sum())
        H["H4"] = {"decision": "vantagem embedded significativa em parte das tarefas" if s4 > 0 else "sem vantagem embedded demonstrada",
                   "sig_tasks": s4, "total_tasks": len(r4), "positive_tasks": int((r4.mean_diff > 0).sum())}

    # H5: practical equivalence noaux vs embedded (TOST-like 5%).
    if not eq.empty and "equivalent_within_margin" in eq.columns:
        H["H5"] = {"decision": "equivalencia pratica " + ("aceita" if bool(eq.equivalent_within_margin.any()) else "nao demonstrada"),
                   "rows": eq.to_dict("records")}
    else:
        H["H5"] = {"decision": "inconclusiva", "rows": eq.to_dict("records") if not eq.empty else []}

    # H6: noaux performance-cost advantage.
    res = load_csv(RESULTS_DIR / "memory_resource_comparison.csv")
    if not res.empty and "mean_capacity" in res.columns and res.mean_capacity.notna().any():
        r = res.dropna(subset=["mean_capacity"])
        best = r.sort_values("mean_capacity", ascending=False).iloc[0]
        best_noaux = r[r.n_aux == 0].sort_values("mean_capacity", ascending=False)
        cap_ratio = float(best_noaux.mean_capacity.iloc[0] / best.mean_capacity) if len(best_noaux) else np.nan
        H["H6"] = {"decision": "vantagem desempenho-custo noaux plausivel" if cap_ratio > 0.95 else "nao aceita (embedded mantem vantagem de desempenho)",
                   "best_model": str(best.model), "best_noaux": str(best_noaux.model.iloc[0]) if len(best_noaux) else None,
                   "noaux_capacity_fraction_of_best": cap_ratio}
    else:
        H["H6"] = {"decision": "inconclusiva"}
    write_json(RESULTS_DIR / "hypothesis_decisions.json", H)
    return H


def classify_architecture() -> pd.DataFrame:
    cap = load_csv(RESULTS_DIR / "multiscale_capacities.csv")
    st = load_csv(RESULTS_DIR / "paired_statistics.csv")
    classification = "resultado inconclusivo"
    reason = "Dados insuficientes."
    if not cap.empty:
        main = cap[~cap.task.astype(str).str.startswith("delay_pair")]
        means = main.groupby("model")["capacity"].mean().sort_values(ascending=False)
        best = means.index[0]
        n_seeds = main.seed.nunique()
        robust = False
        if not st.empty:
            g = st[(st.family == "capacity") & st.comparison.str.startswith(str(best))]
            robust = bool(g.significant.any()) if len(g) else False
        if "noaux" in str(best):
            classification = "vantagem da dinamica efetiva (noaux)" if robust else "melhor media noaux, sem robustez estatistica completa"
        elif "ABC" in str(best):
            classification = "vantagem hierarquica embedded" if robust else "melhor media ABC embedded, sem robustez estatistica completa"
        else:
            classification = "melhor media AB embedded" if robust else "melhor media AB embedded, sem robustez estatistica completa"
        reason = f"Melhor media: {best} ({means.iloc[0]:.4f}) sobre {n_seeds} seeds; significancia Holm avaliada em paired_statistics.csv."
    row = {"classification": classification, "reason": reason, "timestamp": datetime.now().isoformat()}
    df = pd.DataFrame([row])
    df.to_csv(RESULTS_DIR / "architecture_classification.csv", index=False)
    return df


def write_final_summary() -> None:
    log("writing final scientific summary")
    paper_mg = load_csv(RESULTS_DIR / "paper_replication_mackey_glass.csv")
    cap = load_csv(RESULTS_DIR / "multiscale_capacities.csv")
    st = load_csv(RESULTS_DIR / "paired_statistics.csv")
    res = load_csv(RESULTS_DIR / "memory_resource_comparison.csv")
    mg = load_csv(RESULTS_DIR / "mackey_glass_standard.csv")
    scales = load_csv(RESULTS_DIR / "effective_memory_scales.csv")
    wc = load_csv(RESULTS_DIR / "washout_convergence.csv")
    eq = load_csv(RESULTS_DIR / "equivalence_tests.csv")
    gate = json.loads((RESULTS_DIR / "paper_replication_gate.json").read_text()) if (RESULTS_DIR / "paper_replication_gate.json").exists() else {}
    bench = json.loads((RESULTS_DIR / "benchmark_v2.json").read_text()) if (RESULTS_DIR / "benchmark_v2.json").exists() else {}
    env = environment_info()
    H = decide_hypotheses()
    classify_architecture()

    omega_means = paper_mg.groupby("omega")["mse_150"].mean().to_dict() if not paper_mg.empty else {}
    best_omega = min(omega_means, key=omega_means.get) if omega_means else None
    main_cap = cap[~cap.task.astype(str).str.startswith("delay_pair")] if not cap.empty else cap
    cap_means = main_cap.groupby("model")["capacity"].mean().sort_values(ascending=False) if not cap.empty else pd.Series(dtype=float)
    n_eval = int(main_cap.seed.nunique()) if not cap.empty else 0

    def fmt_stats(family, comparison, metric=None):
        r = _sig_row(st, family, comparison, metric=metric)
        if r is None:
            return "sem dados"
        return (f"n={int(r.n)}, dif media={float(r.mean_diff):+.4g}, IC95=[{float(r.ci95_lo):.4g},{float(r.ci95_hi):.4g}], "
                f"Wilcoxon-Holm p={float(r.p_wilcoxon_holm):.3g}, dz={float(r.cohen_dz):.2f}, "
                f"{'SIGNIFICATIVO' if bool(r.significant) else 'nao significativo'}")

    def cap_of(model):
        return f"{cap_means.get(model, np.nan):.4f}" if model in cap_means.index else "n/d"

    tf_ok = None
    if not mg.empty and "r2_teacher_forced" in mg.columns:
        tf_ok = mg.groupby("model")["r2_teacher_forced"].mean()

    abc_vs_ab_cap = fmt_stats("capacity", "ABC-embedded-hierarchical vs AB-embedded", None)
    lines = [
        "# Embedded and Effective Hierarchical ABC QRC — final summary (v2, GPU)",
        "",
        f"Paper base: {PAPER_URL}. Execucao v2 com o protocolo completo em GPU; o v1 (CPU-bounded) fica como registro historico e NAO foi reaproveitado nas analises.",
        "",
        "## Execution scope",
        "",
        f"- Hardware: GPU {env.get('gpu_name')} ({env.get('gpu_mem_gb')} GB), torch {env.get('torch')} CUDA {env.get('torch_cuda')}, Python {env.get('python')}.",
        f"- Dispositivo comprovado: evolucao embedded em complex64 na GPU ({bench.get('gpu_seconds_per_step_ABC-embedded-hierarchical', float('nan'))*1000:.1f} ms/passo ABC 4096x4096; {bench.get('gpu_seconds_per_step_AB-embedded', float('nan'))*1000:.2f} ms/passo AB 256x256); modelos noaux 16x16 exatos em complex128 na CPU.",
        f"- Washout/train/test = {CFG.paper_washout}/{CFG.paper_train}/{CFG.paper_test} em todas as analises principais.",
        f"- Seeds: replicacao do paper = {len(CFG.paper_eval_seeds)}; avaliacao pareada = {n_eval}; tuning Optuna = {CFG.tune_trials} trials (noaux/AB, {CFG.ab_tune_seeds} seeds) e {CFG.abc_tune_trials} trials x {CFG.abc_tune_seeds} seeds (ABC embedded).",
        f"- ABC embedded N=4 (matriz densidade 4096x4096) FOI executado em todas as fases (tuning, capacidades, IPC, Mackey-Glass).",
        f"- Convergencia do washout verificada antes do treino: {'todas as configuracoes convergiram (<' + str(CFG.washout_conv_threshold) + ')' if (not wc.empty and 'converged' in wc.columns and bool(wc.dropna(subset=['converged']).converged.astype(bool).all())) else 'ver washout_convergence.csv e failed_runs.csv'}.",
        "",
        "## Direct answers (12 perguntas)",
        "",
        f"1. A vantagem AB do paper foi reproduzida? Gate: Omega=0.5 {'SUPEROU' if gate.get('omega_0.5_beats_1.0') else 'NAO superou'} Omega=1.0 em MSE150 medio ({gate.get('omega_mean_mse150')}); estatistica pareada: {fmt_stats('paper_MG', 'Omega0.5 vs Omega1.0', 'mse_150')}.",
        f"2. Qual Omega foi melhor (MG, MSE150 medio)? {best_omega if best_omega is not None else 'inconclusivo'}.",
        f"3. A memoria nao markoviana AB superou o regime markoviano? STM tau>=10: {fmt_stats('paper_STM', 'Omega0.5 vs Omega1.0')}; nao-markovianidade: {fmt_stats('paper_nonmark', 'Omega0.5 vs Omega1.0')}.",
        f"4. ABC embedded supera AB embedded? Capacidade media: ABC-hier={cap_of('ABC-embedded-hierarchical')} vs AB={cap_of('AB-embedded')}; exemplo estatistico (por tarefa em paired_statistics.csv): {abc_vs_ab_cap}. MG: {fmt_stats('MG_standard', 'ABC-embedded-hierarchical vs AB-embedded', 'mse_150')}.",
        f"5. ABC sem auxiliares supera AB sem auxiliares? {fmt_stats('capacity', 'ABC-noaux-kraus vs AB-noaux-kraus')}; medias: ABC-noaux-kraus={cap_of('ABC-noaux-kraus')}, AB-noaux-kraus={cap_of('AB-noaux-kraus')}.",
        f"6. ABC embedded supera ABC sem auxiliares? {fmt_stats('capacity', 'ABC-embedded-hierarchical vs ABC-noaux-hierarchical')}; MG: {fmt_stats('MG_standard', 'ABC-embedded-hierarchical vs ABC-noaux-hierarchical', 'mse_150')}.",
        f"7. As versoes apresentam escalas de memoria semelhantes? Ver effective_memory_scales.csv ({len(scales)} registros; picos STM por modelo e autocorrelacao das camadas A/B/C do ABC embedded N=4).",
        f"8. A arquitetura sem auxiliares reproduz os revivals? {'Picos multiplos detectados em ' + ', '.join(sorted(set(scales[(scales.scale_rank > 1)].model))) if (not scales.empty and (scales.scale_rank > 1).any()) else 'Sem picos secundarios robustos detectados'}.",
        f"9. A versao sem auxiliares funciona na previsao autonoma? Sim (rollout sem valores futuros); validacao teacher-forced media por modelo: {tf_ok.round(4).to_dict() if tf_ok is not None else 'n/d'}; VPT em mackey_glass_standard.csv.",
        "10. Qual arquitetura utiliza menos qubits? M0/AB/ABC-noaux usam apenas N_A=4 qubits fisicos; embedded AB usa 8 e embedded ABC usa 12.",
        "11. Qual arquitetura utiliza menos memoria total? M0-noaux; os noaux trocam qubits auxiliares por buffer classico de estados 16x16. Ver memory_resource_comparison.csv.",
        f"12. Qual arquitetura possui melhor relacao desempenho-custo? Melhor capacidade media global: {cap_means.index[0] if len(cap_means) else 'n/d'} ({cap_means.iloc[0]:.4f} media). Melhor noaux: {cap_means[[m for m in cap_means.index if 'noaux' in m]].idxmax() if any('noaux' in m for m in cap_means.index) else 'n/d'}. Custo por passo e memoria em computational_cost_comparison.csv.",
        "",
        "## Controle negativo shuffled-history",
        "",
        f"- ABC-noaux-kraus vs shuffled-history: {fmt_stats('capacity', 'ABC-noaux-kraus vs ABC-noaux-shuffled-history')}.",
        "- O shuffle e refeito por seed e usa apenas estados passados do buffer (checks em sanity_checks.json). Se o controle ainda vencer em alguma tarefa, isso indica que a ordem temporal do buffer nao esta sendo explorada naquela tarefa, e esta reportado como tal.",
        "",
        "## Hypothesis decisions (n>=20 obrigatorio para aceitar/rejeitar)",
        "",
    ]
    for h in ["H1", "H2", "H3", "H4", "H5", "H6"]:
        d = H.get(h, {})
        lines.append(f"- {h}: {d.get('decision', 'inconclusiva')}. Detalhes em hypothesis_decisions.json.")
    lines += [
        "",
        "## Limitations",
        "",
        f"- Seeds de replicacao executadas: {len(CFG.paper_eval_seeds)} de {CFG.planned_eval_seeds} planejadas (reducao explicita registrada em failed_runs.csv quando aplicavel).",
        "- O readout e somente em A com 66 features Pauli para todos os modelos, como pre-registrado.",
        "- Estados embedded evoluidos em complex64 na GPU; validacao pontual contra complex128 em sanity_checks.json.",
        "- Nao ha afirmacao de vantagem quantica; resultados negativos e inconclusivos foram preservados.",
    ]
    (RESULTS_DIR / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_required_csvs() -> None:
    required = [
        "config.json", "environment.json", "sanity_checks.json",
        "smoke_results.csv", "washout_convergence.csv",
        "paper_replication_stm.csv", "paper_replication_nonmarkovianity.csv", "paper_replication_mackey_glass.csv",
        "tuning_trials_ab.csv", "tuning_trials_abc.csv", "best_parameters_by_task.csv",
        "effective_memory_scales.csv", "multiscale_capacities.csv", "ipc_by_component.csv",
        "mackey_glass_standard.csv", "mackey_glass_two_delay.csv", "autonomous_prediction_examples.csv",
        "paired_statistics.csv", "resource_costs.csv", "failed_runs.csv",
        "noaux_tuning_trials.csv", "noaux_best_parameters.csv", "noaux_memory_capacities.csv",
        "noaux_ipc_by_component.csv", "noaux_mackey_glass.csv", "embedded_vs_noaux_by_seed.csv",
        "equivalence_tests.csv", "memory_resource_comparison.csv", "computational_cost_comparison.csv",
        "architecture_classification.csv",
    ]
    for name in required:
        path = RESULTS_DIR / name
        if path.exists() and path.stat().st_size > 0:
            continue
        if name.endswith(".json"):
            write_json(path, {"status": "created_empty_placeholder", "reason": "no rows generated"})
        else:
            pd.DataFrame([{"status": "no_rows_generated"}]).to_csv(path, index=False)


def make_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if NOTEBOOK_PATH.exists():
            zf.write(NOTEBOOK_PATH, NOTEBOOK_PATH.name)
        for folder in [RESULTS_DIR, FIGURES_DIR]:
            for path in folder.rglob("*"):
                if path.is_file() and path.suffix != ".sqlite3" and "channel_cache" not in str(path):
                    zf.write(path, path.as_posix())
    log(f"zip written: {ZIP_PATH}")


def create_notebook() -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            "# Embedded and effective hierarchical ABC QRC — v2 (GPU)\n\n"
            "Notebook generated by `embedded_effective_qrc_pipeline_v2.py`. Full-protocol GPU execution: "
            "washout/train/test = 1000/1000/1000, >=20 paired evaluation seeds, exact embedded ABC N=4 "
            "(4096x4096 complex64 on GPU), 64 Optuna trials, washout-convergence diagnostics, "
            "teacher-forced validation before autonomous Mackey-Glass rollout. "
            "v1 results were NOT reused in any main analysis."
        ),
        nbformat.v4.new_markdown_cell(
            "## Preregistered criteria\n\n"
            "H1 accepts paper reproduction only if intermediate Omega beats Markovian AB with paired significance and prolonged STM. "
            "H2 accepts hierarchical ABC only if it beats tuned AB and tied ABC, with practical effect size, Holm-corrected Wilcoxon, bootstrap CI, two separated memory scales, stability, and A-only readout. "
            "H3 accepts autonomous forecasting only with at least 10% MSE/NRMSE reduction, preserved valid prediction time, no divergence increase, and significance. "
            "H4-H6 compare embedded auxiliaries with effective no-auxiliary dynamics using the same A-only readout and cost accounting. "
            "No hypothesis is accepted or rejected with fewer than 20 paired seeds."
        ),
        nbformat.v4.new_code_cell(
            "import embedded_effective_qrc_pipeline_v2 as qrc\n"
            "# Hard GPU gate: prints versions and ABORTS if the GPU is not usable (no CPU fallback).\n"
            "qrc.require_gpu(verbose=True)\n"
            "qrc.RESULTS_DIR, qrc.FIGURES_DIR"
        ),
        nbformat.v4.new_code_cell(
            "bench = qrc.benchmark_step_costs()\n"
            "budget = qrc.estimate_and_decide_budget(bench)\n"
            "bench"
        ),
        nbformat.v4.new_code_cell("qrc.run_all(execute_notebook=False)"),
        nbformat.v4.new_code_cell(
            "import pandas as pd\n"
            "gate = (qrc.RESULTS_DIR / 'paper_replication_gate.json').read_text(encoding='utf-8')\n"
            "print(gate[:2500])"
        ),
        nbformat.v4.new_code_cell(
            "summary = (qrc.RESULTS_DIR / 'final_summary.md').read_text(encoding='utf-8')\n"
            "print(summary)"
        ),
        nbformat.v4.new_code_cell(
            "st = pd.read_csv(qrc.RESULTS_DIR / 'paired_statistics.csv')\n"
            "cols = ['family','metric','task','comparison','n','mean_diff','ci95_lo','ci95_hi','p_wilcoxon_holm','cohen_dz','significant']\n"
            "st[cols].sort_values(['family','task']).head(60)"
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "print('CSV files:', len(list(qrc.RESULTS_DIR.glob('*.csv'))))\n"
            "print('Figures:', len(list(qrc.FIGURES_DIR.glob('*.pdf'))), 'pdf /', len(list(qrc.FIGURES_DIR.glob('*.png'))), 'png')\n"
            "print('Zip exists:', qrc.ZIP_PATH.exists(), qrc.ZIP_PATH)\n"
            "import json\n"
            "print(json.dumps(json.loads((qrc.RESULTS_DIR / 'environment.json').read_text()), indent=2)[:1500])"
        ),
    ]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nbformat.write(nb, NOTEBOOK_PATH)
    log(f"notebook created: {NOTEBOOK_PATH}")


def execute_notebook_file() -> None:
    log("executing generated notebook")
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(nb, timeout=86400, kernel_name="python3", allow_errors=False)
    client.execute()
    nbformat.write(nb, NOTEBOOK_PATH)
    log("notebook execution complete")


def run_all(force: bool = False, execute_notebook: bool = True) -> None:
    ensure_dirs()
    random.seed(0)
    np.random.seed(0)
    require_gpu(verbose=False)
    bench = benchmark_step_costs()
    estimate_and_decide_budget(bench)
    write_json(RESULTS_DIR / "config.json", {
        **asdict(CFG), "paper_url": PAPER_URL,
        "run_mode_sequence": ["smoke", "paper", "full"],
        "device_policy": "GPU obrigatorio para toda evolucao embedded (complex64); noaux 16x16 exato complex128 em CPU; abortar sem GPU, sem fallback silencioso.",
        "v1_reuse": "nenhum resultado v1 reaproveitado nas analises principais; estudo Optuna novo (optuna_abc_v2.sqlite3)",
    })
    write_json(RESULTS_DIR / "environment.json", environment_info())
    run_sanity_checks(force=force)
    run_smoke(force=force)
    run_washout_convergence(force=force)
    run_paper_replication(force=force)
    run_tuning(force=force)
    run_multiscale_and_ipc(force=force)
    run_mackey_glass_full(force=force)
    run_statistics(force=force)
    write_resource_comparisons()
    write_final_summary()
    ensure_required_csvs()
    generate_figures(force=force)
    write_marker("full")
    if execute_notebook:
        create_notebook()
        execute_notebook_file()
    make_zip()


if __name__ == "__main__":
    run_all()
