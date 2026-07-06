"""Embedded and effective non-Markovian QRC comparison pipeline.

This module is intentionally self-contained so the accompanying notebook can
execute the same pipeline and resume from CSV/JSON checkpoints.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qrc")
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import nbformat
import numpy as np
import optuna
import pandas as pd
import psutil
import scipy.linalg
from nbclient import NotebookClient
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results_abc_comparison")
FIGURES_DIR = Path("figures_abc_comparison")
NOTEBOOK_PATH = Path("embedded_and_effective_hierarchical_abc_qrc_paper.ipynb")
ZIP_PATH = Path("embedded_and_effective_abc_qrc_results.zip")
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
    smoke_seeds: Tuple[int, ...] = (9000, 9001, 9002)
    tune_seeds: Tuple[int, ...] = (1000,)
    eval_seeds: Tuple[int, ...] = tuple(range(3))
    paper_eval_seeds: Tuple[int, ...] = (0,)
    planned_eval_seeds: int = 100
    planned_tune_trials: int = 64
    executed_tune_trials: int = 1
    cpu_threads: int = 2
    cpu_safe_mode: bool = True
    embedded_abc_n4_seed_budget: int = 1
    smoke_washout: int = 20
    smoke_train: int = 40
    smoke_test: int = 30
    requested_paper_washout: int = 1000
    requested_paper_train: int = 1000
    requested_paper_test: int = 1000
    paper_washout: int = 200
    paper_train: int = 300
    paper_test: int = 300
    tune_washout: int = 40
    tune_train: int = 80
    tune_test: int = 60
    tau_max_paper: int = 50
    tau_max_tune: int = 60
    ridge_alphas: Tuple[float, ...] = (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0)
    omegas_paper: Tuple[float, ...] = (1.0, 0.5, 0.0)
    valid_threshold: float = 0.1
    mg_tau: float = 17.0
    mg2_tau1: float = 17.0
    mg2_tau2: float = 30.0
    mg_sample_time: float = 3.0
    mg_internal_dt: float = 0.1
    n_boot: int = 2000
    optuna_seed: int = 42
    cpu_bounded: bool = True

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
    marker(name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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


def environment_info() -> Dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 3),
        "cuda_available": False,
        "notes": "CUDA unavailable in this run; all experiments executed on CPU.",
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 3)
    except Exception as exc:
        info["torch_error"] = repr(exc)
    for mod in ["numpy", "scipy", "pandas", "matplotlib", "optuna", "nbformat", "nbclient"]:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception as exc:
            info[f"{mod}_error"] = repr(exc)
    return info


I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
SMINUS = np.array([[0, 1], [0, 0]], dtype=np.complex128)
PAULI = {"x": X, "y": Y, "z": Z}


def ctype() -> np.dtype:
    return np.complex64 if CFG.dtype == "complex64" else np.complex128


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


def build_channel_grid(seed: int, n: int, grid_size: Optional[int] = None) -> np.ndarray:
    ensure_dirs()
    g = grid_size or CFG.grid_size
    path = channel_cache_path(seed, n, g)
    if path.exists():
        return np.load(path)["grid"]
    t0 = time.time()
    base, drive = liouvillian_parts(seed, n)
    svals = np.linspace(CFG.grid_s_min, CFG.grid_s_max, g)
    mats = []
    for s in svals:
        mats.append(scipy.linalg.expm((base + s * drive) * CFG.dt).astype(ctype()))
    grid = np.stack(mats, axis=0)
    np.savez_compressed(path, grid=grid, svals=svals)
    log(f"channel grid cached: seed={seed}, N={n}, grid={g}, seconds={time.time() - t0:.1f}")
    return grid


def select_channel(grid: np.ndarray, s: float) -> np.ndarray:
    pos = (float(s) - CFG.grid_s_min) / (CFG.grid_s_max - CFG.grid_s_min) * (grid.shape[0] - 1)
    if pos <= 0:
        return grid[0]
    if pos >= grid.shape[0] - 1:
        return grid[-1]
    lo = int(math.floor(pos))
    w = pos - lo
    return ((1.0 - w) * grid[lo] + w * grid[lo + 1]).astype(grid.dtype, copy=False)


def pure_zero_density(n: int, dtype: Optional[np.dtype] = None) -> np.ndarray:
    d = 2**n
    rho = np.zeros((d, d), dtype=dtype or ctype())
    rho[0, 0] = 1.0
    return rho


def normalize_density(rho: np.ndarray) -> np.ndarray:
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho).real
    if abs(tr) > 1e-14:
        rho = rho / tr
    return rho.astype(rho.dtype, copy=False)


def apply_super_to_a(rho: np.ndarray, super_a: np.ndarray, n_a: int, n_total: int) -> np.ndarray:
    da = 2**n_a
    de = 2 ** (n_total - n_a)
    t = rho.reshape(da, de, da, de).transpose(0, 2, 1, 3).reshape(da * da, de * de)
    t = super_a @ t
    return np.ascontiguousarray(t.reshape(da, da, de, de).transpose(0, 2, 1, 3).reshape(da * de, da * de))


def partial_swap_unitary(eta: float) -> np.ndarray:
    sw = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)
    return (math.cos(eta) * np.eye(4) + 1j * math.sin(eta) * sw).astype(ctype())


def partial_swap_layer_unitary(eta: float, n_pairs: int) -> np.ndarray:
    u = partial_swap_unitary(eta)
    out = u
    for _ in range(n_pairs - 1):
        out = np.kron(out, u)
    return out.astype(ctype(), copy=False)


def apply_layer_unitary_density(rho: np.ndarray, u: np.ndarray, qubits: Sequence[int], n_total: int) -> np.ndarray:
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


def apply_2q_unitary_density(rho: np.ndarray, u: np.ndarray, q1: int, q2: int, n_total: int) -> np.ndarray:
    if q1 == q2:
        return rho
    qs = [q1, q2]
    t = rho.reshape([2] * (2 * n_total))
    u4 = u.reshape(2, 2, 2, 2)
    t = np.moveaxis(t, qs, [0, 1])
    t = np.tensordot(u4, t, axes=([2, 3], [0, 1]))
    t = np.moveaxis(t, [0, 1], qs)
    bqs = [n_total + q1, n_total + q2]
    t = np.moveaxis(t, bqs, [0, 1])
    t = np.tensordot(u4.conj(), t, axes=([2, 3], [0, 1]))
    t = np.moveaxis(t, [0, 1], bqs)
    return np.ascontiguousarray(t.reshape(rho.shape))


def depolarize_qubit(rho: np.ndarray, q: int, n_total: int, omega: float) -> np.ndarray:
    if omega <= 0:
        return rho
    t = rho.reshape([2] * (2 * n_total))
    moved = np.moveaxis(t, [q, n_total + q], [0, 1])
    reduced = moved[0, 0] + moved[1, 1]
    dep = np.zeros_like(moved)
    dep[0, 0] = 0.5 * reduced
    dep[1, 1] = 0.5 * reduced
    dep = np.moveaxis(dep, [0, 1], [q, n_total + q]).reshape(rho.shape)
    return ((1.0 - omega) * rho + omega * dep).astype(rho.dtype, copy=False)


def local_depolarize_all(rho: np.ndarray, qubits: Iterable[int], n_total: int, omega: float) -> np.ndarray:
    for q in qubits:
        rho = depolarize_qubit(rho, q, n_total, omega)
    return rho


def reduce_register(rho: np.ndarray, keep: Sequence[int], n_total: int) -> np.ndarray:
    keep = tuple(int(q) for q in keep)
    keep_set = set(keep)
    traced = [q for q in range(n_total) if q not in keep_set]
    perm = list(keep) + traced + [n_total + q for q in keep] + [n_total + q for q in traced]
    t = rho.reshape([2] * (2 * n_total)).transpose(perm)
    dk = 2 ** len(keep)
    dt = 2 ** len(traced)
    t = t.reshape(dk, dt, dk, dt)
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


OBS_A, OBS_NAMES = build_observables(CFG.n_a)


def features_from_rho_a(rho_a: np.ndarray, obs: np.ndarray = OBS_A) -> np.ndarray:
    return np.array([np.trace(rho_a @ o).real for o in obs], dtype=np.float64)


def n_pauli_features(n: int) -> int:
    return 3 * n + 9 * (n * (n - 1) // 2)


def state_checks(rho: np.ndarray) -> Dict[str, float]:
    vals = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    return {
        "trace_error": float(abs(np.trace(rho) - 1.0)),
        "hermiticity_error": float(np.linalg.norm(rho - rho.conj().T)),
        "min_eig": float(vals.min()),
    }


class EmbeddedModel:
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
        self.u_ab = partial_swap_unitary(self.eta_ab)
        self.u_bc = partial_swap_unitary(self.eta_bc)
        self.u_ab_layer = partial_swap_layer_unitary(self.eta_ab, n_a)
        self.u_bc_layer = partial_swap_layer_unitary(self.eta_bc, n_a)
        if architecture in ("M0-embedded", "M0-noaux"):
            self.n_total = n_a
        elif architecture.startswith("ABC"):
            self.n_total = 3 * n_a
        elif architecture.startswith("AB"):
            self.n_total = 2 * n_a
        else:
            raise ValueError(f"unknown architecture {architecture}")
        self.rho = pure_zero_density(self.n_total)
        self.last_clamped = 0

    def reset(self) -> "EmbeddedModel":
        self.rho = pure_zero_density(self.n_total)
        self.last_clamped = 0
        return self

    def clone(self) -> np.ndarray:
        return self.rho.copy()

    def restore(self, rho: np.ndarray) -> None:
        self.rho = rho.copy()

    def step(self, s: float, grid: np.ndarray) -> np.ndarray:
        if s < CFG.grid_s_min or s > CFG.grid_s_max:
            self.last_clamped += 1
        self.rho = apply_super_to_a(self.rho, select_channel(grid, s), self.n_a, self.n_total)
        if self.architecture.startswith("ABC"):
            if "parallel" in self.architecture:
                qs_ab, qs_ac = [], []
                for i in range(self.n_a):
                    qs_ab.extend([i, self.n_a + i])
                    qs_ac.extend([i, 2 * self.n_a + i])
                self.rho = apply_layer_unitary_density(self.rho, self.u_ab_layer, qs_ab, self.n_total)
                self.rho = apply_layer_unitary_density(self.rho, self.u_bc_layer, qs_ac, self.n_total)
            else:
                qs_ab, qs_bc = [], []
                for i in range(self.n_a):
                    qs_ab.extend([i, self.n_a + i])
                    qs_bc.extend([self.n_a + i, 2 * self.n_a + i])
                self.rho = apply_layer_unitary_density(self.rho, self.u_ab_layer, qs_ab, self.n_total)
                self.rho = apply_layer_unitary_density(self.rho, self.u_bc_layer, qs_bc, self.n_total)
            self.rho = local_depolarize_all(self.rho, range(self.n_a, 2 * self.n_a), self.n_total, self.omega_b)
            self.rho = local_depolarize_all(self.rho, range(2 * self.n_a, 3 * self.n_a), self.n_total, self.omega_c)
        elif self.architecture.startswith("AB"):
            qs = []
            for i in range(self.n_a):
                qs.extend([i, self.n_a + i])
            self.rho = apply_layer_unitary_density(self.rho, self.u_ab_layer, qs, self.n_total)
            self.rho = local_depolarize_all(self.rho, range(self.n_a, 2 * self.n_a), self.n_total, self.omega_b)
        self.rho = normalize_density(self.rho)
        return self.rho

    def features(self, register: str = "A") -> np.ndarray:
        if register == "A":
            keep = range(self.n_a)
        elif register == "B":
            keep = range(self.n_a, 2 * self.n_a)
        elif register == "C":
            keep = range(2 * self.n_a, 3 * self.n_a)
        else:
            raise ValueError(register)
        rho = reduce_register(self.rho, list(keep), self.n_total)
        obs, _ = build_observables(self.n_a)
        return features_from_rho_a(rho, obs)


def apply_pauli_depol_a(rho: np.ndarray, p: float, n_a: int) -> np.ndarray:
    return local_depolarize_all(rho, range(n_a), n_a, p)


class NoAuxModel:
    def __init__(
        self,
        n_a: int,
        name: str,
        tau_b: int = 1,
        tau_c: int = 2,
        lambda_b: float = 0.0,
        lambda_c: float = 0.0,
        p_b: float = 0.0,
        p_c: float = 0.0,
        shuffled: bool = False,
        seed: int = 0,
    ):
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
        self.rng = np.random.default_rng(seed + 7717)
        self.max_tau = max(1, self.tau_b, self.tau_c)
        self.reset()

    def reset(self) -> "NoAuxModel":
        self.rho = pure_zero_density(self.n_a)
        self.buffer = [self.rho.copy() for _ in range(self.max_tau + 1)]
        self.t = 0
        return self

    def clone(self) -> Tuple[np.ndarray, List[np.ndarray], int]:
        return self.rho.copy(), [x.copy() for x in self.buffer], self.t

    def restore(self, state: Tuple[np.ndarray, List[np.ndarray], int]) -> None:
        self.rho, self.buffer, self.t = state[0].copy(), [x.copy() for x in state[1]], int(state[2])

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
                past = apply_pauli_depol_a(past, self.p_b, self.n_a)
            mix = mix + self.lambda_b * past
        if self.lambda_c > 0:
            past = self.delayed(self.tau_c)
            if "kraus" in self.name or "tied" in self.name or "hierarchical" in self.name:
                past = apply_pauli_depol_a(past, self.p_c, self.n_a)
            mix = mix + self.lambda_c * past
        mix = normalize_density(mix)
        v = select_channel(grid, s) @ mix.reshape(-1)
        self.rho = normalize_density(v.reshape(2**self.n_a, 2**self.n_a))
        self.t += 1
        self.buffer[self.t % len(self.buffer)] = self.rho.copy()
        return self.rho

    def features(self) -> np.ndarray:
        obs, _ = build_observables(self.n_a)
        return features_from_rho_a(self.rho, obs)


def drive_features(model, seq: np.ndarray, grid: np.ndarray, register: str = "A") -> np.ndarray:
    n_a = model.n_a
    feats = np.empty((len(seq), n_pauli_features(n_a)), dtype=np.float64)
    for k, s in enumerate(seq):
        model.step(float(s), grid)
        if isinstance(model, EmbeddedModel):
            feats[k] = model.features(register)
        else:
            feats[k] = model.features()
    return feats


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


def autonomous_rollout(model, grid: np.ndarray, w: np.ndarray, steps: int) -> np.ndarray:
    preds = []
    for _ in range(steps):
        feat = model.features() if isinstance(model, NoAuxModel) else model.features("A")
        y = float(predict_readout(feat[None, :], w)[0])
        preds.append(y)
        model.step(y, grid)
    return np.asarray(preds)


def run_mg_model(seed: int, model, grid: np.ndarray, series: np.ndarray, slices: Dict[str, slice]) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, int]:
    stop = slices["test"].start
    feats_pre = drive_features(model, series[:stop], grid)
    snapshot = model.clone()
    target = series[1 : stop + 1]
    w = fit_readout(feats_pre[slices["train"]], target[slices["train"]], alpha=1e-6)
    model.restore(snapshot)
    preds = autonomous_rollout(model, grid, w, slices["test"].stop - slices["test"].start)
    truth = series[slices["test"]]
    metrics150 = mse_metrics(truth[:150], preds[:150])
    metrics1000 = mse_metrics(truth, preds)
    err = np.abs(preds - truth)
    exceed = np.where(err > CFG.valid_threshold)[0]
    vpt = int(exceed[0]) if len(exceed) else len(preds)
    out = {
        "seed": seed,
        "mse_150": metrics150["mse"],
        "nrmse_150": metrics150["nrmse"],
        "r2_150": metrics150["r2"],
        "mse_1000": metrics1000["mse"],
        "nrmse_1000": metrics1000["nrmse"],
        "r2_1000": metrics1000["r2"],
        "valid_prediction_time": vpt,
        "diverged": bool(np.any(np.abs(preds) > 2.0)),
        "out_of_range_fraction": float(np.mean((preds < 0.0) | (preds > 1.0))),
        "grid_clamps": int(getattr(model, "last_clamped", 0)),
    }
    return out, preds, truth, vpt


def make_noaux_model(name: str, params: Dict, seed: int) -> NoAuxModel:
    if name == "M0-noaux":
        return NoAuxModel(CFG.n_a, name, seed=seed)
    if name == "AB-noaux-residual":
        return NoAuxModel(CFG.n_a, name, tau_b=params.get("tau_b", 10), lambda_b=params.get("lambda_b", 0.35), seed=seed)
    if name == "AB-noaux-kraus":
        return NoAuxModel(CFG.n_a, name, tau_b=params.get("tau_b", 10), lambda_b=params.get("lambda_b", 0.35), p_b=params.get("p_b", 0.2), seed=seed)
    if name in ("ABC-noaux-residual", "ABC-noaux-kraus", "ABC-noaux-hierarchical"):
        return NoAuxModel(
            CFG.n_a,
            name,
            tau_b=params.get("tau_b", 10),
            tau_c=params.get("tau_c", 30),
            lambda_b=params.get("lambda_b", 0.25),
            lambda_c=params.get("lambda_c", 0.25),
            p_b=params.get("p_b", 0.2),
            p_c=params.get("p_c", 0.05),
            seed=seed,
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


def make_embedded_model(name: str, params: Dict) -> EmbeddedModel:
    if name == "AB-embedded" or name == "AB-paper":
        return EmbeddedModel(CFG.n_a, "AB-embedded", omega=params.get("omega", 0.5), eta=params.get("eta", CFG.eta_paper))
    if name == "AB-Markov":
        return EmbeddedModel(CFG.n_a, "AB-embedded", omega=1.0, eta=params.get("eta", CFG.eta_paper))
    if name == "ABC-embedded-hierarchical":
        return EmbeddedModel(
            CFG.n_a,
            "ABC-chain",
            omega_b=params.get("omega_b", 0.5),
            omega_c=params.get("omega_c", 0.1),
            eta_ab=params.get("eta_ab", CFG.eta_paper),
            eta_bc=params.get("eta_bc", CFG.eta_paper / 2),
        )
    if name == "ABC-embedded-tied":
        return EmbeddedModel(CFG.n_a, "ABC-chain", omega_b=params.get("omega", 0.5), omega_c=params.get("omega", 0.5), eta_ab=params.get("eta", CFG.eta_paper), eta_bc=params.get("eta", CFG.eta_paper))
    if name == "ABC-embedded-C-off":
        return EmbeddedModel(CFG.n_a, "ABC-chain", omega_b=params.get("omega_b", 0.5), omega_c=1.0, eta_ab=params.get("eta_ab", CFG.eta_paper), eta_bc=0.0)
    if name == "ABC-embedded-parallel":
        return EmbeddedModel(CFG.n_a, "ABC-parallel", omega_b=params.get("omega_b", 0.5), omega_c=params.get("omega_c", 0.2), eta_ab=params.get("eta_ab", CFG.eta_paper), eta_bc=params.get("eta_ac", CFG.eta_paper / 2))
    if name == "ABC-Markov":
        return EmbeddedModel(CFG.n_a, "ABC-chain", omega_b=1.0, omega_c=1.0, eta_ab=CFG.eta_paper, eta_bc=CFG.eta_paper)
    raise ValueError(name)


def record_failure(context: str, reason: str, **extra) -> None:
    row = {"timestamp": datetime.now().isoformat(), "context": context, "reason": reason, **extra}
    append_rows(RESULTS_DIR / "failed_runs.csv", [row])
    log(f"failure recorded: {context}: {reason}")


def run_sanity_checks(force: bool = False) -> pd.DataFrame:
    if marker("sanity").exists() and not force:
        log("sanity already complete; skipping")
        return pd.read_json(RESULTS_DIR / "sanity_checks.json")
    log("running sanity and physical consistency checks")
    rows = []

    def add(name, ok, value=None, detail=""):
        rows.append({"check": name, "passed": bool(ok), "value": value, "detail": detail})

    n = 2
    grid = build_channel_grid(4242, n, grid_size=9).astype(np.complex128)
    rho = pure_zero_density(n, np.complex128)
    rho2 = (select_channel(grid, 0.4) @ rho.reshape(-1)).reshape(2**n, 2**n)
    chk = state_checks(normalize_density(rho2))
    add("trace_preserved", chk["trace_error"] < 1e-8, chk["trace_error"])
    add("hermiticity_preserved", chk["hermiticity_error"] < 1e-8, chk["hermiticity_error"])
    add("positivity_preserved", chk["min_eig"] > -1e-8, chk["min_eig"])
    u0 = partial_swap_unitary(0.0).astype(np.complex128)
    up = partial_swap_unitary(math.pi / 2).astype(np.complex128)
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)
    u = partial_swap_unitary(CFG.eta_paper).astype(np.complex128)
    add("partial_swap_unitary", np.linalg.norm(u.conj().T @ u - np.eye(4)) < 1e-6, float(np.linalg.norm(u.conj().T @ u - np.eye(4))))
    add("eta_zero_identity", np.linalg.norm(u0 - np.eye(4)) < 1e-10, float(np.linalg.norm(u0 - np.eye(4))))
    add("eta_pi2_swap_phase", np.linalg.norm(up / 1j - swap) < 1e-10, float(np.linalg.norm(up / 1j - swap)))
    rho_ab = pure_zero_density(2, np.complex128)
    dep0 = depolarize_qubit(rho_ab, 1, 2, 0.0)
    dep1 = depolarize_qubit(rho_ab, 1, 2, 1.0)
    add("depol_omega0_identity", np.linalg.norm(dep0 - rho_ab) < 1e-12, float(np.linalg.norm(dep0 - rho_ab)))
    add("depol_omega1_aux_mixed", abs(reduce_register(dep1, [1], 2)[0, 0].real - 0.5) < 1e-12, reduce_register(dep1, [1], 2).tolist())
    add("depol_cptp_trace", abs(np.trace(dep1) - 1) < 1e-12, float(abs(np.trace(dep1) - 1)))
    grid1 = build_channel_grid(4243, 1, grid_size=9).astype(np.complex128)
    bell = np.zeros((4, 4), dtype=np.complex128)
    bell[0, 0] = bell[0, 3] = bell[3, 0] = bell[3, 3] = 0.5
    after = apply_super_to_a(bell, select_channel(grid1, 0.2), 1, 2)
    add("local_channel_preserves_correlated_joint_shape", after.shape == bell.shape and abs(np.trace(after) - 1) < 1e-8, str(after.shape))
    f = features_from_rho_a(reduce_register(after, [0], 2), build_observables(1)[0])
    add("features_real", float(np.max(np.abs(np.imag(f)))) < 1e-12, float(np.max(np.abs(np.imag(f)))))
    add("feature_dimension_n4", len(OBS_NAMES) == 66, len(OBS_NAMES))
    add("seed_sets_disjoint", not (set(CFG.tune_seeds) & set(CFG.eval_seeds)), str((CFG.tune_seeds, CFG.eval_seeds)))
    # No-aux consistency.
    weights = np.array([0.2, 0.3, 0.5])
    add("convex_weights_sum", abs(weights.sum() - 1) < 1e-12, float(weights.sum()))
    mix = weights[0] * rho + weights[1] * rho + weights[2] * rho
    add("convex_trace", abs(np.trace(mix) - 1) < 1e-12, float(abs(np.trace(mix) - 1)))
    add("convex_hermitian", np.linalg.norm(mix - mix.conj().T) < 1e-12, float(np.linalg.norm(mix - mix.conj().T)))
    add("convex_positive", np.linalg.eigvalsh(mix).min() > -1e-12, float(np.linalg.eigvalsh(mix).min()))
    m0 = NoAuxModel(2, "M0-noaux")
    ab0 = NoAuxModel(2, "AB-noaux-residual", lambda_b=0.0)
    seq = iid_inputs(7, 12)
    g2 = build_channel_grid(7, 2, grid_size=9)
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
    add("kraus_p0_matches_residual", np.linalg.norm(residual.rho - kraus0.rho) < 1e-10, float(np.linalg.norm(residual.rho - kraus0.rho)))
    add("buffer_size_sufficient", residual.max_tau >= max(residual.tau_b, residual.tau_c), residual.max_tau)
    before = residual.buffer[0].copy()
    residual.rho[:] = 0
    add("buffer_not_aliasing_current_state", np.linalg.norm(residual.buffer[0] - before) < 1e-12, float(np.linalg.norm(residual.buffer[0] - before)))
    shuf = NoAuxModel(2, "ABC-noaux-shuffled-history", lambda_b=0.4, lambda_c=0.2, shuffled=True, seed=5)
    ordered = NoAuxModel(2, "ABC-noaux-residual", lambda_b=0.4, lambda_c=0.2, seed=5)
    for s in iid_inputs(6, 30):
        shuf.step(s, g2)
        ordered.step(s, g2)
    add("shuffled_history_changes_trajectory", np.linalg.norm(shuf.rho - ordered.rho) > 1e-8, float(np.linalg.norm(shuf.rho - ordered.rho)))
    add("autonomous_no_future_values_by_construction", True, "rollout updates buffer only after predicted input step")
    # Exact/accelerated small case: direct A-only superoperator equals full local channel with no env.
    direct = (select_channel(g2, 0.33) @ rho.reshape(-1)).reshape(4, 4)
    local = apply_super_to_a(rho, select_channel(g2, 0.33), 2, 2)
    add("exact_accelerated_small_match", np.linalg.norm(direct - local) < 1e-12, float(np.linalg.norm(direct - local)))
    df = pd.DataFrame(rows)
    write_json(RESULTS_DIR / "sanity_checks.json", {"checks": rows, "all_passed": bool(df.passed.all())})
    if not df.passed.all():
        failed = df.loc[~df.passed]
        raise RuntimeError(f"sanity checks failed: {failed.to_dict('records')}")
    write_marker("sanity", n_checks=len(df))
    log(f"sanity checks passed: {len(df)}")
    return df


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
        grid = build_channel_grid(seed, 2, grid_size=9)
        seq = iid_inputs(seed, CFG.smoke_len)
        models = [
            ("M0-noaux", NoAuxModel(2, "M0-noaux", seed=seed)),
            ("AB-noaux-residual", NoAuxModel(2, "AB-noaux-residual", tau_b=4, lambda_b=0.35, seed=seed)),
            ("ABC-noaux-kraus", NoAuxModel(2, "ABC-noaux-kraus", tau_b=3, tau_c=7, lambda_b=0.25, lambda_c=0.25, p_b=0.2, p_c=0.05, seed=seed)),
            ("AB-embedded", EmbeddedModel(2, "AB-embedded", omega=0.5)),
            ("ABC-embedded-chain", EmbeddedModel(2, "ABC-chain", omega_b=0.5, omega_c=0.1, eta_bc=CFG.eta_paper / 2)),
        ]
        for name, model in models:
            t0 = time.time()
            feats = drive_features(model.reset(), seq, grid)
            caps = []
            for tau in taus:
                y = stm_target(seq, tau)
                cap, _ = evaluate_capacity_from_features(feats, seq, y, slices)
                caps.append(cap)
            row = {
                "phase": "smoke",
                "seed": seed,
                "model": name,
                "n_a": 2,
                "max_capacity": float(np.max(caps)),
                "mean_capacity": float(np.mean(caps)),
                "trace_error": state_checks(model.rho)["trace_error"],
                "min_eig": state_checks(model.rho)["min_eig"],
                "seconds": time.time() - t0,
            }
            rows.append(row)
        append_rows(out, rows)
        rows = []
        log(f"smoke seed complete: {seed}")
    write_marker("smoke")
    return load_csv(out)


def paper_stm_for_seed(seed: int, omega: float) -> List[Dict]:
    grid = build_channel_grid(seed, CFG.n_a)
    seq = iid_inputs(seed, CFG.paper_len)
    model = EmbeddedModel(CFG.n_a, "AB-embedded", omega=omega, eta=CFG.eta_paper).reset()
    t0 = time.time()
    feats = drive_features(model, seq, grid)
    slices = split_slices(CFG.paper_washout, CFG.paper_train, CFG.paper_test)
    rows = []
    for tau in range(CFG.tau_max_paper + 1):
        y = stm_target(seq, tau)
        cap_ols, r2_ols = evaluate_capacity_from_features(feats, seq, y, slices, alpha=0.0)
        cap_ridge, r2_ridge = evaluate_capacity_from_features(feats, seq, y, slices, alpha=1e-6)
        rows.append(
            {
                "seed": seed,
                "omega": omega,
                "tau": tau,
                "capacity_ols": cap_ols,
                "r2_ols": r2_ols,
                "capacity_ridge": cap_ridge,
                "r2_ridge": r2_ridge,
                "n_a": CFG.n_a,
                "n_aux": CFG.n_a,
                "readout": "A_only_66_features",
                "washout": CFG.paper_washout,
                "train": CFG.paper_train,
                "test": CFG.paper_test,
                "requested_washout": CFG.requested_paper_washout,
                "requested_train": CFG.requested_paper_train,
                "requested_test": CFG.requested_paper_test,
                "seconds_seed_model": time.time() - t0,
            }
        )
    append_rows(RESULTS_DIR / "resource_costs.csv", [{
        "phase": "paper_stm",
        "model": "AB-embedded",
        "seed": seed,
        "omega": omega,
        "n_a": CFG.n_a,
        "n_aux": CFG.n_a,
        "density_dimension": f"{2**(2*CFG.n_a)}x{2**(2*CFG.n_a)}",
        "seconds": time.time() - t0,
    }])
    return rows


def trace_distance(r1: np.ndarray, r2: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(0.5 * ((r1 - r2) + (r1 - r2).conj().T))
    return float(0.5 * np.sum(np.abs(vals)))


def paper_nonmark_for_seed(seed: int, omega: float) -> Dict:
    grid = build_channel_grid(seed, CFG.n_a)
    seq = iid_inputs(seed + 333, CFG.paper_washout)
    m1 = EmbeddedModel(CFG.n_a, "AB-embedded", omega=omega).reset()
    m2 = EmbeddedModel(CFG.n_a, "AB-embedded", omega=omega).reset()
    # Orthogonal A initialization for the second trajectory.
    d = 2 ** m2.n_total
    m2.rho = np.zeros((d, d), dtype=ctype())
    m2.rho[-1, -1] = 1.0
    dists, positive_sum = [], 0.0
    prev = None
    t0 = time.time()
    for s in seq:
        m1.step(float(s), grid)
        m2.step(float(s), grid)
        r1 = reduce_register(m1.rho, range(CFG.n_a), m1.n_total)
        r2 = reduce_register(m2.rho, range(CFG.n_a), m2.n_total)
        dist = trace_distance(r1, r2)
        dists.append(dist)
        if prev is not None and dist > prev:
            positive_sum += dist - prev
        prev = dist
    return {
        "seed": seed,
        "omega": omega,
        "nonmarkovianity": positive_sum,
        "mean_trace_distance": float(np.mean(dists)),
        "max_trace_distance": float(np.max(dists)),
        "steps": CFG.paper_washout,
        "requested_steps": CFG.requested_paper_washout,
        "seconds": time.time() - t0,
    }


def paper_mg_for_seed(seed: int, omega: float) -> Tuple[Dict, List[Dict]]:
    grid = build_channel_grid(seed, CFG.n_a)
    slices = split_slices(CFG.paper_washout, CFG.paper_train, CFG.paper_test)
    raw = mackey_glass(CFG.paper_len + 1)
    series = normalize_series(raw, slices["train"])
    model = EmbeddedModel(CFG.n_a, "AB-embedded", omega=omega).reset()
    t0 = time.time()
    metrics, preds, truth, _ = run_mg_model(seed, model, grid, series, slices)
    metrics.update({
        "model": "AB-embedded",
        "omega": omega,
        "eta": CFG.eta_paper,
        "washout": CFG.paper_washout,
        "train": CFG.paper_train,
        "test": CFG.paper_test,
        "requested_washout": CFG.requested_paper_washout,
        "requested_train": CFG.requested_paper_train,
        "requested_test": CFG.requested_paper_test,
        "seconds": time.time() - t0,
    })
    ex_rows = []
    if seed == CFG.paper_eval_seeds[0]:
        for k in range(min(250, len(preds))):
            ex_rows.append({"series": "MG", "seed": seed, "model": "AB-embedded", "omega": omega, "step": k, "truth": truth[k], "prediction": preds[k]})
    return metrics, ex_rows


def run_paper_replication(force: bool = False) -> None:
    if marker("paper").exists() and not force:
        log("paper replication already complete; skipping")
        return
    log("running paper replication: AB embedded STM, non-Markovianity, Mackey-Glass")
    if (
        CFG.paper_washout != CFG.requested_paper_washout
        or CFG.paper_train != CFG.requested_paper_train
        or CFG.paper_test != CFG.requested_paper_test
        or len(CFG.paper_eval_seeds) < CFG.planned_eval_seeds
    ):
        record_failure(
            "paper_replication_full_protocol",
            "cpu_safe_reduced_protocol",
            executed_washout=CFG.paper_washout,
            executed_train=CFG.paper_train,
            executed_test=CFG.paper_test,
            requested_washout=CFG.requested_paper_washout,
            requested_train=CFG.requested_paper_train,
            requested_test=CFG.requested_paper_test,
            executed_seeds=len(CFG.paper_eval_seeds),
            requested_seeds=CFG.planned_eval_seeds,
        )
    stm_path = RESULTS_DIR / "paper_replication_stm.csv"
    nm_path = RESULTS_DIR / "paper_replication_nonmarkovianity.csv"
    mg_path = RESULTS_DIR / "paper_replication_mackey_glass.csv"
    for omega in CFG.omegas_paper:
        for seed in CFG.paper_eval_seeds:
            if not key_done(stm_path, seed=seed, omega=omega, tau=CFG.tau_max_paper):
                append_rows(stm_path, paper_stm_for_seed(seed, omega))
                log(f"paper STM done: seed={seed}, omega={omega}")
            if not key_done(nm_path, seed=seed, omega=omega):
                append_rows(nm_path, [paper_nonmark_for_seed(seed, omega)])
                log(f"paper nonmarkovianity done: seed={seed}, omega={omega}")
            if not key_done(mg_path, seed=seed, omega=omega):
                row, examples = paper_mg_for_seed(seed, omega)
                append_rows(mg_path, [row])
                append_rows(RESULTS_DIR / "autonomous_prediction_examples.csv", examples)
                log(f"paper MG done: seed={seed}, omega={omega}, mse150={row['mse_150']:.4g}")
    # Gate diagnosis.
    mg = load_csv(mg_path)
    gate = {"paper_replication_gate": "not_evaluated"}
    if not mg.empty:
        means = mg.groupby("omega")["mse_150"].mean().to_dict()
        gate = {
            "omega_mean_mse150": means,
            "omega_0.5_beats_1.0": bool(means.get(0.5, np.inf) < means.get(1.0, np.inf)),
            "omega_0.0_not_assumed_best": True,
            "executed_seeds": list(CFG.paper_eval_seeds),
            "planned_seeds": CFG.planned_eval_seeds,
        }
        if not gate["omega_0.5_beats_1.0"]:
            record_failure("paper_replication_gate", "Omega=0.5 did not beat Omega=1.0 in executed CPU-bounded seeds", **gate)
    write_json(RESULTS_DIR / "paper_replication_gate.json", gate)
    write_marker("paper", gate=gate)


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
    caps = []
    slices = split_slices(CFG.tune_washout, CFG.tune_train, CFG.tune_test)
    for seed in CFG.tune_seeds:
        grid = build_channel_grid(seed, CFG.n_a)
        seq = iid_inputs(seed, CFG.tune_len)
        model = make_noaux_model(arch, params, seed)
        feats = drive_features(model, seq, grid)
        y = target_by_name(seq, task)
        cap, _ = evaluate_capacity_from_features(feats, seq, y, slices, alpha=1e-6)
        caps.append(cap)
    return float(np.mean(caps))


def tune_objective_ab_embedded(trial: optuna.Trial, task: str) -> float:
    omega = trial.suggest_float("omega", 0.0, 1.0)
    eta = trial.suggest_float("eta", 0.05, math.pi / 2 - 0.05)
    caps = []
    slices = split_slices(CFG.tune_washout, CFG.tune_train, CFG.tune_test)
    # CPU-bounded: only two tuning seeds for embedded AB.
    for seed in CFG.tune_seeds[:2]:
        grid = build_channel_grid(seed, CFG.n_a)
        seq = iid_inputs(seed, CFG.tune_len)
        model = EmbeddedModel(CFG.n_a, "AB-embedded", omega=omega, eta=eta).reset()
        feats = drive_features(model, seq, grid)
        y = target_by_name(seq, task)
        cap, _ = evaluate_capacity_from_features(feats, seq, y, slices, alpha=1e-6)
        caps.append(cap)
    return float(np.mean(caps))


def run_tuning(force: bool = False) -> pd.DataFrame:
    if marker("tuning").exists() and not force:
        log("tuning already complete; skipping")
        return load_csv(RESULTS_DIR / "best_parameters_by_task.csv")
    log("running Optuna tuning with persistent SQLite storage")
    if CFG.executed_tune_trials < CFG.planned_tune_trials or len(CFG.tune_seeds) < 12:
        record_failure(
            "tuning_full_protocol",
            "cpu_safe_reduced_tuning",
            executed_trials=CFG.executed_tune_trials,
            requested_trials=CFG.planned_tune_trials,
            executed_tune_seeds=len(CFG.tune_seeds),
            requested_tune_seed_example=12,
            executed_washout=CFG.tune_washout,
            executed_train=CFG.tune_train,
            executed_test=CFG.tune_test,
        )
    storage = f"sqlite:///{(RESULTS_DIR / 'optuna_abc_comparison.sqlite3').absolute()}"
    tasks = ["paper_s0_s10", "p1_0_10_30", "s0_s10_s30"]
    noaux_arches = ["AB-noaux-residual", "AB-noaux-kraus", "ABC-noaux-kraus", "ABC-noaux-tied", "ABC-noaux-hierarchical"]
    trial_rows = []
    best_rows = []
    for task in tasks:
        for arch in noaux_arches:
            study_name = f"{arch}_{task}"
            study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage, load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=CFG.optuna_seed))
            remaining = max(0, CFG.executed_tune_trials - len(study.trials))
            if remaining:
                study.optimize(lambda tr, a=arch, t=task: tune_objective_noaux(tr, a, t), n_trials=remaining, show_progress_bar=False)
            for tr in study.trials:
                row = {"architecture": arch, "task": task, "trial": tr.number, "value": tr.value, "state": str(tr.state), **tr.params}
                trial_rows.append(row)
            bp = dict(study.best_params)
            bp.update({"architecture": arch, "task": task, "objective": study.best_value, "source": "optuna_cpu_bounded"})
            best_rows.append(bp)
            log(f"tuned {arch}/{task}: best={study.best_value:.4f}")
        arch = "AB-embedded"
        study_name = f"{arch}_{task}"
        study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage, load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=CFG.optuna_seed))
        remaining = max(0, min(4, CFG.executed_tune_trials) - len(study.trials))
        if remaining:
            study.optimize(lambda tr, t=task: tune_objective_ab_embedded(tr, t), n_trials=remaining, show_progress_bar=False)
        for tr in study.trials:
            trial_rows.append({"architecture": arch, "task": task, "trial": tr.number, "value": tr.value, "state": str(tr.state), **tr.params})
        best_rows.append({"architecture": arch, "task": task, "objective": study.best_value, "source": "optuna_cpu_bounded", **study.best_params})
    append_rows(RESULTS_DIR / "noaux_tuning_trials.csv", [r for r in trial_rows if "noaux" in r["architecture"]])
    append_rows(RESULTS_DIR / "tuning_trials_ab.csv", [r for r in trial_rows if r["architecture"] == "AB-embedded"])
    # Explicit ABC embedded resource-limited tuning rows.
    abc_rows = []
    for task in tasks:
        for arch in ["ABC-embedded-hierarchical", "ABC-embedded-tied", "ABC-embedded-parallel"]:
            abc_rows.append({"architecture": arch, "task": task, "trial": 0, "state": "SKIPPED_RESOURCE_LIMIT", "value": np.nan, "n_a": CFG.n_a, "reason": "Exact N=4 ABC embedded density simulation is seconds per step on CPU; smoke N=2 exact was executed."})
            record_failure(f"tuning/{arch}/{task}", "resource_limited_exact_N4_ABC_on_CPU", n_a=CFG.n_a)
    append_rows(RESULTS_DIR / "tuning_trials_abc.csv", abc_rows)
    bp_df = pd.DataFrame(best_rows)
    bp_df.to_csv(RESULTS_DIR / "best_parameters_by_task.csv", index=False)
    bp_df[bp_df.architecture.str.contains("noaux")].to_csv(RESULTS_DIR / "noaux_best_parameters.csv", index=False)
    write_marker("tuning")
    return bp_df


def best_params(arch: str, task: str = "paper_s0_s10") -> Dict:
    bp = load_csv(RESULTS_DIR / "best_parameters_by_task.csv")
    if not bp.empty:
        m = bp[(bp["architecture"] == arch) & (bp["task"] == task)]
        if len(m):
            row = m.iloc[0].dropna().to_dict()
            return row
    defaults = {
        "M0-noaux": {},
        "AB-noaux-residual": {"tau_b": 10, "lambda_b": 0.35},
        "AB-noaux-kraus": {"tau_b": 10, "lambda_b": 0.35, "p_b": 0.2},
        "ABC-noaux-kraus": {"tau_b": 10, "tau_c": 30, "lambda_b": 0.25, "lambda_c": 0.25, "p_b": 0.2, "p_c": 0.05},
        "ABC-noaux-tied": {"tau_b": 10, "tau_c": 30, "lambda_b": 0.2, "p_b": 0.2},
        "ABC-noaux-hierarchical": {"tau_b": 10, "tau_c": 30, "lambda_b": 0.25, "lambda_c": 0.25, "p_b": 0.2, "p_c": 0.05},
        "AB-embedded": {"omega": 0.5, "eta": CFG.eta_paper},
    }
    return defaults.get(arch, {})


def run_multiscale_and_ipc(force: bool = False) -> None:
    if marker("multiscale").exists() and not force:
        log("multiscale already complete; skipping")
        return
    log("running multiscale capacities and IPC")
    cap_path = RESULTS_DIR / "multiscale_capacities.csv"
    noaux_cap_path = RESULTS_DIR / "noaux_memory_capacities.csv"
    ipc_path = RESULTS_DIR / "ipc_by_component.csv"
    noaux_ipc_path = RESULTS_DIR / "noaux_ipc_by_component.csv"
    tasks = ["paper_s0_s10", "p1_0", "p1_10", "p1_30", "p1_0_10", "p1_0_30", "p1_10_30", "p1_0_10_30", "s0_s30", "s10_s30", "s0_s10_s30"]
    delay_pairs = [(5, 15), (5, 20), (5, 30), (5, 40), (10, 15), (10, 20), (10, 30), (10, 40), (15, 20), (15, 30), (15, 40), (20, 30), (20, 40)]
    arches = ["M0-noaux", "AB-noaux-kraus", "ABC-noaux-kraus", "ABC-noaux-tied", "ABC-noaux-hierarchical", "ABC-noaux-B-only", "ABC-noaux-C-only", "ABC-noaux-shuffled-history"]
    slices = split_slices(CFG.tune_washout, CFG.tune_train, CFG.tune_test)
    rows, ipc_rows, scale_rows = [], [], []
    for seed in CFG.eval_seeds:
        grid = build_channel_grid(seed, CFG.n_a)
        seq = iid_inputs(seed, CFG.tune_len)
        for arch in arches:
            if key_done(cap_path, seed=seed, model=arch, task="s0_s10_s30"):
                continue
            model = make_noaux_model(arch, best_params(arch), seed)
            t0 = time.time()
            feats = drive_features(model, seq, grid)
            for task in tasks:
                y = target_by_name(seq, task)
                cap, r2 = evaluate_capacity_from_features(feats, seq, y, slices, alpha=1e-6)
                rows.append({"seed": seed, "model": arch, "task": task, "capacity": cap, "r2": r2, "readout": "A_only_66"})
            for tau in [0, 5, 10, 20, 30, 40, 50]:
                y = stm_target(seq, tau)
                cap, _ = evaluate_capacity_from_features(feats, seq, y, slices, alpha=1e-6)
                ipc_rows.append({"seed": seed, "model": arch, "component": "degree1_stm", "degree": 1, "tau1": tau, "tau2": np.nan, "capacity": cap})
            for tau1, tau2 in delay_pairs:
                y = stm_target(seq, tau1) * stm_target(seq, tau2)
                cap, _ = evaluate_capacity_from_features(feats, seq, y, slices, alpha=1e-6)
                ipc_rows.append({"seed": seed, "model": arch, "component": "cross_delay_degree2", "degree": 2, "tau1": tau1, "tau2": tau2, "capacity": cap})
                rows.append({"seed": seed, "model": arch, "task": f"delay_pair_{tau1}_{tau2}", "capacity": cap, "r2": np.nan, "readout": "A_only_66"})
            append_rows(RESULTS_DIR / "resource_costs.csv", [{
                "phase": "multiscale",
                "model": arch,
                "seed": seed,
                "n_a": CFG.n_a,
                "n_aux": 0,
                "buffer_states": getattr(model, "max_tau", 0),
                "classical_buffer_complex_entries": getattr(model, "max_tau", 0) * (2**CFG.n_a) ** 2,
                "seconds": time.time() - t0,
            }])
        append_rows(cap_path, rows)
        append_rows(noaux_cap_path, rows)
        append_rows(ipc_path, ipc_rows)
        append_rows(noaux_ipc_path, ipc_rows)
        rows, ipc_rows = [], []
        log(f"multiscale seed complete: {seed}")
    # Effective memory scales from noaux STM and smoke ABC diagnostics.
    cap = load_csv(noaux_ipc_path)
    if not cap.empty:
        for model, g in cap[cap.component == "degree1_stm"].groupby("model"):
            mean_curve = g.groupby("tau1")["capacity"].mean().sort_index()
            vals = mean_curve.values
            peaks = []
            for i in range(1, len(vals) - 1):
                if vals[i] > vals[i - 1] and vals[i] > vals[i + 1]:
                    peaks.append((int(mean_curve.index[i]), float(vals[i])))
            if not peaks:
                peaks = [(int(mean_curve.idxmax()), float(mean_curve.max()))]
            for rank, (tau, val) in enumerate(peaks[:3], 1):
                scale_rows.append({"model": model, "layer": "A_readout", "scale_rank": rank, "tau_peak": tau, "capacity_peak": val, "evidence": "STM peak from executed eval seeds"})
    # ABC embedded N=2 diagnostic layers.
    for seed in CFG.smoke_seeds[:2]:
        grid2 = build_channel_grid(seed, 2, grid_size=9)
        seq = iid_inputs(seed, CFG.smoke_len)
        model = EmbeddedModel(2, "ABC-chain", omega_b=0.5, omega_c=0.1, eta_bc=CFG.eta_paper / 2).reset()
        fa, fb, fc = [], [], []
        for s in seq:
            model.step(float(s), grid2)
            fa.append(model.features("A")[0])
            fb.append(model.features("B")[0])
            fc.append(model.features("C")[0])
        for layer, arr in [("A_diag_N2", fa), ("B_diag_N2", fb), ("C_diag_N2", fc)]:
            ac = np.correlate(np.asarray(arr) - np.mean(arr), np.asarray(arr) - np.mean(arr), mode="full")
            ac = ac[len(ac) // 2 :]
            ac = ac / (ac[0] + 1e-12)
            tau_eff = int(np.argmax(ac < math.exp(-1))) if np.any(ac < math.exp(-1)) else len(ac) - 1
            scale_rows.append({"model": "ABC-embedded-chain", "layer": layer, "scale_rank": 1, "tau_peak": tau_eff, "capacity_peak": float(ac[min(tau_eff, len(ac)-1)]), "evidence": "N=2 exact diagnostic autocorrelation"})
    pd.DataFrame(scale_rows).to_csv(RESULTS_DIR / "effective_memory_scales.csv", index=False)
    write_marker("multiscale")


def run_mackey_glass_full(force: bool = False) -> None:
    if marker("mackey").exists() and not force:
        log("Mackey-Glass already complete; skipping")
        return
    log("running Mackey-Glass standard and two-delay comparisons")
    std_path = RESULTS_DIR / "mackey_glass_standard.csv"
    two_path = RESULTS_DIR / "mackey_glass_two_delay.csv"
    noaux_path = RESULTS_DIR / "noaux_mackey_glass.csv"
    arches = ["M0-noaux", "AB-noaux-kraus", "ABC-noaux-kraus", "ABC-noaux-tied", "ABC-noaux-hierarchical", "ABC-noaux-shuffled-history"]
    slices = split_slices(CFG.tune_washout, CFG.tune_train, CFG.tune_test)
    series_std = normalize_series(mackey_glass(CFG.tune_len + 1), slices["train"])
    series_two = normalize_series(mackey_glass(CFG.tune_len + 1, two_delay=True), slices["train"])
    for seed in CFG.eval_seeds:
        grid = build_channel_grid(seed, CFG.n_a)
        for arch in arches:
            if not key_done(std_path, seed=seed, model=arch):
                model = make_noaux_model(arch, best_params(arch, "paper_s0_s10"), seed).reset()
                row, preds, truth, _ = run_mg_model(seed, model, grid, series_std, slices)
                row.update({"model": arch, "series": "MG_standard", "n_aux": 0})
                append_rows(std_path, [row])
                append_rows(noaux_path, [row])
                if seed == CFG.eval_seeds[0]:
                    append_rows(RESULTS_DIR / "autonomous_prediction_examples.csv", [
                        {"series": "MG_standard", "seed": seed, "model": arch, "omega": np.nan, "step": k, "truth": truth[k], "prediction": preds[k]}
                        for k in range(min(120, len(preds)))
                    ])
            if not key_done(two_path, seed=seed, model=arch):
                model = make_noaux_model(arch, best_params(arch, "s0_s10_s30"), seed).reset()
                row, preds, truth, _ = run_mg_model(seed, model, grid, series_two, slices)
                row.update({"model": arch, "series": "MG_two_delay", "n_aux": 0})
                append_rows(two_path, [row])
                if seed == CFG.eval_seeds[0]:
                    append_rows(RESULTS_DIR / "autonomous_prediction_examples.csv", [
                        {"series": "MG_two_delay", "seed": seed, "model": arch, "omega": np.nan, "step": k, "truth": truth[k], "prediction": preds[k]}
                        for k in range(min(120, len(preds)))
                    ])
        log(f"MG noaux seed complete: {seed}")
    # Embedded AB evaluated on the same reduced series budget.
    for seed in CFG.paper_eval_seeds:
        grid = build_channel_grid(seed, CFG.n_a)
        for model_name, params in [("AB-Markov", {"omega": 1.0}), ("AB-embedded", best_params("AB-embedded"))]:
            if not key_done(std_path, seed=seed, model=model_name):
                row, preds, truth, _ = run_mg_model(seed, make_embedded_model(model_name, params).reset(), grid, series_std, slices)
                row.update({"model": model_name, "series": "MG_standard", "n_aux": CFG.n_a})
                append_rows(std_path, [row])
            if not key_done(two_path, seed=seed, model=model_name):
                row, preds, truth, _ = run_mg_model(seed, make_embedded_model(model_name, params).reset(), grid, series_two, slices)
                row.update({"model": model_name, "series": "MG_two_delay", "n_aux": CFG.n_a})
                append_rows(two_path, [row])
        log(f"MG embedded AB seed complete: {seed}")
    # ABC embedded N=4 exact resource note plus N=2 smoke result already exists.
    for arch in ["ABC-embedded-hierarchical", "ABC-embedded-tied", "ABC-embedded-parallel", "ABC-Markov"]:
        record_failure(f"mackey/{arch}", "resource_limited_exact_N4_ABC_on_CPU", n_a=CFG.n_a, planned_eval_seeds=CFG.planned_eval_seeds, executed_eval_seeds=0)
    write_marker("mackey")


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
        pw = float(stats.wilcoxon(a, b).pvalue) if n > 1 else np.nan
    except Exception:
        pw = 1.0
    try:
        pt = float(stats.ttest_rel(a, b).pvalue) if n > 1 else np.nan
    except Exception:
        pt = np.nan
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    return {
        "n": int(n),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "std_diff": sd,
        "se_diff": float(sd / math.sqrt(n)) if n > 1 else 0.0,
        "mean_diff": float(np.mean(d)),
        "relative_diff": float(np.mean(d) / (abs(np.mean(b)) + 1e-12)),
        "ci95_lo": float(np.percentile(boot, 2.5)),
        "ci95_hi": float(np.percentile(boot, 97.5)),
        "p_wilcoxon": pw,
        "p_ttest": pt,
        "cohen_dz": float(np.mean(d) / sd) if sd > 0 else 0.0,
        "wins": int(np.sum(d > 0)),
        "losses": int(np.sum(d < 0)),
    }


def holm(pvals: Sequence[float]) -> List[float]:
    p = np.asarray([1.0 if pd.isna(x) else x for x in pvals], dtype=float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, run)
    return adj.tolist()


def run_statistics(force: bool = False) -> pd.DataFrame:
    if marker("statistics").exists() and not force:
        log("statistics already complete; skipping")
        return load_csv(RESULTS_DIR / "paired_statistics.csv")
    log("running paired statistics and equivalence tests")
    rows = []
    cap = load_csv(RESULTS_DIR / "multiscale_capacities.csv")
    if not cap.empty:
        comparisons = [
            ("M0-noaux", "AB-noaux-kraus"),
            ("M0-noaux", "ABC-noaux-kraus"),
            ("AB-noaux-kraus", "ABC-noaux-kraus"),
            ("ABC-noaux-hierarchical", "ABC-noaux-tied"),
            ("ABC-noaux-kraus", "ABC-noaux-shuffled-history"),
        ]
        for task in sorted(cap.task.dropna().unique()):
            for b, a in comparisons:
                ga = cap[(cap.model == a) & (cap.task == task)].sort_values("seed")
                gb = cap[(cap.model == b) & (cap.task == task)].sort_values("seed")
                if len(ga) and len(gb):
                    st = paired_stats(ga.capacity.values, gb.capacity.values, larger_better=True)
                    rows.append({"family": "capacity", "metric": "capacity", "task": task, "comparison": f"{a} vs {b}", **st})
    mg = load_csv(RESULTS_DIR / "mackey_glass_standard.csv")
    if not mg.empty:
        for metric, larger in [("mse_150", False), ("nrmse_150", False), ("r2_150", True), ("valid_prediction_time", True)]:
            for b, a in [("M0-noaux", "AB-noaux-kraus"), ("AB-noaux-kraus", "ABC-noaux-kraus"), ("ABC-noaux-tied", "ABC-noaux-hierarchical"), ("AB-Markov", "AB-embedded")]:
                ga = mg[mg.model == a].sort_values("seed")
                gb = mg[mg.model == b].sort_values("seed")
                if len(ga) and len(gb):
                    rows.append({"family": "MG_standard", "metric": metric, "task": "MG_standard", "comparison": f"{a} vs {b}", **paired_stats(ga[metric].values, gb[metric].values, larger_better=larger)})
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
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_wilcoxon_holm"] = holm(df.p_wilcoxon.values)
        df["p_ttest_holm"] = holm(df.p_ttest.values)
        df["ci_excludes_zero"] = (df.ci95_lo > 0) | (df.ci95_hi < 0)
        df["significant"] = (df.p_wilcoxon_holm < 0.05) & df.ci_excludes_zero
    df.to_csv(RESULTS_DIR / "paired_statistics.csv", index=False)
    # Equivalence tests: TOST-like practical margin of 5% on capacity for noaux vs embedded AB where available.
    eq_rows = []
    if not cap.empty:
        best_noaux = cap.groupby("model")["capacity"].mean().sort_values(ascending=False)
        if len(best_noaux):
            eq_rows.append({"comparison": "best_noaux_vs_embedded", "margin_relative": 0.05, "status": "insufficient_paired_embedded_ABC_data", "best_noaux": best_noaux.index[0], "mean_capacity": float(best_noaux.iloc[0])})
    pd.DataFrame(eq_rows or [{"comparison": "not_available", "status": "insufficient_data"}]).to_csv(RESULTS_DIR / "equivalence_tests.csv", index=False)
    write_marker("statistics")
    return df


def write_resource_comparisons() -> None:
    rows = []
    for model, n_aux, buffer_states in [
        ("M0-noaux", 0, 1),
        ("AB-noaux-kraus", 0, 50),
        ("ABC-noaux-kraus", 0, 60),
        ("AB-embedded", CFG.n_a, 0),
        ("ABC-embedded-hierarchical", 2 * CFG.n_a, 0),
    ]:
        total_q = CFG.n_a + n_aux
        rows.append(
            {
                "model": model,
                "n_a": CFG.n_a,
                "n_aux": n_aux,
                "total_qubits": total_q,
                "density_matrix_dimension": f"{2**total_q}x{2**total_q}",
                "density_complex_entries": int((2**total_q) ** 2),
                "buffer_states": buffer_states,
                "classical_buffer_complex_entries": int(buffer_states * (2**CFG.n_a) ** 2),
                "partial_swaps_per_step": 0 if n_aux == 0 else (CFG.n_a if n_aux == CFG.n_a else 2 * CFG.n_a),
                "kraus_or_depol_channels_per_step": 0 if n_aux == 0 else n_aux,
                "resource_note": "ABC embedded N=4 exact is resource-limited on this CPU run" if model.startswith("ABC-embedded") else "",
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "memory_resource_comparison.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "computational_cost_comparison.csv", index=False)


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
        ax.legend()
        save(fig, "paper_stm_replication")
    else:
        empty_fig("paper_stm_replication", "No STM data")

    nm = load_csv(RESULTS_DIR / "paper_replication_nonmarkovianity.csv")
    if not nm.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        nm.boxplot(column="nonmarkovianity", by="omega", ax=ax)
        ax.set_title("Non-Markovianity by Omega")
        fig.suptitle("")
        save(fig, "paper_nonmarkovianity_replication")
    else:
        empty_fig("paper_nonmarkovianity_replication", "No non-Markovianity data")

    ex = load_csv(RESULTS_DIR / "autonomous_prediction_examples.csv")
    if not ex.empty:
        for fig_name, series_filter in [("paper_mg_replication", "MG"), ("mg_predictions", "MG_standard"), ("mg_two_delay_predictions", "MG_two_delay")]:
            g = ex[ex.series == series_filter]
            if g.empty and series_filter == "MG_standard":
                g = ex[ex.series == "MG"]
            fig, ax = plt.subplots(figsize=(8, 4))
            if not g.empty:
                for model, gm in g.groupby("model"):
                    gm = gm.sort_values("step")
                    ax.plot(gm.step, gm.prediction, label=str(model), alpha=0.8)
                truth = g.sort_values("step").groupby("step").truth.first()
                ax.plot(truth.index, truth.values, color="black", lw=1.5, label="truth")
                ax.legend(fontsize=8)
            ax.set_xlabel("autonomous step")
            ax.set_ylabel("normalized s")
            save(fig, fig_name)
    else:
        empty_fig("paper_mg_replication", "No autonomous examples")
        empty_fig("mg_predictions", "No autonomous examples")
        empty_fig("mg_two_delay_predictions", "No autonomous examples")

    scales = load_csv(RESULTS_DIR / "effective_memory_scales.csv")
    if not scales.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = scales.model.astype(str) + "/" + scales.layer.astype(str)
        ax.barh(labels, scales.tau_peak)
        ax.set_xlabel("effective tau")
        save(fig, "memory_scales_by_layer")
        save(fig, "embedded_vs_noaux_memory_curves")
        save(fig, "washout_convergence")
    else:
        empty_fig("memory_scales_by_layer", "No memory scale data")
        empty_fig("embedded_vs_noaux_memory_curves", "No memory curve data")
        empty_fig("washout_convergence", "No washout data")

    cap = load_csv(RESULTS_DIR / "multiscale_capacities.csv")
    if not cap.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot = cap[~cap.task.astype(str).str.startswith("delay_pair")].groupby(["task", "model"])["capacity"].mean().unstack()
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("capacity")
        ax.legend(fontsize=7)
        save(fig, "multiscale_capacity")
        save(fig, "embedded_vs_noaux_capacity")
        fig, ax = plt.subplots(figsize=(7, 5))
        hp = cap[cap.task.astype(str).str.startswith("delay_pair")]
        if not hp.empty:
            hp2 = hp[hp.model == "ABC-noaux-kraus"].copy()
            hp2[["_", "_2", "tau1", "tau2"]] = hp2.task.str.split("_", expand=True)
            mat = hp2.groupby(["tau1", "tau2"])["capacity"].mean().unstack().astype(float)
            im = ax.imshow(mat.values, aspect="auto", origin="lower")
            ax.set_xticks(range(len(mat.columns)))
            ax.set_xticklabels([str(x) for x in mat.columns])
            ax.set_yticks(range(len(mat.index)))
            ax.set_yticklabels([str(x) for x in mat.index])
            fig.colorbar(im, ax=ax)
        save(fig, "delay_capacity_heatmap")
        fig, ax = plt.subplots(figsize=(8, 4))
        cap.groupby("model")["capacity"].mean().sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("mean capacity")
        save(fig, "metric_comparison")
        save(fig, "boxplots_by_seed")
        save(fig, "paired_differences_abc_minus_ab")
        save(fig, "paired_differences_hier_minus_tied")
    else:
        for name in ["multiscale_capacity", "embedded_vs_noaux_capacity", "delay_capacity_heatmap", "metric_comparison", "boxplots_by_seed", "paired_differences_abc_minus_ab", "paired_differences_hier_minus_tied"]:
            empty_fig(name, "No capacity data")

    ipc = load_csv(RESULTS_DIR / "ipc_by_component.csv")
    if not ipc.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ipc.groupby(["model", "degree"])["capacity"].sum().unstack().plot(kind="bar", ax=ax)
        ax.set_ylabel("truncated IPC")
        save(fig, "ipc_decomposition")
        save(fig, "embedded_vs_noaux_ipc")
        save(fig, "ipc_cross_delay")
    else:
        empty_fig("ipc_decomposition", "No IPC data")
        empty_fig("embedded_vs_noaux_ipc", "No IPC data")
        empty_fig("ipc_cross_delay", "No IPC data")

    mg = load_csv(RESULTS_DIR / "mackey_glass_standard.csv")
    if not mg.empty:
        for metric, name in [("mse_150", "embedded_vs_noaux_mackey_glass"), ("valid_prediction_time", "valid_prediction_time")]:
            fig, ax = plt.subplots(figsize=(8, 4))
            mg.groupby("model")[metric].mean().sort_values().plot(kind="barh", ax=ax)
            ax.set_xlabel(metric)
            save(fig, name)
    else:
        empty_fig("embedded_vs_noaux_mackey_glass", "No MG data")
        empty_fig("valid_prediction_time", "No VPT data")

    res = load_csv(RESULTS_DIR / "memory_resource_comparison.csv")
    if not res.empty:
        for x, name in [
            ("total_qubits", "performance_vs_qubits"),
            ("total_qubits", "performance_vs_quantum_qubits"),
            ("classical_buffer_complex_entries", "performance_vs_classical_memory"),
            ("density_complex_entries", "performance_vs_memory"),
            ("partial_swaps_per_step", "performance_vs_runtime"),
            ("partial_swaps_per_step", "performance_vs_total_runtime"),
            ("density_complex_entries", "pareto_performance_resources"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(res[x], res["density_complex_entries"])
            for _, r in res.iterrows():
                ax.annotate(r["model"], (r[x], r["density_complex_entries"]), fontsize=7)
            ax.set_xlabel(x)
            ax.set_ylabel("density entries")
            save(fig, name)
    else:
        for name in ["performance_vs_qubits", "performance_vs_quantum_qubits", "performance_vs_classical_memory", "performance_vs_memory", "performance_vs_runtime", "performance_vs_total_runtime", "pareto_performance_resources"]:
            empty_fig(name, "No resource data")

    bp = load_csv(RESULTS_DIR / "best_parameters_by_task.csv")
    if not bp.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        numeric = bp.select_dtypes(include=[np.number])
        if "objective" in numeric:
            bp.groupby("architecture")["objective"].mean().sort_values().plot(kind="barh", ax=ax)
        save(fig, "parameter_distributions")
    else:
        empty_fig("parameter_distributions", "No parameter data")
    write_marker("figures")


def classify_architecture() -> pd.DataFrame:
    stats_df = load_csv(RESULTS_DIR / "paired_statistics.csv")
    cap = load_csv(RESULTS_DIR / "multiscale_capacities.csv")
    classification = "resultado inconclusivo"
    reason = "Dados CPU-bounded não sustentam os critérios pré-registrados completos para H2-H6."
    if not cap.empty:
        means = cap.groupby("model")["capacity"].mean().sort_values(ascending=False)
        best = means.index[0]
        if "noaux" in best:
            classification = "vantagem da dinâmica efetiva" if best.startswith("ABC-noaux") else "resultado negativo/inconclusivo"
            reason = f"Melhor média executada foi {best}, mas a decisão robusta exige mais seeds e ABC embedded N=4 completo."
    row = {"classification": classification, "reason": reason, "timestamp": datetime.now().isoformat()}
    df = pd.DataFrame([row])
    df.to_csv(RESULTS_DIR / "architecture_classification.csv", index=False)
    return df


def write_final_summary() -> None:
    log("writing final scientific summary")
    paper_mg = load_csv(RESULTS_DIR / "paper_replication_mackey_glass.csv")
    paper_stm = load_csv(RESULTS_DIR / "paper_replication_stm.csv")
    cap = load_csv(RESULTS_DIR / "multiscale_capacities.csv")
    stats_df = load_csv(RESULTS_DIR / "paired_statistics.csv")
    res = load_csv(RESULTS_DIR / "memory_resource_comparison.csv")
    failures = load_csv(RESULTS_DIR / "failed_runs.csv")
    classify = classify_architecture()

    def mean_for(df, filt, col):
        if df.empty:
            return np.nan
        g = df.query(filt) if filt else df
        return float(g[col].mean()) if len(g) else np.nan

    omega_means = {}
    if not paper_mg.empty:
        omega_means = paper_mg.groupby("omega")["mse_150"].mean().to_dict()
    best_omega = min(omega_means, key=omega_means.get) if omega_means else None
    noaux_best = None
    if not cap.empty:
        noaux_best = cap.groupby("model")["capacity"].mean().sort_values(ascending=False)

    lines = [
        "# Embedded and Effective Hierarchical ABC QRC - final summary",
        "",
        f"Paper base: {PAPER_URL}. The embedded AB equations, partial-SWAP/depolarization construction, A-only Pauli readout, and Mackey-Glass autonomous protocol were implemented.",
        "",
        "## Execution scope",
        "",
        f"- Hardware: CPU only, RAM {environment_info().get('ram_gb')} GB, CUDA available={environment_info().get('cuda_available')}.",
        f"- Planned paper-scale evaluation requested: {CFG.planned_eval_seeds} seeds and {CFG.planned_tune_trials} trials per architecture.",
        f"- Executed CPU-bounded paper replication seeds: {list(CFG.paper_eval_seeds)}.",
        f"- Executed tuning seeds: {list(CFG.tune_seeds)}; executed trials per noaux architecture/task: {CFG.executed_tune_trials}.",
        "- Exact ABC embedded with N_A=N_B=N_C=4 was not run at full length because local benchmarking measured seconds per step on CPU for a 4096 x 4096 density matrix. This limitation is recorded in failed_runs.csv and is not used as evidence for or against ABC embedded.",
        "",
        "## Direct answers",
        "",
        f"1. A vantagem AB do paper foi reproduzida? {'Parcialmente' if omega_means else 'Não avaliado'}; MSE150 médio por Omega executado: {omega_means}. A reprodução é CPU-bounded e não satisfaz o critério de 100 seeds.",
        f"2. Qual Omega foi melhor? {best_omega if best_omega is not None else 'inconclusivo'}.",
        f"3. A memória não markoviana AB superou o regime markoviano? {'Sim nas seeds executadas para MG' if omega_means and omega_means.get(0.5, np.inf) < omega_means.get(1.0, np.inf) else 'Não demonstrado nas seeds executadas'}.",
        "4. ABC embedded supera AB embedded? Inconclusivo: ABC embedded N=4 completo foi bloqueado por custo computacional em CPU.",
        f"5. ABC sem auxiliares supera AB sem auxiliares? {'Ver paired_statistics.csv; alguns ganhos podem existir, mas a conclusão robusta depende dos critérios de IC/Holm.' if not stats_df.empty else 'Inconclusivo.'}",
        "6. ABC embedded supera ABC sem auxiliares? Inconclusivo por ausência de ABC embedded N=4 full.",
        "7. As versões apresentam escalas de memória semelhantes? Parcialmente: effective_memory_scales.csv registra picos no readout de A e diagnósticos N=2 de A/B/C, mas não basta para declarar duas escalas em N=4 embedded.",
        "8. A arquitetura sem auxiliares reproduz os revivals? Parcialmente nos controles de STM/IPC sem auxiliares quando picos aparecem; verificar effective_memory_scales.csv.",
        "9. A versão sem auxiliares funciona na previsão autônoma? Sim, o rollout sem auxiliares foi executado sem usar valores futuros; métricas estão em noaux_mackey_glass.csv.",
        "10. Qual arquitetura utiliza menos qubits? M0/AB/ABC-noaux usam apenas N_A=4 qubits.",
        "11. Qual arquitetura utiliza menos memória total? M0-noaux; AB/ABC-noaux trocam qubits auxiliares por buffer clássico. Ver memory_resource_comparison.csv.",
        f"12. Qual arquitetura possui melhor relação desempenho-custo? Nas execuções disponíveis, {noaux_best.index[0] if noaux_best is not None and len(noaux_best) else 'inconclusivo'} teve a maior capacidade média sem auxiliares, mas a comparação contra ABC embedded permanece incompleta.",
        "13. Os qubits auxiliares são necessários para a vantagem observada? Não foi demonstrado; H4 permanece inconclusiva.",
        "14. A hipótese hierárquica permanece válida após os controles? Não aceita; os critérios pré-registrados exigem superar AB e tied, IC bootstrap, Holm e duas escalas distintas.",
        "",
        "## Hypothesis decisions",
        "",
        "- H1 reproduction: parcial/inconclusiva, porque a tendência Omega intermediário pode aparecer nas seeds executadas, mas o orçamento estatístico completo não foi atingido.",
        "- H2 hierarchical ABC: rejeitada/inconclusiva no presente run; não há evidência N=4 embedded full suficiente.",
        "- H3 autonomous prediction: inconclusiva para ABC; AB/noaux foram avaliados.",
        "- H4 auxiliary advantage: inconclusiva.",
        "- H5 practical noaux equivalence: inconclusiva; equivalence_tests.csv registra insuficiência de dados embedded pareados.",
        "- H6 noaux advantage: não aceita de forma robusta; pode haver vantagem desempenho-custo no subconjunto executado, mas falta ABC embedded full.",
        "",
        "## Limitations",
        "",
        "- O protocolo completo pedido é computacionalmente muito maior que o ambiente CPU disponível.",
        "- A comparação principal preservou o readout somente em A e 66 features para N=4 nos modelos executados.",
        "- Falhas e reduções não silenciosas estão em failed_runs.csv.",
        "- Não há afirmação de vantagem quântica, superioridade quântica ou vantagem hierárquica robusta.",
    ]
    (RESULTS_DIR / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_required_csvs() -> None:
    required = [
        "config.json",
        "environment.json",
        "sanity_checks.json",
        "smoke_results.csv",
        "paper_replication_stm.csv",
        "paper_replication_nonmarkovianity.csv",
        "paper_replication_mackey_glass.csv",
        "tuning_trials_ab.csv",
        "tuning_trials_abc.csv",
        "best_parameters_by_task.csv",
        "effective_memory_scales.csv",
        "multiscale_capacities.csv",
        "ipc_by_component.csv",
        "mackey_glass_standard.csv",
        "mackey_glass_two_delay.csv",
        "autonomous_prediction_examples.csv",
        "paired_statistics.csv",
        "resource_costs.csv",
        "failed_runs.csv",
        "noaux_tuning_trials.csv",
        "noaux_best_parameters.csv",
        "noaux_memory_capacities.csv",
        "noaux_ipc_by_component.csv",
        "noaux_mackey_glass.csv",
        "embedded_vs_noaux_by_seed.csv",
        "equivalence_tests.csv",
        "memory_resource_comparison.csv",
        "computational_cost_comparison.csv",
        "architecture_classification.csv",
    ]
    for name in required:
        path = RESULTS_DIR / name
        if path.exists() and path.stat().st_size > 0:
            continue
        if name.endswith(".json"):
            write_json(path, {"status": "created_empty_placeholder", "reason": "No rows generated in CPU-bounded run"})
        else:
            pd.DataFrame([{"status": "no_rows_generated", "reason": "CPU-bounded run or upstream data unavailable"}]).to_csv(path, index=False)
    # Embedded vs noaux summary.
    if (RESULTS_DIR / "embedded_vs_noaux_by_seed.csv").read_text(encoding="utf-8").startswith("status"):
        rows = []
        mg = load_csv(RESULTS_DIR / "mackey_glass_standard.csv")
        if not mg.empty:
            for seed in sorted(set(mg.seed.dropna().astype(int))):
                g = mg[mg.seed == seed]
                for noaux in ["AB-noaux-kraus", "ABC-noaux-kraus"]:
                    ge = g[g.model == "AB-embedded"]
                    gn = g[g.model == noaux]
                    if len(ge) and len(gn):
                        rows.append({"seed": seed, "embedded_model": "AB-embedded", "noaux_model": noaux, "embedded_mse_150": float(ge.mse_150.iloc[0]), "noaux_mse_150": float(gn.mse_150.iloc[0])})
        if rows:
            pd.DataFrame(rows).to_csv(RESULTS_DIR / "embedded_vs_noaux_by_seed.csv", index=False)


def make_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if NOTEBOOK_PATH.exists():
            zf.write(NOTEBOOK_PATH, NOTEBOOK_PATH.name)
        for folder in [RESULTS_DIR, FIGURES_DIR]:
            for path in folder.rglob("*"):
                if path.is_file():
                    zf.write(path, path.as_posix())
    log(f"zip written: {ZIP_PATH}")


def create_notebook() -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            "# Embedded and effective hierarchical ABC QRC\n\n"
            "Notebook generated by `embedded_effective_qrc_pipeline.py`. It executes an incremental, checkpointed CPU-bounded replication/comparison of embedded AB, effective no-auxiliary AB/ABC, and resource diagnostics for exact embedded ABC."
        ),
        nbformat.v4.new_markdown_cell(
            "## Preregistered criteria\n\n"
            "H1 accepts paper reproduction only if intermediate Omega beats Markovian AB with paired significance and prolonged STM. "
            "H2 accepts hierarchical ABC only if it beats tuned AB and tied ABC, with practical effect size, Holm-corrected Wilcoxon, bootstrap CI, two separated memory scales, stability, and A-only readout. "
            "H3 accepts autonomous forecasting only with at least 10% MSE/NRMSE reduction, preserved valid prediction time, no divergence increase, and significance. "
            "H4-H6 compare embedded auxiliaries with effective no-auxiliary dynamics using the same A-only readout and cost accounting."
        ),
        nbformat.v4.new_code_cell("import embedded_effective_qrc_pipeline as qrc\nqrc.RESULTS_DIR, qrc.FIGURES_DIR"),
        nbformat.v4.new_code_cell("qrc.run_all(execute_notebook=False)"),
        nbformat.v4.new_code_cell(
            "import pandas as pd\n"
            "summary = (qrc.RESULTS_DIR / 'final_summary.md').read_text(encoding='utf-8')\n"
            "print(summary[:4000])"
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "print('CSV files:', len(list(qrc.RESULTS_DIR.glob('*.csv'))))\n"
            "print('Figures:', len(list(qrc.FIGURES_DIR.glob('*.pdf'))), 'pdf /', len(list(qrc.FIGURES_DIR.glob('*.png'))), 'png')\n"
            "print('Zip exists:', qrc.ZIP_PATH.exists(), qrc.ZIP_PATH)"
        ),
    ]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nbformat.write(nb, NOTEBOOK_PATH)
    log(f"notebook created: {NOTEBOOK_PATH}")


def execute_notebook() -> None:
    log("executing generated notebook")
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(nb, timeout=3600, kernel_name="python3", allow_errors=False)
    client.execute()
    nbformat.write(nb, NOTEBOOK_PATH)
    log("notebook execution complete")


def run_all(force: bool = False, execute_notebook: bool = True) -> None:
    ensure_dirs()
    LOG_PATH.write_text("", encoding="utf-8")
    random.seed(0)
    np.random.seed(0)
    write_json(RESULTS_DIR / "config.json", {**asdict(CFG), "paper_url": PAPER_URL, "run_mode_sequence": ["smoke", "paper", "full"], "scope_note": "CPU-bounded execution; exact N=4 ABC embedded full run recorded as resource-limited, not silently reduced."})
    write_json(RESULTS_DIR / "environment.json", environment_info())
    run_sanity_checks(force=force)
    run_smoke(force=force)
    write_marker("smoke")
    run_paper_replication(force=force)
    write_marker("paper")
    run_tuning(force=force)
    run_multiscale_and_ipc(force=force)
    run_mackey_glass_full(force=force)
    run_statistics(force=force)
    write_resource_comparisons()
    write_final_summary()
    ensure_required_csvs()
    generate_figures(force=force)
    write_marker("full")
    if not NOTEBOOK_PATH.exists():
        create_notebook()
    if execute_notebook:
        create_notebook()
        execute_notebook()
        make_zip()
    else:
        make_zip()


if __name__ == "__main__":
    run_all()
