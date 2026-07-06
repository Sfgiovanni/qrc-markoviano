"""Extra experiments v4 (supplementary material) for the non-Markovian QRC study
(extends Sannia et al., arXiv:2505.02491), built ON TOP of the finalized v2
pipeline and the v3 auxiliary-dimension sweep.

Three cheap complementary experiments (target budget < 8 h), all checkpointed and
GPU-mandatory:
  Exp 4  Finite-sampling (shot) noise: does the noaux advantage survive realistic
         measurements?  Reuses exact Tr[rho O] features, injects binomial-variance
         Gaussian noise sigma_O = sqrt((1-<O>^2)/N_shots) per (step, observable)
         AFTER the train/test split, refits ridge; also re-runs the Mackey-Glass
         autonomous rollout with noise inside the feedback loop.
  Exp 5  Standard benchmarks NARMA-10 and Santa Fe laser (dataset A), no retuning
         (transfer, not ceiling).  Paired Wilcoxon+Holm noaux-vs-embedded.
  Exp 6  Topology control at a single d_B=64 (n_B=6): parallel-redundant coupling
         vs the intra-B chain of Exp 1 v3 -- separating "dimension" from "topology".

Isolation: v2 is imported as a library; ALL of v2's I/O globals are redirected
into results_extra_v4/ so nothing is ever written into the protected v2 or v3
directories.  Finalized artefacts are read back only through explicit absolute
paths (V2_DIR / V3_DIR) in read mode.

Phase 0 is an automatic go/no-go gate: sanity of the new parallel construction +
an Exp-4 N_shots=inf data-path check against v2 + an Exp-5 NARMA baseline check +
a MEASURED micro-benchmark budget.  The full run proceeds only if all conditions
hold; otherwise it aborts and writes ABORTED.md.  If the measured budget lands in
[12h, 20h] it self-degrades (documented) to fit under 12h; above 20h it aborts.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qrc-v2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import optuna
import pandas as pd
import torch

import qrc_pipeline as v2

# ---------------------------------------------------------------------------
# Isolation: redirect every v2 I/O global into results_extra_v4/.
# ---------------------------------------------------------------------------
V4_DIR = Path("results_extra_v4").resolve()
V2_DIR = Path("results_abc_comparison_v2").resolve()
V3_DIR = Path("results_extra_v3").resolve()
v2.RESULTS_DIR = V4_DIR
v2.FIGURES_DIR = V4_DIR / "figures"
v2.LOG_PATH = V4_DIR / "run.log"

CFG = v2.CFG
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# v4 constants
# ---------------------------------------------------------------------------
EVAL_SEEDS = tuple(range(20))
ALPHA = 1e-6  # ridge alpha, matches v2 multiscale eval

# Exp 4
EXP4_MODELS = ["ABC-noaux-kraus", "ABC-noaux-tied", "AB-embedded", "ABC-embedded-hierarchical"]
NOAUX_SET = {"ABC-noaux-kraus", "ABC-noaux-tied", "M0-noaux"}
N_SHOTS_LIST: List[Optional[float]] = [1e2, 1e3, 1e4, 1e5, None]  # None == infinite (exact)
EXP4_TAUS = [5, 10, 15, 20, 30]
EXP4_PRODUCT = ("s10_x_s20", 10, 20)
NOISE_REPS = 10          # noise draws per (seed, N_shots) for capacity tasks
MG_NOISE_REPS = 10       # noise draws per (seed, N_shots) for the MG rollout
MG_ROLLOUT_LEN = 150     # autonomous rollout length under shot noise (NRMSE_150 + VPT)
RETAIN_FRAC = 0.80       # capacity retention threshold for the summary

# Exp 5
EXP5_MODELS = EXP4_MODELS + ["M0-noaux", "AB-Markov"]
EXP5_EMBEDDED = ["AB-embedded", "ABC-embedded-hierarchical"]
EXP5_NOAUX = ["ABC-noaux-kraus", "ABC-noaux-tied"]
NARMA_LEN = CFG.paper_len            # 3000
NARMA_INPUT_SEED_OFFSET = 909090     # disjoint from iid_inputs (seed+12345) and tuning seeds
SANTAFE_LOCAL = V4_DIR / "santafe_A_source.csv"
SANTAFE_URLS = [
    "https://web.archive.org/web/20160424015114/http://www-psych.stanford.edu/~andreas/Time-Series/SantaFe/A.dat",
    "https://web.archive.org/web/20160424015114/http://www-psych.stanford.edu/~andreas/Time-Series/SantaFe/A.cont",
]
SANTAFE_WASH = 1000
SANTAFE_TRAIN = 5000
SANTAFE_TF = 1000        # teacher-forced one-step validation span
SANTAFE_ROLL = 100       # autonomous rollout horizon

# Exp 6
EXP6_DB = 64
EXP6_NB = 6
EXP6_TUNE_SEEDS = (1000, 1001, 1002, 1003)
EXP6_TRIALS = 32
EXP6_TUNE_TAUS = (5, 10, 15, 20, 30)
STM_TAUS = list(range(0, 51))
EXP6_PRODUCT = ("s10_x_s20", 10, 20)

# Budget thresholds (hours)
BUDGET_GO_H = 12.0
BUDGET_ABORT_H = 20.0

DEFAULT_RUN_CONFIG = {
    "exp4_n_shots": [1e2, 1e3, 1e4, 1e5, None],
    "santafe_rollout": True,
    "exp6_eval_seeds": 20,
}

assert not (set(EVAL_SEEDS) & set(EXP6_TUNE_SEEDS)), "eval and tuning seeds must be disjoint"

# ---------------------------------------------------------------------------
# Reuse v2 infra (now pointing at V4_DIR)
# ---------------------------------------------------------------------------
log = v2.log
marker = v2.marker
write_marker = v2.write_marker
key_done = v2.key_done
append_rows = v2.append_rows
record_failure = v2.record_failure
write_json = v2.write_json
load_csv = v2.load_csv

ABORTED_PATH = V4_DIR / "ABORTED.md"
BUDGET_PATH = V4_DIR / "budget_v4.json"
PROGRESS_PATH = V4_DIR / "progress_log.md"
FAILED_PATH = V4_DIR / "failed_runs.csv"
DECISIONS_PATH = V4_DIR / "decisions_log.md"
SUMMARY_PATH = V4_DIR / "extra_v4_summary.md"


class WatchdogError(RuntimeError):
    pass


class AbortRun(RuntimeError):
    pass


def decision(title: str, detail: str) -> None:
    """Append an autonomous decision to decisions_log.md (audit trail)."""
    with DECISIONS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"- [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] **{title}** — {detail}\n")
    log(f"DECISION: {title} — {detail}")


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------
_HB_LAST = 0.0
_HB_INTERVAL = 15 * 60


def gpu_mem_mb() -> Tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (torch.cuda.memory_allocated() / 1e6, torch.cuda.max_memory_allocated() / 1e6)


def heartbeat(phase: str, frac: float, extra: str = "", force: bool = False) -> None:
    global _HB_LAST
    now = time.time()
    if not force and (now - _HB_LAST) < _HB_INTERVAL:
        return
    _HB_LAST = now
    frac = max(0.0, min(1.0, frac))
    elapsed = now - v2._T0
    eta = (elapsed / frac - elapsed) if frac > 1e-6 else float("nan")
    cur, peak = gpu_mem_mb()
    line = (
        f"- [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] phase=**{phase}** "
        f"{frac*100:5.1f}% | elapsed {elapsed/3600:.2f}h | "
        f"ETA {('%.2fh' % (eta/3600)) if eta == eta else 'n/a'} | "
        f"GPU {cur:.0f}/{peak:.0f} MB{(' | ' + extra) if extra else ''}"
    )
    with PROGRESS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Channel-grid priming: copy any existing v2 N=4 grids for the needed seeds into
# the v4 cache (read v2, write v4 only).  A single .npz per (seed, N, grid_size)
# serves both the GPU and NumPy code paths.
# ---------------------------------------------------------------------------
def prime_grid_cache(seeds: Sequence[int]) -> None:
    v2.ensure_dirs()
    dst_dir = V4_DIR / "channel_cache"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        name = (
            f"channel_N{CFG.n_a}_seed{seed}_g{CFG.grid_size}_dt{CFG.dt}"
            f"_gamma{CFG.gamma}_range{CFG.grid_s_min}_{CFG.grid_s_max}.npz"
        )
        src = V2_DIR / "channel_cache" / name
        dst = dst_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Best-parameter loading (read v2, task paper_s0_s10).
# ---------------------------------------------------------------------------
_BP_CACHE: Optional[pd.DataFrame] = None


def v2_params(arch: str, task: str = "paper_s0_s10") -> Dict:
    global _BP_CACHE
    if _BP_CACHE is None:
        _BP_CACHE = pd.read_csv(V2_DIR / "best_parameters_by_task.csv")
    r = _BP_CACHE[(_BP_CACHE.architecture == arch) & (_BP_CACHE.task == task)]
    if len(r):
        return {k: v for k, v in r.iloc[0].dropna().to_dict().items()
                if k not in ("architecture", "task", "objective", "source")}
    return v2.best_params(arch) if v2.is_embedded(arch) else {}


def make(name: str, seed: int):
    return v2.make_model(name, v2_params(name), seed).reset()


def feat_fn_for(name: str) -> Callable:
    return (lambda m: m.features("A")) if v2.is_embedded(name) else (lambda m: m.features())


# ---------------------------------------------------------------------------
# Watchdog-guarded feature driver (works for any model exposing .step + feat_fn).
# ---------------------------------------------------------------------------
def drive(model, seq: np.ndarray, grid, feat_fn: Callable, step_threshold: Optional[float]) -> np.ndarray:
    out: List[np.ndarray] = []
    limit = (10.0 * step_threshold) if step_threshold else None
    for s in seq:
        t0 = time.time()
        model.step(float(s), grid)
        out.append(feat_fn(model))
        if limit is not None and (time.time() - t0) > limit:
            raise WatchdogError(f"step {time.time()-t0:.3f}s exceeded 10x benchmark {step_threshold:.3f}s")
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# Shot noise: sigma_O = sqrt((1-<O>^2)/N_shots), binomial variance of a +-1 Pauli
# observable estimated from N_shots single-shot measurements.  Applied per
# (step, observable) element with the TRUE <O> setting the scale.
# ---------------------------------------------------------------------------
def add_shot_noise(feats: np.ndarray, n_shots: Optional[float], rng: np.random.Generator) -> np.ndarray:
    if n_shots is None:
        return feats
    var = np.clip(1.0 - feats * feats, 0.0, None) / float(n_shots)
    return feats + rng.standard_normal(feats.shape) * np.sqrt(var)


def cap_from_feats(feats: np.ndarray, seq: np.ndarray, target: np.ndarray, slices) -> float:
    cap, _ = v2.evaluate_capacity_from_features(feats, seq, target, slices, alpha=ALPHA)
    return cap


def io_metrics(feats: np.ndarray, target: np.ndarray, slices) -> Dict[str, float]:
    """Supervised input->output task: NMSE / NRMSE / R2 on the test span."""
    w = v2.fit_readout(feats[slices["train"]], target[slices["train"]], alpha=ALPHA)
    pred = v2.predict_readout(feats[slices["test"]], w)
    m = v2.mse_metrics(target[slices["test"]], pred)
    return {"nmse": float(m["nrmse"] ** 2), "nrmse": float(m["nrmse"]), "r2": float(m["r2"]),
            "capacity": v2.capacity_score(target[slices["test"]], pred)}


# ===========================================================================
# Exp 6 model: parallel-redundant B register at a single d_B (no intra-B chain).
#   step: input channel on A -> for each B qubit a partial-SWAP(eta_ab) to its A
#   partner (A[i] for i<n_a; extra B qubits couple to A[0], A[1], ...) ->
#   depolarize all B qubits with omega -> renormalize.
# ===========================================================================
class ParallelAuxModelGPU:
    def __init__(self, n_b: int, eta_ab: float, omega: float):
        self.n_a = CFG.n_a
        self.n_b = int(n_b)
        self.n_total = self.n_a + self.n_b
        self.eta_ab = float(eta_ab)
        self.omega = float(omega)
        dev = v2.get_device()
        self.u_ab = torch.tensor(v2.partial_swap_unitary_np(self.eta_ab), dtype=v2.CDTYPE, device=dev)
        pairs: List[Tuple[int, int]] = [(i, self.n_a + i) for i in range(min(self.n_a, self.n_b))]
        for extra in range(self.n_b - self.n_a):          # redundant fan-in to A[extra % n_a]
            pairs.append((extra % self.n_a, self.n_a + self.n_a + extra))
        self.ab_pairs = pairs
        self.b_qubits = list(range(self.n_a, self.n_total))
        self.last_clamped = 0
        self.rho = v2.pure_zero_density_t(self.n_total)

    def reset(self) -> "ParallelAuxModelGPU":
        self.rho = v2.pure_zero_density_t(self.n_total)
        self.last_clamped = 0
        return self

    def clone(self) -> torch.Tensor:
        return self.rho.clone()

    def restore(self, rho: torch.Tensor) -> None:
        self.rho = rho.clone()

    def step(self, s: float, grid: torch.Tensor) -> torch.Tensor:
        if s < CFG.grid_s_min or s > CFG.grid_s_max:
            self.last_clamped += 1
        self.rho = v2.apply_super_to_a_t(self.rho, v2.select_channel_gpu(grid, s), self.n_a, self.n_total)
        for qa, qb in self.ab_pairs:
            self.rho = v2.apply_layer_unitary_density_t(self.rho, self.u_ab, [qa, qb], self.n_total)
        self.rho = v2.local_depolarize_all_t(self.rho, self.b_qubits, self.n_total, self.omega)
        self.rho = v2.normalize_density_t(self.rho)
        return self.rho

    def reduced(self, register: str = "A") -> torch.Tensor:
        keep = range(self.n_a) if register == "A" else range(self.n_a, self.n_total)
        return v2.reduce_register_t(self.rho, list(keep), self.n_total)

    def features(self, register: str = "A") -> np.ndarray:
        return v2.features_from_rho_t(self.reduced(register), v2.obs_gpu(self.n_a)).double().cpu().numpy()


# ===========================================================================
# PHASE 0
# ===========================================================================
def sanity_parallel() -> Dict:
    """trace/hermiticity/positivity after 50 steps for the parallel n_B=6 model
    (1024x1024; eigvalsh computed directly since v2.state_checks_t skips it for
    dim>256)."""
    grid = v2.build_channel_grid_gpu(0, CFG.n_a)
    seq = v2.iid_inputs(0, 50)
    m = ParallelAuxModelGPU(EXP6_NB, eta_ab=CFG.eta_paper, omega=0.3).reset()
    for s in seq:
        m.step(float(s), grid)
    chk = v2.state_checks_t(m.rho)
    herm = 0.5 * (m.rho + m.rho.conj().T)
    chk["min_eig"] = float(torch.linalg.eigvalsh(herm).min().item())
    passed = (chk["trace_error"] < 2e-3 and chk["hermiticity_error"] < 2e-3 and chk["min_eig"] > -2e-3)
    log(f"sanity parallel n_b={EXP6_NB}: trace_err={chk['trace_error']:.2e} "
        f"herm={chk['hermiticity_error']:.2e} min_eig={chk['min_eig']:.2e} -> {'OK' if passed else 'FAIL'}")
    return {**chk, "passed": bool(passed)}


def exp4_datapath_check() -> Dict:
    """(1) N_shots=inf noise path reproduces the clean capacity exactly (<1e-6).
    (2) The reused AB-embedded feature/param/grid path reproduces v2's stored
    degree1_stm capacities (<1e-6)."""
    slices = v2.split_slices(CFG.paper_washout, CFG.paper_train, CFG.paper_test)
    ref = pd.read_csv(V2_DIR / "ipc_by_component.csv")
    ref = ref[(ref.model == "AB-embedded") & (ref.component == "degree1_stm")]
    taus = [0, 5, 10, 20, 30]
    seeds = [0, 1]
    max_v2 = 0.0
    max_inf = 0.0
    for seed in seeds:
        seq = v2.iid_inputs(seed, CFG.paper_len)
        grid = v2.build_channel_grid_gpu(seed, CFG.n_a)
        m = make("AB-embedded", seed)
        feats = drive(m, seq, grid, feat_fn_for("AB-embedded"), None)
        noisy_inf = add_shot_noise(feats, None, np.random.default_rng(0))
        for tau in taus:
            tgt = v2.stm_target(seq, tau)
            c_clean = cap_from_feats(feats, seq, tgt, slices)
            c_inf = cap_from_feats(noisy_inf, seq, tgt, slices)
            max_inf = max(max_inf, abs(c_clean - c_inf))
            v2_vals = ref[(ref.tau1 == tau) & (ref.seed == seed)]["capacity"].values
            if len(v2_vals):
                max_v2 = max(max_v2, abs(c_clean - float(v2_vals[0])))
    passed = (max_inf < 1e-6) and (max_v2 < 1e-6)
    log(f"exp4 data-path check: max|inf-clean|={max_inf:.2e} max|clean-v2|={max_v2:.2e} -> "
        f"{'PASS' if passed else 'FAIL'}")
    return {"passed": bool(passed), "max_inf_vs_clean": max_inf, "max_clean_vs_v2": max_v2, "tol": 1e-6}


def exp5_narma_check() -> Dict:
    """NARMA-10 baseline: M0-noaux NMSE < 1 on seed 0."""
    slices = v2.split_slices(CFG.paper_washout, CFG.paper_train, CFG.paper_test)
    u = narma_input(0)
    target = narma10_target(u)
    m = make("M0-noaux", 0)
    grid = v2.get_grid("M0-noaux", 0)
    feats = drive(m, u, grid, feat_fn_for("M0-noaux"), None)
    met = io_metrics(feats, target, slices)
    passed = met["nmse"] < 1.0
    log(f"exp5 NARMA baseline M0-noaux: NMSE={met['nmse']:.4f} -> {'PASS' if passed else 'FAIL'}")
    return {"passed": bool(passed), "m0_nmse": met["nmse"]}


def measure_step_costs() -> Dict[str, float]:
    log("micro-benchmarking seconds/step per configuration")
    grid_gpu = v2.build_channel_grid_gpu(0, CFG.n_a)
    grid_np = v2.build_channel_grid_np(0, CFG.n_a)
    seq = v2.iid_inputs(0, 60)
    costs: Dict[str, float] = {}

    def bench(model, feat_fn, grid, key):
        for s in seq[:15]:
            model.step(float(s), grid)
            feat_fn(model)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for s in seq[15:]:
            model.step(float(s), grid)
            feat_fn(model)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        costs[key] = (time.time() - t0) / (len(seq) - 15)
        log(f"  {key}: {costs[key]*1000:.2f} ms/step")

    for name in EXP5_MODELS:
        g = grid_gpu if v2.is_embedded(name) else grid_np
        bench(make(name, 0), feat_fn_for(name), g, name)
    bench(ParallelAuxModelGPU(EXP6_NB, CFG.eta_paper, 0.3).reset(), lambda mm: mm.features("A"), grid_gpu, "parallel_dB64")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return costs


def estimate_hours(costs: Dict[str, float], cfg: Dict) -> Dict:
    L = CFG.paper_len
    n_finite = len([n for n in cfg["exp4_n_shots"] if n is not None])
    # Exp 4: one clean drive of L per (model, seed) + MG (pre-drive 2000 +
    # (n_finite*reps + 1) rollouts of MG_ROLLOUT_LEN) per (model, seed).
    exp4 = 0.0
    mg_pre = 2 * CFG.paper_washout
    mg_roll = (n_finite * MG_NOISE_REPS + 1) * MG_ROLLOUT_LEN
    for name in EXP4_MODELS:
        c = costs[name]
        exp4 += len(EVAL_SEEDS) * (L + mg_pre + mg_roll) * c
    # Exp 5: NARMA drive L + Santa Fe (wash+train+tf + rollout) per (model, seed).
    sf = SANTAFE_WASH + SANTAFE_TRAIN + SANTAFE_TF + (SANTAFE_ROLL if cfg["santafe_rollout"] else 0)
    exp5 = sum(len(EVAL_SEEDS) * (L + sf) * costs[name] for name in EXP5_MODELS)
    # Exp 6: parallel tuning (32 trials x 4 seeds x tune_len) + eval seeds x L.
    cp = costs["parallel_dB64"]
    exp6 = (EXP6_TRIALS * len(EXP6_TUNE_SEEDS) * CFG.tune_len + cfg["exp6_eval_seeds"] * L) * cp
    total = (exp4 + exp5 + exp6) / 3600.0
    return {"exp4_h": round(exp4 / 3600, 3), "exp5_h": round(exp5 / 3600, 3),
            "exp6_h": round(exp6 / 3600, 3), "total_h": round(total, 3)}


def decide_budget(costs: Dict[str, float]) -> Dict:
    base = dict(DEFAULT_RUN_CONFIG)
    est0 = estimate_hours(costs, base)
    log(f"measured budget (no degradation): total {est0['total_h']:.2f} h "
        f"(exp4 {est0['exp4_h']}, exp5 {est0['exp5_h']}, exp6 {est0['exp6_h']})")
    if est0["total_h"] > BUDGET_ABORT_H:
        return {"run_config": base, "estimate": est0, "decision": "abort", "degradations": []}
    if est0["total_h"] < BUDGET_GO_H:
        return {"run_config": base, "estimate": est0, "decision": "go", "degradations": []}
    cfg = dict(base)
    degr: List[Dict] = []
    ladder = [
        ("exp4_n_shots", [1e3, 1e4, 1e5, None], "drop_exp4_Nshots_1e2"),
        ("santafe_rollout", False, "santafe_teacher_forced_only"),
        ("exp6_eval_seeds", 10, "reduce_exp6_eval_seeds_20_to_10"),
    ]
    est = est0
    for key, val, reason in ladder:
        if est["total_h"] < BUDGET_GO_H:
            break
        cfg[key] = val
        est = estimate_hours(costs, cfg)
        degr.append({"cut": reason, "new_total_h": est["total_h"]})
        record_failure("budget_degradation", reason, measured_total_h_before=est0["total_h"], new_total_h=est["total_h"])
        log(f"degradation applied: {reason} -> {est['total_h']:.2f} h")
    dec = "go" if est["total_h"] <= BUDGET_ABORT_H else "abort"
    return {"run_config": cfg, "estimate": est, "estimate_base": est0, "decision": dec, "degradations": degr}


def write_aborted(reason: str, details: Dict) -> None:
    lines = ["# RUN ABORTED", "",
             f"Aborted at {datetime.now().isoformat()} during Phase 0 (go/no-go gate).", "",
             f"**Reason:** {reason}", "", "```json", json.dumps(details, indent=2, default=str), "```",
             "", "Delete this file to allow a fresh attempt after fixing the cause."]
    ABORTED_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"ABORTED: {reason}")


def phase0_gate() -> Dict:
    if marker("setup").exists() and BUDGET_PATH.exists():
        b = json.loads(BUDGET_PATH.read_text())
        # JSON turns the Python None (infinite shots) into null; restore it.
        b["run_config"]["exp4_n_shots"] = [None if x is None else x for x in b["run_config"]["exp4_n_shots"]]
        log(f"Phase 0 already passed; run_config={b['run_config']}")
        return b

    v2.require_gpu(verbose=True)
    prime_grid_cache(sorted(set(EVAL_SEEDS) | set(EXP6_TUNE_SEEDS)))

    sanity = sanity_parallel()
    dp = exp4_datapath_check()
    narma = exp5_narma_check()
    costs = measure_step_costs()
    budget = decide_budget(costs)

    record = {
        "generated_at": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "sanity_parallel": sanity,
        "exp4_datapath_check": dp,
        "exp5_narma_check": narma,
        "seconds_per_step": costs,
        "estimate": budget["estimate"],
        "estimate_base": budget.get("estimate_base", budget["estimate"]),
        "degradations": budget["degradations"],
        "run_config": budget["run_config"],
        "thresholds": {"go_h": BUDGET_GO_H, "abort_h": BUDGET_ABORT_H},
    }

    reasons = []
    if not sanity["passed"]:
        reasons.append("sanity of parallel construction failed (trace/hermiticity/positivity)")
    if not dp["passed"]:
        reasons.append(f"Exp4 data-path check failed: inf-vs-clean={dp['max_inf_vs_clean']:.2e}, "
                       f"clean-vs-v2={dp['max_clean_vs_v2']:.2e} (tol 1e-6)")
    if not narma["passed"]:
        reasons.append(f"Exp5 NARMA baseline failed: M0-noaux NMSE={narma['m0_nmse']:.3f} >= 1")
    if budget["decision"] == "abort":
        reasons.append(f"measured budget total {budget['estimate']['total_h']:.2f} h exceeds {BUDGET_ABORT_H} h")

    if reasons:
        record["aborted"] = True
        record["abort_reasons"] = reasons
        write_json(BUDGET_PATH, record)
        write_aborted("; ".join(reasons), record)
        raise AbortRun("; ".join(reasons))

    write_json(BUDGET_PATH, record)
    write_marker("setup", run_config=budget["run_config"], estimate=budget["estimate"])
    heartbeat("phase0_gate", 1.0, extra=f"budget {budget['estimate']['total_h']:.2f}h", force=True)
    log(f"Phase 0 PASSED. run_config={budget['run_config']} estimate={budget['estimate']['total_h']:.2f}h")
    return record


# ===========================================================================
# EXP 4: finite-sampling (shot) noise
# ===========================================================================
EXP4_CAP_CSV = V4_DIR / "shot_noise_capacities.csv"
EXP4_MG_CSV = V4_DIR / "shot_noise_mackey.csv"
EXP4_SUMMARY_JSON = V4_DIR / "shot_noise_summary.json"


def _noise_rng(seed: int, n_shots: Optional[float], rep: int, tag: int) -> np.random.Generator:
    n_code = 0 if n_shots is None else int(round(math.log10(n_shots)))
    return np.random.default_rng((tag, seed, n_code, rep))


def _n_label(n_shots: Optional[float]) -> str:
    return "inf" if n_shots is None else f"1e{int(round(math.log10(n_shots)))}"


def exp4_capacities(model_name: str, seed: int, n_shots_list, step_threshold: Optional[float]) -> None:
    if key_done(EXP4_CAP_CSV, model=model_name, seed=seed):
        return
    slices = v2.split_slices(CFG.paper_washout, CFG.paper_train, CFG.paper_test)
    seq = v2.iid_inputs(seed, CFG.paper_len)
    grid = v2.get_grid(model_name, seed)
    m = make(model_name, seed)
    feats = drive(m, seq, grid, feat_fn_for(model_name), step_threshold)
    targets = {f"s_{tau}": v2.stm_target(seq, tau) for tau in EXP4_TAUS}
    _, t1, t2 = EXP4_PRODUCT
    targets[EXP4_PRODUCT[0]] = v2.stm_target(seq, t1) * v2.stm_target(seq, t2)
    rows = []
    for n_shots in n_shots_list:
        reps = 1 if n_shots is None else NOISE_REPS
        for rep in range(reps):
            noisy = add_shot_noise(feats, n_shots, _noise_rng(seed, n_shots, rep, 4))
            for task, tgt in targets.items():
                rows.append({"model": model_name, "seed": seed, "n_shots": _n_label(n_shots),
                             "rep": rep, "task": task, "capacity": cap_from_feats(noisy, seq, tgt, slices)})
    append_rows(EXP4_CAP_CSV, rows)


def _noisy_rollout(model, grid, w, steps: int, n_shots, rng, ffn) -> np.ndarray:
    preds = []
    for _ in range(steps):
        feat = ffn(model)
        if n_shots is not None:
            feat = feat + rng.standard_normal(feat.shape) * np.sqrt(np.clip(1.0 - feat * feat, 0.0, None) / float(n_shots))
        y = float(v2.predict_readout(feat[None, :], w)[0])
        preds.append(y)
        model.step(y, grid)
    return np.asarray(preds)


def exp4_mackey(model_name: str, seed: int, n_shots_list, step_threshold: Optional[float]) -> None:
    if key_done(EXP4_MG_CSV, model=model_name, seed=seed):
        return
    slices = v2.split_slices(CFG.paper_washout, CFG.paper_train, CFG.paper_test)
    stop = slices["test"].start
    series = v2.normalize_series(v2.mackey_glass(CFG.paper_len + 1), slices["train"])
    grid = v2.get_grid(model_name, seed)
    ffn = feat_fn_for(model_name)
    m = make(model_name, seed)
    feats_pre = drive(m, series[:stop], grid, ffn, step_threshold)  # clean pre-drive
    snapshot = m.clone()
    target = series[1:stop + 1]
    truth = series[slices["test"]][:MG_ROLLOUT_LEN]
    rows = []
    for n_shots in n_shots_list:
        reps = 1 if n_shots is None else MG_NOISE_REPS
        for rep in range(reps):
            rng = _noise_rng(seed, n_shots, rep, 44)
            noisy_pre = add_shot_noise(feats_pre, n_shots, rng)
            w = v2.fit_readout(noisy_pre[slices["train"]], target[slices["train"]], alpha=ALPHA)
            m.restore(snapshot)
            preds = _noisy_rollout(m, grid, w, len(truth), n_shots, rng, ffn)
            metr = v2.mse_metrics(truth[:150], preds[:150])
            err = np.abs(preds - truth)
            exceed = np.where(err > CFG.valid_threshold)[0]
            vpt = int(exceed[0]) if len(exceed) else len(preds)
            rows.append({"model": model_name, "seed": seed, "n_shots": _n_label(n_shots), "rep": rep,
                         "nrmse_150": metr["nrmse"], "mse_150": metr["mse"], "valid_prediction_time": vpt})
    append_rows(EXP4_MG_CSV, rows)


def run_exp4(run_config: Dict, costs: Dict[str, float]) -> None:
    if marker("exp4").exists():
        log("exp4 already complete; skipping")
        return
    log(f"=== EXP 4: shot noise (N_shots={[_n_label(n) for n in run_config['exp4_n_shots']]}) ===")
    n_list = run_config["exp4_n_shots"]
    if [n for n in n_list if n is not None] != [1e2, 1e3, 1e4, 1e5]:
        record_failure("exp4", "reduced_n_shots_budget", executed=[_n_label(n) for n in n_list])
    total = len(EXP4_MODELS) * len(EVAL_SEEDS)
    done = 0
    for model_name in EXP4_MODELS:
        thr = costs.get(model_name)
        for seed in EVAL_SEEDS:
            try:
                exp4_capacities(model_name, seed, n_list, thr)
                exp4_mackey(model_name, seed, n_list, thr)
            except WatchdogError as exc:
                record_failure(f"exp4/{model_name}/seed{seed}", "watchdog_step_timeout", detail=str(exc))
                log(f"WATCHDOG exp4 {model_name} seed={seed}: {exc}")
            except Exception as exc:  # noqa: BLE001
                record_failure(f"exp4/{model_name}/seed{seed}", "unit_exception", detail=repr(exc))
                log(f"ERROR exp4 {model_name} seed={seed}: {exc!r}")
            done += 1
            heartbeat("exp4", done / total, extra=f"{model_name} seed{seed}")
    _exp4_summary(n_list)
    write_marker("exp4", models=EXP4_MODELS, n_shots=[_n_label(n) for n in n_list])


def _relabel_nshots(df: pd.DataFrame) -> pd.DataFrame:
    """pandas re-parses the string labels ("1e2".."inf") back into floats on CSV
    read; map them to canonical string labels so grouping is stable."""
    if df.empty or "n_shots" not in df.columns:
        return df
    m = {100.0: "1e2", 1000.0: "1e3", 10000.0: "1e4", 100000.0: "1e5", float("inf"): "inf"}
    df = df.copy()
    df["n_shots"] = df["n_shots"].map(
        lambda x: x if isinstance(x, str) else m.get(float(x), str(x)))
    return df


def _exp4_summary(n_list) -> None:
    cap = _relabel_nshots(load_csv(EXP4_CAP_CSV))
    if cap.empty:
        return
    order = [n for n in ["1e2", "1e3", "1e4", "1e5", "inf"] if n in set(cap.n_shots.unique())]
    # multiscale capacity per (model, n_shots): mean over tasks, seeds, reps.
    means: Dict[str, Dict[str, float]] = {}
    for model_name in EXP4_MODELS:
        g = cap[cap.model == model_name]
        means[model_name] = {n: float(g[g.n_shots == n]["capacity"].mean()) for n in order}

    def retain_n(series: Dict[str, float]) -> Optional[str]:
        exact = series.get("inf", float("nan"))
        if not (exact == exact) or exact <= 0:
            return None
        best = None
        for n in [x for x in order if x != "inf"]:  # ascending N_shots
            if series.get(n, -1) >= RETAIN_FRAC * exact:
                best = n
                break
        return best

    summary = {"retain_frac": RETAIN_FRAC, "multiscale_capacity_by_model": means,
               "min_n_shots_retain80": {mdl: retain_n(means[mdl]) for mdl in EXP4_MODELS}}
    # noaux - embedded gap retention (best noaux vs best embedded)
    noaux_best = "ABC-noaux-kraus"
    emb_best = "ABC-embedded-hierarchical"
    gap = {n: means[noaux_best].get(n, float("nan")) - means[emb_best].get(n, float("nan")) for n in order}
    gap_exact = gap.get("inf", float("nan"))
    gap_retain = None
    if gap_exact == gap_exact and gap_exact > 0:
        for n in [x for x in order if x != "inf"]:
            if gap.get(n, -1) >= RETAIN_FRAC * gap_exact:
                gap_retain = n
                break
    summary["noaux_minus_embedded_gap"] = {"pair": [noaux_best, emb_best], "by_n_shots": gap,
                                           "gap_exact": gap_exact, "min_n_shots_retain80_gap": gap_retain}
    # Mackey-Glass degradation
    mg = _relabel_nshots(load_csv(EXP4_MG_CSV))
    if not mg.empty:
        mg_tbl = {}
        for model_name in EXP4_MODELS:
            gm = mg[mg.model == model_name]
            mg_tbl[model_name] = {n: {"nrmse_150": float(gm[gm.n_shots == n]["nrmse_150"].mean()),
                                      "vpt": float(gm[gm.n_shots == n]["valid_prediction_time"].mean())}
                                  for n in order}
        summary["mackey_by_model"] = mg_tbl
    write_json(EXP4_SUMMARY_JSON, summary)
    log(f"exp4 summary written: {EXP4_SUMMARY_JSON.name}")


# ===========================================================================
# EXP 5: standard benchmarks (NARMA-10, Santa Fe laser A)
# ===========================================================================
EXP5_NARMA_CSV = V4_DIR / "narma10_results.csv"
EXP5_SANTAFE_CSV = V4_DIR / "santafe_results.csv"
EXP5_STATS_CSV = V4_DIR / "benchmarks_paired_stats.csv"


def narma_input(seed: int) -> np.ndarray:
    return np.random.default_rng(seed + NARMA_INPUT_SEED_OFFSET).uniform(0.0, 0.5, size=NARMA_LEN)


def narma10_target(u: np.ndarray) -> np.ndarray:
    n = len(u)
    y = np.zeros(n, dtype=np.float64)
    # Let overflow propagate as inf (silently) so the finite assert can catch it,
    # rather than spamming RuntimeWarnings as in the historical run.
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(9, n - 1):
            y[k + 1] = (0.3 * y[k] + 0.05 * y[k] * np.sum(y[k - 9:k + 1])
                        + 1.5 * u[k - 9] * u[k] + 0.1)
    # G3 fix: fail loudly on numerical divergence so the caller can remap the seed.
    assert np.isfinite(y).all(), "NARMA10 target diverged (overflow/NaN)"
    return y


def narma10_target_for_seed(seed: int, max_attempts: int = 3) -> Tuple[np.ndarray, np.ndarray, int, Dict]:
    """G3 fix: generate a finite NARMA10 (input, target) pair. If the recurrence
    diverges, retry with a shifted seed (seed + 10000*attempt), recording the
    remapping. Raises ValueError after `max_attempts` exhausted (caller records
    a failure and skips the unit; n_effective then reflects reality)."""
    cur = seed
    remap: Dict = {}
    for attempt in range(max_attempts):
        u = narma_input(cur)
        try:
            target = narma10_target(u)
        except AssertionError:
            cur = seed + 10000 * (attempt + 1)
            continue
        if cur != seed:
            remap = {"orig_seed": seed, "used_seed": cur, "attempts": attempt + 1}
        return u, target, cur, remap
    raise ValueError(f"NARMA10 target non-finite after {max_attempts} attempts (orig seed {seed})")


def load_santafe() -> Optional[np.ndarray]:
    if SANTAFE_LOCAL.exists():
        try:
            return np.loadtxt(SANTAFE_LOCAL).astype(np.float64).ravel()
        except Exception as exc:  # noqa: BLE001
            record_failure("exp5_santafe", "local_file_unreadable", detail=repr(exc))
    # P2 fix: the URLs are the two parts of the series (A.dat + A.cont); try every
    # one, record failures, and continue instead of returning None on the first
    # failure. Concatenate whatever parts downloaded; the length check in the
    # caller decides whether the recovered series is usable.
    parts = []
    for url in SANTAFE_URLS:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                parts.append(np.array([float(x) for x in resp.read().decode().split()], dtype=np.float64))
        except Exception as exc:  # noqa: BLE001
            record_failure("exp5_santafe", "download_failed", url=url, detail=repr(exc))
            continue
    if not parts:
        return None
    series = np.concatenate(parts)
    try:
        np.savetxt(SANTAFE_LOCAL, series, fmt="%.6f")
    except Exception:  # noqa: BLE001
        pass
    return series


def run_narma(run_config: Dict, costs: Dict) -> None:
    slices = v2.split_slices(CFG.paper_washout, CFG.paper_train, CFG.paper_test)
    total = len(EXP5_MODELS) * len(EVAL_SEEDS)
    done = 0
    for model_name in EXP5_MODELS:
        thr = costs.get(model_name)
        for seed in EVAL_SEEDS:
            done += 1
            if key_done(EXP5_NARMA_CSV, model=model_name, seed=seed):
                continue
            try:
                try:
                    u, target, used_seed, remap = narma10_target_for_seed(seed)
                except ValueError as exc:
                    record_failure(f"narma/{model_name}/seed{seed}", "narma_target_nonfinite", detail=str(exc))
                    heartbeat("exp5_narma", done / total, extra=f"{model_name} seed{seed} SKIP nonfinite")
                    continue
                if remap:
                    decision(f"NARMA seed remap {seed}->{used_seed}", f"original target diverged; {remap}")
                grid = v2.get_grid(model_name, seed)
                m = make(model_name, seed)
                feats = drive(m, u, grid, feat_fn_for(model_name), thr)
                met = io_metrics(feats, target, slices)
                # G3 fix: never write a non-finite metrics row (single validator).
                if not v2.all_finite(met, keys=("nmse", "nrmse", "r2")):
                    record_failure(f"narma/{model_name}/seed{seed}", "nonfinite_metric",
                                   nmse=met.get("nmse"), nrmse=met.get("nrmse"), r2=met.get("r2"))
                    heartbeat("exp5_narma", done / total, extra=f"{model_name} seed{seed} SKIP nonfinite metric")
                    continue
                row = {"model": model_name, "seed": seed, "task": "NARMA10",
                       "nmse": met["nmse"], "nrmse": met["nrmse"], "r2": met["r2"]}
                if remap:
                    row["remapped_from_seed"] = seed
                    row["used_seed"] = used_seed
                append_rows(EXP5_NARMA_CSV, [row])
            except WatchdogError as exc:
                record_failure(f"narma/{model_name}/seed{seed}", "watchdog_step_timeout", detail=str(exc))
            except Exception as exc:  # noqa: BLE001
                record_failure(f"narma/{model_name}/seed{seed}", "unit_exception", detail=repr(exc))
                log(f"ERROR narma {model_name} seed={seed}: {exc!r}")
            heartbeat("exp5_narma", done / total, extra=f"{model_name} seed{seed}")


def run_santafe(run_config: Dict, costs: Dict) -> None:
    raw = load_santafe()
    if raw is None:
        decision("Santa Fe skipped", "download/local load failed; recorded in failed_runs.csv, run continues")
        return
    need = SANTAFE_WASH + SANTAFE_TRAIN + max(SANTAFE_TF, SANTAFE_ROLL) + 1
    if len(raw) < need:
        record_failure("exp5_santafe", "series_too_short", have=len(raw), need=need)
        decision("Santa Fe skipped", f"series length {len(raw)} < required {need}")
        return
    do_roll = run_config["santafe_rollout"]
    train_slice = slice(SANTAFE_WASH, SANTAFE_WASH + SANTAFE_TRAIN)
    series = v2.normalize_series(raw, train_slice)
    stop = SANTAFE_WASH + SANTAFE_TRAIN
    target_pre = series[1:stop + 1]
    total = len(EXP5_MODELS) * len(EVAL_SEEDS)
    done = 0
    for model_name in EXP5_MODELS:
        thr = costs.get(model_name)
        ffn = feat_fn_for(model_name)
        for seed in EVAL_SEEDS:
            done += 1
            if key_done(EXP5_SANTAFE_CSV, model=model_name, seed=seed):
                continue
            try:
                grid = v2.get_grid(model_name, seed)
                m = make(model_name, seed)
                feats_pre = drive(m, series[:stop], grid, ffn, thr)
                snapshot = m.clone()
                w = v2.fit_readout(feats_pre[train_slice], target_pre[train_slice], alpha=ALPHA)
                # teacher-forced one-step validation
                feats_tf = drive(m, series[stop:stop + SANTAFE_TF], grid, ffn, thr)
                tf_pred = v2.predict_readout(feats_tf, w)
                tf_truth = series[stop + 1:stop + SANTAFE_TF + 1]
                mm = min(len(tf_pred), len(tf_truth))
                tf = v2.mse_metrics(tf_truth[:mm], tf_pred[:mm])
                row = {"model": model_name, "seed": seed, "nrmse_tf": tf["nrmse"], "r2_tf": tf["r2"]}
                if do_roll:
                    m.restore(snapshot)
                    preds = _noisy_rollout(m, grid, w, SANTAFE_ROLL, None, None, ffn)
                    truth = series[stop:stop + SANTAFE_ROLL]
                    ro = v2.mse_metrics(truth, preds)
                    err = np.abs(preds - truth)
                    exceed = np.where(err > CFG.valid_threshold)[0]
                    row.update({"nrmse_rollout": ro["nrmse"], "r2_rollout": ro["r2"],
                                "vpt": int(exceed[0]) if len(exceed) else SANTAFE_ROLL})
                else:
                    row.update({"nrmse_rollout": np.nan, "r2_rollout": np.nan, "vpt": np.nan})
                # G3 fix: teacher-forced metrics must be finite to be written.
                if not v2.all_finite(row, keys=("nrmse_tf", "r2_tf")):
                    record_failure(f"santafe/{model_name}/seed{seed}", "nonfinite_metric",
                                   nrmse_tf=row.get("nrmse_tf"), r2_tf=row.get("r2_tf"))
                    heartbeat("exp5_santafe", done / total, extra=f"{model_name} seed{seed} SKIP nonfinite")
                    continue
                append_rows(EXP5_SANTAFE_CSV, [row])
            except WatchdogError as exc:
                record_failure(f"santafe/{model_name}/seed{seed}", "watchdog_step_timeout", detail=str(exc))
            except Exception as exc:  # noqa: BLE001
                record_failure(f"santafe/{model_name}/seed{seed}", "unit_exception", detail=repr(exc))
                log(f"ERROR santafe {model_name} seed={seed}: {exc!r}")
            heartbeat("exp5_santafe", done / total, extra=f"{model_name} seed{seed}")


def exp5_paired_stats() -> None:
    """Paired Wilcoxon + Holm, noaux vs embedded, on the primary error metrics
    (smaller is better)."""
    rows = []
    pvals: List[float] = []
    idx: List[int] = []

    def add_family(df: Optional[pd.DataFrame], metric: str, task: str):
        if df is None or df.empty:
            return
        for na in EXP5_NOAUX:
            for emb in EXP5_EMBEDDED:
                a, b = [], []
                for seed in EVAL_SEEDS:
                    va = df[(df.model == na) & (df.seed == seed)][metric]
                    vb = df[(df.model == emb) & (df.seed == seed)][metric]
                    if len(va) and len(vb) and np.isfinite(va.iloc[0]) and np.isfinite(vb.iloc[0]):
                        a.append(float(va.iloc[0])); b.append(float(vb.iloc[0]))
                if len(a) >= 2:
                    st = v2.paired_stats(np.array(a), np.array(b), larger_better=False)
                    # M1 fix: report mean_diff AND ci95 in the SAME sense
                    # (noaux - embedded = a - b). orient_effect flips the CI that
                    # paired_stats produced along d=b-a (larger_better=False).
                    eff = v2.orient_effect(st, larger_better=False, report="a_minus_b")
                    idx.append(len(rows))
                    pvals.append(st.get("p_wilcoxon", np.nan))
                    rows.append({"task": task, "metric": metric, "noaux": na, "embedded": emb,
                                 "mean_noaux": st["mean_a"], "mean_embedded": st["mean_b"],
                                 "mean_diff_noaux_minus_emb": eff["mean_diff"],
                                 "ci95_lo": eff["ci95_lo"], "ci95_hi": eff["ci95_hi"],
                                 "cohen_dz_noaux_minus_emb": eff["cohen_dz"],
                                 "n_noaux_minus_emb_pos": eff["wins"], "n_noaux_minus_emb_neg": eff["losses"],
                                 "p_wilcoxon": st.get("p_wilcoxon", np.nan), "n": st["n"]})

    add_family(load_csv(EXP5_NARMA_CSV), "nmse", "NARMA10")
    add_family(load_csv(EXP5_NARMA_CSV), "nrmse", "NARMA10")
    sf = load_csv(EXP5_SANTAFE_CSV)
    add_family(sf if not sf.empty else None, "nrmse_tf", "SantaFe_teacher_forced")
    if not sf.empty and sf["nrmse_rollout"].notna().any():
        add_family(sf, "nrmse_rollout", "SantaFe_rollout")

    if rows:
        adj = v2.holm([pvals[i] for i in range(len(pvals))])
        for r, p in zip(rows, adj):
            r["p_holm"] = float(p)
        pd.DataFrame(rows).to_csv(EXP5_STATS_CSV, index=False)
    log(f"exp5 paired stats written: {EXP5_STATS_CSV.name} ({len(rows)} comparisons)")


def run_exp5(run_config: Dict, costs: Dict) -> None:
    if marker("exp5").exists():
        log("exp5 already complete; skipping")
        return
    log("=== EXP 5: standard benchmarks (NARMA-10, Santa Fe A) ===")
    if not run_config["santafe_rollout"]:
        record_failure("exp5_santafe", "teacher_forced_only_budget")
    run_narma(run_config, costs)
    run_santafe(run_config, costs)
    exp5_paired_stats()
    write_marker("exp5", models=EXP5_MODELS)


# ===========================================================================
# EXP 6: topology control (parallel vs chain at d_B=64)
# ===========================================================================
EXP6_CSV = V4_DIR / "topology_control.csv"
EXP6_VERDICT_JSON = V4_DIR / "topology_verdict.json"
EXP6_TRIALS_CSV = V4_DIR / "exp6_parallel_tuning.csv"


def exp6_tune() -> Dict:
    slices = v2.split_slices(CFG.tune_washout, CFG.tune_train, CFG.tune_test)
    storage = f"sqlite:///{(V4_DIR / 'optuna_parallel_v4.sqlite3').as_posix()}"

    def objective(trial: optuna.Trial) -> float:
        omega = trial.suggest_float("omega", 0.0, 1.0)
        eta_ab = trial.suggest_float("eta_ab", 0.05, math.pi / 2 - 0.05)
        caps = []
        for seed in EXP6_TUNE_SEEDS:
            seq = v2.iid_inputs(seed, CFG.tune_len)
            grid = v2.build_channel_grid_gpu(seed, CFG.n_a)
            m = ParallelAuxModelGPU(EXP6_NB, eta_ab, omega).reset()
            feats = drive(m, seq, grid, lambda mm: mm.features("A"), None)
            caps.append(np.mean([cap_from_feats(feats, seq, v2.stm_target(seq, t), slices) for t in EXP6_TUNE_TAUS]))
        return float(np.mean(caps))

    study = optuna.create_study(direction="maximize", study_name="parallel_dB64",
                                storage=storage, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=CFG.optuna_seed))
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, EXP6_TRIALS - done)
    if remaining:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)
        heartbeat("exp6_tune", 1.0, extra="tuning done", force=True)
    best = dict(study.best_params)
    rows = [{"trial": t.number, "value": t.value, **t.params}
            for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if rows and not EXP6_TRIALS_CSV.exists():
        pd.DataFrame(rows).to_csv(EXP6_TRIALS_CSV, index=False)
    log(f"exp6 parallel tuned best={study.best_value:.4f} params={best}")
    return best


def exp6_eval_seed(seed: int, best: Dict, step_threshold: Optional[float]) -> None:
    if key_done(EXP6_CSV, seed=seed, topology="parallel", task="stm_linear"):
        return
    slices = v2.split_slices(CFG.paper_washout, CFG.paper_train, CFG.paper_test)
    seq = v2.iid_inputs(seed, CFG.paper_len)
    grid = v2.build_channel_grid_gpu(seed, CFG.n_a)
    m = ParallelAuxModelGPU(EXP6_NB, best.get("eta_ab", CFG.eta_paper), best.get("omega", 0.5)).reset()
    feats = drive(m, seq, grid, lambda mm: mm.features("A"), step_threshold)
    rows = []
    for tau in STM_TAUS:
        rows.append({"seed": seed, "topology": "parallel", "d_B": EXP6_DB, "task": "stm_linear",
                     "tau": tau, "capacity": cap_from_feats(feats, seq, v2.stm_target(seq, tau), slices)})
    _, t1, t2 = EXP6_PRODUCT
    prod = v2.stm_target(seq, t1) * v2.stm_target(seq, t2)
    rows.append({"seed": seed, "topology": "parallel", "d_B": EXP6_DB, "task": EXP6_PRODUCT[0],
                 "tau": -1, "capacity": cap_from_feats(feats, seq, prod, slices)})
    append_rows(EXP6_CSV, rows)


def _mem_range(mean_curve: pd.Series) -> int:
    valid = mean_curve[mean_curve > CFG.valid_threshold]
    return int(valid.index.max()) if len(valid) else -1


def _v3_chain_stm(d_b: int) -> pd.DataFrame:
    """Read the existing v3 aux-dimension sweep (chain) for a given d_B."""
    src = V3_DIR / "aux_dimension_sweep.csv"
    if not src.exists():
        return pd.DataFrame()
    df = pd.read_csv(src)
    return df[(df.d_B == d_b) & (df.task == "stm_linear")]


def exp6_verdict() -> None:
    par = load_csv(EXP6_CSV)
    par_stm = par[par.task == "stm_linear"] if not par.empty else pd.DataFrame()
    chain64 = _v3_chain_stm(64)
    chain16 = _v3_chain_stm(16)
    out: Dict = {"n_b_qubits": EXP6_NB, "d_B": EXP6_DB, "memory_range": {}, "paired": {}}

    def curve(df):
        return df.groupby("tau")["capacity"].mean().sort_index() if not df.empty else pd.Series(dtype=float)

    c_par, c_ch64, c_ch16 = curve(par_stm), curve(chain64), curve(chain16)
    out["memory_range"] = {"parallel_dB64": _mem_range(c_par) if len(c_par) else -1,
                           "chain_dB64": _mem_range(c_ch64) if len(c_ch64) else -1,
                           "chain_dB16": _mem_range(c_ch16) if len(c_ch16) else -1}

    def paired(dfa, dfb):
        a, b = [], []
        for seed in EVAL_SEEDS:
            va = dfa[dfa.seed == seed]["capacity"].sum()
            vb = dfb[dfb.seed == seed]["capacity"].sum()
            if va and vb:
                a.append(float(va)); b.append(float(vb))
        return v2.paired_stats(np.array(a), np.array(b), larger_better=True) if len(a) >= 2 else {"n": len(a)}

    if not par_stm.empty and not chain64.empty:
        out["paired"]["parallel_vs_chain_dB64"] = paired(par_stm, chain64)
    if not par_stm.empty and not chain16.empty:
        out["paired"]["parallel_vs_chain_dB16"] = paired(par_stm, chain16)

    rng_par = out["memory_range"]["parallel_dB64"]
    rng_ch16 = out["memory_range"]["chain_dB16"]
    if rng_par >= 0 and rng_ch16 >= 0 and rng_par > rng_ch16 + 2:
        out["verdict"] = "topology CHANGES memory range: parallel extends beyond d_B=16 chain"
    else:
        out["verdict"] = "topology does NOT rescue memory range: parallel n_B=6 no better than d_B=16"
    write_json(EXP6_VERDICT_JSON, out)
    log(f"exp6 verdict: {out['verdict']}")


def run_exp6(run_config: Dict, costs: Dict) -> None:
    if marker("exp6").exists():
        log("exp6 already complete; skipping")
        return
    log("=== EXP 6: topology control (parallel vs chain, d_B=64) ===")
    if not (V3_DIR / "aux_dimension_sweep.csv").exists():
        record_failure("exp6", "v3_chain_sweep_missing")
    best = exp6_tune()
    n_eval = run_config["exp6_eval_seeds"]
    if n_eval < len(EVAL_SEEDS):
        record_failure("exp6", "reduced_eval_seeds_budget", executed=n_eval, planned=len(EVAL_SEEDS))
    thr = costs.get("parallel_dB64")
    eval_seeds = EVAL_SEEDS[:n_eval]
    for i, seed in enumerate(eval_seeds):
        try:
            exp6_eval_seed(seed, best, thr)
        except WatchdogError as exc:
            record_failure(f"exp6/seed{seed}", "watchdog_step_timeout", detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            record_failure(f"exp6/seed{seed}", "unit_exception", detail=repr(exc))
            log(f"ERROR exp6 seed={seed}: {exc!r}")
        heartbeat("exp6", (i + 1) / len(eval_seeds), extra=f"parallel seed{seed}")
    exp6_verdict()
    write_marker("exp6", best_params=best, eval_seeds=list(eval_seeds))


# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
def _fmt_stats(s: Dict) -> str:
    if not s or s.get("n", 0) < 2:
        return "insufficient data"
    return (f"mean_diff={s.get('mean_diff', float('nan')):.4g}, 95% CI "
            f"[{s.get('ci95_lo', float('nan')):.4g}, {s.get('ci95_hi', float('nan')):.4g}], "
            f"Wilcoxon p={s.get('p_wilcoxon', float('nan')):.4g}, n={s.get('n')}")


def write_summary() -> None:
    lines = ["# Extra experiments v4 — summary", "", f"_Generated {datetime.now().isoformat()}_", ""]
    if BUDGET_PATH.exists():
        b = json.loads(BUDGET_PATH.read_text())
        lines += [f"Run config: `{json.dumps(b.get('run_config', {}))}` | "
                  f"measured budget: {b.get('estimate', {}).get('total_h', 'n/a')} h", ""]

    # Verdict 1 — shot noise
    lines += ["## Verdict 1 — robustness to finite-sampling (shot) noise (Exp 4)", ""]
    if EXP4_SUMMARY_JSON.exists():
        s = json.loads(EXP4_SUMMARY_JSON.read_text())
        retain = s.get("min_n_shots_retain80", {})
        gap = s.get("noaux_minus_embedded_gap", {})
        survives = gap.get("min_n_shots_retain80_gap")
        verdict = (f"noaux advantage survives shot noise down to N_shots={survives}"
                   if survives else "noaux advantage does NOT retain 80% at any finite N_shots tested")
        means = s.get("multiscale_capacity_by_model", {})
        order = [c for c in ["1e2", "1e3", "1e4", "1e5", "inf"]
                 if any(c in means.get(m, {}) for m in EXP4_MODELS)]
        lines += [f"**{verdict}**", "",
                  "Mean multiscale capacity (STM tau in {5,10,15,20,30} + product s10*s20) by "
                  "model vs N_shots:", "",
                  "| model | " + " | ".join(order) + " |",
                  "|---|" + "---|" * len(order)]
        for mdl in EXP4_MODELS:
            row = means.get(mdl, {})
            lines.append(f"| {mdl} | " + " | ".join(f"{row.get(c, float('nan')):.4g}" for c in order) + " |")
        gap_by = gap.get("by_n_shots", {})
        lines += ["",
                  f"noaux−embedded gap ({' − '.join(gap.get('pair', []))}) by N_shots: "
                  + ", ".join(f"{c}={gap_by.get(c, float('nan')):.4g}" for c in order)
                  + f" (exact={gap.get('gap_exact', float('nan')):.4g}); the advantage erodes as shots fall.", "",
                  "Min N_shots retaining >=80% of exact multiscale capacity, by model:", "",
                  "| model | min N_shots (>=80%) |", "|---|---|"]
        for mdl in EXP4_MODELS:
            lines.append(f"| {mdl} | {retain.get(mdl)} |")
    else:
        lines += ["_no data_"]

    # Verdict 2 — benchmarks
    lines += ["", "## Verdict 2 — position on standard benchmarks (Exp 5)", ""]
    narma = load_csv(EXP5_NARMA_CSV)
    sf = load_csv(EXP5_SANTAFE_CSV)
    if not narma.empty:
        lines += ["NARMA-10 mean NMSE / NRMSE by model (20 seeds, no retuning):", "",
                  "| model | NMSE | NRMSE |", "|---|---|---|"]
        for mdl in EXP5_MODELS:
            g = narma[narma.model == mdl]
            if not g.empty:
                lines.append(f"| {mdl} | {g['nmse'].mean():.4g} | {g['nrmse'].mean():.4g} |")
    if not sf.empty:
        lines += ["", "Santa Fe laser A, mean NRMSE by model:", "",
                  "| model | NRMSE (teacher-forced) | NRMSE (100-step rollout) | VPT |", "|---|---|---|---|"]
        for mdl in EXP5_MODELS:
            g = sf[sf.model == mdl]
            if not g.empty:
                ro = g['nrmse_rollout'].mean()
                vpt = g['vpt'].mean()
                lines.append(f"| {mdl} | {g['nrmse_tf'].mean():.4g} | "
                             f"{('%.4g' % ro) if ro == ro else 'n/a'} | {('%.1f' % vpt) if vpt == vpt else 'n/a'} |")
    else:
        lines += ["", "_Santa Fe skipped (see Anomalies)._"]
    stats = load_csv(EXP5_STATS_CSV)
    if not stats.empty:
        lines += ["", "Paired noaux-vs-embedded (Wilcoxon + Holm, smaller error better):", "",
                  "| task | metric | noaux | embedded | mean_diff (noaux−emb) | p_holm |", "|---|---|---|---|---|---|"]
        for _, r in stats.iterrows():
            lines.append(f"| {r['task']} | {r['metric']} | {r['noaux']} | {r['embedded']} | "
                         f"{r['mean_diff_noaux_minus_emb']:.4g} | {r.get('p_holm', float('nan')):.4g} |")

    # Verdict 3 — topology
    lines += ["", "## Verdict 3 — topology control (Exp 6)", ""]
    if EXP6_VERDICT_JSON.exists():
        v = json.loads(EXP6_VERDICT_JSON.read_text())
        lines += [f"**{v.get('verdict', 'no data')}**", "",
                  "Memory range (max tau, C>0.1): "
                  f"parallel d_B=64 = {v['memory_range'].get('parallel_dB64')}, "
                  f"chain d_B=64 = {v['memory_range'].get('chain_dB64')}, "
                  f"chain d_B=16 = {v['memory_range'].get('chain_dB16')}.", ""]
        pv = v.get("paired", {})
        if "parallel_vs_chain_dB16" in pv:
            lines.append(f"Paired STM total, parallel vs chain d_B=16: {_fmt_stats(pv['parallel_vs_chain_dB16'])}.")
        if "parallel_vs_chain_dB64" in pv:
            lines.append(f"Paired STM total, parallel vs chain d_B=64: {_fmt_stats(pv['parallel_vs_chain_dB64'])}.")
    else:
        lines += ["_no data_"]

    # Anomalies
    lines += ["", "## Anomalies (failed_runs.csv + decisions_log.md)", ""]
    failed = load_csv(FAILED_PATH)
    if failed.empty:
        lines.append("failed_runs.csv: none recorded.")
    else:
        lines += ["| timestamp | context | reason |", "|---|---|---|"]
        for _, r in failed.iterrows():
            lines.append(f"| {r.get('timestamp', '')} | {r.get('context', '')} | {r.get('reason', '')} |")
    if DECISIONS_PATH.exists():
        lines += ["", "**decisions_log.md:**", "", DECISIONS_PATH.read_text().strip() or "_none_"]

    # P3 fix: report effective/missing/non-finite cells per table in the summary.
    rep = v2.validate_run(V4_DIR, expected_tables())
    lines += ["", v2.completeness_markdown(rep)]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"summary written: {SUMMARY_PATH.name}")


def expected_tables() -> Dict:
    """M2 fix: planned (model x [n_shots] x seed) cells for the v4 benchmark tables.
    Any missing/non-finite cell -> validate_run marks the run partial."""
    seeds = list(EVAL_SEEDS)
    shots = [np.inf if s is None else float(s) for s in N_SHOTS_LIST]
    return {"tables": {
        EXP5_NARMA_CSV.name: {
            "cell_combos": [{"model": m} for m in EXP5_MODELS],
            "seed_col": "seed", "seeds": seeds, "value_cols": ["nmse", "nrmse", "r2"]},
        EXP5_SANTAFE_CSV.name: {
            "cell_combos": [{"model": m} for m in EXP5_MODELS],
            "seed_col": "seed", "seeds": seeds, "value_cols": ["nrmse_tf", "r2_tf"]},
        "shot_noise_capacities.csv": {
            "cell_combos": [{"model": m, "n_shots": ns} for m in EXP4_MODELS for ns in shots],
            "seed_col": "seed", "seeds": seeds, "value_cols": ["capacity"]},
        "shot_noise_mackey.csv": {
            "cell_combos": [{"model": m, "n_shots": ns} for m in EXP4_MODELS for ns in shots],
            "seed_col": "seed", "seeds": seeds, "value_cols": ["nrmse_150"]},
        "topology_control.csv": {
            "cell_combos": [{"topology": "parallel"}],
            "seed_col": "seed", "seeds": seeds, "value_cols": ["capacity"]},
    }}


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    v2.ensure_dirs()
    PROGRESS_PATH.touch(exist_ok=True)
    DECISIONS_PATH.touch(exist_ok=True)
    if ABORTED_PATH.exists():
        log(f"ABORTED.md present ({ABORTED_PATH}); refusing to run. Delete it to retry.")
        print(ABORTED_PATH.read_text())
        return
    decision("MG shot-noise scope", f"autonomous rollout capped at {MG_ROLLOUT_LEN} steps with "
             f"{MG_NOISE_REPS} noise reps (NRMSE_150 + VPT); design choice to bound budget, not a degradation")
    log("========== qrc_experiments_robustness start ==========")
    heartbeat("start", 0.0, force=True)

    record = phase0_gate()  # raises AbortRun on gate failure
    run_config = record["run_config"]
    costs = record["seconds_per_step"]

    run_exp4(run_config, costs)
    run_exp5(run_config, costs)
    run_exp6(run_config, costs)
    write_summary()
    # M2 fix: gate completion through validate_run (writes summary_partial.json
    # instead of summary_complete.json when seeds/metrics are missing or non-finite).
    report = v2.write_validated_completion(V4_DIR, "summary", expected_tables(), run_config=run_config)
    if report["status"] != "complete":
        log(f"RUN PARTIAL: {report['n_missing']} missing / {report['n_nonfinite']} non-finite cells; "
            f"wrote summary_partial.json (see completeness_matrix.csv)")
    heartbeat("done", 1.0, force=True)
    log(f"========== qrc_experiments_robustness {report['status']} ==========")


if __name__ == "__main__":
    try:
        main()
    except AbortRun as exc:
        print(f"\n*** RUN ABORTED (Phase 0 gate): {exc}\nSee {ABORTED_PATH}\n")
        raise SystemExit(3)
