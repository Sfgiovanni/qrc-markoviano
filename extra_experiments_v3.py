"""Extra experiments v3 for the non-Markovian QRC study (extends Sannia et al.,
arXiv:2505.02491), built ON TOP of the finalized v2 pipeline.

Three complementary experiments, all checkpointed and GPU-mandatory:
  Exp 1  Auxiliary-dimension sweep (memory range vs d_B) -- AB-embedded with a
         variable-size B register (A[i]<->B[i] partial-SWAPs plus an intra-B
         nearest-neighbour partial-SWAP chain when n_B>4).
  Exp 2  Budget-matched retuning of ABC-embedded-hierarchical (anti-subtuning).
  Exp 3  Readout localized to A / B / C (+full) to separate a STORAGE failure
         from a BACKFLOW failure.

Isolation: v2 is imported as a library; ALL of v2's I/O globals are redirected
into results_extra_v3/ so nothing is ever written into results_abc_comparison_v2/.
The finalized v2 artefacts are read back only through explicit absolute paths in
read mode.

Phase 0 is an automatic go/no-go gate: consistency gate + sanity of the new
construction + a MEASURED micro-benchmark budget. The full run proceeds only if
all conditions hold; otherwise it aborts and writes ABORTED.md. If the measured
budget lands in [12h, 20h] it self-degrades (documented) to fit under 12h; above
20h it aborts.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from datetime import datetime, timedelta
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

import embedded_effective_qrc_pipeline_v2 as v2

# ---------------------------------------------------------------------------
# Isolation: redirect every v2 I/O global into results_extra_v3/. v2's functions
# reference these names as module globals at call time, so this reroutes grid
# caches, logs, markers, appended CSVs and recorded failures away from the
# protected v2 directory. v2 artefacts are read only via V2_DIR (absolute).
# ---------------------------------------------------------------------------
V3_DIR = Path("results_extra_v3").resolve()
V2_DIR = Path("results_abc_comparison_v2").resolve()
v2.RESULTS_DIR = V3_DIR
v2.FIGURES_DIR = V3_DIR / "figures"
v2.LOG_PATH = V3_DIR / "run.log"

CFG = v2.CFG
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# v3 constants
# ---------------------------------------------------------------------------
DB_LIST = [2, 4, 8, 16, 32, 64]
EXP1_TUNE_SEEDS = (1000, 1001, 1002, 1003)
EXP1_TUNE_TAUS = (5, 10, 15, 20, 30)          # objective: mean linear STM over these
EXP1_DELAY_TAUS = (5, 10, 15, 20, 30, 40)     # reported pure-delay tasks
EXP1_PRODUCT = ("s10_x_s20", 10, 20)          # product task s_{k-10}*s_{k-20}
STM_TAUS = list(range(0, 51))
GATE_TAUS = [0, 5, 10, 20, 30, 40, 50]        # taus stored by v2 multiscale (degree1_stm)
GATE_SEEDS = (0, 1, 2, 3)
GATE_TOL = 1e-3
EVAL_SEEDS = tuple(range(20))
EXP2_TUNE_SEEDS = tuple(range(1000, 1008))    # 8 tuning seeds, disjoint from EVAL_SEEDS
EXP2_TRIALS = 64
EXP2_TUNE_TASK = "paper_s0_s10"               # task whose best-params drive v2 eval
EXP3_TAUS = list(range(0, 51))

# Budget thresholds (hours)
BUDGET_GO_H = 12.0
BUDGET_ABORT_H = 20.0

ALPHA = 1e-6                                    # ridge alpha, matches v2 multiscale eval

# Default (undegraded) run configuration; may be tightened by Phase 0.
DEFAULT_RUN_CONFIG = {
    "exp1_trials": 32,
    "dB64_eval_seeds": 20,
    "exp3_readouts": ["A_only", "B_only", "C_only", "full"],
}

# ---------------------------------------------------------------------------
# Small helpers (reuse v2 infra, which now points at V3_DIR)
# ---------------------------------------------------------------------------
log = v2.log
marker = v2.marker
write_marker = v2.write_marker
key_done = v2.key_done
append_rows = v2.append_rows
record_failure = v2.record_failure
write_json = v2.write_json
load_csv = v2.load_csv

ABORTED_PATH = V3_DIR / "ABORTED.md"
BUDGET_PATH = V3_DIR / "budget_v3.json"
PROGRESS_PATH = V3_DIR / "progress_log.md"
FAILED_PATH = V3_DIR / "failed_runs.csv"
SUMMARY_PATH = V3_DIR / "extra_experiments_summary.md"


class WatchdogError(RuntimeError):
    pass


class AbortRun(RuntimeError):
    pass


def nb_of(d_b: int) -> int:
    return int(round(math.log2(d_b)))


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
# Channel grid: read v2 cache (copy) if present, else build into V3 cache.
# Never writes into V2_DIR.
# ---------------------------------------------------------------------------
def prime_grid_cache(seeds: Sequence[int]) -> None:
    """Copy any existing v2 N=4 channel grids for `seeds` into the v3 cache so
    the reused v2 grid builders find them locally (read v2, write v3 only)."""
    v2.ensure_dirs()
    dst_dir = V3_DIR / "channel_cache"
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
# Exp 1 model: AB-embedded with a variable-size B register.
#   step: input channel on A -> A[i]<->B[i] partial-SWAP(eta_ab) for
#   i<min(n_a,n_b) -> intra-B nearest-neighbour partial-SWAP(eta_bb) chain when
#   n_b>n_a -> depolarize all B qubits with omega -> renormalize.
# At n_b==n_a and eta_bb==0 this reduces EXACTLY to v2 AB-embedded.
# ---------------------------------------------------------------------------
class AuxSweepModelGPU:
    def __init__(self, n_b: int, eta_ab: float, eta_bb: float, omega: float):
        self.n_a = CFG.n_a
        self.n_b = int(n_b)
        self.n_total = self.n_a + self.n_b
        self.eta_ab = float(eta_ab)
        self.eta_bb = float(eta_bb)
        self.omega = float(omega)
        dev = v2.get_device()
        self.u_ab = torch.tensor(v2.partial_swap_unitary_np(self.eta_ab), dtype=v2.CDTYPE, device=dev)
        self.u_bb = torch.tensor(v2.partial_swap_unitary_np(self.eta_bb), dtype=v2.CDTYPE, device=dev)
        self.ab_pairs = [(i, self.n_a + i) for i in range(min(self.n_a, self.n_b))]
        self.bb_pairs: List[Tuple[int, int]] = []
        if self.n_b > self.n_a:
            b0 = self.n_a
            self.bb_pairs = [(b0 + j, b0 + j + 1) for j in range(self.n_b - 1)]
        self.b_qubits = list(range(self.n_a, self.n_total))
        self.last_clamped = 0
        self.rho = v2.pure_zero_density_t(self.n_total)

    def reset(self) -> "AuxSweepModelGPU":
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
        for qa, qb in self.bb_pairs:
            self.rho = v2.apply_layer_unitary_density_t(self.rho, self.u_bb, [qa, qb], self.n_total)
        self.rho = v2.local_depolarize_all_t(self.rho, self.b_qubits, self.n_total, self.omega)
        self.rho = v2.normalize_density_t(self.rho)
        return self.rho

    def reduced(self, register: str = "A") -> torch.Tensor:
        keep = range(self.n_a) if register == "A" else range(self.n_a, self.n_total)
        return v2.reduce_register_t(self.rho, list(keep), self.n_total)

    def features_t(self, register: str = "A") -> torch.Tensor:
        return v2.features_from_rho_t(self.reduced(register), v2.obs_gpu(self.n_a))

    def features(self, register: str = "A") -> np.ndarray:
        return self.features_t(register).double().cpu().numpy()


# ---------------------------------------------------------------------------
# Efficient full 1&2-body Pauli features (no dense n_total observables).
# ---------------------------------------------------------------------------
_SINGLE_OPS_T: Optional[torch.Tensor] = None
_TWO_OPS_T: Optional[torch.Tensor] = None


def _pauli_op_tensors():
    global _SINGLE_OPS_T, _TWO_OPS_T
    if _SINGLE_OPS_T is None:
        dev = v2.get_device()
        singles = np.stack([v2.PAULI["x"], v2.PAULI["y"], v2.PAULI["z"]])
        two = [np.kron(v2.PAULI[a], v2.PAULI[b]) for a in "xyz" for b in "xyz"]
        _SINGLE_OPS_T = torch.tensor(singles, dtype=v2.CDTYPE, device=dev)
        _TWO_OPS_T = torch.tensor(np.stack(two), dtype=v2.CDTYPE, device=dev)
    return _SINGLE_OPS_T, _TWO_OPS_T


def full_pauli_features(rho: torch.Tensor, n_total: int) -> np.ndarray:
    single, two = _pauli_op_tensors()
    vals: List[torch.Tensor] = []
    for q in range(n_total):
        r = v2.reduce_register_t(rho, [q], n_total)
        vals.append(torch.einsum("kij,ji->k", single, r).real)
    for i in range(n_total):
        for j in range(i + 1, n_total):
            r = v2.reduce_register_t(rho, [i, j], n_total)
            vals.append(torch.einsum("kij,ji->k", two, r).real)
    return torch.cat(vals).double().cpu().numpy()


# ---------------------------------------------------------------------------
# Watchdog-guarded driver. `feat_fns` maps a readout name to a callable
# (model)->1d np.array. Returns {name: (T, nf) array}.
# ---------------------------------------------------------------------------
def drive_collect(
    model,
    seq: np.ndarray,
    grid,
    feat_fns: Dict[str, Callable],
    step_threshold: Optional[float],
) -> Dict[str, np.ndarray]:
    out: Dict[str, List[np.ndarray]] = {k: [] for k in feat_fns}
    limit = (10.0 * step_threshold) if step_threshold else None
    for s in seq:
        t0 = time.time()
        model.step(float(s), grid)
        for name, fn in feat_fns.items():
            out[name].append(fn(model))
        if limit is not None and (time.time() - t0) > limit:
            raise WatchdogError(f"step {time.time()-t0:.3f}s exceeded 10x benchmark {step_threshold:.3f}s")
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def cap_from_feats(feats: np.ndarray, seq: np.ndarray, target: np.ndarray, slices) -> float:
    cap, _ = v2.evaluate_capacity_from_features(feats, seq, target, slices, alpha=ALPHA)
    return cap


# ===========================================================================
# PHASE 0: sanity + consistency gate + measured micro-benchmark + budget gate.
# ===========================================================================
def sanity_new_construction() -> Dict:
    """|trace-1|<2e-3, hermiticity<2e-3, min-eig>-2e-3 after 50 steps for
    n_B in {2,4,6} (d_B in {4,16,64})."""
    results = {}
    grid = v2.build_channel_grid_gpu(0, CFG.n_a)
    seq = v2.iid_inputs(0, 50)
    ok = True
    for n_b in (2, 4, 6):
        m = AuxSweepModelGPU(n_b, eta_ab=CFG.eta_paper, eta_bb=CFG.eta_paper / 2, omega=0.3).reset()
        for s in seq:
            m.step(float(s), grid)
        chk = v2.state_checks_t(m.rho)
        # v2.state_checks_t skips eigvalsh for dim>256; compute min eigenvalue
        # directly so positivity is genuinely checked for n_b=6 (1024x1024).
        herm = 0.5 * (m.rho + m.rho.conj().T)
        chk["min_eig"] = float(torch.linalg.eigvalsh(herm).min().item())
        passed = (chk["trace_error"] < 2e-3 and chk["hermiticity_error"] < 2e-3 and chk["min_eig"] > -2e-3)
        ok = ok and passed
        results[f"n_b={n_b}"] = {**chk, "passed": passed}
        log(f"sanity n_b={n_b}: trace_err={chk['trace_error']:.2e} herm={chk['hermiticity_error']:.2e} min_eig={chk['min_eig']:.2e} -> {'OK' if passed else 'FAIL'}")
    results["passed"] = ok
    return results


def consistency_gate() -> Dict:
    """n_B=4, eta_bb=0 with v2 AB-embedded best-params must reproduce the v2
    AB-embedded degree1_stm capacity curve within GATE_TOL (mean over 4 seeds)."""
    bp = pd.read_csv(V2_DIR / "best_parameters_by_task.csv")
    row = bp[(bp.architecture == "AB-embedded") & (bp.task == "paper_s0_s10")].iloc[0]
    omega, eta = float(row["omega"]), float(row["eta"])
    log(f"consistency gate: AB-embedded v2 best-params omega={omega:.6f} eta={eta:.6f}")
    ref = pd.read_csv(V2_DIR / "ipc_by_component.csv")
    ref = ref[(ref.model == "AB-embedded") & (ref.component == "degree1_stm")]

    slices = v2.split_slices(CFG.paper_washout, CFG.paper_train, CFG.paper_test)
    my_curve: Dict[int, List[float]] = {t: [] for t in GATE_TAUS}
    for seed in GATE_SEEDS:
        seq = v2.iid_inputs(seed, CFG.paper_len)
        grid = v2.build_channel_grid_gpu(seed, CFG.n_a)
        m = AuxSweepModelGPU(n_b=CFG.n_a, eta_ab=eta, eta_bb=0.0, omega=omega).reset()
        feats = drive_collect(m, seq, grid, {"A": lambda mm: mm.features("A")}, None)["A"]
        for tau in GATE_TAUS:
            my_curve[tau].append(cap_from_feats(feats, seq, v2.stm_target(seq, tau), slices))

    diffs = {}
    max_abs = 0.0
    for tau in GATE_TAUS:
        v2_vals = ref[(ref.tau1 == tau) & (ref.seed.isin(GATE_SEEDS))]["capacity"].values
        my_mean = float(np.mean(my_curve[tau]))
        v2_mean = float(np.mean(v2_vals)) if len(v2_vals) else float("nan")
        d = abs(my_mean - v2_mean)
        diffs[str(tau)] = {"my_mean": my_mean, "v2_mean": v2_mean, "abs_diff": d}
        max_abs = max(max_abs, d)
        log(f"gate tau={tau:2d}: mine={my_mean:.4f} v2={v2_mean:.4f} |d|={d:.2e}")
    passed = max_abs < GATE_TOL
    log(f"consistency gate max|diff|={max_abs:.2e} tol={GATE_TOL:.0e} -> {'PASS' if passed else 'FAIL'}")
    return {"passed": passed, "max_abs_diff": max_abs, "tol": GATE_TOL, "per_tau": diffs,
            "v2_params": {"omega": omega, "eta": eta}}


def measure_step_costs() -> Dict[str, float]:
    """Real seconds/step per configuration (no linear extrapolation)."""
    log("micro-benchmarking seconds/step per configuration")
    grid = v2.build_channel_grid_gpu(0, CFG.n_a)
    seq = v2.iid_inputs(0, 60)
    costs: Dict[str, float] = {}

    def bench(model, feat_fn, key):
        for s in seq[:15]:
            model.step(float(s), grid)
            feat_fn(model)
        torch.cuda.synchronize()
        t0 = time.time()
        for s in seq[15:]:
            model.step(float(s), grid)
            feat_fn(model)
        torch.cuda.synchronize()
        costs[key] = (time.time() - t0) / (len(seq) - 15)
        log(f"  {key}: {costs[key]*1000:.2f} ms/step (dim {2**model.n_total})")

    for d_b in DB_LIST:
        n_b = nb_of(d_b)
        m = AuxSweepModelGPU(n_b, CFG.eta_paper, CFG.eta_paper / 2, 0.3).reset()
        bench(m, lambda mm: mm.features_t("A"), f"aux_dB{d_b}")

    ab = v2.make_embedded_model("AB-embedded", {"omega": 0.14, "eta": CFG.eta_paper}).reset()
    bench(ab, lambda mm: mm.features_t("A"), "ab_A")
    bench(ab, lambda mm: full_pauli_features(mm.rho, mm.n_total), "ab_full")

    abc = v2.make_embedded_model("ABC-embedded-hierarchical", v2.best_params("ABC-embedded-hierarchical")).reset()
    # best_params reads from V3 (empty) -> falls back to v2 defaults; fine for timing.
    bench(abc, lambda mm: mm.features_t("A"), "abc_A")
    bench(abc, lambda mm: full_pauli_features(mm.rho, mm.n_total), "abc_full")
    torch.cuda.empty_cache()
    return costs


def estimate_hours(costs: Dict[str, float], cfg: Dict) -> Dict:
    """Estimate wall-clock hours per phase from MEASURED step costs and a run
    configuration (which encodes any degradation)."""
    tune_steps = CFG.tune_len            # steps per tuning drive
    L = CFG.paper_len                    # 3000
    mg_steps = 2 * CFG.paper_washout + 2 * CFG.paper_train  # ~ drive(2000)+tf(1000)+rollout(1000)

    # Exp 1
    exp1 = 0.0
    for d_b in DB_LIST:
        c = costs[f"aux_dB{d_b}"]
        n_eval = cfg["dB64_eval_seeds"] if d_b == 64 else len(EVAL_SEEDS)
        tune = cfg["exp1_trials"] * len(EXP1_TUNE_SEEDS) * tune_steps
        ev = n_eval * L
        exp1 += (tune + ev) * c
    # Exp 2
    exp2_tune = EXP2_TRIALS * len(EXP2_TUNE_SEEDS) * tune_steps * costs["abc_A"]
    exp2_eval = len(EVAL_SEEDS) * (L + mg_steps) * costs["abc_A"]
    exp2 = exp2_tune + exp2_eval
    # Exp 3
    ab_key = "ab_full" if "full" in cfg["exp3_readouts"] else "ab_A"
    abc_key = "abc_full" if "full" in cfg["exp3_readouts"] else "abc_A"
    exp3 = len(EVAL_SEEDS) * L * (costs[ab_key] + costs[abc_key])

    total = (exp1 + exp2 + exp3) / 3600.0
    return {
        "exp1_h": round(exp1 / 3600, 3),
        "exp2_h": round(exp2 / 3600, 3),
        "exp3_h": round(exp3 / 3600, 3),
        "total_h": round(total, 3),
    }


def decide_budget(costs: Dict[str, float]) -> Dict:
    """Return {run_config, estimate, decision}. Degrade in the mandated order to
    fit under BUDGET_GO_H; abort above BUDGET_ABORT_H."""
    base = dict(DEFAULT_RUN_CONFIG)
    est0 = estimate_hours(costs, base)
    log(f"measured budget (no degradation): total {est0['total_h']:.2f} h "
        f"(exp1 {est0['exp1_h']}, exp2 {est0['exp2_h']}, exp3 {est0['exp3_h']})")

    if est0["total_h"] > BUDGET_ABORT_H:
        return {"run_config": base, "estimate": est0, "decision": "abort", "degradations": []}
    if est0["total_h"] < BUDGET_GO_H:
        return {"run_config": base, "estimate": est0, "decision": "go", "degradations": []}

    # In [12h, 20h]: degrade in order (a)->(b)->(c) until under 12h.
    cfg = dict(base)
    degr: List[Dict] = []
    ladder = [
        ("dB64_eval_seeds", 10, "reduce_dB64_eval_seeds_20_to_10"),
        ("exp3_readouts", ["A_only", "B_only", "C_only"], "drop_exp3_full_readout"),
        ("exp1_trials", 24, "reduce_exp1_trials_32_to_24"),
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
    decision = "go" if est["total_h"] <= BUDGET_ABORT_H else "abort"
    return {"run_config": cfg, "estimate": est, "estimate_base": est0, "decision": decision, "degradations": degr}


def write_aborted(reason: str, details: Dict) -> None:
    lines = [
        "# RUN ABORTED",
        "",
        f"Aborted at {datetime.now().isoformat()} during Phase 0 (go/no-go gate).",
        "",
        f"**Reason:** {reason}",
        "",
        "```json",
        json.dumps(details, indent=2, default=str),
        "```",
        "",
        "Delete this file to allow a fresh attempt after fixing the cause.",
    ]
    ABORTED_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"ABORTED: {reason}")


def phase0_gate() -> Dict:
    """Run the automatic gate. On pass, persist budget_v3.json (with the chosen
    run_config) and write the setup marker. On fail, write ABORTED.md and raise."""
    if marker("setup").exists() and BUDGET_PATH.exists():
        b = json.loads(BUDGET_PATH.read_text())
        log(f"Phase 0 already passed; run_config={b['run_config']}")
        return b

    v2.require_gpu(verbose=True)
    prime_grid_cache(sorted(set(EXP1_TUNE_SEEDS) | set(EVAL_SEEDS) | set(EXP2_TUNE_SEEDS) | set(GATE_SEEDS)))

    sanity = sanity_new_construction()
    gate = consistency_gate()
    costs = measure_step_costs()
    budget = decide_budget(costs)

    record = {
        "generated_at": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "sanity": sanity,
        "consistency_gate": gate,
        "seconds_per_step": costs,
        "estimate": budget["estimate"],
        "estimate_base": budget.get("estimate_base", budget["estimate"]),
        "degradations": budget["degradations"],
        "run_config": budget["run_config"],
        "thresholds": {"go_h": BUDGET_GO_H, "abort_h": BUDGET_ABORT_H},
        "note_exp1_trials_raised": "Exp1 tuning uses 32 trials per d_B (raised from 16) as requested; recorded here, not in failed_runs.",
    }

    reasons = []
    if not sanity["passed"]:
        reasons.append("sanity of new construction failed (trace/hermiticity/positivity)")
    if not gate["passed"]:
        reasons.append(f"consistency gate failed: max|diff|={gate['max_abs_diff']:.2e} >= {GATE_TOL:.0e}")
    if budget["decision"] == "abort":
        reasons.append(f"measured budget total {budget['estimate']['total_h']:.2f} h exceeds {BUDGET_ABORT_H} h")

    if reasons:
        record["aborted"] = True
        record["abort_reasons"] = reasons
        write_json(BUDGET_PATH, record)
        write_aborted("; ".join(reasons), record)
        raise AbortRun("; ".join(reasons))

    write_json(BUDGET_PATH, record)
    write_marker("setup", run_config=budget["run_config"], estimate=budget["estimate"],
                 gate_max_abs_diff=gate["max_abs_diff"])
    heartbeat("phase0_gate", 1.0, extra=f"budget {budget['estimate']['total_h']:.2f}h", force=True)
    log(f"Phase 0 PASSED. run_config={budget['run_config']} estimate={budget['estimate']['total_h']:.2f}h")
    return record


# ===========================================================================
# EXP 1: auxiliary-dimension sweep
# ===========================================================================
EXP1_SWEEP_CSV = V3_DIR / "aux_dimension_sweep.csv"
EXP1_SUMMARY_JSON = V3_DIR / "aux_dimension_summary.json"
EXP1_TRIALS_CSV = V3_DIR / "exp1_tuning_trials.csv"


def exp1_tune(d_b: int, n_trials: int) -> Dict:
    n_b = nb_of(d_b)
    slices = v2.split_slices(CFG.tune_washout, CFG.tune_train, CFG.tune_test)
    storage = f"sqlite:///{(V3_DIR / 'optuna_exp1_v3.sqlite3').as_posix()}"

    def objective(trial: optuna.Trial) -> float:
        omega = trial.suggest_float("omega", 0.0, 1.0)
        eta_ab = trial.suggest_float("eta_ab", 0.05, math.pi / 2 - 0.05)
        eta_bb = trial.suggest_float("eta_bb", 0.05, math.pi / 2 - 0.05) if n_b > CFG.n_a else 0.0
        caps = []
        for seed in EXP1_TUNE_SEEDS:
            seq = v2.iid_inputs(seed, CFG.tune_len)
            grid = v2.build_channel_grid_gpu(seed, CFG.n_a)
            m = AuxSweepModelGPU(n_b, eta_ab, eta_bb, omega).reset()
            feats = drive_collect(m, seq, grid, {"A": lambda mm: mm.features("A")}, None)["A"]
            caps.append(np.mean([cap_from_feats(feats, seq, v2.stm_target(seq, t), slices) for t in EXP1_TUNE_TAUS]))
        return float(np.mean(caps))

    study = optuna.create_study(direction="maximize", study_name=f"aux_dB{d_b}",
                                storage=storage, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=CFG.optuna_seed))
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - done)
    if remaining:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    best = dict(study.best_params)
    if "eta_bb" not in best:
        best["eta_bb"] = 0.0
    rows = [{"d_B": d_b, "n_b_qubits": n_b, "trial": t.number, "value": t.value, **t.params}
            for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if rows and not key_done(EXP1_TRIALS_CSV, d_B=d_b):
        append_rows(EXP1_TRIALS_CSV, rows)
    log(f"exp1 d_B={d_b}: tuned best={study.best_value:.4f} params={best}")
    return best


def exp1_eval_seed(d_b: int, seed: int, best: Dict, n_total_bench: float) -> None:
    n_b = nb_of(d_b)
    if key_done(EXP1_SWEEP_CSV, d_B=d_b, seed=seed, task=EXP1_PRODUCT[0]):
        return
    slices = v2.split_slices(CFG.paper_washout, CFG.paper_train, CFG.paper_test)
    seq = v2.iid_inputs(seed, CFG.paper_len)
    grid = v2.build_channel_grid_gpu(seed, CFG.n_a)
    m = AuxSweepModelGPU(n_b, best.get("eta_ab", CFG.eta_paper), best.get("eta_bb", 0.0), best.get("omega", 0.5)).reset()
    feats = drive_collect(m, seq, grid, {"A": lambda mm: mm.features("A")}, n_total_bench)["A"]
    rows = []
    for tau in STM_TAUS:
        cap = cap_from_feats(feats, seq, v2.stm_target(seq, tau), slices)
        rows.append({"d_B": d_b, "n_b_qubits": n_b, "seed": seed, "task": "stm_linear", "tau": tau, "capacity": cap})
    _, t1, t2 = EXP1_PRODUCT
    prod = v2.stm_target(seq, t1) * v2.stm_target(seq, t2)
    cap_p = cap_from_feats(feats, seq, prod, slices)
    rows.append({"d_B": d_b, "n_b_qubits": n_b, "seed": seed, "task": EXP1_PRODUCT[0], "tau": -1, "capacity": cap_p})
    append_rows(EXP1_SWEEP_CSV, rows)


def run_exp1(run_config: Dict, costs: Dict[str, float]) -> None:
    if marker("exp1").exists():
        log("exp1 already complete; skipping")
        return
    log(f"=== EXP 1: auxiliary-dimension sweep (trials={run_config['exp1_trials']}) ===")
    n_trials = run_config["exp1_trials"]
    total_units = sum((run_config["dB64_eval_seeds"] if d == 64 else len(EVAL_SEEDS)) for d in DB_LIST)
    done_units = 0
    for d_b in DB_LIST:
        best = exp1_tune(d_b, n_trials)
        eval_seeds = EVAL_SEEDS[: run_config["dB64_eval_seeds"]] if d_b == 64 else EVAL_SEEDS
        if d_b == 64 and run_config["dB64_eval_seeds"] < len(EVAL_SEEDS):
            record_failure("exp1_dB64", "reduced_eval_seeds_budget", executed=run_config["dB64_eval_seeds"], planned=len(EVAL_SEEDS))
        bench = costs.get(f"aux_dB{d_b}")
        for seed in eval_seeds:
            try:
                exp1_eval_seed(d_b, seed, best, bench)
            except WatchdogError as exc:
                record_failure(f"exp1/d_B{d_b}/seed{seed}", "watchdog_step_timeout", detail=str(exc))
                log(f"WATCHDOG exp1 d_B={d_b} seed={seed}: {exc}")
            except Exception as exc:  # noqa: BLE001
                record_failure(f"exp1/d_B{d_b}/seed{seed}", "unit_exception", detail=repr(exc))
                log(f"ERROR exp1 d_B={d_b} seed={seed}: {exc!r}")
            done_units += 1
            heartbeat("exp1", done_units / total_units, extra=f"d_B={d_b} seed={seed}")
        write_marker(f"exp1_dB{d_b}", best_params=best, eval_seeds=list(eval_seeds))
    _exp1_summary()
    write_marker("exp1", trials=n_trials)


def _exp1_summary() -> None:
    df = load_csv(EXP1_SWEEP_CSV)
    if df.empty:
        return
    summary = {"memory_valid_threshold": CFG.valid_threshold, "per_d_B": {}}
    stm = df[df.task == "stm_linear"]
    prod = df[df.task == EXP1_PRODUCT[0]]
    for d_b in DB_LIST:
        g = stm[stm.d_B == d_b]
        if g.empty:
            continue
        mean_curve = g.groupby("tau")["capacity"].mean().sort_index()
        valid = mean_curve[mean_curve > CFG.valid_threshold]
        mem_range = int(valid.index.max()) if len(valid) else -1
        total_stm = float(mean_curve.sum())
        pg = prod[prod.d_B == d_b]["capacity"]
        summary["per_d_B"][str(d_b)] = {
            "n_b_qubits": nb_of(d_b),
            "memory_range_tau": mem_range,
            "total_linear_stm": total_stm,
            "product_s10_s20_mean": float(pg.mean()) if len(pg) else float("nan"),
            "n_seeds": int(g["seed"].nunique()),
        }
    write_json(EXP1_SUMMARY_JSON, summary)
    log(f"exp1 summary written: {EXP1_SUMMARY_JSON.name}")


# ===========================================================================
# EXP 2: budget-matched retuning of ABC-embedded-hierarchical
# ===========================================================================
EXP2_EVAL_CSV = V3_DIR / "abc_retuned_evaluation.csv"
EXP2_VS_JSON = V3_DIR / "abc_retuned_vs_v2.json"
EXP2_CONV_CSV = V3_DIR / "optuna_convergence_curves.csv"


def exp2_extract_convergence() -> None:
    """Read best-value-vs-trial curves from the EXISTING v2 optuna sqlite
    (read-only via a copy) for the ABC-embedded studies."""
    if EXP2_CONV_CSV.exists():
        return
    src = V2_DIR / "optuna_abc_v2.sqlite3"
    if not src.exists():
        record_failure("exp2_convergence", "v2_optuna_sqlite_missing")
        return
    tmp = V3_DIR / "_v2_optuna_readonly.sqlite3"
    shutil.copy2(src, tmp)
    storage = f"sqlite:///{tmp.as_posix()}"
    rows = []
    try:
        summaries = optuna.get_all_study_summaries(storage=storage)
        for s in summaries:
            if "ABC-embedded" not in s.study_name:
                continue
            st = optuna.load_study(study_name=s.study_name, storage=storage)
            best = -math.inf
            for t in sorted(st.trials, key=lambda x: x.number):
                if t.value is None:
                    continue
                best = max(best, t.value)
                rows.append({"study": s.study_name, "trial": t.number, "value": t.value, "best_so_far": best})
    finally:
        tmp.unlink(missing_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(EXP2_CONV_CSV, index=False)
    log(f"exp2 convergence curves extracted: {len(rows)} rows from v2 sqlite")


def exp2_tune() -> Dict:
    storage = f"sqlite:///{(V3_DIR / 'optuna_abc_v3.sqlite3').as_posix()}"
    CFG.abc_tune_seeds = len(EXP2_TUNE_SEEDS)  # 8; objective uses CFG.tune_seeds[:8] = 1000..1007
    study = optuna.create_study(direction="maximize",
                                study_name=f"ABC-embedded-hierarchical_{EXP2_TUNE_TASK}_v3retune",
                                storage=storage, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=CFG.optuna_seed))
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, EXP2_TRIALS - done)
    if remaining:
        log(f"exp2 retuning ABC-hier: {remaining} trials remaining ({EXP2_TRIALS} x {len(EXP2_TUNE_SEEDS)} seeds)")
        study.optimize(lambda tr: v2.tune_objective_abc_embedded(tr, "ABC-embedded-hierarchical", EXP2_TUNE_TASK),
                       n_trials=remaining, show_progress_bar=False)
        heartbeat("exp2_tune", 1.0, extra="tuning done", force=True)
    log(f"exp2 retuned best={study.best_value:.4f} params={study.best_params}")
    return dict(study.best_params)


def exp2_eval(best: Dict) -> None:
    slices = v2.split_slices(CFG.paper_washout, CFG.paper_train, CFG.paper_test)
    series_std = v2.normalize_series(v2.mackey_glass(CFG.paper_len + 1), slices["train"])
    for i, seed in enumerate(EVAL_SEEDS):
        if key_done(EXP2_EVAL_CSV, seed=seed, kind="mackey_standard"):
            heartbeat("exp2_eval", (i + 1) / len(EVAL_SEEDS), extra=f"seed {seed} cached")
            continue
        try:
            seq = v2.iid_inputs(seed, CFG.paper_len)
            grid = v2.build_channel_grid_gpu(seed, CFG.n_a)
            m = v2.make_embedded_model("ABC-embedded-hierarchical", best).reset()
            feats = drive_collect(m, seq, grid, {"A": lambda mm: mm.features("A")}, None)["A"]
            rows = []
            for task in v2.MULTISCALE_TASKS:
                cap = cap_from_feats(feats, seq, v2.target_by_name(seq, task), slices)
                rows.append({"seed": seed, "kind": "multiscale", "task": task, "capacity": cap})
            for tau in GATE_TAUS:
                cap = cap_from_feats(feats, seq, v2.stm_target(seq, tau), slices)
                rows.append({"seed": seed, "kind": "stm", "task": f"stm_{tau}", "tau": tau, "capacity": cap})
            # Mackey-Glass standard (fresh model instance for the MG protocol)
            m_mg = v2.make_embedded_model("ABC-embedded-hierarchical", best).reset()
            mg_row, _, _, _ = v2.run_mg_model(seed, m_mg, grid, series_std, slices)
            rows.append({"seed": seed, "kind": "mackey_standard", "task": "MG_standard",
                         "capacity": np.nan, "mse_150": mg_row["mse_150"], "nrmse_150": mg_row["nrmse_150"],
                         "r2_150": mg_row["r2_150"], "valid_prediction_time": mg_row["valid_prediction_time"]})
            append_rows(EXP2_EVAL_CSV, rows)
        except WatchdogError as exc:
            record_failure(f"exp2/seed{seed}", "watchdog_step_timeout", detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            record_failure(f"exp2/seed{seed}", "unit_exception", detail=repr(exc))
            log(f"ERROR exp2 seed={seed}: {exc!r}")
        heartbeat("exp2_eval", (i + 1) / len(EVAL_SEEDS), extra=f"seed {seed}")


def exp2_compare(best: Dict) -> None:
    """Paired Wilcoxon + bootstrap CI: retuned (v3) vs v2 numbers, same 20 seeds."""
    new = load_csv(EXP2_EVAL_CSV)
    if new.empty:
        return
    out: Dict = {"retuned_params": best, "v2_params": {}, "comparisons": {}}
    bp = pd.read_csv(V2_DIR / "best_parameters_by_task.csv")
    r = bp[(bp.architecture == "ABC-embedded-hierarchical") & (bp.task == EXP2_TUNE_TASK)]
    if len(r):
        out["v2_params"] = r.iloc[0].dropna().to_dict()

    # STM total per seed
    v2_ipc = pd.read_csv(V2_DIR / "ipc_by_component.csv")
    v2_stm = v2_ipc[(v2_ipc.model == "ABC-embedded-hierarchical") & (v2_ipc.component == "degree1_stm")]
    new_stm = new[new.kind == "stm"]
    a, b = [], []
    for seed in EVAL_SEEDS:
        na = new_stm[(new_stm.seed == seed) & (new_stm.tau.isin(GATE_TAUS))]["capacity"]
        vb = v2_stm[(v2_stm.seed == seed) & (v2_stm.tau1.isin(GATE_TAUS))]["capacity"]
        if len(na) and len(vb):
            a.append(float(na.sum())); b.append(float(vb.sum()))
    if a:
        out["comparisons"]["stm_total"] = v2.paired_stats(np.array(a), np.array(b), larger_better=True)

    # Multiscale tasks
    v2_ms = pd.read_csv(V2_DIR / "multiscale_capacities.csv")
    v2_ms = v2_ms[v2_ms.model == "ABC-embedded-hierarchical"]
    new_ms = new[new.kind == "multiscale"]
    for task in v2.MULTISCALE_TASKS:
        a, b = [], []
        for seed in EVAL_SEEDS:
            na = new_ms[(new_ms.seed == seed) & (new_ms.task == task)]["capacity"]
            vb = v2_ms[(v2_ms.seed == seed) & (v2_ms.task == task)]["capacity"]
            if len(na) and len(vb):
                a.append(float(na.iloc[0])); b.append(float(vb.iloc[0]))
        if a:
            out["comparisons"][f"multiscale::{task}"] = v2.paired_stats(np.array(a), np.array(b), larger_better=True)

    # Mackey-Glass standard: mse_150 (smaller is better)
    v2_mg = pd.read_csv(V2_DIR / "mackey_glass_standard.csv")
    v2_mg = v2_mg[v2_mg.model == "ABC-embedded-hierarchical"]
    new_mg = new[new.kind == "mackey_standard"]
    a, b = [], []
    for seed in EVAL_SEEDS:
        na = new_mg[new_mg.seed == seed]["mse_150"]
        vb = v2_mg[v2_mg.seed == seed]["mse_150"]
        if len(na) and len(vb):
            a.append(float(na.iloc[0])); b.append(float(vb.iloc[0]))
    if a:
        out["comparisons"]["mackey_mse150"] = v2.paired_stats(np.array(a), np.array(b), larger_better=False)

    write_json(EXP2_VS_JSON, out)
    log(f"exp2 comparison written: {EXP2_VS_JSON.name}")


def run_exp2(run_config: Dict) -> None:
    if marker("exp2").exists():
        log("exp2 already complete; skipping")
        return
    log("=== EXP 2: budget-matched ABC-embedded-hierarchical retuning ===")
    exp2_extract_convergence()
    best = exp2_tune()
    exp2_eval(best)
    exp2_compare(best)
    write_marker("exp2", trials=EXP2_TRIALS, tune_seeds=list(EXP2_TUNE_SEEDS), best_params=best)


# ===========================================================================
# EXP 3: readout localized to A / B / C (+full)
# ===========================================================================
EXP3_CSV = V3_DIR / "readout_location_stm.csv"


def _exp3_feat_fns(model_name: str, n_total: int, readouts: List[str]) -> Dict[str, Callable]:
    fns: Dict[str, Callable] = {}
    if "A_only" in readouts:
        fns["A_only"] = lambda mm: mm.features("A")
    if "B_only" in readouts:
        fns["B_only"] = lambda mm: mm.features("B")
    if model_name.startswith("ABC") and "C_only" in readouts:
        fns["C_only"] = lambda mm: mm.features("C")
    if "full" in readouts:
        fns["full"] = lambda mm: full_pauli_features(mm.rho, n_total)
    return fns


def run_exp3(run_config: Dict) -> None:
    if marker("exp3").exists():
        log("exp3 already complete; skipping")
        return
    log(f"=== EXP 3: readout location (readouts={run_config['exp3_readouts']}) ===")
    if run_config["exp3_readouts"] != DEFAULT_RUN_CONFIG["exp3_readouts"]:
        record_failure("exp3", "reduced_readouts_budget", executed=run_config["exp3_readouts"], planned=DEFAULT_RUN_CONFIG["exp3_readouts"])
    slices = v2.split_slices(CFG.paper_washout, CFG.paper_train, CFG.paper_test)
    bp = pd.read_csv(V2_DIR / "best_parameters_by_task.csv")

    def v2_best(arch: str) -> Dict:
        r = bp[(bp.architecture == arch) & (bp.task == "paper_s0_s10")]
        return r.iloc[0].dropna().to_dict() if len(r) else v2.best_params(arch)

    models = [("AB-embedded", v2_best("AB-embedded")),
              ("ABC-embedded-hierarchical", v2_best("ABC-embedded-hierarchical"))]
    total = len(models) * len(EVAL_SEEDS)
    done = 0
    for model_name, params in models:
        for seed in EVAL_SEEDS:
            done += 1
            if key_done(EXP3_CSV, model=model_name, seed=seed, readout="A_only"):
                heartbeat("exp3", done / total, extra=f"{model_name} seed{seed} cached")
                continue
            try:
                seq = v2.iid_inputs(seed, CFG.paper_len)
                grid = v2.build_channel_grid_gpu(seed, CFG.n_a)
                m = v2.make_embedded_model(model_name, params).reset()
                fns = _exp3_feat_fns(model_name, m.n_total, run_config["exp3_readouts"])
                bench_key = "abc_A" if model_name.startswith("ABC") else "ab_A"
                # No hard watchdog here: full readout genuinely lengthens each step.
                feats = drive_collect(m, seq, grid, fns, None)
                rows = []
                for readout, fmat in feats.items():
                    for tau in EXP3_TAUS:
                        cap = cap_from_feats(fmat, seq, v2.stm_target(seq, tau), slices)
                        rows.append({"model": model_name, "readout": readout, "seed": seed, "tau": tau, "capacity": cap})
                append_rows(EXP3_CSV, rows)
            except WatchdogError as exc:
                record_failure(f"exp3/{model_name}/seed{seed}", "watchdog_step_timeout", detail=str(exc))
            except Exception as exc:  # noqa: BLE001
                record_failure(f"exp3/{model_name}/seed{seed}", "unit_exception", detail=repr(exc))
                log(f"ERROR exp3 {model_name} seed={seed}: {exc!r}")
            heartbeat("exp3", done / total, extra=f"{model_name} seed{seed}")
    write_marker("exp3", models=[m for m, _ in models], readouts=run_config["exp3_readouts"])


# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
def _verdict_scale_law() -> Tuple[str, Dict]:
    s = load_csv(EXP1_SWEEP_CSV)
    if s.empty:
        return "no data", {}
    stm = s[s.task == "stm_linear"]
    ranges = {}
    for d_b in DB_LIST:
        g = stm[stm.d_B == d_b]
        if g.empty:
            continue
        mc = g.groupby("tau")["capacity"].mean().sort_index()
        valid = mc[mc > CFG.valid_threshold]
        ranges[d_b] = int(valid.index.max()) if len(valid) else -1
    # Paired Wilcoxon on per-seed total STM: largest d_B vs d_B=16 (v2 baseline).
    stats_pair = {}
    if 64 in stm.d_B.values and 16 in stm.d_B.values:
        a, b = [], []
        for seed in EVAL_SEEDS:
            ga = stm[(stm.d_B == 64) & (stm.seed == seed)]["capacity"].sum()
            gb = stm[(stm.d_B == 16) & (stm.seed == seed)]["capacity"].sum()
            if ga and gb:
                a.append(float(ga)); b.append(float(gb))
        if a:
            stats_pair = v2.paired_stats(np.array(a), np.array(b), larger_better=True)
    increasing = [ranges[d] for d in DB_LIST if d in ranges]
    verdict = "memory range increases with d_B" if len(increasing) >= 2 and increasing[-1] > increasing[0] else \
              "memory range does NOT scale up with d_B"
    return verdict, {"memory_range_by_dB": ranges, "stm_total_dB64_vs_dB16": stats_pair}


def _verdict_subtuning() -> Tuple[str, Dict]:
    if not EXP2_VS_JSON.exists():
        return "no data", {}
    d = json.loads(EXP2_VS_JSON.read_text())
    comps = d.get("comparisons", {})
    stm = comps.get("stm_total", {})
    ms = comps.get("multiscale::s0_s10_s30", {})
    improved = False
    for key in ("stm_total", "multiscale::s0_s10_s30", "multiscale::p1_0_10_30"):
        c = comps.get(key, {})
        if c and c.get("p_wilcoxon", 1.0) < 0.05 and c.get("mean_diff", 0.0) > 0:
            improved = True
    verdict = ("subtuning WAS a factor: 64x8 retuning significantly improves the retuned model"
               if improved else
               "NOT subtuning: extra tuning budget does not significantly improve ABC-embedded-hierarchical")
    return verdict, {"stm_total": stm, "multiscale_s0_s10_s30": ms, "all": comps}


def _verdict_mechanism() -> Tuple[str, Dict]:
    df = load_csv(EXP3_CSV)
    if df.empty:
        return "no data", {}
    info = {}
    verdicts = []
    for model_name in df.model.unique():
        g = df[df.model == model_name]
        tau_band = [t for t in EXP3_TAUS if 25 <= t <= 35]
        by_ro = {}
        for ro in g.readout.unique():
            band = g[(g.readout == ro) & (g.tau.isin(tau_band))]["capacity"]
            by_ro[ro] = float(band.mean()) if len(band) else float("nan")
        info[model_name] = by_ro
        a_val = by_ro.get("A_only", 0.0)
        b_val = max(by_ro.get("B_only", 0.0), by_ro.get("C_only", 0.0))
        if b_val > 5 * max(a_val, 1e-6) and b_val > 0.05:
            verdicts.append(f"{model_name}: BACKFLOW failure (info at B/C, tau~30, but not in A)")
        elif max(a_val, b_val) < 0.05:
            verdicts.append(f"{model_name}: STORAGE failure (tau~30 absent everywhere)")
        else:
            verdicts.append(f"{model_name}: partial/ambiguous (A~{a_val:.3f}, B|C~{b_val:.3f})")
    return "; ".join(verdicts), info


def write_summary() -> None:
    v_scale, d_scale = _verdict_scale_law()
    v_sub, d_sub = _verdict_subtuning()
    v_mech, d_mech = _verdict_mechanism()
    failed = load_csv(FAILED_PATH)

    lines = ["# Extra experiments v3 — summary", "",
             f"_Generated {datetime.now().isoformat()}_", ""]
    if BUDGET_PATH.exists():
        b = json.loads(BUDGET_PATH.read_text())
        lines += [f"Run config: `{json.dumps(b.get('run_config', {}))}` | "
                  f"measured budget: {b.get('estimate', {}).get('total_h', 'n/a')} h", ""]

    lines += ["## Verdict 1 — memory range vs auxiliary dimension (Exp 1)", "",
              f"**{v_scale}**", "", "| d_B | n_B qubits | memory range (max tau, C>0.1) |",
              "|---|---|---|"]
    for d_b, rng in d_scale.get("memory_range_by_dB", {}).items():
        lines.append(f"| {d_b} | {nb_of(int(d_b))} | {rng} |")
    sp = d_scale.get("stm_total_dB64_vs_dB16", {})
    if sp:
        lines += ["", f"Paired STM total, d_B=64 vs d_B=16: mean_diff={sp.get('mean_diff'):.4f}, "
                  f"95% CI [{sp.get('ci95_lo'):.4f}, {sp.get('ci95_hi'):.4f}], "
                  f"Wilcoxon p={sp.get('p_wilcoxon'):.4g}, n={sp.get('n')}."]

    lines += ["", "## Verdict 2 — subtuning defense (Exp 2)", "", f"**{v_sub}**", ""]
    stm = d_sub.get("stm_total", {})
    if stm:
        lines += [f"STM total (retuned vs v2): mean_diff={stm.get('mean_diff'):.4f}, "
                  f"95% CI [{stm.get('ci95_lo'):.4f}, {stm.get('ci95_hi'):.4f}], "
                  f"Wilcoxon p={stm.get('p_wilcoxon'):.4g}, n={stm.get('n')}."]

    lines += ["", "## Verdict 3 — mechanism: storage vs backflow (Exp 3)", "", f"**{v_mech}**", "",
              "Mean STM capacity in the tau in [25,35] band by readout:", ""]
    lines += ["```json", json.dumps(d_mech, indent=2, default=str), "```"]

    lines += ["", "## Anomalies (failed_runs.csv)", ""]
    if failed.empty:
        lines.append("None recorded.")
    else:
        lines += ["| timestamp | context | reason |", "|---|---|---|"]
        for _, r in failed.iterrows():
            lines.append(f"| {r.get('timestamp','')} | {r.get('context','')} | {r.get('reason','')} |")

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    write_json(V3_DIR / "verdicts.json", {"scale_law": {"verdict": v_scale, "data": d_scale},
                                          "subtuning": {"verdict": v_sub, "data": d_sub},
                                          "mechanism": {"verdict": v_mech, "data": d_mech}})
    log(f"summary written: {SUMMARY_PATH.name}")


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    v2.ensure_dirs()
    PROGRESS_PATH.touch(exist_ok=True)
    if ABORTED_PATH.exists():
        log(f"ABORTED.md present ({ABORTED_PATH}); refusing to run. Delete it to retry.")
        print(ABORTED_PATH.read_text())
        return
    log("========== extra_experiments_v3 start ==========")
    heartbeat("start", 0.0, force=True)

    record = phase0_gate()  # raises AbortRun on gate failure
    run_config = record["run_config"]
    costs = record["seconds_per_step"]

    run_exp1(run_config, costs)
    run_exp2(run_config)
    run_exp3(run_config)
    write_summary()
    write_marker("summary")
    heartbeat("done", 1.0, force=True)
    log("========== extra_experiments_v3 complete ==========")


if __name__ == "__main__":
    try:
        main()
    except AbortRun as exc:
        print(f"\n*** RUN ABORTED (Phase 0 gate): {exc}\nSee {ABORTED_PATH}\n")
        raise SystemExit(3)
