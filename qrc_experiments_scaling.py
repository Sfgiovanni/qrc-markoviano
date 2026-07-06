"""Extra experiments v5 -- Exp 7: does the memory horizon track the dynamical
scales?  (Extends Sannia et al., arXiv:2505.02491; built on the v2 pipeline.)

The project has shown the partial-swap embedding memory horizon (~tau=21 at
gamma=0.1, eta=pi/4) does NOT move with auxiliary dimension (v3) nor topology
(v4): "the limit is dynamical, not dimensional".  Exp 7 closes the claim from the
POSITIVE side: show the horizon DOES move when the dynamical scales (gamma, eta)
change, ideally by a predictable law (tau_mem ~ 1/gamma^x).

Model: AB-embedded only, d_B=16 (n_B=4), the v2 paired coupling.  Two 1-D sweeps
around the reference (gamma=0.1, eta=pi/4):
  * gamma in {0.02, 0.05, 0.1, 0.2}, eta = pi/4
  * eta in {pi/8, pi/4, 3pi/8}, gamma = 0.1
(6 unique configurations; the reference point is shared.)

Per configuration: regenerate the gamma-dependent channel cache; grid-tune Omega
(9 values, 4 tuning seeds); run a PAIRED M0-noaux control at the same gamma
(only gamma matters for M0); use an ADAPTIVE ESP washout; evaluate C(tau) for
tau=0..80 over 20 paired seeds with an A-only 66-feature ridge readout.

Isolation: v2's I/O globals are redirected into results_extra_v5/.  v3/v4 artefacts
are read only via absolute paths.  Phase 0 is an automatic go/no-go gate.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qrc-v2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy.linalg
import torch

import qrc_pipeline as v2

# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------
V5_DIR = Path("results_extra_v5").resolve()
V2_DIR = Path("results_abc_comparison_v2").resolve()
V3_DIR = Path("results_extra_v3").resolve()
V4_DIR = Path("results_extra_v4").resolve()
v2.RESULTS_DIR = V5_DIR
v2.FIGURES_DIR = V5_DIR / "figures"
v2.LOG_PATH = V5_DIR / "run.log"

CFG = v2.CFG

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_A = CFG.n_a                       # 4
ETA_REF = math.pi / 4
GAMMAS = [0.02, 0.05, 0.1, 0.2]
ETAS = [math.pi / 8, math.pi / 4, 3 * math.pi / 8]
ETA_LABELS = {math.pi / 8: "pi8", math.pi / 4: "pi4", 3 * math.pi / 8: "3pi8"}
GAMMA_REF = 0.1
REF_CONFIG = (0.1, ETA_REF)

EVAL_SEEDS = tuple(range(20))
TUNE_SEEDS = (1000, 1001, 1002, 1003)
OMEGA_GRID = [round(x, 4) for x in np.linspace(0.1, 0.9, 9)]
TAUS = list(range(0, 81))
THRESHOLDS = [0.05, 0.1, 0.2]
MAIN_THR = 0.1
ALPHA = 1e-6
N_BOOT = 2000

ESP_TOL = 1e-3
ESP_MAX_STEPS = 5000
ESP_OMEGA = 0.1                     # ESP measured at the grid's LOWEST omega (worst case:
                                    # least B depolarization -> slowest to forget the initial
                                    # state -> longest washout), so the adaptive washout is
                                    # valid for any tuned omega >= 0.1. Conservative choice.

REF_HORIZON_LO, REF_HORIZON_HI = 19, 23   # Phase 0 gate window (v3 gave 21)
# The spec's reference (gamma=0.1, eta=pi/4) does NOT give tau~21: measured ~15.
# v3's d_B=16 horizon of 21 uses eta_ab=1.33879, omega=0.06988 (from the v3 log).
# The Phase 0 "reproduces the old infrastructure" gate is therefore run against v3's
# ACTUAL config (its true intent), not eta=pi/4. See decisions_log.md.
V3_REPRO_ETA = 1.3387885644633968
V3_REPRO_OMEGA = 0.06988003641298665
V3_REPRO_SEEDS = tuple(range(10))

BUDGET_GO_H = 10.0
BUDGET_ABORT_H = 16.0

DEFAULT_RUN_CONFIG = {"gammas": list(GAMMAS), "eval_seeds": 20, "etas": list(ETAS)}

assert not (set(EVAL_SEEDS) & set(TUNE_SEEDS)), "eval and tuning seeds must be disjoint"

# reuse v2 infra (now redirected)
log = v2.log
marker = v2.marker
write_marker = v2.write_marker
key_done = v2.key_done
append_rows = v2.append_rows
record_failure = v2.record_failure
write_json = v2.write_json
load_csv = v2.load_csv

ABORTED_PATH = V5_DIR / "ABORTED.md"
BUDGET_PATH = V5_DIR / "budget_v5.json"
PROGRESS_PATH = V5_DIR / "progress_log.md"
FAILED_PATH = V5_DIR / "failed_runs.csv"
DECISIONS_PATH = V5_DIR / "decisions_log.md"
STATE_PATH = V5_DIR / "config_state.json"
STM_CSV = V5_DIR / "dynamical_sweep_stm.csv"
HORIZONS_CSV = V5_DIR / "horizons.csv"
FITS_JSON = V5_DIR / "scaling_fits.json"
SUMMARY_PATH = V5_DIR / "scaling_law_summary.md"


class WatchdogError(RuntimeError):
    pass


class AbortRun(RuntimeError):
    pass


def cfg_id(gamma: float, eta: float) -> str:
    return f"g{gamma}_e{ETA_LABELS[eta]}"


def decision(title: str, detail: str) -> None:
    with DECISIONS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"- [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] **{title}** — {detail}\n")
    log(f"DECISION: {title} — {detail}")


# ---------------------------------------------------------------------------
# Persistent per-config state (washout, omega_opt, tau_fm, esp_conv)
# ---------------------------------------------------------------------------
def load_state() -> Dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: Dict) -> None:
    write_json(STATE_PATH, state)


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
    line = (f"- [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] phase=**{phase}** "
            f"{frac*100:5.1f}% | elapsed {elapsed/3600:.2f}h | "
            f"ETA {('%.2fh' % (eta/3600)) if eta == eta else 'n/a'} | "
            f"GPU {cur:.0f}/{peak:.0f} MB{(' | ' + extra) if extra else ''}")
    with PROGRESS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------------------
# gamma switching: set CFG.gamma.  Since the M3 fix, v2._GRID_GPU_CACHE keys on
# (seed,n,g,gamma,dt,s_min,s_max), so switching gamma no longer returns stale
# channels.  We still clear defensively to bound memory during the sweep.
# ---------------------------------------------------------------------------
def set_gamma(gamma: float) -> None:
    CFG.gamma = float(gamma)
    v2._GRID_GPU_CACHE.clear()


def ensure_gamma_grids(gamma: float, seeds: Sequence[int]) -> None:
    """Build (or reuse) the channel grids for `gamma`.  For gamma=0.1 copy the v2
    cache after verifying one grid against a fresh rebuild; otherwise build."""
    v2.ensure_dirs()
    (V5_DIR / "channel_cache").mkdir(parents=True, exist_ok=True)
    if abs(gamma - 0.1) < 1e-12:
        _reuse_v2_gamma01(seeds)
    for seed in seeds:
        v2.build_channel_grid_np(seed, N_A)   # cached (respects CFG.gamma)


def _reuse_v2_gamma01(seeds: Sequence[int]) -> None:
    tag = f"_g{CFG.grid_size}_dt{CFG.dt}_gamma0.1_range{CFG.grid_s_min}_{CFG.grid_s_max}.npz"
    verified = False
    for seed in seeds:
        name = f"channel_N{N_A}_seed{seed}{tag}"
        src = V2_DIR / "channel_cache" / name
        dst = V5_DIR / "channel_cache" / name
        if not src.exists() or dst.exists():
            continue
        if not verified:
            # verify v2 grid == fresh rebuild for this seed before trusting the cache
            base, drive = v2.liouvillian_parts(seed, N_A)
            svals = np.linspace(CFG.grid_s_min, CFG.grid_s_max, CFG.grid_size)
            fresh = np.stack([scipy.linalg.expm((base + s * drive) * CFG.dt) for s in svals], axis=0)
            v2grid = np.load(src)["grid"]
            diff = float(np.max(np.abs(fresh - v2grid)))
            if diff > 1e-9:
                record_failure("gamma01_cache", "v2_cache_mismatch", max_abs_diff=diff)
                decision("gamma=0.1 cache", f"v2 cache mismatch (max|diff|={diff:.2e}); rebuilding fresh")
                return
            decision("gamma=0.1 cache", f"reused v2 channel cache (verified max|diff|={diff:.2e} < 1e-9)")
            verified = True
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Models / drives
# ---------------------------------------------------------------------------
def make_ab(omega: float, eta: float) -> "v2.EmbeddedModelGPU":
    return v2.make_embedded_model("AB-embedded", {"omega": omega, "eta": eta}).reset()


def make_m0(seed: int):
    return v2.make_noaux_model("M0-noaux", {}, seed).reset()


def drive_feats(model, seq: np.ndarray, grid, feat_fn: Callable, step_threshold: Optional[float]) -> np.ndarray:
    out: List[np.ndarray] = []
    limit = (10.0 * step_threshold) if step_threshold else None
    for s in seq:
        t0 = time.time()
        model.step(float(s), grid)
        out.append(feat_fn(model))
        if limit is not None and (time.time() - t0) > limit:
            raise WatchdogError(f"step {time.time()-t0:.3f}s exceeded 10x benchmark {step_threshold:.3f}s")
    return np.asarray(out, dtype=np.float64)


def cap_from_feats(feats: np.ndarray, seq: np.ndarray, target: np.ndarray, slices) -> float:
    cap, _ = v2.evaluate_capacity_from_features(feats, seq, target, slices, alpha=ALPHA)
    return cap


# ---------------------------------------------------------------------------
# ESP / adaptive washout
# ---------------------------------------------------------------------------
def esp_washout(gamma: float, eta: float, omega: float = ESP_OMEGA, seed: int = 0) -> Tuple[Optional[int], Optional[int]]:
    """Drive two extreme initial states (|0><0| and maximally mixed) with the same
    input; return (washout, conv_step) where washout = max(1000, 3*conv_step), or
    (None, None) if ESP does not converge (|diff|<ESP_TOL) within ESP_MAX_STEPS."""
    grid = v2.build_channel_grid_gpu(seed, N_A)
    seq = v2.iid_inputs(seed, ESP_MAX_STEPS)
    m1 = make_ab(omega, eta)
    m2 = make_ab(omega, eta)
    d = 2 ** m2.n_total
    m2.rho = (torch.eye(d, dtype=v2.CDTYPE, device=v2.get_device()) / d)
    conv = None
    for k, s in enumerate(seq):
        m1.step(float(s), grid)
        m2.step(float(s), grid)
        diff = float(torch.linalg.norm(m1.rho - m2.rho).item())
        if diff < ESP_TOL:
            conv = k + 1
            break
    if conv is None:
        return None, None
    return max(1000, 3 * conv), conv


# ---------------------------------------------------------------------------
# Evaluation of a single model's C(tau) curve over seeds (checkpointed per seed)
# ---------------------------------------------------------------------------
def eval_curve(config_id: str, gamma: float, eta: float, omega: float, model_tag: str,
               washout: int, seeds: Sequence[int], step_threshold: Optional[float]) -> None:
    train, test = CFG.paper_train, CFG.paper_test
    slices = v2.split_slices(washout, train, test)
    length = washout + train + test
    for seed in seeds:
        if key_done(STM_CSV, config=config_id, model=model_tag, seed=seed):
            continue
        try:
            seq = v2.iid_inputs(seed, length)
            if model_tag == "M0":
                grid = v2.build_channel_grid_np(seed, N_A)
                m = make_m0(seed)
                feats = drive_feats(m, seq, grid, lambda mm: mm.features(), step_threshold)
            else:
                grid = v2.build_channel_grid_gpu(seed, N_A)
                m = make_ab(omega, eta)
                feats = drive_feats(m, seq, grid, lambda mm: mm.features("A"), step_threshold)
            rows = [{"config": config_id, "gamma": gamma, "eta": eta, "omega_opt": omega,
                     "model": model_tag, "seed": seed, "tau": tau,
                     "capacity": cap_from_feats(feats, seq, v2.stm_target(seq, tau), slices)}
                    for tau in TAUS]
            append_rows(STM_CSV, rows)
        except WatchdogError as exc:
            record_failure(f"{config_id}/{model_tag}/seed{seed}", "watchdog_step_timeout", detail=str(exc))
            log(f"WATCHDOG {config_id} {model_tag} seed={seed}: {exc}")
        except Exception as exc:  # noqa: BLE001
            record_failure(f"{config_id}/{model_tag}/seed{seed}", "unit_exception", detail=repr(exc))
            log(f"ERROR {config_id} {model_tag} seed={seed}: {exc!r}")


def mean_curve(config_id: str, model_tag: str) -> pd.Series:
    df = load_csv(STM_CSV)
    if df.empty:
        return pd.Series(dtype=float)
    g = df[(df.config == config_id) & (df.model == model_tag)]
    if g.empty:
        return pd.Series(dtype=float)
    return g.groupby("tau")["capacity"].mean().sort_index()


def horizon(curve: pd.Series, threshold: float) -> int:
    if curve is None or len(curve) == 0:
        return -1
    valid = curve[curve > threshold]
    return int(valid.index.max()) if len(valid) else -1


# ---------------------------------------------------------------------------
# Omega grid tuning (objective = total STM over tau > tau_FM, on tuning seeds)
# ---------------------------------------------------------------------------
def tune_omega(gamma: float, eta: float, washout: int, tau_fm: int, step_threshold: Optional[float]) -> Tuple[float, List[Dict]]:
    slices = v2.split_slices(washout, CFG.paper_train, CFG.paper_test)
    length = washout + CFG.paper_train + CFG.paper_test
    lo = max(tau_fm + 1, 5)
    obj_taus = [t for t in TAUS if lo <= t <= 80] or list(TAUS)
    # precompute driven features per (omega, seed) is expensive; drive per omega/seed.
    trials = []
    best_omega, best_val = OMEGA_GRID[0], -math.inf
    for omega in OMEGA_GRID:
        vals = []
        for seed in TUNE_SEEDS:
            seq = v2.iid_inputs(seed, length)
            grid = v2.build_channel_grid_gpu(seed, N_A)
            m = make_ab(omega, eta)
            feats = drive_feats(m, seq, grid, lambda mm: mm.features("A"), step_threshold)
            vals.append(float(np.sum([cap_from_feats(feats, seq, v2.stm_target(seq, t), slices) for t in obj_taus])))
        mean_val = float(np.mean(vals))
        trials.append({"gamma": gamma, "eta": eta, "omega": omega, "objective": mean_val,
                       "obj_tau_lo": lo, "obj_tau_hi": 80})
        if mean_val > best_val:
            best_val, best_omega = mean_val, omega
    log(f"tune omega {cfg_id(gamma, eta)}: best omega={best_omega} obj={best_val:.4f} (tau in [{lo},80])")
    return best_omega, trials


# ===========================================================================
# Per-gamma / per-config processing (idempotent, checkpointed)
# ===========================================================================
TUNE_TRIALS_CSV = V5_DIR / "omega_tuning_trials.csv"


def process_gamma(gamma: float, etas: List[float], eval_seeds: Sequence[int],
                  costs: Dict[str, float]) -> None:
    state = load_state()
    set_gamma(gamma)
    ensure_gamma_grids(gamma, sorted(set(eval_seeds) | set(TUNE_SEEDS) | {0}))

    # ESP washout at reference eta (gamma-dominated) gates the whole gamma.
    gk = f"gamma_{gamma}"
    if gk not in state or state[gk].get("washout") is None:
        w, conv = esp_washout(gamma, ETA_REF)
        state.setdefault(gk, {})
        state[gk].update({"washout": w, "esp_conv": conv})
        save_state(state)
        if w is None:
            record_failure(f"gamma_{gamma}", "esp_violated", detail=f"no ESP convergence in {ESP_MAX_STEPS} steps")
            decision(f"gamma={gamma} skipped", f"ESP not reached within {ESP_MAX_STEPS} steps (reportable result)")
            return
        log(f"gamma={gamma}: ESP conv={conv} -> washout={w}")
    if state[gk].get("washout") is None:
        return
    w_gamma = int(state[gk]["washout"])
    thr_bench = costs.get("AB-embedded")

    # M0 control (only gamma matters); shared tau_FM for all etas at this gamma.
    m0_id = f"M0_g{gamma}"
    eval_curve(m0_id, gamma, ETA_REF, np.nan, "M0", w_gamma, eval_seeds, costs.get("M0-noaux"))
    tau_fm = horizon(mean_curve(m0_id, "M0"), MAIN_THR)
    state[gk]["tau_fm"] = tau_fm
    save_state(state)
    log(f"gamma={gamma}: tau_FM (M0) = {tau_fm}")

    for eta in etas:
        cid = cfg_id(gamma, eta)
        st = state.get(cid, {})
        # per-config ESP (eta changes mixing); fall back to gamma washout.
        if "washout" not in st:
            w_cfg, conv = esp_washout(gamma, eta)
            if w_cfg is None:
                record_failure(cid, "esp_violated", detail=f"no ESP in {ESP_MAX_STEPS} steps")
                decision(f"{cid} skipped", "ESP not reached (reportable)")
                st = {"washout": None, "esp_conv": None}
                state[cid] = st
                save_state(state)
                continue
            st = {"washout": w_cfg, "esp_conv": conv}
            state[cid] = st
            save_state(state)
        if st.get("washout") is None:
            continue
        w_cfg = int(st["washout"])
        # Omega tuning
        if "omega_opt" not in st:
            omega_opt, trials = tune_omega(gamma, eta, w_cfg, tau_fm, thr_bench)
            if not TUNE_TRIALS_CSV.exists() or not key_done(TUNE_TRIALS_CSV, gamma=gamma, eta=eta):
                append_rows(TUNE_TRIALS_CSV, trials)
            st["omega_opt"] = omega_opt
            state[cid] = st
            save_state(state)
        omega_opt = float(st["omega_opt"])
        # Embedded evaluation
        eval_curve(cid, gamma, eta, omega_opt, "AB-embedded", w_cfg, eval_seeds, thr_bench)
        heartbeat("eval", 0.5, extra=f"{cid} omega={omega_opt} washout={w_cfg}", force=True)


# ===========================================================================
# PHASE 0
# ===========================================================================
def sanity_state(gamma: float) -> Dict:
    set_gamma(gamma)
    v2.build_channel_grid_np(0, N_A)
    grid = v2.build_channel_grid_gpu(0, N_A)
    seq = v2.iid_inputs(0, 60)
    m = make_ab(0.5, ETA_REF)
    for s in seq:
        m.step(float(s), grid)
    chk = v2.state_checks_t(m.rho)
    passed = (chk["trace_error"] < 2e-3 and chk["hermiticity_error"] < 2e-3 and chk["min_eig"] > -2e-3)
    log(f"sanity gamma={gamma}: trace={chk['trace_error']:.2e} herm={chk['hermiticity_error']:.2e} "
        f"min_eig={chk['min_eig']:.2e} -> {'OK' if passed else 'FAIL'}")
    return {**chk, "passed": bool(passed)}


def measure_costs() -> Dict[str, float]:
    log("micro-benchmarking seconds/step")
    set_gamma(GAMMA_REF)
    v2.build_channel_grid_np(0, N_A)
    grid = v2.build_channel_grid_gpu(0, N_A)
    grid_np = v2.build_channel_grid_np(0, N_A)
    seq = v2.iid_inputs(0, 60)
    costs: Dict[str, float] = {}

    def bench(model, ffn, g, key):
        for s in seq[:15]:
            model.step(float(s), g); ffn(model)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for s in seq[15:]:
            model.step(float(s), g); ffn(model)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        costs[key] = (time.time() - t0) / (len(seq) - 15)
        log(f"  {key}: {costs[key]*1000:.2f} ms/step")

    bench(make_ab(0.5, ETA_REF), lambda mm: mm.features("A"), grid, "AB-embedded")
    bench(make_m0(0), lambda mm: mm.features(), grid_np, "M0-noaux")
    return costs


def estimate_hours(costs: Dict[str, float], cfg: Dict, washouts: Dict[float, int]) -> Dict:
    L = CFG.paper_train + CFG.paper_test
    n_eval = cfg["eval_seeds"]
    c_ab, c_m0 = costs["AB-embedded"], costs["M0-noaux"]
    total = 0.0
    # cache building (approx 5 s/grid for new gammas; gamma=0.1 reused)
    for gamma in cfg["gammas"]:
        if abs(gamma - 0.1) > 1e-12:
            total += (n_eval + len(TUNE_SEEDS)) * 5.0
    for gamma in cfg["gammas"]:
        w = washouts.get(gamma, 2000)
        drive = w + L
        # M0 eval
        total += n_eval * drive * c_m0
        etas = cfg["etas"] if abs(gamma - 0.1) < 1e-12 else [ETA_REF]
        for _eta in etas:
            total += len(OMEGA_GRID) * len(TUNE_SEEDS) * drive * c_ab   # omega tuning
            total += n_eval * drive * c_ab                             # embedded eval
        # ESP measurement cost (a few drives up to ESP_MAX_STEPS)
        total += 2 * ESP_MAX_STEPS * c_ab
    return {"total_h": round(total / 3600.0, 3)}


def decide_budget(costs: Dict[str, float], washouts: Dict[float, int]) -> Dict:
    base = dict(DEFAULT_RUN_CONFIG)
    est0 = estimate_hours(costs, base, washouts)
    log(f"measured budget (no degradation): total {est0['total_h']:.2f} h")
    if est0["total_h"] > BUDGET_ABORT_H:
        return {"run_config": base, "estimate": est0, "decision": "abort", "degradations": []}
    if est0["total_h"] < BUDGET_GO_H:
        return {"run_config": base, "estimate": est0, "decision": "go", "degradations": []}
    cfg = dict(base)
    degr = []
    ladder = [
        ("gammas", [g for g in GAMMAS if g != 0.02], "drop_gamma_0.02"),
        ("eval_seeds", 12, "reduce_eval_seeds_20_to_12"),
        ("etas", [e for e in ETAS if abs(e - 3 * math.pi / 8) > 1e-9], "drop_eta_3pi8"),
    ]
    est = est0
    for key, val, reason in ladder:
        if est["total_h"] < BUDGET_GO_H:
            break
        cfg[key] = val
        est = estimate_hours(costs, cfg, washouts)
        degr.append({"cut": reason, "new_total_h": est["total_h"]})
        record_failure("budget_degradation", reason, before_h=est0["total_h"], after_h=est["total_h"])
        log(f"degradation applied: {reason} -> {est['total_h']:.2f} h")
    dec = "go" if est["total_h"] <= BUDGET_ABORT_H else "abort"
    return {"run_config": cfg, "estimate": est, "estimate_base": est0, "decision": dec, "degradations": degr}


def write_aborted(reason: str, details: Dict) -> None:
    ABORTED_PATH.write_text("\n".join([
        "# RUN ABORTED", "", f"Aborted at {datetime.now().isoformat()} during Phase 0.", "",
        f"**Reason:** {reason}", "", "```json", json.dumps(details, indent=2, default=str), "```",
        "", "Delete this file to retry after fixing the cause."]), encoding="utf-8")
    log(f"ABORTED: {reason}")


def _infra_repro_tau() -> int:
    """Reproduce v3's d_B=16 AB-embedded horizon at gamma=0.1 (eta/omega from v3)."""
    slices = v2.split_slices(1000, CFG.paper_train, CFG.paper_test)
    length = 1000 + CFG.paper_train + CFG.paper_test
    curves = []
    for sd in V3_REPRO_SEEDS:
        seq = v2.iid_inputs(sd, length)
        grid = v2.build_channel_grid_gpu(sd, N_A)
        m = make_ab(V3_REPRO_OMEGA, V3_REPRO_ETA)
        feats = drive_feats(m, seq, grid, lambda mm: mm.features("A"), None)
        curves.append([cap_from_feats(feats, seq, v2.stm_target(seq, t), slices) for t in TAUS])
    mc = np.array(curves).mean(axis=0)
    idx = np.where(mc > MAIN_THR)[0]
    return int(TAUS[int(idx.max())]) if len(idx) else -1


def phase0_gate() -> Dict:
    if marker("setup").exists() and BUDGET_PATH.exists():
        b = json.loads(BUDGET_PATH.read_text())
        log(f"Phase 0 already passed; run_config={b['run_config']}")
        return b

    v2.require_gpu(verbose=True)

    # sanity per gamma
    sanity = {f"gamma_{g}": sanity_state(g) for g in GAMMAS}

    # ESP at worst case gamma=0.02 (before generating all caches)
    set_gamma(0.02)
    v2.build_channel_grid_np(0, N_A)
    w002, conv002 = esp_washout(0.02, ETA_REF)
    esp002 = {"washout": w002, "conv": conv002, "violated": w002 is None}
    if w002 is None:
        decision("gamma=0.02 ESP", f"no convergence in {ESP_MAX_STEPS} steps; gamma=0.02 will be skipped per rule 4")

    costs = measure_costs()

    # washouts for the budget estimate (measure once per gamma at ref eta)
    washouts: Dict[float, int] = {}
    for g in GAMMAS:
        if g == 0.02 and w002 is None:
            continue
        set_gamma(g)
        v2.build_channel_grid_np(0, N_A)
        w, _ = esp_washout(g, ETA_REF)
        washouts[g] = w if w is not None else 5000
    budget = decide_budget(costs, washouts)
    run_config = budget["run_config"]
    if w002 is None and 0.02 in run_config["gammas"]:
        run_config["gammas"] = [g for g in run_config["gammas"] if g != 0.02]

    # Infrastructure-reproduction gate. The spec ties this to (gamma=0.1, eta=pi/4)
    # expecting tau~21, but eta=pi/4 measures ~15; v3's tau=21 uses eta=1.339,
    # omega=0.07. The gate's INTENT is "new infra reproduces old infra", so we test
    # v3's ACTUAL d_B=16 config (documented deviation; the specified sweeps are
    # unchanged and eta=pi/4 is still evaluated as a normal data point).
    decision("Reference gate eta correction",
             "spec's reference (gamma=0.1, eta=pi/4) gives tau_mem~15, not 21; v3's tau=21 "
             f"uses eta={V3_REPRO_ETA:.4f}, omega={V3_REPRO_OMEGA:.4f}. Reproduction gate run "
             "against v3's real config; sweeps kept as specified (eta in {pi/8,pi/4,3pi/8}).")
    set_gamma(GAMMA_REF)
    ensure_gamma_grids(GAMMA_REF, sorted(set(EVAL_SEEDS) | set(TUNE_SEEDS) | {0}))
    ref_tau = _infra_repro_tau()
    ref_ok = REF_HORIZON_LO <= ref_tau <= REF_HORIZON_HI
    log(f"infra reproduction (v3 d_B=16 config): tau_mem={ref_tau} "
        f"(window [{REF_HORIZON_LO},{REF_HORIZON_HI}]) -> {'OK' if ref_ok else 'FAIL'}")

    record = {
        "generated_at": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "sanity": sanity, "esp_gamma002": esp002, "washouts": washouts,
        "seconds_per_step": costs, "estimate": budget["estimate"],
        "estimate_base": budget.get("estimate_base", budget["estimate"]),
        "degradations": budget["degradations"], "run_config": run_config,
        "infra_repro_tau_mem": ref_tau, "infra_repro_config": {"eta": V3_REPRO_ETA, "omega": V3_REPRO_OMEGA},
        "reference_window": [REF_HORIZON_LO, REF_HORIZON_HI],
        "thresholds": {"go_h": BUDGET_GO_H, "abort_h": BUDGET_ABORT_H},
    }

    reasons = []
    if not all(s["passed"] for s in sanity.values()):
        reasons.append("state sanity failed for some gamma")
    if not ref_ok:
        reasons.append(f"infra-repro tau_mem={ref_tau} outside [{REF_HORIZON_LO},{REF_HORIZON_HI}] "
                       "(new infrastructure does not reproduce v3)")
    if budget["decision"] == "abort":
        reasons.append(f"measured budget {budget['estimate']['total_h']:.2f} h exceeds {BUDGET_ABORT_H} h")
    if reasons:
        record["aborted"] = True
        record["abort_reasons"] = reasons
        write_json(BUDGET_PATH, record)
        write_aborted("; ".join(reasons), record)
        raise AbortRun("; ".join(reasons))

    write_json(BUDGET_PATH, record)
    write_marker("setup", run_config=run_config, estimate=budget["estimate"], reference_tau_mem=ref_tau)
    heartbeat("phase0_gate", 1.0, extra=f"budget {budget['estimate']['total_h']:.2f}h, ref_tau={ref_tau}", force=True)
    log(f"Phase 0 PASSED. run_config={run_config} estimate={budget['estimate']['total_h']:.2f}h ref_tau={ref_tau}")
    return record


# ===========================================================================
# Horizons + scaling fits
# ===========================================================================
def _per_seed_curves(config_id: str, model_tag: str) -> Dict[int, np.ndarray]:
    df = load_csv(STM_CSV)
    g = df[(df.config == config_id) & (df.model == model_tag)]
    out = {}
    for seed, gs in g.groupby("seed"):
        s = gs.set_index("tau")["capacity"].sort_index()
        out[int(seed)] = s.reindex(TAUS).values
    return out


def _boot_horizon(curves: Dict[int, np.ndarray], threshold: float) -> Tuple[float, float, float]:
    seeds = sorted(curves)
    mat = np.array([curves[s] for s in seeds])  # (nseed, ntau)
    point = _hz(mat.mean(axis=0), threshold)
    rng = np.random.default_rng(20260706)
    vals = []
    n = len(seeds)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        vals.append(_hz(mat[idx].mean(axis=0), threshold))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def _hz(curve: np.ndarray, threshold: float) -> float:
    idx = np.where(curve > threshold)[0]
    return float(TAUS[int(idx.max())]) if len(idx) else -1.0


def build_horizons() -> pd.DataFrame:
    rows = []
    state = load_state()
    df = load_csv(STM_CSV)
    if df.empty:
        return pd.DataFrame()
    for config_id in sorted(df.config.unique()):
        model_tag = "M0" if config_id.startswith("M0_") else "AB-embedded"
        curves = _per_seed_curves(config_id, model_tag)
        if not curves:
            continue
        sub = df[df.config == config_id].iloc[0]
        for thr in THRESHOLDS:
            pt, lo, hi = _boot_horizon(curves, thr)
            rows.append({"config": config_id, "model": model_tag,
                         "gamma": float(sub.gamma), "eta": float(sub.eta),
                         "threshold": thr, "tau_mem": pt, "ci95_lo": lo, "ci95_hi": hi,
                         "n_seeds": len(curves)})
    out = pd.DataFrame(rows)
    out.to_csv(HORIZONS_CSV, index=False)
    return out


def _loglog_fit(x: np.ndarray, y: np.ndarray) -> Dict:
    ok = (x > 0) & (y > 0)
    if ok.sum() < 2:
        return {"exponent": None, "intercept": None, "r2": None, "n": int(ok.sum())}
    lx, ly = np.log(x[ok]), np.log(y[ok])
    b, a = np.polyfit(lx, ly, 1)
    pred = a + b * lx
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"exponent": float(b), "intercept": float(a), "r2": float(r2), "n": int(ok.sum())}


def build_fits() -> Dict:
    fits: Dict = {}
    df = load_csv(STM_CSV)
    if df.empty:
        write_json(FITS_JSON, {"error": "no data"})
        return {}
    gammas = sorted({float(g) for g in df[df.model == "AB-embedded"].gamma.unique()})
    gam_sweep = [g for g in gammas]  # all gammas present at eta=pi/4
    # per-gamma reference-eta horizons (point + per-seed for bootstrap of the slope)
    tau_by_gamma: Dict[float, float] = {}
    fm_by_gamma: Dict[float, float] = {}
    curves_ab: Dict[float, Dict[int, np.ndarray]] = {}
    curves_m0: Dict[float, Dict[int, np.ndarray]] = {}
    for g in gam_sweep:
        cid = cfg_id(g, ETA_REF)
        cab = _per_seed_curves(cid, "AB-embedded")
        cm0 = _per_seed_curves(f"M0_g{g}", "M0")
        if not cab:
            continue
        curves_ab[g] = cab
        curves_m0[g] = cm0
        tau_by_gamma[g] = _hz(np.array(list(cab.values())).mean(axis=0), MAIN_THR)
        fm_by_gamma[g] = _hz(np.array(list(cm0.values())).mean(axis=0), MAIN_THR) if cm0 else float("nan")

    gs = np.array(sorted(tau_by_gamma))
    inv_g = 1.0 / gs
    tau_arr = np.array([tau_by_gamma[g] for g in gs])
    fm_arr = np.array([fm_by_gamma[g] for g in gs])

    # (a) tau_mem ~ 1/gamma
    fit_a = _loglog_fit(inv_g, tau_arr)
    # (b) excess (tau_mem - tau_FM) ~ 1/gamma
    excess = tau_arr - fm_arr
    fit_b = _loglog_fit(inv_g, excess)

    # bootstrap slope CIs over seeds
    def boot_slope(curves_map, subtract_m0):
        rng = np.random.default_rng(20260706)
        slopes = []
        gg = sorted(curves_map)
        seed_lists = {g: sorted(curves_map[g]) for g in gg}
        for _ in range(N_BOOT):
            xs, ys = [], []
            for g in gg:
                sl = seed_lists[g]
                idx = rng.integers(0, len(sl), size=len(sl))
                mat = np.array([curves_map[g][sl[i]] for i in idx])
                tm = _hz(mat.mean(axis=0), MAIN_THR)
                if subtract_m0 and g in curves_m0 and curves_m0[g]:
                    sl0 = sorted(curves_m0[g])
                    idx0 = rng.integers(0, len(sl0), size=len(sl0))
                    mat0 = np.array([curves_m0[g][sl0[i]] for i in idx0])
                    tm = tm - _hz(mat0.mean(axis=0), MAIN_THR)
                if tm > 0:
                    xs.append(1.0 / g); ys.append(tm)
            if len(xs) >= 2:
                b, _a = np.polyfit(np.log(xs), np.log(ys), 1)
                slopes.append(float(b))
        if not slopes:
            return None, None
        return float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))

    a_lo, a_hi = boot_slope(curves_ab, subtract_m0=False)
    b_lo, b_hi = boot_slope(curves_ab, subtract_m0=True)
    fit_a.update({"exp_ci95_lo": a_lo, "exp_ci95_hi": a_hi})
    fit_b.update({"exp_ci95_lo": b_lo, "exp_ci95_hi": b_hi})

    # (c) tau_mem vs eta at gamma=0.1
    eta_points = []
    for e in ETAS:
        cid = cfg_id(GAMMA_REF, e)
        c = _per_seed_curves(cid, "AB-embedded")
        if c:
            eta_points.append({"eta": e, "eta_label": ETA_LABELS[e],
                               "tau_mem": _hz(np.array(list(c.values())).mean(axis=0), MAIN_THR)})
    eta_taus = [p["tau_mem"] for p in eta_points]
    if len(eta_taus) >= 3:
        if eta_taus[1] >= eta_taus[0] and eta_taus[1] >= eta_taus[2] and (eta_taus[1] > eta_taus[0] or eta_taus[1] > eta_taus[2]):
            eta_trend = "interior optimum near eta=pi/4"
        elif eta_taus[0] <= eta_taus[1] <= eta_taus[2]:
            eta_trend = "increases with eta (slower swap retains more)"
        elif eta_taus[0] >= eta_taus[1] >= eta_taus[2]:
            eta_trend = "decreases with eta"
        else:
            eta_trend = "non-monotonic / no clear trend"
    else:
        eta_trend = "insufficient eta points"

    # (d) consistency with v3 (dimension) and v4 (topology), all at gamma=0.1
    consistency = {"reference_tau_mem_gamma01": tau_by_gamma.get(GAMMA_REF)}
    try:
        v3 = json.loads((V3_DIR / "aux_dimension_summary.json").read_text())
        consistency["v3_dimension_memory_range"] = {k: v.get("memory_range_tau")
                                                    for k, v in v3.get("per_d_B", {}).items()}
    except Exception:  # noqa: BLE001
        consistency["v3_dimension_memory_range"] = None
    try:
        v4 = json.loads((V4_DIR / "topology_verdict.json").read_text())
        consistency["v4_topology_memory_range"] = v4.get("memory_range")
    except Exception:  # noqa: BLE001
        consistency["v4_topology_memory_range"] = None
    ref = tau_by_gamma.get(GAMMA_REF)
    if ref is not None:
        allpts = []
        d3 = consistency.get("v3_dimension_memory_range") or {}
        allpts += [v for v in d3.values() if isinstance(v, (int, float)) and v > 0]
        d4 = consistency.get("v4_topology_memory_range") or {}
        allpts += [v for v in d4.values() if isinstance(v, (int, float)) and v > 0]
        consistency["all_gamma01_points"] = allpts
        consistency["consistent"] = bool(allpts) and all(abs(v - ref) <= 5 for v in allpts)

    fits = {
        "gamma_sweep": {"gammas": list(gs), "tau_mem": list(tau_arr), "tau_FM": list(fm_arr),
                        "excess": list(excess)},
        "fit_a_tau_vs_inv_gamma": fit_a,
        "fit_b_excess_vs_inv_gamma": fit_b,
        "eta_sweep": {"points": eta_points, "trend": eta_trend},
        "consistency_v3_v4": consistency,
    }
    write_json(FITS_JSON, fits)
    return fits


# ===========================================================================
# SUMMARY
# ===========================================================================
def write_summary() -> None:
    hz = build_horizons()
    fits = build_fits()
    lines = ["# Exp 7 — dynamical scaling of the memory horizon (v5)", "",
             f"_Generated {datetime.now().isoformat()}_", ""]
    if BUDGET_PATH.exists():
        b = json.loads(BUDGET_PATH.read_text())
        lines += [f"Run config: `{json.dumps(b.get('run_config', {}))}` | "
                  f"budget: {b.get('estimate', {}).get('total_h', 'n/a')} h | "
                  f"reference tau_mem={b.get('infra_repro_tau_mem')}", ""]  # P1 fix: correct key

    # Horizons table (threshold 0.1)
    lines += ["## Horizons (threshold C>0.1)", "",
              "| config | gamma | eta | model | tau_mem | 95% CI |", "|---|---|---|---|---|---|"]
    if not hz.empty:
        h1 = hz[hz.threshold == MAIN_THR].sort_values(["model", "gamma", "eta"])
        for _, r in h1.iterrows():
            eta_lab = ETA_LABELS.get(r["eta"], f"{r['eta']:.3f}")
            lines.append(f"| {r['config']} | {r['gamma']} | {eta_lab} | {r['model']} | "
                         f"{r['tau_mem']:.0f} | [{r['ci95_lo']:.0f}, {r['ci95_hi']:.0f}] |")

    # Fits
    fa = fits.get("fit_a_tau_vs_inv_gamma", {})
    fb = fits.get("fit_b_excess_vs_inv_gamma", {})
    lines += ["", "## Scaling fits", ""]
    if fa.get("exponent") is not None:
        lines += [f"- **(a) tau_mem ~ (1/gamma)^x**: x = {fa['exponent']:.3f} "
                  f"[95% CI {fa.get('exp_ci95_lo')}, {fa.get('exp_ci95_hi')}], R²={fa.get('r2'):.3f}."]
    if fb.get("exponent") is not None:
        lines += [f"- **(b) (tau_mem − tau_FM) ~ (1/gamma)^x**: x = {fb['exponent']:.3f} "
                  f"[95% CI {fb.get('exp_ci95_lo')}, {fb.get('exp_ci95_hi')}], R²={fb.get('r2'):.3f}."]
    es = fits.get("eta_sweep", {})
    lines += [f"- **(c) eta sweep (gamma=0.1)**: {es.get('trend')}. "
              f"points: {[(p['eta_label'], p['tau_mem']) for p in es.get('points', [])]}"]
    gsw = fits.get("gamma_sweep", {})
    if gsw:
        lines += ["", "gamma sweep (eta=pi/4): "
                  + ", ".join(f"gamma={g}: tau_mem={t:.0f} (M0 {f:.0f})"
                              for g, t, f in zip(gsw["gammas"], gsw["tau_mem"], gsw["tau_FM"]))]

    # Verdict
    lines += ["", "## Verdict", ""]
    if fa.get("exponent") is not None and fa.get("r2", 0) is not None and fa.get("r2", 0) >= 0.9:
        lines.append(f"**The horizon tracks 1/gamma^x with x={fa['exponent']:.2f} "
                     f"[{fa.get('exp_ci95_lo')}, {fa.get('exp_ci95_hi')}] (R²={fa['r2']:.2f}) — "
                     f"the memory limit is dynamical: it moves with gamma as predicted.**")
    elif fa.get("exponent") is not None:
        lines.append(f"**The horizon moves with gamma (x={fa['exponent']:.2f}, R²={fa.get('r2')}) "
                     f"but the power law is not clean (R²<0.9).**")
    else:
        lines.append("**No clean scaling law could be fit.**")

    # Consistency
    cons = fits.get("consistency_v3_v4", {})
    lines += ["", "## Consistency with v3 (dimension) and v4 (topology), all gamma=0.1", "",
              f"Reference tau_mem(gamma=0.1) = {cons.get('reference_tau_mem_gamma01')}. "
              f"v3/v4 gamma=0.1 points = {cons.get('all_gamma01_points')}. "
              f"Consistent (within ±5): {cons.get('consistent')}."]

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
    rep = v2.validate_run(V5_DIR, expected_tables(EVAL_SEEDS))
    lines += ["", v2.completeness_markdown(rep)]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"summary written: {SUMMARY_PATH.name}")


def expected_tables(eval_seeds: Sequence[int]) -> Dict:
    """G2 fix: the planned (config x model x seed) cells the sweep must produce.
    AB-embedded is evaluated at every gamma with eta=pi/4, plus the eta sweep at
    the reference gamma; M0 is the noaux baseline per gamma. If any planned cell
    is missing/non-finite, validate_run marks the run partial (no summary_complete)."""
    combos: List[Dict] = []
    for g in GAMMAS:
        combos.append({"config": cfg_id(g, ETA_REF), "model": "AB-embedded"})
        combos.append({"config": f"M0_g{g}", "model": "M0"})
    for e in ETAS:
        if abs(e - ETA_REF) > 1e-12:
            combos.append({"config": cfg_id(GAMMA_REF, e), "model": "AB-embedded"})
    return {"tables": {
        STM_CSV.name: {"cell_combos": combos, "seed_col": "seed",
                       "seeds": list(eval_seeds), "value_cols": ["capacity"]},
    }}


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    v2.ensure_dirs()
    PROGRESS_PATH.touch(exist_ok=True)
    DECISIONS_PATH.touch(exist_ok=True)
    if ABORTED_PATH.exists():
        log(f"ABORTED.md present; refusing to run. Delete it to retry.")
        print(ABORTED_PATH.read_text())
        return
    log("========== qrc_experiments_scaling start ==========")
    decision("ESP washout omega", f"ESP measured at omega={ESP_OMEGA} (grid minimum, worst case for "
             "washout); tuned omega is >= this, so the adaptive washout is conservative.")
    heartbeat("start", 0.0, force=True)

    record = phase0_gate()
    run_config = record["run_config"]
    costs = record["seconds_per_step"]
    eval_seeds = EVAL_SEEDS[: run_config["eval_seeds"]]
    if run_config["eval_seeds"] < len(EVAL_SEEDS):
        record_failure("eval", "reduced_eval_seeds_budget", executed=run_config["eval_seeds"])

    for gamma in run_config["gammas"]:
        etas = run_config["etas"] if abs(gamma - GAMMA_REF) < 1e-12 else [ETA_REF]
        try:
            process_gamma(gamma, etas, eval_seeds, costs)
        except AbortRun:
            raise
        except Exception as exc:  # noqa: BLE001
            record_failure(f"gamma_{gamma}", "gamma_processing_exception", detail=repr(exc))
            log(f"ERROR processing gamma={gamma}: {exc!r}")
        heartbeat("sweep", 0.9, extra=f"gamma={gamma} done", force=True)

    write_summary()
    # G2/M2 fix: gate the completion marker through validate_run. If any planned
    # (config x model x seed) cell is missing/non-finite (e.g. gamma=0.2 lost to a
    # watchdog abort), this writes summary_partial.json instead of _complete.json.
    report = v2.write_validated_completion(V5_DIR, "summary", expected_tables(eval_seeds),
                                           run_config=run_config)
    if report["status"] != "complete":
        log(f"RUN PARTIAL: {report['n_missing']} missing / {report['n_nonfinite']} non-finite cells; "
            f"wrote summary_partial.json (see completeness_matrix.csv)")
    heartbeat("done", 1.0, force=True)
    log(f"========== qrc_experiments_scaling {report['status']} ==========")


if __name__ == "__main__":
    try:
        main()
    except AbortRun as exc:
        print(f"\n*** RUN ABORTED (Phase 0 gate): {exc}\nSee {ABORTED_PATH}\n")
        raise SystemExit(3)
