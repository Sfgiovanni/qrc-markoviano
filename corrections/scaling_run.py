"""C1 (G2): gamma=0.2 AB-embedded reexecution in an isolated dir.

Reuses v5.process_gamma (ESP washout, M0 control, 9-omega tuning, AB-embedded
eval, STM tau 0..80) with all output paths redirected to
results_corrections_v6/c1_gamma02/. Never writes the immutable v5 dir.
M0_g0.2 (already computed in v5) is pre-seeded to save compute; gamma=0.2
channels are copied from the v5 cache. Watchdog disabled (costs={}) to avoid
spurious aborts on the shared GPU (conservative; logged).
"""
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import qrc_pipeline as v2  # noqa: E402
import qrc_experiments_scaling as v5  # noqa: E402

C1 = Path(ROOT) / "results_corrections_v6" / "c1_gamma02"
(C1 / "channel_cache").mkdir(parents=True, exist_ok=True)

# ---- redirect every v5 output path to the isolated dir --------------------
v2.RESULTS_DIR = C1
v2.FIGURES_DIR = C1 / "figures"
v2.LOG_PATH = C1 / "run.log"
v5.STM_CSV = C1 / "dynamical_sweep_stm.csv"
v5.HORIZONS_CSV = C1 / "horizons.csv"
v5.STATE_PATH = C1 / "config_state.json"
v5.TUNE_TRIALS_CSV = C1 / "omega_tuning_trials.csv"
v5.SUMMARY_PATH = C1 / "scaling_law_summary.md"
v5.PROGRESS_PATH = C1 / "progress_log.md"
v5.DECISIONS_PATH = C1 / "decisions_log.md"
v2.ensure_dirs()

V5_CACHE = Path(ROOT) / "results_extra_v5" / "channel_cache"
GAMMA = 0.2
SEEDS = list(range(20))


def copy_gamma02_channels():
    need = sorted(set(SEEDS) | set(v5.TUNE_SEEDS) | {0})
    tag = f"_g{v2.CFG.grid_size}_dt{v2.CFG.dt}_gamma{GAMMA}_range{v2.CFG.grid_s_min}_{v2.CFG.grid_s_max}.npz"
    n = 0
    for sd in need:
        name = f"channel_N{v5.N_A}_seed{sd}{tag}"
        src, dst = V5_CACHE / name, C1 / "channel_cache" / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            n += 1
    print(f"copied {n} gamma=0.2 channel grids from v5 cache")


def preseed_m0():
    """Copy v5's M0_g0.2 STM rows so process_gamma reuses them (tau_FM)."""
    src = Path(ROOT) / "results_extra_v5" / "dynamical_sweep_stm.csv"
    df = pd.read_csv(src)
    m0 = df[df.config == "M0_g0.2"]
    if len(m0):
        m0.to_csv(v5.STM_CSV, index=False)
        print(f"pre-seeded {len(m0)} M0_g0.2 STM rows ({m0.seed.nunique()} seeds)")


def main():
    copy_gamma02_channels()
    preseed_m0()
    v5.decision("C1 watchdog", "watchdog disabled (costs={}) to avoid spurious step-timeout "
                "aborts on the shared GPU; conservative for a re-execution")
    # process_gamma reads a state file; make sure it starts clean for gamma=0.2.
    v5.process_gamma(GAMMA, [v5.ETA_REF], SEEDS, costs={})
    # Report AB-embedded gamma=0.2 STM presence.
    cid = v5.cfg_id(GAMMA, v5.ETA_REF)
    df = pd.read_csv(v5.STM_CSV)
    ab = df[(df.config == cid) & (df.model == "AB-embedded")]
    print(f"DONE: {cid} AB-embedded seeds={sorted(ab.seed.unique())} ({ab.seed.nunique()}/20)")
    curve = ab.groupby("tau").capacity.mean().sort_index()
    tau_mem = v5.horizon(curve, v5.MAIN_THR)
    print(f"g0.2_epi4 AB-embedded tau_mem(thr=0.1) = {tau_mem}")
    with open(C1 / "c1_run_result.txt", "w") as fh:
        fh.write(f"config={cid}\nseeds={ab.seed.nunique()}\ntau_mem_thr0.1={tau_mem}\n")


if __name__ == "__main__":
    main()
