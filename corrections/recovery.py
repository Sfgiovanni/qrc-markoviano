"""C2 (M2): recover watchdog/NaN-lost seeds, integrated ONLY into v6 CSVs.

Units: v5 g0.02 AB-embedded seed15; v4 narma AB-embedded seed8 + narma seed1
(regenerated via A2, 6 models); v4 exp4 (shot noise) AB-embedded seed0; v4 exp6
parallel seed12. Each unit is isolated in try/except; a re-failure is recorded
and the run continues (n_effective reports reality). Reads originals read-only.
"""
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import qrc_pipeline as v2  # noqa: E402
import qrc_experiments_robustness as v4  # noqa: E402
import qrc_experiments_scaling as v5  # noqa: E402

CFG = v2.CFG
V6 = Path(ROOT) / "results_corrections_v6"
C2 = V6 / "c2_recovery"
(C2 / "channel_cache").mkdir(parents=True, exist_ok=True)
v2.RESULTS_DIR = C2
v2.FIGURES_DIR = C2 / "figures"
v2.LOG_PATH = C2 / "run.log"
v2.ensure_dirs()

V4_DIR = Path(ROOT) / "results_extra_v4"
V5_DIR = Path(ROOT) / "results_extra_v5"
STATUS = {}


def copy_channel(src_cache, seed, gamma):
    tag = f"channel_N4_seed{seed}_g{CFG.grid_size}_dt{CFG.dt}_gamma{gamma}_range{CFG.grid_s_min}_{CFG.grid_s_max}.npz"
    src, dst = Path(src_cache) / tag, C2 / "channel_cache" / tag
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
    return dst.exists()


def unit(name):
    def deco(fn):
        try:
            fn()
            STATUS[name] = "recovered"
            print(f"[OK] {name}")
        except Exception as exc:  # noqa: BLE001
            STATUS[name] = f"failed: {exc!r}"
            v2.record_failure(f"c2/{name}", "recovery_failed", detail=repr(exc))
            print(f"[FAIL] {name}: {exc!r}")
            traceback.print_exc()
    return deco


def recover_v5_seed15():
    for sd in (15,):
        copy_channel(V5_DIR / "channel_cache", sd, 0.02)
    v5.STM_CSV = C2 / "v5_g0.02_seed15_stm.csv"
    v5.set_gamma(0.02)
    v5.eval_curve("g0.02_epi4", 0.02, v5.ETA_REF, 0.1, "AB-embedded", 1000, [15], None)
    df = pd.read_csv(v5.STM_CSV)
    assert (df.seed == 15).any(), "seed15 not written"


def recover_narma():
    for sd in (0, 1, 8):
        copy_channel(V4_DIR / "channel_cache", sd, 0.1)
    v5.set_gamma(0.1)
    out = C2 / "narma10_results_recovered.csv"
    orig = pd.read_csv(V4_DIR / "narma10_results.csv")
    finite = orig[orig.nmse.notna()].copy()          # drop the 6 seed=1 NaN rows
    finite.to_csv(out, index=False)
    v4.EXP5_NARMA_CSV = out
    slices = v2.split_slices(CFG.paper_washout, CFG.paper_train, CFG.paper_test)
    targets = [(m, 1) for m in v4.EXP5_MODELS] + [("AB-embedded", 8)]
    for model, seed in targets:
        u, target, used_seed, remap = v4.narma10_target_for_seed(seed)
        grid = v2.get_grid(model, seed)
        m = v4.make(model, seed)
        feats = v4.drive(m, u, grid, v4.feat_fn_for(model), None)
        met = v4.io_metrics(feats, target, slices)
        if not v2.all_finite(met, keys=("nmse", "nrmse", "r2")):
            v2.record_failure(f"c2/narma/{model}/seed{seed}", "nonfinite_metric")
            continue
        # Keep the 6-column schema identical to the pre-seeded rows (append_rows
        # does not align columns). Remaps are logged, not added as extra columns.
        row = {"model": model, "seed": seed, "task": "NARMA10",
               "nmse": met["nmse"], "nrmse": met["nrmse"], "r2": met["r2"]}
        if remap:
            v2.record_failure(f"c2/narma/{model}/seed{seed}", "narma_seed_remapped",
                              used_seed=used_seed, **remap)
        v4.append_rows(out, [row])
    df = pd.read_csv(out)
    print(f"  narma recovered: {df.seed.nunique()} seeds, {len(df)} rows, "
          f"all finite={bool(df.nmse.notna().all())}")


def recover_shot_noise_seed0():
    copy_channel(V4_DIR / "channel_cache", 0, 0.1)
    v5.set_gamma(0.1)
    v4.EXP4_CAP_CSV = C2 / "shot_noise_capacities_seed0.csv"
    v4.EXP4_MG_CSV = C2 / "shot_noise_mackey_seed0.csv"   # NOTE: real global is EXP4_MG_CSV
    n_list = [100.0, 1000.0, 10000.0, 100000.0, None]
    v4.exp4_capacities("AB-embedded", 0, n_list, None)
    v4.exp4_mackey("AB-embedded", 0, n_list, None)
    cap = pd.read_csv(v4.EXP4_CAP_CSV)
    print(f"  shot_noise seed0: capacities rows={len(cap)}, n_shots={sorted(cap.n_shots.unique())}")


def recover_topology_seed12():
    copy_channel(V4_DIR / "channel_cache", 12, 0.1)
    v5.set_gamma(0.1)
    v4.EXP6_CSV = C2 / "topology_control_seed12.csv"
    best = {"eta_ab": 1.3240, "omega": 0.0581}   # argmax of exp6_parallel_tuning value
    v4.exp6_eval_seed(12, best, None)
    df = pd.read_csv(v4.EXP6_CSV)
    assert (df.seed == 12).any(), "seed12 not written"
    print(f"  topology seed12: rows={len(df)}")


def main():
    unit("v5_g0.02_seed15")(recover_v5_seed15)
    unit("v4_narma_seed1_seed8")(recover_narma)
    unit("v4_shot_noise_seed0")(recover_shot_noise_seed0)
    unit("v4_topology_seed12")(recover_topology_seed12)
    lines = ["# C2 (M2) — recuperação de seeds perdidas", ""]
    for k, v in STATUS.items():
        lines.append(f"- **{k}**: {v}")
    (C2 / "c2_status.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
