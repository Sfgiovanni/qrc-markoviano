"""C2 finish: rerun the two units that didn't complete cleanly the first time:
narma seed1+seed8 (schema fixed) and shot_noise_mackey seed0 (EXP4_MG_CSV
redirect fixed). Isolated; originals read-only.
"""
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import embedded_effective_qrc_pipeline_v2 as v2  # noqa: E402
import extra_experiments_v4 as v4  # noqa: E402
import extra_experiments_v5 as v5  # noqa: E402

CFG = v2.CFG
C2 = Path(ROOT) / "results_corrections_v6" / "c2_recovery"
v2.RESULTS_DIR = C2
v2.FIGURES_DIR = C2 / "figures"
v2.LOG_PATH = C2 / "run.log"
V4_DIR = Path(ROOT) / "results_extra_v4"


def narma():
    v5.set_gamma(0.1)
    out = C2 / "narma10_results_recovered.csv"
    orig = pd.read_csv(V4_DIR / "narma10_results.csv")
    finite = orig[orig.nmse.notna()].copy()
    finite.to_csv(out, index=False)
    slices = v2.split_slices(CFG.paper_washout, CFG.paper_train, CFG.paper_test)
    targets = [(m, 1) for m in v4.EXP5_MODELS] + [("AB-embedded", 8)]
    for model, seed in targets:
        u, target, used_seed, remap = v4.narma10_target_for_seed(seed)
        grid = v2.get_grid(model, seed)
        m = v4.make(model, seed)
        feats = v4.drive(m, u, grid, v4.feat_fn_for(model), None)
        met = v4.io_metrics(feats, target, slices)
        if not v2.all_finite(met, keys=("nmse", "nrmse", "r2")):
            print(f"  nonfinite {model} seed{seed}")
            continue
        v4.append_rows(out, [{"model": model, "seed": seed, "task": "NARMA10",
                              "nmse": met["nmse"], "nrmse": met["nrmse"], "r2": met["r2"]}])
        if remap:
            print(f"  narma seed{seed} remapped -> {used_seed}")
    df = pd.read_csv(out)
    print(f"NARMA recovered: {df.seed.nunique()} seeds, {len(df)} rows, all finite={bool(df.nmse.notna().all())}")


def shot_noise_mackey():
    v5.set_gamma(0.1)
    v4.EXP4_MG_CSV = C2 / "shot_noise_mackey_seed0.csv"
    v4.exp4_mackey("AB-embedded", 0, [100.0, 1000.0, 10000.0, 100000.0, None], None)
    df = pd.read_csv(v4.EXP4_MG_CSV)
    print(f"shot_noise_mackey seed0: {len(df)} rows, n_shots={sorted(df.n_shots.unique())}")


if __name__ == "__main__":
    narma()
    shot_noise_mackey()
    print("C2 finish done")
