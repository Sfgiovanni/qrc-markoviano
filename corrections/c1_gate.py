"""C1 gate: reproduce the gamma=0.1 reference horizon with the corrected code.

Runs v5._infra_repro_tau() (AB-embedded, V3-repro eta/omega, seeds 0-9, STM
tau 0..80, threshold 0.1) in an ISOLATED results dir so the immutable v5 dir is
never written. Passes iff tau_mem in [19,23] (recorded value: 21).
"""
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import embedded_effective_qrc_pipeline_v2 as v2  # noqa: E402
import extra_experiments_v5 as v5  # noqa: E402

C1_DIR = Path(ROOT) / "results_corrections_v6" / "c1_gamma02"
v2.RESULTS_DIR = C1_DIR
v2.FIGURES_DIR = C1_DIR / "figures"
v2.LOG_PATH = C1_DIR / "run.log"
v2.ensure_dirs()


def main():
    v5.set_gamma(v5.GAMMA_REF)  # 0.1; clears in-memory GPU cache
    ref_tau = v5._infra_repro_tau()
    lo, hi = v5.REF_HORIZON_LO, v5.REF_HORIZON_HI
    ok = lo <= ref_tau <= hi
    print(f"C1_GATE ref_tau={ref_tau} window=[{lo},{hi}] -> {'PASS' if ok else 'FAIL'}")
    with open(C1_DIR / "c1_gate_result.txt", "w") as fh:
        fh.write(f"ref_tau={ref_tau}\nwindow=[{lo},{hi}]\nresult={'PASS' if ok else 'FAIL'}\n")
    sys.exit(0 if ok else 7)


if __name__ == "__main__":
    main()
