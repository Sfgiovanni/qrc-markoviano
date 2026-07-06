"""A3 (M1) unit tests: mean_diff and ci95 must share the same sense.

Covers both paired_stats modes (larger_better True/False) via orient_effect.
Run: python3 tests_corrections/test_a3_paired_sign.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import embedded_effective_qrc_pipeline_v2 as v2


def _synthetic():
    # noaux errors clearly larger than embedded errors -> noaux-embedded > 0.
    rng = np.random.default_rng(0)
    embedded = rng.uniform(0.10, 0.20, size=30)
    noaux = embedded + rng.uniform(0.05, 0.15, size=30)  # always larger
    return noaux, embedded


def test_error_metric_mode_signs_agree():
    # Error metric: smaller is better -> larger_better=False, a=noaux, b=embedded.
    noaux, embedded = _synthetic()
    st = v2.paired_stats(noaux, embedded, larger_better=False)
    eff = v2.orient_effect(st, larger_better=False, report="a_minus_b")
    # noaux - embedded should be positive (noaux worse), CI both positive.
    assert eff["mean_diff"] > 0, eff
    assert eff["ci95_lo"] > 0 and eff["ci95_hi"] > 0, eff
    assert eff["ci95_lo"] <= eff["mean_diff"] <= eff["ci95_hi"], eff
    # mean_diff must equal mean_noaux - mean_embedded
    assert abs(eff["mean_diff"] - (st["mean_a"] - st["mean_b"])) < 1e-9
    # oriented wins = count where (noaux - embedded) > 0. noaux always larger -> 30.
    assert eff["wins"] == 30 and eff["losses"] == 0


def test_score_metric_mode_signs_agree():
    # Score metric: larger is better -> larger_better=True.
    rng = np.random.default_rng(1)
    a = rng.uniform(0.5, 0.6, size=25)          # method A scores
    b = a - rng.uniform(0.05, 0.1, size=25)     # method B always lower
    st = v2.paired_stats(a, b, larger_better=True)
    eff = v2.orient_effect(st, larger_better=True, report="a_minus_b")
    # a - b > 0, CI positive, and no flip should have occurred.
    assert eff["mean_diff"] > 0
    assert eff["ci95_lo"] > 0 and eff["ci95_hi"] > 0
    assert abs(eff["mean_diff"] - st["mean_diff"]) < 1e-12  # no flip
    assert abs(eff["ci95_lo"] - st["ci95_lo"]) < 1e-12


def test_flip_is_consistent_between_reports():
    noaux, embedded = _synthetic()
    st = v2.paired_stats(noaux, embedded, larger_better=False)
    a_minus_b = v2.orient_effect(st, larger_better=False, report="a_minus_b")
    b_minus_a = v2.orient_effect(st, larger_better=False, report="b_minus_a")
    assert abs(a_minus_b["mean_diff"] + b_minus_a["mean_diff"]) < 1e-9
    assert abs(a_minus_b["ci95_lo"] + b_minus_a["ci95_hi"]) < 1e-9
    assert abs(a_minus_b["ci95_hi"] + b_minus_a["ci95_lo"]) < 1e-9


def test_empty():
    eff = v2.orient_effect({"n": 0}, larger_better=False)
    assert np.isnan(eff["mean_diff"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} A3 tests passed.")
