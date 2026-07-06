"""A2 (G3) unit tests: NARMA10 finiteness guard + single metric validator.

Run: python3 -m pytest tests_corrections/test_a2_narma_finite.py -q
 or: python3 tests_corrections/test_a2_narma_finite.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import embedded_effective_qrc_pipeline_v2 as v2
import extra_experiments_v4 as v4


def test_all_finite_dict_and_iter():
    assert v2.all_finite({"a": 1.0, "b": 2.0}) is True
    assert v2.all_finite({"a": 1.0, "b": np.nan}) is False
    assert v2.all_finite({"a": 1.0, "b": np.inf}) is False
    # keys selector ignores unlisted (possibly nan) fields
    assert v2.all_finite({"nrmse_tf": 0.1, "r2_tf": 0.9, "vpt": np.nan}, keys=("nrmse_tf", "r2_tf")) is True
    assert v2.all_finite([1.0, 2.0, 3.0]) is True
    assert v2.all_finite([1.0, np.nan]) is False
    assert v2.all_finite({"a": None}, keys=("a",)) is False


def test_narma_target_asserts_on_divergence():
    # A deliberately divergent input drives the recurrence to overflow.
    u = np.full(60, 1e6, dtype=np.float64)
    raised = False
    try:
        v4.narma10_target(u)
    except AssertionError:
        raised = True
    assert raised, "narma10_target must assert on non-finite output"


def test_narma_target_for_seed_returns_finite():
    # The review reports seed=1 diverging in the historical run. The guarded
    # generator must always return a finite target (directly or via remap).
    for seed in (0, 1, 2, 8):
        u, target, used_seed, remap = v4.narma10_target_for_seed(seed)
        assert np.isfinite(target).all(), f"seed {seed} target must be finite"
        assert np.isfinite(u).all()
        if used_seed != seed:
            assert remap.get("orig_seed") == seed and remap.get("used_seed") == used_seed


def test_narma_target_for_seed_raises_when_hopeless():
    # Monkeypatch narma_input to always produce a diverging series -> exhausts retries.
    orig = v4.narma_input
    try:
        v4.narma_input = lambda s: np.full(60, 1e6, dtype=np.float64)
        raised = False
        try:
            v4.narma10_target_for_seed(123, max_attempts=3)
        except ValueError:
            raised = True
        assert raised
    finally:
        v4.narma_input = orig


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} A2 tests passed.")
