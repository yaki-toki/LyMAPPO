"""P6 (obs 정규화) 단위 테스트 — encode_obs 가 큰 큐/HOL/Z 를 압축하는지.

    PYTHONDONTWRITEBYTECODE=1 python -B tests/test_encode_obs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.networks import encode_obs  # noqa: E402

N_STA, N_LINKS = 5, 3


def _obs(queue_len: float, hol_s: float, z: float) -> dict:
    return {
        "csi": np.full((N_STA, N_LINKS), 3.0, dtype=np.float32),
        "queue": np.full((N_STA, N_LINKS, 4), queue_len, dtype=np.float32),
        "hol_age": np.full((N_STA, N_LINKS), hol_s, dtype=np.float32),
        "cbr": np.zeros(N_LINKS, dtype=np.float32),
        "Z_99": np.full(N_STA, z, dtype=np.float32),
        "Z_99_9": np.full(N_STA, z, dtype=np.float32),
    }


def test_dim_includes_link_mask() -> None:
    vec = encode_obs(_obs(0.0, 0.0, 0.0))
    expected = (
        N_STA * N_LINKS + N_STA * N_LINKS * 4 + N_STA * N_LINKS
        + N_LINKS + N_STA * 2 + N_LINKS  # P7: link_mask
    )
    assert vec.shape == (expected,), vec.shape


def test_large_values_compressed() -> None:
    """binding/persist 에서 나오는 극단값 (큐 5000, HOL 180ms, Z 100)."""
    vec = encode_obs(_obs(5000.0, 0.18, 100.0))
    assert float(np.abs(vec).max()) < 50.0, float(np.abs(vec).max())


def test_zero_obs_zero_features_except_mask() -> None:
    vec = encode_obs(_obs(0.0, 0.0, 0.0))
    # csi (앞 15개) 와 link_mask fallback (뒤 3개=1) 제외 전부 0.
    assert np.allclose(vec[N_STA * N_LINKS:-N_LINKS], 0.0)
    assert np.allclose(vec[-N_LINKS:], 1.0)  # mask 없으면 대칭 (전부 1)


def test_link_mask_passthrough() -> None:
    obs = _obs(0.0, 0.0, 0.0)
    obs["link_mask"] = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    vec = encode_obs(obs)
    assert vec[-N_LINKS:].tolist() == [0.0, 1.0, 1.0]


def test_monotone_in_queue() -> None:
    small = encode_obs(_obs(1.0, 0.0, 0.0))
    large = encode_obs(_obs(1000.0, 0.0, 0.0))
    q_slice = slice(N_STA * N_LINKS, N_STA * N_LINKS + N_STA * N_LINKS * 4)
    assert (large[q_slice] > small[q_slice]).all()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
