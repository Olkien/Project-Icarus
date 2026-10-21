import numpy as np

from fly.bench import EPISODE_STEPS, FIELDS, append_csv, peak_memory_mb, random_network, run_bench


def test_random_network_shape_and_signs():
    net = random_network(1000, 20, seed=1)
    assert net.n == 1000
    assert 0.95 * 20_000 <= net.weights.nnz <= 20_000
    out_degree = np.diff(net.weights.tocsc().indptr)
    assert out_degree.max() <= 20
    pre_sign = np.sign(np.asarray(net.weights.sum(axis=0)).ravel())
    inhibitory = (pre_sign < 0).mean()
    assert 0.15 < inhibitory < 0.25
    assert net.weights.dtype == np.float32


def test_random_network_seeded():
    a = random_network(500, 10, seed=3)
    b = random_network(500, 10, seed=3)
    c = random_network(500, 10, seed=4)
    assert (a.weights != b.weights).nnz == 0
    assert (a.weights != c.weights).nnz > 0


def test_run_bench_reports_fields(tmp_path):
    row = run_bench(500, 20, ticks=2, warmup_ticks=1)
    assert set(row) == set(FIELDS)
    assert row["steps_per_s"] > 0
    assert row["episode_s"] > 0
    assert abs(row["episode_s"] - EPISODE_STEPS / row["steps_per_s"]) < 0.1
    assert row["mean_rate_hz"] > 0
    path = tmp_path / "bench.csv"
    append_csv(row, path)
    append_csv(row, path)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_peak_memory_positive():
    assert peak_memory_mb() > 0
