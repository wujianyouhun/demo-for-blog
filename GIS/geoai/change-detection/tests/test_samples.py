"""Tests for training sample generation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_generate_synthetic_samples_counts_match(tmp_path):
    from cdd.sample_builder import generate_synthetic_samples, sample_counts

    result = generate_synthetic_samples(tmp_path, num_samples=4, tile_size=64)
    counts = sample_counts(tmp_path)

    assert result["mode"] == "synthetic"
    assert counts == {"time_a": 4, "time_b": 4, "labels": 4}
