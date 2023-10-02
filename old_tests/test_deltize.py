import numpy as np

import pytest

from ice_simplex_assimilate.deltize import RawSample, HeightBounds, \
    process_sample, post_process_sample, process_ensemble, post_process_ensemble

@pytest.fixture
def h_bnd():
    return HeightBounds.from_interval_widths(np.array([1, 2, 3]))

def test_create_height_bounds_from_interval_widths(h_bnd):
    HeightBounds.from_interval_widths(np.array([1, 2, 3]))

def test_create_height_bounds():
    HeightBounds(np.array([0, 1, 3, 6]))

def test_create_raw_sample(h_bnd):
    area = np.array([0.1, 0.2, 0.3])
    volume = np.array([0.05, 0.4, 1.0])
    snow = None
    RawSample(area, volume, snow)

def test_process_sample(h_bnd):
    area = np.array([0.1, 0.2, 0.3])
    volume = np.array([0.05, 0.4, 1.0])
    raw_sample = RawSample(area, volume)
    sample = process_sample(raw_sample, h_bnd)
    assert np.isclose(1, sample.sum(), atol=1e-20)

def post_process_inverts_process(h_bnd):
    area = np.array([0.1, 0.2, 0.3])
    volume = np.array([0.05, 0.4, 1.0])
    raw_sample = RawSample(area, volume)
    sample = process_sample(raw_sample, h_bnd)
    post_sample = post_process_sample(sample, h_bnd)
    assert np.allclose(raw_sample.area, post_sample.area)
    assert np.allclose(raw_sample.volume, post_sample.volume)

