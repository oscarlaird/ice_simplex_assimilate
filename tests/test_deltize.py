import numpy as np
from hypothesis import given

from dirichlet_assimilate import deltize

from tests import custom_strategies

@given(custom_strategies.sample_w_h_bounds_strategy())
def test_inversion_possible(sample_w_h_bounds):
    n, h_bnd, sample = sample_w_h_bounds

    raw_sample = deltize.post_process_sample(sample, h_bnd)
    # legal area -- area vector lies on the simplex
    assert np.isclose(raw_sample.area.sum(), 1) or raw_sample.area.sum() < 1
    assert np.all(raw_sample.area >= 0)
    # legal ice volumes in each category
    assert np.all(np.logical_and(h_bnd[:-1] * raw_sample.area <= raw_sample.volume,
                                 raw_sample.volume <= h_bnd[1:] * raw_sample.area)), (h_bnd, raw_sample.area, raw_sample.volume)
    assert np.all(raw_sample.volume >= 0)

@given(custom_strategies.sample_w_h_bounds_strategy())
def test_process_inverts_post_process(sample_w_h_bounds):
    n, h_bnd, sample = sample_w_h_bounds
    raw_sample = deltize.post_process_sample(sample, h_bnd)
    after_sample = deltize.process_sample(raw_sample, h_bnd)
    assert np.allclose(sample, after_sample, atol=1e-7), f'orig_sample: {sample}\nafter_sample: {after_sample}'