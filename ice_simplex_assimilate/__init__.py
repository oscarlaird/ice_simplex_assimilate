import numpy as np
from numpy.typing import NDArray
import warnings

from simplex_assimilate.utils.quantize import quantize
from simplex_assimilate.fixed_point import ONE

OUTPUT_WARN_THRESHOLD = int(1e-9 * ONE)

class HeightBounds(np.ndarray):
    min_interval = 1e-7

    def __new__(cls, input_array):
        a = np.asarray(input_array).view(cls)
        assert np.all(a[1:] - a[:-1] >= cls.min_interval), f'Height bounds must be provided in sorted order' \
                                                           f'and spaced by more than {cls.min_interval}: {a}'
        assert a[0] == 0, f'Lowest height bound must be 0.0, not {a}'
        assert a.ndim == 1, f'Height bounds must be a vector, but ndim={a.ndim}'
        return a

    @property
    def intervals(self):
        return list(zip(self[:-1], self[1:]))

    @classmethod
    def from_interval_widths(cls, intervals: np.ndarray):
        ''' Create height bounds from a vector of interval widths '''
        assert np.all(intervals >= cls.min_interval)  # check that all intervals are greater than the minimum
        a = np.cumsum(intervals)  # heights are the cumulative sum of the intervals
        a = np.insert(a, 0, 0)  # insert height of zero at the beginning
        return HeightBounds(a)

def check_raw_sample_legal(area: NDArray[np.float32], volume: NDArray[np.float32], height_bounds: HeightBounds):
    """ Check that a raw sample of area and volume is legal i.e. ah₁ ≤ v ≤ ah₂ """
    assert len(area) == len(volume) == len(height_bounds.intervals)
    legal = True
    legal &= np.all(area >= 0) and np.all(volume >= 0)
    legal &= area.sum() <= 1
    for i, interval in enumerate(height_bounds.intervals):
        legal &= interval[0] * area[i] <= volume[i] <= interval[1] * area[i]
    return legal


def pre_process_sample(area: NDArray[np.float32], volume: NDArray[np.float32], height_bounds: HeightBounds, threshold=1e-6) -> NDArray[
    np.uint32]:
    """ Convert a raw sample of area and  volume to a deltized representation suitable for the simplex_assimilate package
    Input:
        -- area
        -- volume
        -- height bounds
    Output:
        -- deltized sample quantized to 32-bit fixed point
    Pipeline:
    1. Assert that the area, volume, and height bounds have legal values (ah₁ ≤ v ≤ ah₂)
    2. Convert to deltized form using the height bounds.
    3. Threshold by pair. Set each pair to zero if their sum is less than 2*threshold
    4. For each pair, move enough mass from the larger component to make the smaller component equal to the threshold
    5. Quantize to 32-bit fixed point
    """
    # 1. Assert that the area, volume, and height bounds have legal values (ah₁ ≤ v ≤ ah₂)
    assert check_raw_sample_legal(area, volume, height_bounds)
    # 2. Convert to deltized form using the height bounds.
    x = np.zeros(2 * len(height_bounds.intervals) + 1, dtype=np.float32)
    area_before = area.sum()
    for i, interval in enumerate(height_bounds.intervals):
        M = np.array([[1., 1., ],
                      interval])
        # M @ x = [area, volume]
        l, r = np.linalg.inv(M) @ np.array([area[i], volume[i]])
        # 3. Threshold by pair. Set each pair to zero if their sum is less than 2*threshold
        # 4. For each pair, move enough mass from the larger component to make the smaller component equal to the threshold
        if l + r < 2 * threshold:
            l = r = 0
        else:
            if l < threshold:
                r -= threshold - l
                l = threshold
            elif r < threshold:
                l -= threshold - r
                r = threshold
        x[2 * i + 1] = l
        x[2 * i + 2] = r
    x *= area_before / x.sum() # insure thresholding doesn't change the amount of ice_area
    x[0] = max(0, 1 - x.sum())  # first component is open water. How much isn't covered in ice?
    # 5. Quantize to 32-bit fixed point
    return quantize(x[None, :])[0]


def post_process_sample(sample: NDArray[np.uint32], height_bounds: HeightBounds) -> (
        NDArray[np.float32], NDArray[np.float32]):
    """ Convert a deltized sample to a raw sample of area and volume
    Input:
        -- deltized sample
        -- height bounds
    Output:
        -- area
        -- volume
    Pipeline:
    1. Assert that the deltized sample sums to 1
    2. Warn if any component is less than 1e-9, but greater than zero.
    3. Dequantize to floating point
    """
    # 1. Assert that the deltized sample sums to 1
    assert sample.sum() == ONE
    # 2. Warn if any component is less than 1e-9, but greater than zero.
    if np.any((0 < sample) & (sample < OUTPUT_WARN_THRESHOLD)):
        warnings.warn(f"Warning: output sample contains components less than 1e-9: {sample}")
    # 3. Dequantize to floating point
    x = sample.astype(np.float32) / ONE
    # 4. Convert to raw form
    area = np.zeros(len(height_bounds.intervals), dtype=np.float32)
    volume = np.zeros(len(height_bounds.intervals), dtype=np.float32)
    for i, interval in enumerate(height_bounds.intervals):
        l_mass = x[2 * i + 1]  # mass of lower component
        u_mass = x[2 * i + 2]  # mass of upper component
        M = np.array([[1., 1., ],
                      interval])
        area[i], volume[i] = M @ np.array([l_mass, u_mass])
    return area, volume


def pre_process_ensemble(areas: NDArray[np.float32], volumes: NDArray[np.float32], height_bounds: HeightBounds, threshold=1e-6) -> \
        NDArray[np.uint32]:
    """ Apply pre_process_sample to a batch of samples """
    vectorized_pps = np.vectorize(pre_process_sample, signature='(n),(n)->(m)', excluded=['height_bounds', 'threshold'])
    # apply f as an aggregate function on axis 1
    return vectorized_pps(areas, volumes, height_bounds=height_bounds, threshold=threshold)


def post_process_ensemble(ensemble: NDArray[np.uint32], height_bounds: HeightBounds) -> NDArray[np.float32]:
    vectorized_pps = np.vectorize(post_process_sample, signature='(n)->(m),(m)', excluded=['height_bounds'])
    return vectorized_pps(ensemble, height_bounds=height_bounds)

