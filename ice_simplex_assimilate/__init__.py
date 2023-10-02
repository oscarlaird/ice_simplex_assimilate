import numpy as np
from numpy.typing import NDArray

SIG_BITS = 31
ONE = np.uint32(2 ** SIG_BITS)
DELTA = np.uint32(1)

THRESHOLD = int(1e-7 * ONE)
OUTPUT_WARN_THRESHOLD = int(1e-9 * ONE)


class HeightBounds(np.ndarray):
    min_interval = 1e-7

    def __new__(cls, input_array):
        a = np.asarray(input_array).view(cls)
        assert np.all(a[1:] - a[:-1] >= cls.min_interval), f'Height bounds must be provided in sorted order' \
                                                           f'and spaced by more than {cls.min_interval}: {a}'
        assert a[0] == 0, f'Lowest height bound should be 0.0, not {a}'
        assert a.ndim == 1, f'Height bounds must be a vector, but ndim={a.ndim}'
        return a

    @property
    def intervals(self):
        return zip(self[:-1], self[1:])

    @classmethod
    def from_interval_widths(cls, intervals: np.ndarray):
        ''' Create height bounds from a vector of interval widths '''
        assert np.all(intervals >= cls.min_interval)  # check that all intervals are greater than the minimum
        a = np.cumsum(intervals)  # heights are the cumulative sum of the intervals
        a = np.insert(a, 0, 0)  # insert height of zero at the beginning
        return HeightBounds(a)

def pre_process_sample(area: NDArray[np.float32], volume: NDArray[np.float32], height_bounds: NDArray[np.float32]) -> NDArray[np.uint32]:
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
    3. Quantize to 32-bit fixed point
    4. Threshold by pair. Set each pair to zero if their sum is less than 2*threshold
    5. For each pair, move enough mass from the larger component to make the smaller component equal to the threshold
    """
    # 1. Assert that the area, volume, and height bounds have legal values (ah₁ ≤ v ≤ ah₂)
    # 2. Convert to deltized form using the height bounds.
    # TODO check that inputing zeros are okay
    assert len(area) == len(volume) == len(height_bounds) - 1
    assert np.all(area >= 0) and np.all(volume >= 0)
    height_bounds = HeightBounds(height_bounds)
    x = []
    for i, interval in enumerate(height_bounds.intervals):
        M = np.array([[1., 1., ],
                      interval])
        x += list(np.linalg.inv(M) @ np.array([area[i], volume[i]]))
        assert x[-1] >= 0 and x[-2] >= 0,  "Illegal area and volume causes negative mass in deltized form. Check ah₁ ≤ v ≤ ah₂."
    x.insert(0, 1 - sum(x))  # first component is open water. How much isn't covered in ice?
    x /= sum(x)  # normalize
    # 3. Quantize to 32-bit fixed point
    x = quantize(x)
    assert x.dtype == np.uint32
    # 4. Threshold by pair. Set each pair to zero if their sum is less than 2*threshold
    for i in range(1, len(x), 2):
        if x[i] + x[i+1] < 2*THRESHOLD:
            x[i] = x[i+1] = 0
    # 5. For each pair, move enough mass from the larger component to make the smaller component equal to the threshold
    for i in range(1, len(x), 2):
        if x[i] < THRESHOLD:
            x[i+1] -= THRESHOLD - x[i]
            x[i] = THRESHOLD
        elif x[i+1] < THRESHOLD:
            x[i] -= THRESHOLD - x[i+1]
            x[i+1] = THRESHOLD
    # 6. Assert that this quantized form still sums to 1
    assert x.sum() == ONE
    return x

def post_process_sample(sample: NDArray[np.uint32], height_bounds: NDArray[np.float32]) -> NDArray[np.float32], NDArray[np.float32]:
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
    # TODO input a height_bounds object
    # 1. Assert that the deltized sample sums to 1
    assert sample.sum() == ONE
    # 2. Warn if any component is less than 1e-9, but greater than zero.
    if np.any(0 < sample & sample < OUTPUT_WARN_THRESHOLD):
        print(f"Warning: output sample contains components less than 1e-9: {sample}")
    # 3. Dequantize to floating point
    x = sample.astype(np.float32) / ONE
    # 4. Convert to raw form
    area = np.zeros_like(height_bounds.intervals, dtype=np.float32)
    volume = np.zeros_like(height_bounds.intervals, dtype=np.float32)
    for i, interval in enumerate(height_bounds.intervals):
        # TODO audit
        M = np.array([[1., 1., ],
                      interval])
        area[i], volume[i] = M @ x[2*i:2*i+2]
    return area, volume

def post_process_ensemble(ensemble: NDArray[np.uint32], height_bounds: NDArray[np.float32]) -> NDArray[np.float32]:
    vectorized_pps = np.vectorize(post_process_sample, signature='(n),()->(m),(m)', excluded=['height_bounds'])
    return vectorized_pps(ensemble, height_bounds=height_bounds)


# CONVERT BACK TO RAW FORM
def post_process_sample(sample: NDArray[np.float64], h_bnd: HeightBounds) -> RawSample:
    assert len(h_bnd) * 2 - 1 == len(sample)
    l, r = sample[1::2], sample[2::2]  # delta size on left and right of each interval
    a = l + r
    v = h_bnd[:-1] * l + h_bnd[1:] * r
    return RawSample(area=np.array(a), volume=np.array(v)


def pre_process_samples(areas: NDArray[np.float32], volumes: NDArray[np.float32],  height_bounds: NDArray[np.float32]) -> NDArray[np.uint32]:
    """ Apply pre_process_sample to a batch of samples """
    vectorized_pps = np.vectorize(pre_process_sample, signature='(n),(n),()->(m)', excluded=['height_bounds'])
    # apply f as an aggregate function on axis 1
    return vectorized_pps(areas, volumes, height_bounds=height_bounds)


def quantize(float_samples: NDArray[np.float64]) -> NDArray[np.uint32]:
    # check inputs lie on the simplex to within tolerance DELTA
    assert float_samples.ndim == 2, "Samples must be a 2D array"
    assert np.all(float_samples >= 0), "Samples must be non-negative"
    assert np.all(1 - DELTA / ONE < float_samples.sum(axis=1)) and np.all(
        float_samples.sum(axis=1) < 1 + DELTA / ONE), f"Samples must sum to 1, to tolerance {DELTA / ONE}"
    # take cumulative sum, round to nearest quantized value, and take differences
    cumsum = np.cumsum(float_samples, axis=1)
    cumsum = np.insert(cumsum, 0, 0, axis=1)
    cumsum = np.round(cumsum * ONE).astype(np.uint32)  # multiply by ONE to convert from float to uint32
    samples = np.diff(cumsum, axis=1)
    if not np.all((samples > 0) == (float_samples > 0)):
        warnings.warn(f"Truncation performed in quantization. Inputs should be thresholded before quantization."
                      f"Recommended threshold is at least 10*Δ={DELTA / ONE * 10}, preferably greater.")
    assert (samples.sum(axis=1) == ONE).all(), "Samples do not sum to 1 after quantization"
    return samples
