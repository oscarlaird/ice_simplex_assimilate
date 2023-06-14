import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List


@dataclass
class RawSample:
    area: np.ndarray
    volume: np.ndarray
    snow: np.ndarray = None

    def __post_init__(self):
        if self.snow is None:
            self.snow = np.zeros_like(self.area)
        if not len(self.area) == len(self.volume) == len(self.snow):
            raise ValueError('Area, Volume, and Snow vectors must have the same length.')

    def threshold(self, a=None, v=None, s=None):
        """ Set area, volume, or snow to zero if below the given threshold """
        self.area[self.area < a] = 0.
        self.volume[self.volume < v] = 0.
        if s:
            self.snow[self.snow < s] = 0.


class HeightBounds(np.ndarray):
    min_interval = 1e-7
    max_interval = 20.0

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


# PROCESS TO DELTIZED FORM
def process_sample(raw_sample: RawSample, h_bnd: HeightBounds) -> NDArray[np.float64]:
    assert len(raw_sample.area) + 1 == len(h_bnd)
    x = []
    for i, interval in enumerate(h_bnd.intervals):
        M = np.array([[1., 1., ],
                      interval])
        x += list(np.linalg.inv(M) @ np.array([raw_sample.area[i], raw_sample.volume[i]]))
    x.insert(0, 1 - sum(x))  # first component is open water. How much isn't covered in ice?
    x = np.array(x, dtype=np.float64)
    x /= x.sum()  # renormalize to one
    return x


def process_ensemble(raw_ensemble: List[RawSample], h_bnd: HeightBounds) -> NDArray[np.float64]:
    return np.array([process_sample(raw_sample, h_bnd) for raw_sample in raw_ensemble], dtype=np.float64)

def build_raw_ensemble(area: np.ndarray, volume: np.ndarray) -> List[RawSample]:
    assert area.shape == volume.shape
    return [RawSample(area=area, volume=volume) for area, volume in zip(area, volume)]


# CONVERT BACK TO RAW FORM
def post_process_sample(sample: NDArray[np.float64], h_bnd: HeightBounds) -> RawSample:
    assert len(h_bnd) * 2 - 1 == len(sample)
    l, r = sample[1::2], sample[2::2]  # delta size on left and right of each interval
    a = l + r
    v = h_bnd[:-1] * l + h_bnd[1:] * r
    return RawSample(area=np.array(a), volume=np.array(v), snow=np.zeros_like(a))


def post_process_ensemble(ensemble: NDArray[np.float64], h_bnd: HeightBounds) -> List[RawSample]:
    raw_samples = []
    for sample in ensemble:
        raw_samples.append(post_process_sample(sample, h_bnd=h_bnd))
    return raw_samples
