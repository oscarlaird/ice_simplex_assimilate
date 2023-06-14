from dataclasses import dataclass
import numpy as np
from typing import List

import scipy.stats

@dataclass
class RawSample:
    area: np.ndarray
    volume: np.ndarray
    snow: np.ndarray = None
    def __post_init__(self):
        if self.snow is None:
            self.snow = np.zeros_like(self.area)
        if not len(self.area)==len(self.volume)==len(self.snow):
            raise ValueError('Area, Volume, and Snow vectors must have the same length.')

    def threshold(self, a=None, v=None, s=None):
        ''' Set area, volume, or snow to zero if below the given threshold '''
        if a:
            self.area[self.area < a]     = 0.
        if v:
            self.volume[self.volume < v] = 0.
        if s:
            self.snow[self.snow < s]     = 0.

@dataclass
class RawEnsemble:
    samples: List[RawSample]

class HeightBounds(np.ndarray):
    min_interval = 1e-7
    max_interval = 20.0

    def __new__(cls, input_array):
        a = np.asarray(input_array).view(cls)
        assert np.all(a[1:] - a[:-1] >= cls.min_interval), f'Height bounds must be provided in sorted order'\
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

@dataclass
class Observation:
    n: int
    r: int

