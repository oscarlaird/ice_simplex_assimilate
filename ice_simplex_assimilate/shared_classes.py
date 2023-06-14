from dataclasses import dataclass
import numpy as np
from typing import List

# load ncurses and do everything to show hello world to the user
# import ncurses
# ncurses.initscr()
# ncurses.printw("Hello World!!!")
# ncurses.refresh() # print it on to the real screen
# ncurses.getch()  # wait for user input
# ncurses.endwin() # restore terminal


import scipy.stats


class Sample(np.ndarray):
    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)
    @property
    def sample_class(self):
        return SampleClass(self > 0)

    def threshold(self, threshold):
        self[self < threshold] = 0

class SampleClass(np.ndarray):
    def __new__(cls, input_array):
        return np.asarray(input_array, dtype=bool).view(cls)

@dataclass
class Ensemble:
    samples: List[Sample]
    def __post_init__(self):
        self.sample_classes = [SampleClass(c) for c in {tuple(s.sample_class) for s in self.samples}] # unique sample classes
        self.sample_classes = sorted(self.sample_classes, key = lambda x: list(x)) # sort by sample class
        self.class_ensembles = [ClassEnsemble(samples=[s for s in self.samples if np.all(s.sample_class==c)]) for c in self.sample_classes]

class ClassEnsemble(Ensemble):
    def __post_init__(self):
        self.sample_class = self.samples[0].sample_class

class UniformSample(np.ndarray):
    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)


@dataclass
class ClassDirichlet:
    alpha: np.ndarray
    sample_class: SampleClass
    scale: float = 1.0

    def __post_init__(self):
        assert np.any(self.sample_class), f'Sample class must have at least one true component.'
        assert len(self.alpha) == np.count_nonzero(self.sample_class),\
            f'alpha ({self.alpha}) does not match number of true components in sample class ({self.sample_class}).'

    @property
    def full_alpha(self):
        a = np.zeros_like(self.sample_class, dtype=float)
        a[self.sample_class] = self.alpha
        return a

    @property
    def full_mean_sample(self):
        return Sample(self.full_alpha / self.full_alpha.sum())

    def to_beta(self):
        # If we are only interested the first component, we might want to simplify the dirichlet to a beta
        alpha = self.alpha[0], self.alpha[1:].sum()
        sample_class = np.array([self.sample_class[0], np.any(self.sample_class[1:])])
        return ClassDirichlet(alpha=alpha, sample_class=sample_class)

    @staticmethod
    def unif():
        return scipy.stats.uniform(0, 1).rvs()

    def marg_cdf(self, x0) -> float:
        # marginal cdf of the first component

        # if the scale=0 i.e. there is no more remaining space, then the distribution is a delta at 0
        if np.isclose(self.scale, 0):
            if np.isclose(x0, 0):
                return self.unif()
            elif x0 < 0:
                return 0
            elif x0 > 0:
                return 1
        y0 = x0 / self.scale  # scale x0 -> y0 in the range [0, 1]


        # if y0 should a zero component then the dist is a delta at zero
        #    1 if x is positive
        #    0 if x is negative
        #    ? if x is 0 (we map to a uniform random variable)
        if not self.sample_class[0]:
            return self.unif() if np.isclose(y0) else 0 if y0<0 else 1
        # if all the remaining components should be zero then the cdf is
        #    0 if x is less than 1
        #    ? if x = 1 (we map to a uniform random var)
        #    1 if x > 1
        elif not np.any(self.sample_class[1:]):
            return self.unif() if np.isclose(y0, 1.) else 0 if y0 < 1 else 1
        else:
            return self.marginal_beta_scaled.cdf(y0)

    def marg_pdf(self, x0):
        y0 = x0 / self.scale
        return self.marginal_beta_scaled.pdf(y0)

    @property
    def marginal_beta_scaled(self):
        ''' marginal distribution of y0=x0/scale. the function marg_cdf handles edge cases '''
        return scipy.stats.beta(self.alpha[0], self.alpha[1:].sum())

    def conditional_dist(self, x0):
        '''Return the conditional class dirichlet distribution conditioned on x0'''
        assert len(self.alpha) > 1, 'Cannot condition a one-dimensional dirichlet distribution'
        return ClassDirichlet(alpha=self.alpha[1:], sample_class=self.sample_class, scale=(self.scale - x0))

    def check_valid(self):
        assert np.all(self.alpha > 0), f'alpha_i must be strictly positive, {self.alpha}'


@dataclass
class MixedDirichlet:
    mixture_weights: np.ndarray
    dirichlets: List[ClassDirichlet]
    scale: float = 1.0

    @property
    def alpha_matrix(self):
        return np.array([cd.full_alpha for cd in self.dirichlets])

    @property
    def class_matrix(self):
        return np.array([cd.sample_class for cd in self.dirichlets])

    def marg_cdf(self, x0) -> float:
        return np.array([cd.marg_cdf(x0) for cd in self.dirichlets]) @ self.mixture_weights

    def inverse_marg_cdf(self, x0) -> float:
        pass

    def conditional_likelihoods(self, x0):
        if np.isclose(x0, 0):
            return self.class_matrix[:, 0] == 0
        elif np.isclose(x0, self.scale):
            return (self.class_matrix[:, 0] > 0) & np.any(self.class_matrix[:,1:], axis=1)
        else:
            return np.array([cd.marg_pdf(x0) if cd.sample_class[0] else 0 for cd in self.dirichlets])

    def conditional_dist(self, x0):
        new_mixing_rates = self.mixture_weights @ self.conditional_likelihoods(x0)
        new_dirichlets = [cd.conditional_dist(x0) for cd in self.dirichlets]
        return MixedDirichlet(mixture_weights=new_mixing_rates, dirichlets=new_dirichlets)

    def check_valid(self):
        for cd in self.dirichlets:
            cd.check_valid()
        assert np.all(self.mixture_weights > 0), f'Mixing rates must be greater than 0, {self.mixture_weights}'
        assert np.isclose(self.mixture_weights.sum(), 1.0), f'Mixing rates must sum to one'
        assert 0 <= self.scale <= 1.0, 'Scale must be between zero and one'




@dataclass
class UniformEnsemble:
    samples: List[UniformSample]

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

