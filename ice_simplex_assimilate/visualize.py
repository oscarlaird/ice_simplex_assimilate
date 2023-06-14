import numpy as np
import scipy
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm

from .shared_classes import Sample, Ensemble, ClassDirichlet, MixedDirichlet, ClassEnsemble, RawSample, RawEnsemble, Observation, HeightBounds

class Visualisation:

    def __init__(self, h_bnd: HeightBounds, ax=None, log_scale=True, figsize=(20,8), bottom=-15):
        self.h_bnd = h_bnd
        self.log_scale = log_scale
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)
        self.width = 0.25
        self.ax = ax
        self.ax.set_xlabel('Height')
        self.ax.set_xlim(-1, h_bnd[-1]+1)
        self.show_bounds()
        self.color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'].__iter__()
        self.offset_cycle = list(np.array([0., 1., -1., 2., -2., 3., -3., 0.5, -0.5, 1.5, -1.5, 2.5, -2.5,]) * 0.02).__iter__()

        if self.log_scale:
            self.ax.set_ylabel('Log area')
            self.bottom = bottom
            self.ax.set_ylim(self.bottom, 0)
        else:
            # set regular scale
            self.ax.set_ylim(0,1)
            self.ax.set_ylabel('Area')
            self.bottom = 0

    def show_bounds(self):
        for bound in self.h_bnd:
            self.ax.axvline(x=bound, color='r')
        self.ax.set_xticks(self.h_bnd)

    @staticmethod
    def pretty_float(f):
        return f'{f:.1e}' if 0<f<0.001 else f'{f:.3f}'

    def show_sample(self, sample: Sample):
        self.add_sample(sample, style='bar', show_labels=True, show_raw=True)

    def show_raw_sample(self, raw_sample: RawSample):
        self.add_raw_sample(raw_sample, style='bar', show_labels=True)

    def show_raw_ensemble(self, raw_ensemble: RawEnsemble, params={}):
        params = {'color': next(self.color_cycle), 's': 50, **params}
        for rs in raw_ensemble.samples:
            self.add_raw_sample(rs, style='scatter', params=params)

    def show_class_dirichlet(self, cd: ClassDirichlet, params={}, confidence=0.95, pi=None):

        params = {'s': 100, 'marker': '_', 'color': next(self.color_cycle), **params}

        # error bounds
        dists = scipy.stats.beta(cd.alpha, cd.alpha.sum()-cd.alpha)  # beta dists of marginal for each component
        a = (1 - confidence)/2
        lower_bound = dists.isf(1 - a)
        upper_bound = dists.isf(a)
        error_bounds = np.zeros((len(cd.full_mean_sample), 2))
        error_bounds[cd.sample_class] = np.column_stack((lower_bound, upper_bound))

        # label (show alpha)
        fms = [f'{f:.2f}'[2:] for f in cd.full_mean_sample]
        label = f'$\\alpha$% = {"|".join(fms)}   $|\\alpha|$ = {cd.alpha.sum(): >5.1f}'
        if pi:  # indicate the number of ensemble members in the dirichlet distribution
            label += f'     $\\pi$ = {int(pi*100): <2}%'
        params['label'] = label

        # draw sample
        self.add_sample(cd.full_mean_sample, style='scatter', params=params, error_bounds=error_bounds)

        # draw legend above graph
        font_prop = fm.FontProperties(family='monospace')
        self.ax.legend(prop=font_prop, loc='lower center', bbox_to_anchor=(0.5, 1.0))

    def show_mixed_dirichlet(self, md: MixedDirichlet, confidence=0.95):
        for cd, mr in zip(md.dirichlets, md.mixture_weights):
            self.show_class_dirichlet(cd, confidence=confidence, pi=mr)

    def show_dirichlet_plus_samples(self, ce: ClassEnsemble, cd: ClassDirichlet, params={}, confidence=0.9, pi=None):
        params = {'color': next(self.color_cycle), **params}
        self.show_class_ensemble(ce, params=params)
        self.show_class_dirichlet(cd, params=params, confidence=confidence, pi=pi)


    def show_mixed_dirichlet_plus_samples(self, en: Ensemble, md: MixedDirichlet, confidence=0.9):
        for ce, cd, mr in zip(en.class_ensembles, md.dirichlets, md.mixture_weights):
            self.show_dirichlet_plus_samples(ce, cd, confidence=confidence, pi=mr)

    def show_class_ensemble(self, ce: ClassEnsemble, params={}):

        params = {'color': next(self.color_cycle), 's': 50, **params}

        for sample in ce.samples:
            self.add_sample(sample, style='scatter', params=params)

    def show_ensemble(self, ensemble: Ensemble):
        for ce in ensemble.class_ensembles:
            self.show_class_ensemble(ce)

    def add_raw_sample(self, rs: RawSample, style='bar', show_labels=False, params=None):
        if params is None:
            params = {}
        x = [np.mean(interval) for interval in self.h_bnd.intervals]
        x.insert(0, 0.)
        top = np.insert(rs.area, 0, 1-sum(rs.area))
        if self.log_scale:
            top = np.log(top)


        if style=='bar':
            bars = self.ax.bar(x=x, height=top-self.bottom, width=self.width, bottom=self.bottom, **params)
            if show_labels:
                for bar, t in zip(bars, top):
                    self.ax.text(bar.get_x() + bar.get_width() / 2, self.bottom + bar.get_height(), self.pretty_float(t), ha='center', va='bottom')
        elif style=='scatter':
            self.ax.scatter(x=x, y=top, **params)
        else:
            raise ValueError(f'Invalid style={style}, must be bar or scatter')


    def add_sample(self, sample, style='bar', show_labels=False, show_raw=False, error_bounds=None, params=None):
        if params is None:
            params = {}

        assert len(sample) == 2*len(self.h_bnd) - 1

        x_r = self.h_bnd      - self.width/2
        x_l = self.h_bnd[:-1] + self.width/2
        x = np.array([x for pair in zip(x_r[:-1], x_l) for x in pair] + [x_r[-1]])
        x = x[sample > 0]
        if error_bounds is not None:
            x += next(self.offset_cycle)

        top = sample[sample > 0]

        if self.log_scale:
            top = np.log(top)

        if style=='bar':
            bars = self.ax.bar(x=x, height=top-self.bottom, width=self.width, bottom=self.bottom, **params)
        elif style=='scatter':
            self.ax.scatter(x=x, y=top, **params)
        else:
            raise ValueError(f'Invalid style={style}, must be bar or scatter')


        if show_labels:
            assert style=='bar', 'labels can only be shown on bar charts. Try show_labels=False or style="bar"'
            labels = np.vectorize(self.pretty_float)(sample[sample>0])
            for bar, label in zip(bars, labels):
                # height = bar.get_height()
                self.ax.text(bar.get_x() + bar.get_width() / 2, self.bottom + bar.get_height(), label, ha='center', va='bottom')

        if show_raw:
            for interval, l, r, in zip(self.h_bnd.intervals, sample[1::2], sample[2::2]):
                x_pos = sum(interval) / 2
                y_pos = self.bottom + 3 if self.log_scale else 0.6
                a, v = l+r, interval[0]*l + interval[1]*r
                self.ax.annotate(
                    f'a={self.pretty_float(a)}\nv={self.pretty_float(v)}',
                    xy=(x_pos, y_pos),
                    ha='center',
                    va='bottom',
                    textcoords='offset points',
                    xytext=(0,5),
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2')
                )

        if error_bounds is not None:
            assert len(sample)==len(error_bounds),\
                f'length of error_bounds {len(error_bounds)} does not match length of sample f{len(sample)}'
            for x_pos, (low, high) in zip(x, error_bounds):
                if self.log_scale:
                    low, high = np.log(low), np.log(high)
                self.ax.vlines(x=x_pos, ymin=low, ymax=high, colors=(params.get('color', 'blue')))
