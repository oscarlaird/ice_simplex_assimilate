import numpy as np
from hypothesis import strategies as st, assume

from dirichlet_assimilate import shared_classes


@st.composite
def height_bounds_strategy(draw, n_pos_categories):
    # create n positive height intervals of at least width min_interval
    intervals = draw(st.lists(st.floats(min_value=shared_classes.HeightBounds.min_interval * 2,
                                        max_value=shared_classes.HeightBounds.max_interval / 2,
                                        allow_nan=False, allow_infinity=False),
                              min_size=n_pos_categories, max_size=n_pos_categories))
    return shared_classes.HeightBounds.from_interval_widths(np.array(intervals))


@st.composite
def sample_class_strategy(draw, num_components):
    nonzero_components = draw(st.lists(st.booleans(), min_size=num_components, max_size=num_components))
    nonzero_components = np.array(nonzero_components)
    assume(np.any(nonzero_components))
    return shared_classes.SampleClass(nonzero_components)


@st.composite
def sample_strategy(draw, num_components=None, using_sample_class=None):
    # todo this allows nonzero and zero mass in the same thickness interval
    if num_components is None:
        num_components = draw(st.integers(min_value=1, max_value=10))
    if using_sample_class is None:
        using_sample_class = draw(sample_class_strategy(num_components))
    num_nonzero = int(np.sum(using_sample_class, dtype=int))
    s = np.zeros(num_components)
    nonzero_components = draw(st.lists(st.floats(min_value=1e-9, max_value=1, allow_nan=False),
                                       min_size=num_nonzero,
                                       max_size=num_nonzero))
    nonzero_components = np.array(nonzero_components)
    nonzero_components /= nonzero_components.sum()
    s[using_sample_class] = nonzero_components
    return shared_classes.Sample(s)


@st.composite
def sample_w_h_bounds_strategy(draw):
    num_pos_intervals = draw(st.integers(min_value=1, max_value=10))  # number of ice thickness categories
    return num_pos_intervals, \
        draw(height_bounds_strategy(num_pos_intervals)), \
        draw(sample_strategy(num_components=num_pos_intervals * 2 + 1))


@st.composite
def class_ensemble_strategy(draw, num_components=None):
    if num_components is None:
        num_components = draw(st.integers(min_value=1, max_value=10))
    num_samples = draw(st.integers(min_value=1, max_value=10))  # num samples in ensemble
    sample_class = draw(sample_class_strategy(num_components=num_components))
    samples = draw(st.lists(sample_strategy(num_components=num_components, using_sample_class=sample_class),
                            min_size=num_samples, max_size=num_samples))
    return shared_classes.ClassEnsemble(samples)


@st.composite
def ensemble_strategy(draw, num_components=None):
    if num_components is None:
        num_components = draw(st.integers(min_value=1, max_value=10))
    num_classes = draw(st.integers(min_value=1, max_value=10))
    ces = draw(st.lists(class_ensemble_strategy(num_components=num_components),
                            min_size=num_classes, max_size=num_classes))
    return shared_classes.Ensemble([s for ce in ces for s in ce.samples])
