from . import shared_classes
import numpy as np

# raw samples
h_bnd = shared_classes.HeightBounds([0., 1., 2., 4.])

area = np.array([[0.20,0.65,0.1],
                 [0.17,0.66,0.11],
                 [0.10,0.65,0.15],
                 [0.00,1.00,0.00],
                 [0.00,0.00,0.00],
                 ])
volume = np.array([[0.08, 1., 0.31],
                   [0.08, 1., 0.31],
                   [0.08, 1., 0.31],
                   [0.00, 1.5, 0.0],
                   [0.00, 0., 0.00],
                ])
snow = np.array([[0.08, 1., 0.31],
                 [0.08, 1., 0.31],
                 [0.08, 1., 0.31],
                 [0.,   0., 0.00],
                 [0.,   0., 0.00],
                ])

raw_ensemble = shared_classes.RawEnsemble(samples = [shared_classes.RawSample(area=a, volume=v, snow=s) for a,v,s in zip(area, volume, snow)])
raw_sample = raw_ensemble.samples[0]

# processed samples
sample = shared_classes.Sample([0.05 , 0.12 , 0.08 , 0.30 , 0.35 , 0.045, 0.055])
class_ensemble = shared_classes.ClassEnsemble(samples=[
    sample,
    shared_classes.Sample([0.06 , 0.09 , 0.08 , 0.32 , 0.34 , 0.065, 0.045]),
    shared_classes.Sample([0.1  , 0.02 , 0.08 , 0.30 , 0.35 , 0.145, 0.005]),
])
ensemble = shared_classes.Ensemble(samples=[
    *class_ensemble.samples,
    shared_classes.Sample([0.00 , 0.00 , 0.00 , 0.50 , 0.50 , 0.000, 0.000]),
    shared_classes.Sample([1.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.000, 0.000]),
])

# Dirichlet distributions
# class_dirichlet   = shared_classes.ClassDirichlet(alpha=np.array([1.5, 2.9, 2.8123, 7., 6.2, 1.5, 1.5]), sample_class=shared_classes.SampleClass([1, 1, 1, 1, 1, 1, 1]))
# class_dirichlet_2 = shared_classes.ClassDirichlet(alpha=np.array([2, 1, 1, 3, 2]), sample_class=shared_classes.SampleClass([1, 1, 1, 1, 1, 0, 0]))
# mixed_dirichlet   = shared_classes.MixedDirichlet(mixing_rates=np.array([0.75, 0.25]), dirichlets=[class_dirichlet_2, class_dirichlet])



