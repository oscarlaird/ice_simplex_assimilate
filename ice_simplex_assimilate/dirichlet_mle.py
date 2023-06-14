from .shared_classes import ClassEnsemble, Ensemble, ClassDirichlet, MixedDirichlet
import numpy as np
import scipy

# DIRICHLET PARAMETER ESTIMATION
def fit_dirichlet(class_ensemble: ClassEnsemble, max_alpha=1e4) -> ClassDirichlet:
    samples = np.array(class_ensemble.samples)
    assert len(samples)>1
    samples = samples[:, class_ensemble.sample_class]  # extract nonzero components
    assert np.all(samples > 0)
    log_avg = np.log(samples).mean(axis=0)
    geo_mean = np.exp(log_avg)
    mean = geo_mean / geo_mean.sum()  # normalized geometric mean
    # if all the samples are in a tight envelope our MLE for alpha will diverge.
    # use max_alpha instead with the geometric mean of the samples
    if np.allclose(samples.min(axis=0), samples.max(axis=0)):
        s = max_alpha
    else:
        # use the alpha of the closed form MLE using Stirling's approximation
        K = sum(class_ensemble.sample_class)
        s = ((K-1)/2) / (np.inner(mean, np.log(mean) - log_avg))
    alpha = s * mean
    '''
    else:
        gammaln, digamma, polygamma = scipy.special.gammaln, scipy.special.digamma, scipy.special.polygamma
        f = lambda alpha: gammaln(alpha.sum()) - gammaln(alpha).sum() + (log_avg * (alpha - 1)).sum()  # likelihood
        alphas = [np.ones_like(log_avg)]  # initialize alpha_0
        for _ in range(5):
            alpha = alphas[-1]
            grad = digamma(alpha.sum()) - digamma(alpha) + log_avg
            hessian = - np.eye(len(alpha)) * (polygamma(1, alpha))
            hessian += polygamma(1, alpha.sum())
            invH = np.linalg.inv(hessian)
            da = - np.dot(invH, grad)
            alphas.append(alpha + da)
    '''
    return ClassDirichlet(alpha=alpha, sample_class=class_ensemble.sample_class)


def fit_mixed_dirichlet(ensemble: Ensemble) -> MixedDirichlet:
    dirichlets = [None for _ in ensemble.class_ensembles]
    for i, ce in enumerate(ensemble.class_ensembles):
        if len(ce.samples)>1:
            dirichlets[i] = fit_dirichlet(ce)
    max_s = max( [sum(d.alpha) for d in dirichlets if d is not None] )
    for i, ce in enumerate(ensemble.class_ensembles):
        if len(ce.samples)==1:
            alpha = ce.samples[0][ce.sample_class] * max_s
            dirichlets[i] = ClassDirichlet(alpha=alpha, sample_class=ce.sample_class)

    mixing_rates = np.array( [len(ce.samples)/len(ensemble.samples) for ce in ensemble.class_ensembles] )

    return MixedDirichlet(mixture_weights=mixing_rates, dirichlets=dirichlets)

