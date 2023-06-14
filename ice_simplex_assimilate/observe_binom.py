import numpy as np
import scipy

from .shared_classes import MixedDirichlet, Observation

# UPDATE ON OBSERVATION
def x0_to_unif(x0, md: MixedDirichlet) -> float:
    pass


def unif_to_post_x0(unif, md: MixedDirichlet, observation: Observation) -> float:
    # N.B. Dirichlet is conjugate prior to multinomial observation, but not to binomial observation
    # so the observation does not give us a mixed dirichlet for the posterior
    # But the distribution for x0 is beta and is conjugate to the observation

    betas = [scipy.stats.beta(  cd.alpha[0]+observation.r, sum(cd.alpha[1:])+(observation.n-observation.r)  ) for cd in md.dirichlets]
    def cdf(x0):
        return sum([beta.cdf(x0) * pi for beta,pi in zip(betas, md.mixture_weights)])
    delta = 1e-10
    return scipy.optimize.root_scalar(lambda x0: cdf(x0)-uniform, bracket=[delta, 1-delta ]).root

def update_x0(x0, md: MixedDirichlet, observation: Observation):
    u = x0_to_unif(x0, md)
    x = unif_to_post_x0(u, md, observation)
    return x
