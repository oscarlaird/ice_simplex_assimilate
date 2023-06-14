from . import dirichlet, dirichlet_mle, deltize, observe_binom, shared_classes

'''

def transport_ensemble(ensemble: Ensemble, observation: Observation):
    mixed_dirichlet = fit_mixed_dirichlet(ensemble)
    uniform_ensemble = uniformize_ensemble(ensemble, mixed_dirichlet)
    post_ensemble = update_ensemble(uniform_ensemble, mixed_dirichlet, observation=observation)
    return post_ensemble

def transport_raw_ensemble(raw_ensemble: RawEnsemble, h_bnd: HeightBounds, observation: Observation):
    ensemble = process_ensemble(raw_ensemble, h_bnd)
    post_ensemble = transport_ensemble(ensemble, observation=observation)
    post_raw_ensemble = post_process_ensemble(post_ensemble, h_bnd)
    return post_raw_ensemble
'''
