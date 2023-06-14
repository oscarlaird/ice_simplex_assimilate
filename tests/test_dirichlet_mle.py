from hypothesis import given, assume, settings

from dirichlet_assimilate import shared_classes, dirichlet_mle

from tests import custom_strategies


@settings(max_examples=20)
@given(custom_strategies.class_ensemble_strategy())
def test_fit_dirichlet(class_ensemble):
    assume(len(class_ensemble.samples) > 1)
    cd = dirichlet_mle.fit_dirichlet(class_ensemble)  # class dirichlet estimate
    # assert 0 < cd.alpha.sum()  # alpha can be all negative
    # assert np.all(cd.alpha >= 0)
    cd.check_valid()

@settings(max_examples=20)
@given(custom_strategies.ensemble_strategy())
def test_fit_mixed_dirichlet(ensemeble: shared_classes.Ensemble):
    # at least one class ensemble has more than one member
    assume(any([len(ce.samples)>1 for ce in ensemeble.class_ensembles]))
    md = dirichlet_mle.fit_mixed_dirichlet(ensemeble)

