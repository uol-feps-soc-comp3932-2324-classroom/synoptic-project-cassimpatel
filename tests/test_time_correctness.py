import pytest
from src.SpectralClustering import SpectralClustering
from conftest import (
    NUM_REPEATS, MAX_TIMEOUT_SECS, INPUT_SIZES, INPUT_NOISES,
    binary_moons_data, run_timeout_fn, dump_result,
    DECOMP_METHODS,
)

@pytest.mark.parametrize('repeats'  , range(NUM_REPEATS))
@pytest.mark.parametrize('n_points' , INPUT_SIZES)
@pytest.mark.parametrize('noise'    , INPUT_NOISES)
def test_default(repeats, n_points, noise):
    # generate data, fit model, dump to results
    X, y_true  = binary_moons_data(n_points, noise)
    model      = SpectralClustering(2)
    t, y_pred  = run_timeout_fn(model.fit, X)

    if t == MAX_TIMEOUT_SECS:
        dump_result(n_points, noise, t)
        pytest.skip('Timed out')

    dump_result(n_points, noise, t, pred_labels = y_pred, ground_truth = y_true)

@pytest.mark.parametrize('repeats'      , range(NUM_REPEATS))
@pytest.mark.parametrize('n_points'     , INPUT_SIZES)
@pytest.mark.parametrize('noise'        , INPUT_NOISES)
@pytest.mark.parametrize('decomp_method', DECOMP_METHODS)
def test_decomposition(repeats, n_points, noise, decomp_method):
    # generate data, fit model, dump to results
    X, y_true  = binary_moons_data(n_points, noise)
    model = SpectralClustering(2, decomposition = decomp_method)
    t, y_pred  = run_timeout_fn(model.fit, X)

    if t == MAX_TIMEOUT_SECS:
        dump_result(n_points, noise, t, experiment = 'Decomposition', variant = decomp_method)
        pytest.skip('Timed out')

    dump_result(n_points, noise, t, experiment = 'Decomposition', variant = decomp_method, pred_labels = y_pred, ground_truth = y_true)