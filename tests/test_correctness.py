import pytest
from src.SpectralClustering import SpectralClustering
from conftest import (
    NUM_REPEATS, MAX_TIMEOUT_SECS, INPUT_SIZES, INPUT_NOISES,
    binary_moons_data, dump_correctness, run_timeout_fn,
    DECOMP_METHODS,
)

@pytest.mark.parametrize('repeats'  , range(NUM_REPEATS))
@pytest.mark.parametrize('n_points' , INPUT_SIZES)
@pytest.mark.parametrize('noise'    , INPUT_NOISES)
def test_correctness_default(repeats, n_points, noise):
    # generate data, calculate metrics, dump to results
    X, y_true  = binary_moons_data(n_points, noise)
    model      = SpectralClustering(2)
    t, y_pred  = run_timeout_fn(model.fit, X)

    if t == MAX_TIMEOUT_SECS:
        dump_correctness(n_points, noise, timed_out = True)
        pytest.skip('Timed out')

    dump_correctness(n_points, noise, pred_labels = y_pred, ground_truth = y_true)

@pytest.mark.parametrize('repeats'      , range(NUM_REPEATS))
@pytest.mark.parametrize('n_points'     , INPUT_SIZES)
@pytest.mark.parametrize('noise'        , INPUT_NOISES)
@pytest.mark.parametrize('decomp_method', DECOMP_METHODS)
def test_correctness_decomposition(repeats, n_points, noise, decomp_method):
    # generate data, calculate metrics, dump to results
    X, y_true  = binary_moons_data(n_points, noise)
    model = SpectralClustering(2, decomposition = decomp_method)
    t, y_pred  = run_timeout_fn(model.fit, X)

    if t == MAX_TIMEOUT_SECS:
        dump_correctness(n_points, noise, timed_out = True, experiment = 'Decomposition', variant = decomp_method)
        pytest.skip('Timed out')

    dump_correctness(n_points, noise, pred_labels = y_pred, ground_truth = y_true, experiment = 'Decomposition', variant = decomp_method)