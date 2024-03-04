import pytest
from src.SpectralClustering import SpectralClustering
from conftest import (
    NUM_REPEATS, MAX_TIMEOUT_SECS, INPUT_SIZES, INPUT_NOISES,
    binary_moons_data, dump_time, run_timeout_fn,
    DECOMP_METHODS,
)

@pytest.mark.parametrize('repeats'  , range(NUM_REPEATS))
@pytest.mark.parametrize('n_points' , INPUT_SIZES)
@pytest.mark.parametrize('noise'    , INPUT_NOISES)
def test_time_default(repeats, n_points, noise):
    # generate data, find time to fit, dump to results
    X, _  = binary_moons_data(n_points, noise)
    model = SpectralClustering(2)
    t, _  = run_timeout_fn(model.fit, X)

    if t == MAX_TIMEOUT_SECS:
        dump_time(n_points, noise, 'TIMED_OUT')
        pytest.skip('Timed out')

    dump_time(n_points, noise, t)

@pytest.mark.parametrize('repeats'      , range(NUM_REPEATS))
@pytest.mark.parametrize('n_points'     , INPUT_SIZES)
@pytest.mark.parametrize('noise'        , INPUT_NOISES)
@pytest.mark.parametrize('decomp_method', DECOMP_METHODS)
def test_time_decomposition(repeats, n_points, noise, decomp_method):
    # generate data, find time to fit, dump to results
    X, _  = binary_moons_data(n_points, noise)
    model = SpectralClustering(2, decomposition = decomp_method)
    t, _  = run_timeout_fn(model.fit, X)

    if t == MAX_TIMEOUT_SECS:
        dump_time(n_points, noise, t, 'Decomposition', decomp_method)
        pytest.skip('Timed out')

    dump_time(n_points, noise, t, 'Decomposition', decomp_method)