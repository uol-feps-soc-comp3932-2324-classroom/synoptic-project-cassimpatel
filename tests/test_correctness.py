import pytest
from src.SpectralClustering import SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score
from conftest import (
    NUM_REPEATS, MAX_TIMEOUT_SECS, INPUT_SIZES, INPUT_NOISES,
    binary_moons_data, dump_correctness, run_timeout_fn,
)

# calculate a set of metrics for correctness
def calc_correctness(pred_labels, ground_truth):
    # TODO: add more measures of correctness
    return {
        'adjusted_rand_score': adjusted_rand_score(ground_truth, pred_labels),
    }


@pytest.mark.parametrize('repeats'  , range(NUM_REPEATS))
@pytest.mark.parametrize('n_points' , INPUT_SIZES)
@pytest.mark.parametrize('noise'    , INPUT_NOISES)
def test_correctness_default(repeats, n_points, noise):
    # generate data, calculate metrics, dump to results
    X, y_true  = binary_moons_data(n_points, noise)
    model      = SpectralClustering(2)
    t, y_pred  = run_timeout_fn(model.fit, X)

    if t == MAX_TIMEOUT_SECS:
        dump_correctness(n_points, noise, 'TIMED_OUT')
        pytest.skip('Timed out')

    metrics = calc_correctness(y_pred, y_true)
    dump_correctness(n_points, noise, metrics['adjusted_rand_score'])