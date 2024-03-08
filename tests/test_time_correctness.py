import pytest
from src.SpectralClustering import SpectralClustering
from conftest import (
    NUM_REPEATS, MAX_TIMEOUT_SECS, INPUT_SIZES, INPUT_NOISES,
    binary_moons_data, run_timeout_fn, dump_result,
    PIPELINE_METHODS
)

@pytest.mark.parametrize('repeats' , range(NUM_REPEATS))
@pytest.mark.parametrize('n_points', INPUT_SIZES)
@pytest.mark.parametrize('noise'   , INPUT_NOISES)
class TestTimeCorrectness:

    def test_default(self, repeats, n_points, noise):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(n_points, noise)
        model      = SpectralClustering(2)
        t, y_pred  = run_timeout_fn(model.fit, X)

        if t == MAX_TIMEOUT_SECS:
            dump_result(n_points, noise, t)
            pytest.skip('Timed out')

        dump_result(n_points, noise, t, pred_labels = y_pred, ground_truth = y_true)

    @pytest.mark.parametrize('method', PIPELINE_METHODS['decomposition'])
    def test_decomposition(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(n_points, noise)
        model      = SpectralClustering(2, decomposition = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        if t == MAX_TIMEOUT_SECS:
            dump_result(n_points, noise, t, 'Decomposition', method)
            pytest.skip('Timed out')

        dump_result(n_points, noise, t, 'Decomposition', method, y_pred, y_true)

    @pytest.mark.parametrize('method', PIPELINE_METHODS['laplacian'])
    def test_laplacian(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(n_points, noise)
        model      = SpectralClustering(2, laplacian = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        if t == MAX_TIMEOUT_SECS:
            dump_result(n_points, noise, t, 'Laplacian', method)
            pytest.skip('Timed out')

        dump_result(n_points, noise, t, 'Laplacian', method, y_pred, y_true)

    @pytest.mark.parametrize('method', PIPELINE_METHODS['affinity'])
    def test_affinity(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(n_points, noise)
        model      = SpectralClustering(2, affinity = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        if t == MAX_TIMEOUT_SECS:
            dump_result(n_points, noise, t, 'Affinity', method)
            pytest.skip('Timed out')

        dump_result(n_points, noise, t, 'Affinity', method, y_pred, y_true)

    @pytest.mark.parametrize('method', PIPELINE_METHODS['refinement'])
    def test_refinement(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(n_points, noise)
        model      = SpectralClustering(2, refinement = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        if t == MAX_TIMEOUT_SECS:
            dump_result(n_points, noise, t, 'Refinement', method)
            pytest.skip('Timed out')

        dump_result(n_points, noise, t, 'Refinement', method, y_pred, y_true)

    @pytest.mark.parametrize('method', PIPELINE_METHODS['standardisation'])
    def test_standardisation(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(n_points, noise)
        model      = SpectralClustering(2, standardisation = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        if t == MAX_TIMEOUT_SECS:
            dump_result(n_points, noise, t, 'Standardisation', method)
            pytest.skip('Timed out')

        dump_result(n_points, noise, t, 'Standardisation', method, y_pred, y_true)