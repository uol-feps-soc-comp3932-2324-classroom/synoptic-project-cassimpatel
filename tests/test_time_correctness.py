import pytest
from src.SpectralClustering import SpectralClustering
from conftest import (
    NUM_REPEATS, MAX_TIMEOUT_SECS, INPUT_SIZES, INPUT_NOISES, REFINEMENT_K_TESTS, REFINEMENT_EPS_TESTS,
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

        dump_result(n_points, noise, t, 'DEFAULT', 'DEFAULT', X, y_pred, y_true)
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['decomposition'])
    def test_decomposition(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(n_points, noise)
        model      = SpectralClustering(2, decomposition = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Decomposition', method, X, y_pred, y_true)
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['laplacian'])
    def test_laplacian(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(n_points, noise)
        model      = SpectralClustering(2, laplacian = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Laplacian', method, X, y_pred, y_true)
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['affinity'])
    def test_affinity(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(n_points, noise)
        model      = SpectralClustering(2, affinity = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Affinity', method, X, y_pred, y_true)
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['refinement'])
    def test_refinement(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(n_points, noise)
        model      = SpectralClustering(2, refinement = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Refinement', method, X, y_pred, y_true)
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['standardisation'])
    def test_standardisation(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(n_points, noise)
        model      = SpectralClustering(2, standardisation = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Standardisation', method, X, y_pred, y_true)
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

@pytest.mark.parametrize('repeats' , range(NUM_REPEATS))
@pytest.mark.parametrize('noise'   , INPUT_NOISES)
class TestRefinementParamImpact:

    @pytest.mark.parametrize('eps', REFINEMENT_EPS_TESTS)
    def test_refinement_eps_impact(self, repeats, noise, eps):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(1000, noise)
        model      = SpectralClustering(2, refinement='eps', eps=eps)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(1000, noise, t, 'Refinement EPS Impact', eps, X, y_pred, y_true)
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('k', REFINEMENT_K_TESTS)
    def test_refinement_k_impact_knn(self, repeats, noise, k):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(1000, noise)
        model      = SpectralClustering(2, refinement='knn', k=k)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(1000, noise, t, 'Refinement k Impact KNN', k, X, y_pred, y_true)
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('k', REFINEMENT_K_TESTS)
    def test_refinement_k_impact_mknn(self, repeats, noise, k):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(1000, noise)
        model      = SpectralClustering(2, refinement='mutual_knn', k=k)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(1000, noise, t, 'Refinement k Impact MKNN', k, X, y_pred, y_true)
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')