import pytest
import numpy as np
from sklearn import datasets
from src.SpectralClustering import SpectralClustering
from conftest import (
    NUM_REPEATS, MAX_TIMEOUT_SECS, INPUT_SIZES, INPUT_NOISES, REFINEMENT_K_TESTS, REFINEMENT_EPS_TESTS,
    binary_moons_data, run_timeout_fn, dump_result,
    PIPELINE_METHODS
)

@pytest.mark.parametrize('repeats' , range(NUM_REPEATS))
@pytest.mark.parametrize('n_points', INPUT_SIZES)
@pytest.mark.parametrize('noise'   , INPUT_NOISES)
class TestTimeCorrectnessTwoMoons:

    def test_default(self, repeats, n_points, noise):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(n_points, noise)
        model      = SpectralClustering(2)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'DEFAULT', 'DEFAULT', X, y_pred, y_true, "Two Moons")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['decomposition'])
    def test_decomposition(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(n_points, noise)
        model      = SpectralClustering(2, decomposition = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Decomposition', method, X, y_pred, y_true, "Two Moons")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['laplacian'])
    def test_laplacian(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(n_points, noise)
        model      = SpectralClustering(2, laplacian = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Laplacian', method, X, y_pred, y_true, "Two Moons")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['affinity'])
    def test_affinity(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(n_points, noise)
        model      = SpectralClustering(2, affinity = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Affinity', method, X, y_pred, y_true, "Two Moons")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['refinement'])
    def test_refinement(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(n_points, noise)
        model      = SpectralClustering(2, refinement = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Refinement', method, X, y_pred, y_true, "Two Moons")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['standardisation'])
    def test_standardisation(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(n_points, noise)
        model      = SpectralClustering(2, standardisation = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Standardisation', method, X, y_pred, y_true, "Two Moons")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

@pytest.mark.parametrize('repeats' , range(NUM_REPEATS))
@pytest.mark.parametrize('noise'   , INPUT_NOISES)
class TestRefinementParamImpactTwoMoons:

    @pytest.mark.parametrize('eps', REFINEMENT_EPS_TESTS)
    def test_refinement_eps_impact(self, repeats, noise, eps):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(1000, noise)
        model      = SpectralClustering(2, refinement='eps', eps=eps)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(1000, noise, t, 'Refinement EPS Impact', eps, X, y_pred, y_true, "Two Moons")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('k', REFINEMENT_K_TESTS)
    def test_refinement_k_impact_knn(self, repeats, noise, k):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(1000, noise)
        model      = SpectralClustering(2, refinement='knn', k=k)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(1000, noise, t, 'Refinement k Impact KNN', k, X, y_pred, y_true, "Two Moons")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('k', REFINEMENT_K_TESTS)
    def test_refinement_k_impact_mknn(self, repeats, noise, k):
        # generate data, fit model, dump to results
        X, y_true  = binary_moons_data(1000, noise)
        model      = SpectralClustering(2, refinement='mutual_knn', k=k)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(1000, noise, t, 'Refinement k Impact MKNN', k, X, y_pred, y_true, "Two Moons")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

@pytest.mark.parametrize('repeats' , range(NUM_REPEATS))
@pytest.mark.parametrize('n_points', INPUT_SIZES)
@pytest.mark.parametrize('noise'   , [0.5, 1, 1.5])
class TestTimeCorrectnessBlobs:

    def test_default(self, repeats, n_points, noise):
        # generate data, fit model, dump to results
        X, y_true  = datasets.make_blobs(n_samples=n_points, random_state=40, cluster_std=noise)
        model      = SpectralClustering(3)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'DEFAULT', 'DEFAULT', X, y_pred, y_true, "Blobs")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['decomposition'])
    def test_decomposition(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = datasets.make_blobs(n_samples=n_points, random_state=40, cluster_std=noise)
        model      = SpectralClustering(3, decomposition = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Decomposition', method, X, y_pred, y_true, "Blobs")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['laplacian'])
    def test_laplacian(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = datasets.make_blobs(n_samples=n_points, random_state=40, cluster_std=noise)
        model      = SpectralClustering(3, laplacian = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Laplacian', method, X, y_pred, y_true, "Blobs")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['affinity'])
    def test_affinity(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = datasets.make_blobs(n_samples=n_points, random_state=40, cluster_std=noise)
        model      = SpectralClustering(3, affinity = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Affinity', method, X, y_pred, y_true, "Blobs")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['refinement'])
    def test_refinement(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = datasets.make_blobs(n_samples=n_points, random_state=40, cluster_std=noise)
        model      = SpectralClustering(3, refinement = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Refinement', method, X, y_pred, y_true, "Blobs")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['standardisation'])
    def test_standardisation(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = datasets.make_blobs(n_samples=n_points, random_state=40, cluster_std=noise)
        model      = SpectralClustering(3, standardisation = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Standardisation', method, X, y_pred, y_true, "Blobs")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

@pytest.mark.parametrize('repeats' , range(NUM_REPEATS))
@pytest.mark.parametrize('noise'   , INPUT_NOISES)
class TestRefinementParamImpactBlobs:

    @pytest.mark.parametrize('k', REFINEMENT_K_TESTS)
    def test_refinement_k_impact_knn(self, repeats, noise, k):
        # generate data, fit model, dump to results
        X, y_true  = datasets.make_blobs(n_samples=1000, random_state=40, cluster_std=noise)
        model      = SpectralClustering(3, refinement='knn', k=k)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(1000, noise, t, 'Refinement k Impact KNN', k, X, y_pred, y_true, "Blobs")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('k', REFINEMENT_K_TESTS)
    def test_refinement_k_impact_mknn(self, repeats, noise, k):
        # generate data, fit model, dump to results
        X, y_true  = datasets.make_blobs(n_samples=1000, random_state=40, cluster_std=noise)
        model      = SpectralClustering(3, refinement='mutual_knn', k=k)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(1000, noise, t, 'Refinement k Impact MKNN', k, X, y_pred, y_true, "Blobs")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

@pytest.mark.parametrize('repeats' , range(NUM_REPEATS))
@pytest.mark.parametrize('n_points', INPUT_SIZES)
@pytest.mark.parametrize('noise'   , [-1])
class TestTimeCorrectnessVariedBlobs:

    def test_default(self, repeats, n_points, noise):
        # generate data, fit model, dump to results
        X, y_true  = datasets.make_blobs(n_samples=n_points, cluster_std=[1.0, 2.5, 0.5], random_state=170)
        model      = SpectralClustering(3)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'DEFAULT', 'DEFAULT', X, y_pred, y_true, "Varied Blobs")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['decomposition'])
    def test_decomposition(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = datasets.make_blobs(n_samples=n_points, cluster_std=[1.0, 2.5, 0.5], random_state=170)
        model      = SpectralClustering(3, decomposition = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Decomposition', method, X, y_pred, y_true, "Varied Blobs")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['laplacian'])
    def test_laplacian(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = datasets.make_blobs(n_samples=n_points, cluster_std=[1.0, 2.5, 0.5], random_state=170)
        model      = SpectralClustering(3, laplacian = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Laplacian', method, X, y_pred, y_true, "Varied Blobs")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['affinity'])
    def test_affinity(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = datasets.make_blobs(n_samples=n_points, cluster_std=[1.0, 2.5, 0.5], random_state=170)
        model      = SpectralClustering(3, affinity = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Affinity', method, X, y_pred, y_true, "Varied Blobs")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['refinement'])
    def test_refinement(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = datasets.make_blobs(n_samples=n_points, cluster_std=[1.0, 2.5, 0.5], random_state=170)
        model      = SpectralClustering(3, refinement = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Refinement', method, X, y_pred, y_true, "Varied Blobs")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['standardisation'])
    def test_standardisation(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true  = datasets.make_blobs(n_samples=n_points, cluster_std=[1.0, 2.5, 0.5], random_state=170)
        model      = SpectralClustering(3, standardisation = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Standardisation', method, X, y_pred, y_true, "Varied Blobs")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

@pytest.mark.parametrize('repeats' , range(NUM_REPEATS))
@pytest.mark.parametrize('noise'   , [-1])
class TestRefinementParamImpactVariedBlobs:

    @pytest.mark.parametrize('k', REFINEMENT_K_TESTS)
    def test_refinement_k_impact_knn(self, repeats, noise, k):
        # generate data, fit model, dump to results
        X, y_true  = datasets.make_blobs(n_samples=1000, cluster_std=[1.0, 2.5, 0.5], random_state=170)
        model      = SpectralClustering(3, refinement='knn', k=k)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(1000, noise, t, 'Refinement k Impact KNN', k, X, y_pred, y_true, "Varied Blobs")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('k', REFINEMENT_K_TESTS)
    def test_refinement_k_impact_mknn(self, repeats, noise, k):
        # generate data, fit model, dump to results
        X, y_true  = datasets.make_blobs(n_samples=1000, cluster_std=[1.0, 2.5, 0.5], random_state=170)
        model      = SpectralClustering(3, refinement='mutual_knn', k=k)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(1000, noise, t, 'Refinement k Impact MKNN', k, X, y_pred, y_true, "Varied Blobs")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

@pytest.mark.parametrize('repeats' , range(NUM_REPEATS))
@pytest.mark.parametrize('n_points', INPUT_SIZES)
@pytest.mark.parametrize('noise'   , [0.5, 0.75, 1, 1.25, 1.5])
class TestTimeCorrectnessAniso:

    def test_default(self, repeats, n_points, noise):
        # generate data, fit model, dump to results
        X, y_true = datasets.make_blobs(n_samples=n_points, random_state=170, cluster_std=noise)
        X = np.dot(X, [[0.6, -0.6], [-0.4, 0.8]])
        model      = SpectralClustering(3)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'DEFAULT', 'DEFAULT', X, y_pred, y_true, "Aniso")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['decomposition'])
    def test_decomposition(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true = datasets.make_blobs(n_samples=n_points, random_state=170, cluster_std=noise)
        X = np.dot(X, [[0.6, -0.6], [-0.4, 0.8]])
        model      = SpectralClustering(3, decomposition = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Decomposition', method, X, y_pred, y_true, "Aniso")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['laplacian'])
    def test_laplacian(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true = datasets.make_blobs(n_samples=n_points, random_state=170, cluster_std=noise)
        X = np.dot(X, [[0.6, -0.6], [-0.4, 0.8]])
        model      = SpectralClustering(3, laplacian = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Laplacian', method, X, y_pred, y_true, "Aniso")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['affinity'])
    def test_affinity(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true = datasets.make_blobs(n_samples=n_points, random_state=170, cluster_std=noise)
        X = np.dot(X, [[0.6, -0.6], [-0.4, 0.8]])
        model      = SpectralClustering(3, affinity = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Affinity', method, X, y_pred, y_true, "Aniso")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['refinement'])
    def test_refinement(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true = datasets.make_blobs(n_samples=n_points, random_state=170, cluster_std=noise)
        X = np.dot(X, [[0.6, -0.6], [-0.4, 0.8]])
        model      = SpectralClustering(3, refinement = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Refinement', method, X, y_pred, y_true, "Aniso")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('method', PIPELINE_METHODS['standardisation'])
    def test_standardisation(self, repeats, n_points, noise, method):
        # generate data, fit model, dump to results
        X, y_true = datasets.make_blobs(n_samples=n_points, random_state=170, cluster_std=noise)
        X = np.dot(X, [[0.6, -0.6], [-0.4, 0.8]])
        model      = SpectralClustering(3, standardisation = method)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(n_points, noise, t, 'Standardisation', method, X, y_pred, y_true, "Aniso")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

@pytest.mark.parametrize('repeats' , range(NUM_REPEATS))
@pytest.mark.parametrize('noise'   , [0.5, 0.75, 1, 1.25, 1.5])
class TestRefinementParamImpactAniso:

    @pytest.mark.parametrize('k', REFINEMENT_K_TESTS)
    def test_refinement_k_impact_knn(self, repeats, noise, k):
        # generate data, fit model, dump to results
        X, y_true = datasets.make_blobs(n_samples=1000, random_state=170, cluster_std=noise)
        X = np.dot(X, [[0.6, -0.6], [-0.4, 0.8]])
        model      = SpectralClustering(3, refinement='knn', k=k)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(1000, noise, t, 'Refinement k Impact KNN', k, X, y_pred, y_true, "Aniso")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')

    @pytest.mark.parametrize('k', REFINEMENT_K_TESTS)
    def test_refinement_k_impact_mknn(self, repeats, noise, k):
        # generate data, fit model, dump to results
        X, y_true = datasets.make_blobs(n_samples=1000, random_state=170, cluster_std=noise)
        X = np.dot(X, [[0.6, -0.6], [-0.4, 0.8]])
        model      = SpectralClustering(3, refinement='mutual_knn', k=k)
        t, y_pred  = run_timeout_fn(model.fit, X)

        dump_result(1000, noise, t, 'Refinement k Impact MKNN', k, X, y_pred, y_true, "Aniso")
        if t == MAX_TIMEOUT_SECS:
            pytest.skip('Timed out')
