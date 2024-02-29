import datetime
from src.data_generation import sklearn_make_moons
import pytest

# environment vars for preventing long runtime and repeats
NUM_REPEATS      = 5
MAX_TIMEOUT_SECS = 60 * 60

# environment variables defining repeatability and input size/noise/num_clusters
RAND_SEED       = 42
INPUT_SIZES     = [x for x in range(100, 3001, 100)]
INPUT_NOISES    = [0.00, 0.05, 0.10, 0.15]
INPUT_NOISE_SIZE_COMBOS = [{'n_points': a, 'noise': b} for a in INPUT_SIZES for b in INPUT_NOISES]
INPUT_NUM_MOONS = [3, 4, 5, 6]


# where to store current run results
RESULTS_DUMP_FOLDER = f"./results/res_{datetime.datetime.now().strftime('%Y_%m_%d_T%H_%M_%S')}"

# complete set of modules available to test for SpectralClustering

# generate the two moons problem
@pytest.fixture(scope='module')
def binary_moons_data(request):
    n_points = request.param['n_points']
    noise = request.param['noise']
    X, labels = sklearn_make_moons(n_points, noise, RAND_SEED)
    return X, labels

# TODO: helper functions: dumping new timing results, correctness etc.