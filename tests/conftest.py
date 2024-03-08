import os
import time
import json
import signal
from datetime import datetime
from importlib import reload

import src
from src.SpectralClustering import SpectralClustering
from sklearn.metrics import cluster
from src.data_generation import sklearn_make_moons


# environment vars for preventing long runtime and repeats
NUM_REPEATS      = 3
# MAX_TIMEOUT_SECS = 60 * 60
MAX_TIMEOUT_SECS = 60

# input size/noise/num_clusters
RAND_SEED       = None
# INPUT_SIZES     = [x for x in range(100, 3001, 100)]
INPUT_SIZES     = [x for x in range(100, 500, 100)]
INPUT_NOISES    = [0.00, 0.05, 0.10, 0.15, 0.2]
INPUT_NUM_MOONS = [3, 4, 5, 6]

# where to store current run results
RESULTS_DUMP_FOLDER = f'./results/res_{datetime.now().strftime("%Y_%m_%d_T%H_%M_%S")}'
RESULTS_DUMP_DOC    = f'{RESULTS_DUMP_FOLDER}/results_dump.json'
RESULTS_REPORT_DOC  = f'{RESULTS_DUMP_FOLDER}/report.html'

# complete set of modules available to test for SpectralClustering
# TODO: configure this to read straight from the module itself
DECOMP_METHODS    = SpectralClustering.COMPONENT_OPTIONS['decomposition'].keys()
LAPLACIAN_METHODS = SpectralClustering.COMPONENT_OPTIONS['laplacian'    ].keys()
AFFINITY_METHODS  = SpectralClustering.COMPONENT_OPTIONS['affinity'     ].keys()
REFINEMENT_METHODS  = SpectralClustering.COMPONENT_OPTIONS['refinement'     ].keys()

# setup before a testing session: make sure dump folders exist for results
def pytest_configure(config):
    # reload custom package installation
    reload(src)
    
    # make sure folders exist, create empty dump file
    os.makedirs(os.path.dirname(RESULTS_DUMP_DOC), exist_ok=True)
    with open(RESULTS_DUMP_DOC, 'w') as f:
        f.write('')

    # set creation of HTML report
    config.option.htmlpath = f'{RESULTS_DUMP_FOLDER}/report.html'
    config.option.self_contained_html = True


# generic handler to time a function call and capture result
def run_timeout_fn(fn, *args, **kwargs):
    # start a signal alarm to stop the function if it runs too long
    def handler(sig, frame):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(MAX_TIMEOUT_SECS)

    # run the function with supplied arguments, capture result value
    try:
        start   = time.perf_counter()
        results = fn(*args, **kwargs)
        end     = time.perf_counter()
        signal.alarm(0)
    except TimeoutError:
        return (MAX_TIMEOUT_SECS, 'TIMEOUT')
    
    exec_time = end - start
    return (exec_time, results)


# generate the two moons problem
def binary_moons_data(n_points, noise):
    X, labels = sklearn_make_moons(n_points, noise, RAND_SEED)
    return X, labels


# calculate a set of metrics for correctness
def calc_correctness(pred_labels, ground_truth):
    # TODO: add more measures of correctness
    return {
        'adjusted_rand_score'         : cluster.adjusted_rand_score(ground_truth, pred_labels),
        'adjusted_mutual_info_score'  : cluster.adjusted_mutual_info_score(ground_truth, pred_labels),
        'normalized_mutual_info_score': cluster.normalized_mutual_info_score(ground_truth, pred_labels),
        'homogeneity_score'           : cluster.homogeneity_score(ground_truth, pred_labels),
        'completeness_score'          : cluster.completeness_score(ground_truth, pred_labels),
        'v_measure_score'             : cluster.v_measure_score(ground_truth, pred_labels),
    }

def dump_result(n_points, noise, time, experiment = 'DEFAULT', variant = 'DEFAULT', pred_labels = None, ground_truth = None,):
    timed_out = time == MAX_TIMEOUT_SECS
    new_entry = {
        'n_points'  : n_points,
        'noise'     : noise,
        'experiment': experiment,
        'variant'   : variant,
        'time'      : time,
        'timed_out' : "True" if timed_out else "False",
        'log_time'  : datetime.now()
    }

    if not timed_out:
        metrics = calc_correctness(pred_labels, ground_truth)
        new_entry.update(metrics)

    with open(RESULTS_DUMP_DOC, 'a') as f:
        entry_json = json.dumps(new_entry, sort_keys=True, default=str)
        f.write(f'{entry_json}\n')