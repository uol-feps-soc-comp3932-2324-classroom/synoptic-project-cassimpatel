import time
import signal
import pytest
from src.SpectralClustering import SpectralClustering
from conftest import NUM_REPEATS, MAX_TIMEOUT_SECS, INPUT_SIZES, INPUT_NOISES, binary_moons_data, dump_time

# generic handler to time a function call and capture result
def time_fn(fn, *args, **kwargs):
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

@pytest.mark.parametrize('repeats'  , range(NUM_REPEATS))
@pytest.mark.parametrize('n_points' , INPUT_SIZES)
@pytest.mark.parametrize('noise'    , INPUT_NOISES)
def test_time_default(repeats, n_points, noise):
    X, _  = binary_moons_data(n_points, noise)
    model = SpectralClustering(2, 'none', 'euclidean', 'eps', 'standard', 'dense', 'single', 'k-means', 'false')
    t, _  = time_fn(model.fit, X)

    if t == MAX_TIMEOUT_SECS:
        dump_time(n_points, noise, 'TIMED_OUT')
        pytest.skip('Timed out')

    dump_time(n_points, noise, t)