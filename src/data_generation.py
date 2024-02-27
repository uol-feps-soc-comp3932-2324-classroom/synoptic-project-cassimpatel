import sklearn.datasets
import numpy as np
import math
import random

def sklearn_make_moons(n_points, sigma, rand_state = None):
    X, labels = sklearn.datasets.make_moons(n_points, shuffle=True, noise=0.1, random_state=rand_state)
    return X, labels

def make_many_moons(n_moons, sigma, n_points, y_shift = 0.5, rand_state = None):
    # TODO: error checking for invalid parameters specified

    # seed if set
    rng = np.random.default_rng(rand_state)

    # calculate number of points per moon
    n_points_per_moon = math.ceil(n_points / n_moons)
    moons = []

    for y in range(n_moons):
        # generate random values over pi range, factor to flip if an odd indexed moon
        q = rng.uniform(0,np.pi,size=n_points_per_moon)
        factor = -1 if y % 2 else 1
        
        # calculate x and y positions based on circle transformations, and cluster ground_truth
        moon = np.zeros((n_points_per_moon, 3))
        moon[:,0] = np.cos(q) + y + 0.3*y
        moon[:,1] = (np.sin(q) * factor) + (factor == -1) * y_shift
        moon[:,2] = y
        moons.append(moon)

        # apply random noise to x and y points
        noise = rng.normal(0, sigma, size=moon[:,:2].shape)
        moon[:,:2] += noise

    # join into np_array, randomise order and trim additional points
    moons = np.concatenate(moons)
    rng.shuffle(moons)
    moons = moons[:n_points, :]

    return moons[:,:2], moons[:,2]
