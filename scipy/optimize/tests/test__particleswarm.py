"""
Unit tests for particle swarm optimisation
"""

import pytest
import numpy as np

from scipy.optimize import particleswarm, OptimizeResult

def rast(x):
    func =  np.sum(x*x - 10*np.cos(2*np.pi*x)) + 10*np.size(x)
    return func
def f(x):
    return np.sum(x**2)

def test_particle_swarm():
    # Test the particle swarm optimisation on a simple quadratic function
    res = particleswarm(rast,50, 1000, 0.8, 1, 1, 2)
    assert res.success

def test_particle_swarm_rast():
    # Test the particle swarm optimisation on the rast function
    res = particleswarm(rast, 50, 1000, 0.8, 1, 1, 2)
    assert res.success
    assert isinstance(res, OptimizeResult)
    assert np.allclose(res.fun, 0, atol=1e-4)

def test_particle_swarm_quadratic():
    # Test the particle swarm optimisation on a simple quadratic function
    res = particleswarm(f, 50, 1000, 0.8, 1, 1, 2)
    assert res.success
    assert isinstance(res, OptimizeResult)
    assert np.allclose(res.fun, 0, atol=1e-4)

def test_particle_swarm_invalid_params():
    # Test the particle swarm optimisation with invalid parameters
    with pytest.raises(ValueError):
        particleswarm(f, -50, 1000, 0.8, 1, 1, 2)