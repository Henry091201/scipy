"""
Unit tests for particle swarm optimisation
"""

import pytest
import numpy as np
from scipy.optimize import rosen

from scipy.optimize import particleswarm, OptimizeResult
# Set the seed for reproducibility

def rast(x):
    func =  np.sum(x*x - 10*np.cos(2*np.pi*x)) + 10*np.size(x)
    return func
def quadratic(x):
    return np.sum(x**2)


def test_particle_swarm_invalid_params():
    # Test the particle swarm optimisation with invalid parameters
    with pytest.raises(ValueError):
        particleswarm(quadratic, -50, 2)

class TestParticleSwarm:
    # Test correctness
    def test_rosenbrock(self):
        # Test the particle swarm optimisation on the rosenbrock function
        res = particleswarm(rast, 50, 2)
        assert res.success
        assert isinstance(res, OptimizeResult)
        assert np.allclose(res.fun, 0, atol=1e-4)
        
    def test_particle_swarm_quadratic(self):
        # Test the particle swarm optimisation on a simple quadratic function
        res = particleswarm(quadratic, 50, 2)
        assert res.success
        assert isinstance(res, OptimizeResult)
        assert np.allclose(res.fun, 0, atol=1e-4)
        
    def test_particle_swarm_rast(self):
        # Test the particle swarm optimisation on the rastrigin function
        res = particleswarm(rast, 50, 2)
        assert res.success
        assert isinstance(res, OptimizeResult)
        assert np.allclose(res.fun, 0, atol=1e-4)
        
    def test_convergence(self):
        # Test the convergence of the particle swarm optimisation
        res = particleswarm(rosen, 50, 2, niter_success=100)
        msg = "Maximum number of iterations at global best reached"
        assert res.success
        assert res.message == msg
        assert isinstance(res, OptimizeResult)
        assert np.allclose(res.fun, 0, atol=1e-4)

    def test_invalid_params(self):
        # Test the particle swarm optimisation with invalid parameters
        msg = "Swarm size must be greater than 0."
        with pytest.raises(ValueError, match=msg):
            particleswarm(quadratic, -50, 2)
        msg = "Maximum number of iterations must be greater than 0."
        with pytest.raises(ValueError, match=msg):
            particleswarm(quadratic,50, 2, max_iter=-1000)
        msg = "Objective function must be callable."
        with pytest.raises(ValueError, match=msg):
            particleswarm("fail",50, 2)
        msg = "Inertia weight must be greater than 0."
        with pytest.raises(ValueError, match=msg):
            particleswarm(quadratic,50, 1000, 2, w=-0.8)
        msg = "Cognitive and social components must be greater than 0."
        with pytest.raises(ValueError, match=msg):
            particleswarm(quadratic,50, 2, c1=-1)
        msg = "Number of dimensions must be greater than 0."
        with pytest.raises(ValueError, match=msg):
            particleswarm(quadratic,50, -1)
        msg = 'Topology must be callable.'
        with pytest.raises(ValueError, match=msg):
            particleswarm(quadratic,50, 2, topology="fail")
        msg = "Maximum velocity must be greater than 0."
        with pytest.raises(ValueError, match=msg):
            particleswarm(quadratic,50, 2, max_velocity=-5)
       
            
    def test_1d(self):
        # Test the particle swarm optimisation in 1D
        res = particleswarm(quadratic, 50, 1)
        assert res.success
        assert isinstance(res, OptimizeResult)
        assert np.allclose(res.fun, 0, atol=1e-4)
        assert res.x.shape == (1,)
        
    def test_2d(self):
        # Test the particle swarm optimisation in 2D
        res = particleswarm(rast, 50, 2)
        assert res.success
        assert isinstance(res, OptimizeResult)
        assert np.allclose(res.fun, 0, atol=1e-4)
        assert res.x.shape == (2,)
        
    def test_3d(self):
        # Test the particle swarm optimisation in 3D
        res = particleswarm(rosen, 50, 3)
        assert res.success
        assert isinstance(res, OptimizeResult)
        assert np.allclose(res.fun, 0, atol=1e-4)
        assert res.x.shape == (3,)
        
        
    