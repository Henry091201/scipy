import pytest
import numpy as np
from scipy.optimize import particleswarm, rosen
from scipy.optimize import OptimizeResult, TestState 

# Helper functions
def rast(x):
    func =  np.sum(x*x - 10*np.cos(2*np.pi*x)) + 10*np.size(x)
    return func

def quadratic(x):
    return np.sum(x**2)

def mock_topology(pso, particle_index):
    attr = pso.get_attributes()
    swarm_size = attr['swarm_size']
    return np.arange(swarm_size)

def mock_dynamic_inertia(pso):
    attr = pso.get_attributes()
    iteration = attr['current_iteration'] 
    max_iter = attr['max_iter']
    return 0.9 - (0.5 / max_iter) * iteration

# Test cases
class TestParticleSwarm:
    def test_rosenbrock(self):
        res = particleswarm(rast, 50, 2)
        assert res.success
        assert isinstance(res, OptimizeResult)
        assert np.allclose(res.fun, 0, atol=1e-4)
        
    def test_particle_swarm_quadratic(self):
        res = particleswarm(quadratic, 50, 2)
        assert res.success
        assert isinstance(res, OptimizeResult)
        assert np.allclose(res.fun, 0, atol=1e-4)
        
    def test_particle_swarm_rast(self):
        res = particleswarm(rast, 50, 2)
        assert res.success
        assert isinstance(res, OptimizeResult)
        assert np.allclose(res.fun, 0, atol=1e-4)
        
    def test_convergence(self):
        res = particleswarm(rosen, 50, 2, niter_success=100)
        msg = "Maximum number of iterations at global best reached"
        assert res.success
        assert res.message == msg
        assert isinstance(res, OptimizeResult)
        assert np.allclose(res.fun, 0, atol=1e-4)

    def test_out_of_bounds(self):
        bounds = ((-10, 10), (-10, 10))
        state_class = TestState(rast, 50, 2, max_iter=1000, w=0.729, c1=1.4, c2=1.4,
                    bounds=bounds, topology='star', seed=-1, niter_success=-1,
                    max_velocity=-1)
        state_class.setup_test()
        
        attr = state_class.get_attributes()
        assert np.all(attr['positions'] >= -10)
        assert np.all(attr['positions'] <= 10)
        assert np.all(attr['pbest_fitnesses'] != np.inf)

        state_class.set_particle_position(0, np.array([100, 100]))
        assert state_class.calculate_particle_fitness(0) == float('inf')

    def test_velocity_clamping(self):
        bounds = ((-10, 10), (-10, 10))
        
        state_class = TestState(rast, 50, 2, max_iter=1000, w=1, c1=1.4, c2=1.4,
                    bounds=bounds, topology='star', seed=-1, niter_success=-1,
                    max_velocity=3)
        state_class.setup_test()
        
        attr = state_class.get_attributes()
        assert np.all(attr['velocities'] >= -3)
        assert np.all(attr['velocities'] <= 3)
        
        state_class.set_particle_velocity(0, np.array([100, 100]))
        attr = state_class.get_attributes()
        assert not np.all(attr['velocities'] <= 3)
        state_class.update_all_velocities()
        assert np.all(attr['velocities'] <= 3)

    def test_topology(self):
        particleswarm(rast, 50, 2, topology="ring")
        particleswarm(rast, 50, 2, topology="star")
        particleswarm(rast, 50, 2, topology=mock_topology)

    def test_invalid_params(self):
        msg = "Swarm size must be greater than 0."
        with pytest.raises(ValueError, match=msg):
            particleswarm(quadratic, -50, 2)
        msg = "Maximum number of iterations must be greater than 0."
        with pytest.raises(ValueError, match=msg):
            particleswarm(quadratic, 50, 2, max_iter=-1000)
        msg = "Objective function must be callable."
        with pytest.raises(ValueError, match=msg):
            particleswarm("fail", 50, 2)
        msg = "Inertia weight must be greater than 0."
        with pytest.raises(ValueError, match=msg):
            particleswarm(quadratic, 50, 1000, 2, w=-0.8)
        msg = "Cognitive and social components must be greater than 0."
        with pytest.raises(ValueError, match=msg):
            particleswarm(quadratic, 50, 2, c1=-1)
        msg = "Number of dimensions must be greater than 0."
        with pytest.raises(ValueError, match=msg):
            particleswarm(quadratic, 50, -1)
        msg = 'Invalid topology. Must be callable or one of \'ring\' or \'star\'.'
        with pytest.raises(ValueError, match=msg):
            particleswarm(quadratic, 50, 2, topology="fail")
        msg = "Maximum velocity must be greater than 0."
        with pytest.raises(ValueError, match=msg):
            particleswarm(quadratic, 50, 2, max_velocity=-5)

    def test_1d(self):
        res = particleswarm(quadratic, 50, 1)
        assert res.success
        assert isinstance(res, OptimizeResult)
        assert np.allclose(res.fun, 0, atol=1e-4)
        assert res.x.shape == (1,)

    def test_2d(self):
        res = particleswarm(rast, 50, 2)
        assert res.success
        assert isinstance(res, OptimizeResult)
        assert np.allclose(res.fun, 0, atol=1e-4)
        assert res.x.shape == (2,)

    def test_3d(self):
        res = particleswarm(rosen, 50, 3)
        assert res.success
        assert isinstance(res, OptimizeResult)
        assert np.allclose(res.fun, 0, atol=1e-3)
        assert res.x.shape == (3,)

# Additional unit tests
def test_particle_swarm_invalid_params():
    with pytest.raises(ValueError):
        particleswarm(quadratic, -50, 2)

def test_array_dimensions():
    state_class = TestState(rast, 50, 2, max_iter=1000, w=0.729, c1=1.4, c2=1.4,
                    bounds=None, topology='star', seed=-1, niter_success=-1,
                    max_velocity=-1)
    
    attr = state_class.get_attributes()
    assert attr['velocities'].shape == (50, 2)
    assert attr['positions'].shape == (50, 2)
    assert attr['pbest_fitnesses'].shape == (50,)
    assert attr['pbest_fitness_positions'].shape == (50, 2)

def test_dynamic_inertia():
    bounds = np.array([[-10, 10], [-10, 10]])
    particleswarm(rast, 50, 2, w=mock_dynamic_inertia)

def test_velocity_initialisation():
    bounds = ((-10, 10), (-10, 10))
    state_class = TestState(rast, 50, 2, max_iter=1000, w=0.729, c1=1.4, c2=1.4,
                    bounds=bounds, topology='star', seed=-1, niter_success=-1,
                    max_velocity=-1)
    
    state_class.setup_test()
    assert np.all(state_class.get_attributes()['velocities'] >= -3)
    assert np.all(state_class.get_attributes()['velocities'] <= 3)

def test_boundary_swarm_size():
    res = particleswarm(quadratic, 1, 2)
    assert res.success

    with pytest.raises(ValueError):
        particleswarm(quadratic, 0, 2)

def test_boundary_dimensions():
    res = particleswarm(quadratic, 50, 1)
    assert res.success

def test_boundary_max_iterations():
    res = particleswarm(quadratic, 50, 2, max_iter=1)
    assert res.success

def test_boundary_parameters():
    res = particleswarm(quadratic, 50, 2, w=0.001, c1=0.001, c2=0.001)
    assert res.success

    res = particleswarm(quadratic, 50, 2, w=100, c1=100, c2=100)
    assert res.success

def test_random_seed():
    seed_value = 42

    res_fixed_seed = particleswarm(quadratic, 50, 2, seed=seed_value)
    res_same_fixed_seed = particleswarm(quadratic, 50, 2, seed=seed_value)

    assert np.allclose(res_fixed_seed.x, res_same_fixed_seed.x)
    assert np.allclose(res_fixed_seed.fun, res_same_fixed_seed.fun)
