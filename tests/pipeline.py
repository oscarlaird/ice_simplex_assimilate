import unittest
from ice_simplex_assimilate import *
from simplex_assimilate import fixed_point

class TestPipeline(unittest.TestCase):

    height_bounds = HeightBounds(np.array([0.0, 2.0, 4.0]))
    def test_no_change(self):
        areas = np.array([[0.5, 0.3],
                          [0.5, 0.3]])
        volumes = np.array([[0.5, 0.85],
                            [0.5, 0.95]])
        observations = np.array([0.2, 0.2])
        new_area, new_volume = transport(areas, volumes, observations, self.height_bounds)
        self.assertTrue(np.allclose(new_area, areas))
        self.assertTrue(np.allclose(new_volume, volumes))

    def test_unprecedented_water(self):
        """ Our samples our 100% ice, but the observations contain open water """
        areas = np.array([[0.5, 0.5],
                          [0.5, 0.5]])
        volumes = np.array([[0.5, 1.5],
                            [0.4, 1.4]])
        expected_area = np.array([[0.4, 0.4],
                                  [0.4, 0.4]])
        expected_volume = np.array([[0.40, 1.20],
                                    [0.32, 1.12]])
        observations = np.array([0.2, 0.2])
        new_area, new_volume = transport(areas, volumes, observations, self.height_bounds)
        self.assertTrue(np.allclose(new_area, expected_area))
        self.assertTrue(np.allclose(new_volume, expected_volume))

    def test_identical_samples(self):
        # check that identical samples don't cause a problem with Dirichlet estimation (e.g. alpha -> infty)
        areas = np.array([[0.4, 0.4],
                          [0.4, 0.4]])
        volumes = np.array([[0.40, 1.20],
                            [0.40, 1.20]])
        observations = np.array([0.5, 0.5])
        new_area, new_volume = transport(areas, volumes, observations, self.height_bounds)
        expected_area = np.array([[0.25, 0.25],
                                  [0.25, 0.25]])
        expected_volume = np.array([[0.25, 0.75],
                                    [0.25, 0.75]])
        self.assertTrue(np.allclose(new_area, expected_area))
        self.assertTrue(np.allclose(new_volume, expected_volume))

    def test_jump_class(self):
        """ Demonstrate that a surprising observation of open water can cause a change of class """
        areas = np.array([[0.4, 0.4],
                          [0.4, 0.4],
                          [0.4, 0],
                          [0.4, 0]])
        volumes = np.array([[0.40, 1.20],
                            [0.45, 1.25],
                            [0.40, 0],
                            [0.45, 0]])
        observations = np.array([0.8, 0.8, 0.1, 0.1])
        np.random.seed(0)
        new_area, new_volume = transport(areas, volumes, observations, self.height_bounds)
        expected_area = np.array([[0.2, 0],
                                  [0.2, 0],
                                  [0.4624, 0.4375],
                                  [0.4458, 0.4541]])
        expected_volume = np.array([[0.2021, 0],
                                    [0.2229, 0],
                                    [0.4690, 1.335],
                                    [0.5033, 1.388]])
        self.assertTrue(np.allclose(new_area, expected_area, atol=1e-3))
        self.assertTrue(np.allclose(new_volume, expected_volume, atol=1e-3))
