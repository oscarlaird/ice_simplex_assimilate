import unittest
from ice_simplex_assimilate import *
from simplex_assimilate.fixed_point import ONE

class TestProcess(unittest.TestCase):

    height_bounds = HeightBounds(np.array([0.0, 2.0, 4.0]))

    def test_check_raw_sample_legal(self):
        area = np.array([0.5, 0.5])
        volume = np.array([0.5, 1.5])
        self.assertTrue(check_raw_sample_legal(area, volume, self.height_bounds))
        area = np.array([0.7, 0.5])
        volume = np.array([0.7, 1.5])
        self.assertFalse(check_raw_sample_legal(area, volume, self.height_bounds))
        area = np.array([0.5, 0.5])
        volume = np.array([1.5, 1.5])  # volume[0] is too large
        self.assertFalse(check_raw_sample_legal(area, volume, self.height_bounds))

    def test_pre_process_sample(self):
        # healthy sample
        area = np.array([0.5, 0.5])
        volume = np.array([0.5, 1.5])
        sample = pre_process_sample(area, volume, self.height_bounds)
        self.assertTrue(np.all(sample == np.array([0, ONE/4, ONE/4, ONE/4, ONE/4])))
        # healthy sample
        area = np.array([1, 0])
        volume = np.array([1, 0])
        sample = pre_process_sample(area, volume, self.height_bounds)
        self.assertTrue(np.all(sample == np.array([0, ONE/2, ONE/2, 0, 0])))
        # mass in second interval should be thresholded
        area = np.array([1 - 1e-7, 1e-7])
        volume = np.array([1 - 1e-7, 3 * 1e-7])
        sample = pre_process_sample(area, volume, self.height_bounds)
        self.assertTrue(np.all(sample == np.array([0, ONE/2, ONE/2, 0, 0])))
        # mass should be moved from the left side to the right side of the interval
        area = np.array([1, 0])
        volume = np.array([0, 0])
        sample = pre_process_sample(area, volume, self.height_bounds)
        self.assertTrue(np.all(sample == np.array([0, ONE - 2147, 2147, 0, 0])))
        # open water implied
        area = np.array([0.5, 0])
        volume = np.array([0.5, 0])
        sample = pre_process_sample(area, volume, self.height_bounds)
        self.assertTrue(np.all(sample == np.array([ONE/2, ONE/4, ONE/4, 0, 0])))

    def test_pre_process_ensemble(self):
        area = np.array([[0.5, 0.5],
                        [1, 0],
                        [1 - 1e-7, 1e-7],
                        [1, 0],
                        [0.5, 0]])
        volume = np.array([[0.5, 1.5],
                           [1, 0],
                           [1 - 1e-7, 3 * 1e-7],
                           [0, 0],
                           [0.5, 0]])
        ensemble = pre_process_ensemble(area, volume, self.height_bounds)
        expected_ensemble = np.array([[0, ONE/4, ONE/4, ONE/4, ONE/4],
                                        [0, ONE/2, ONE/2, 0, 0],
                                        [0, ONE/2, ONE/2, 0, 0],
                                        [0, ONE - 2147, 2147, 0, 0],
                                        [ONE/2, ONE/4, ONE/4, 0, 0]])
        self.assertTrue(np.all(ensemble == expected_ensemble))

    def test_post_process_sample(self):
        # healthy sample
        sample = np.array([0, ONE/4, ONE/4, ONE/4, ONE/4])
        area, volume = post_process_sample(sample, self.height_bounds)
        self.assertTrue(np.all(area == np.array([0.5, 0.5])))
        self.assertTrue(np.all(volume == np.array([0.5, 1.5])))
        # sample with all open water
        sample = np.array([ONE, 0, 0, 0, 0])
        area, volume = post_process_sample(sample, self.height_bounds)
        self.assertTrue(np.all(area == np.array([0, 0])))
        self.assertTrue(np.all(volume == np.array([0, 0])))

    def test_post_process_ensemble(self):
        ensemble = np.array([[0, ONE/4, ONE/4, ONE/4, ONE/4],
                             [ONE, 0, 0, 0, 0]])
        area, volume = post_process_ensemble(ensemble, self.height_bounds)
        expected_area = np.array([[0.5, 0.5],
                                    [0, 0]])
        expected_volume = np.array([[0.5, 1.5],
                                    [0, 0]])
        self.assertTrue(np.all(area == expected_area))
        self.assertTrue(np.all(volume == expected_volume))