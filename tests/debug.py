import numpy as np
import ice_simplex_assimilate

prior_data = np.load('../prior.npz')
prior_area = prior_data['areas']
prior_volume = prior_data['volumes']
open_frac = prior_data['open_frac'][0]
height_bounds = ice_simplex_assimilate.HeightBounds(prior_data['height_bounds'])

x_0 = np.ones(len(prior_area)) * open_frac
ice_simplex_assimilate.transport(prior_area, prior_volume, x_0, height_bounds)