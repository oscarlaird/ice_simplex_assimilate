

def load_matlab(path):
    """Load a Matlab file into a dictionary.

    Parameters
    ----------
    path : str
        Path to the Matlab file.

    Returns
    -------
    dict
        Dictionary containing the Matlab file data.
    """
    import scipy.io as sio
    return sio.loadmat(path)