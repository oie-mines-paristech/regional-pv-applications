from os import path
from typing import Optional, Union

import numpy as np


def test_input_paths(in_ssrd_path: str, in_t2m_path: str) -> None:
    """
    Checks if user-defined input paths exist and point to .nc files.

    Parameters
    ----------
    in_ssrd_path : str
        Path to ssrd (solar radiation) input.
    in_t2m_path : str
        Path to t2m (air temperature) input.

    Returns
    -------
    None

    """
    vars_ = ["ssrd", "t2m"]
    for i, file_path in enumerate([in_ssrd_path, in_t2m_path]):
        msg = f"{vars_[i]} path points to a non-existing directory"
        directory = path.dirname(file_path)
        assert path.exists(directory), msg

        msg = f"{vars_[i]} path does not point to a .nc file"
        assert file_path.endswith(".nc"), msg


def test_output_paths(out_path: Optional[str]) -> None:
    """
    Checks if user-defined output path points to .nc & existing directory.

    Parameters
    ----------
    out_path : Optional[str]
        User-defined output path.

    Returns
    -------
    None

    """
    msg = "if to_save=True, out_path must be str"
    assert isinstance(out_path, str), msg

    msg = "out_path does not point to a .nc file"
    assert out_path.endswith(".nc"), msg

    msg = "out_path points to a non-existing directory"
    directory = path.dirname(out_path)
    assert path.exists(directory), msg


def test_file_lists(
    in_ssrd: list[str],
    in_t2m: list[str],
    out_filenames: list[str],
) -> None:
    """
    Asserts that number of ssrd input files matches that of t2m ones.

    Parameters
    ----------
    in_ssrd : list[str]
        User-defined list of ssrd input filus.
    in_t2m : list[str]
        User-defined list of t2m input filus.
    out_filenames : list[str]
        User-defined list of output filenames. Default is 'default',
        which based on ssrd input name.

    Returns
    -------
    None

    """
    message = "SSRD file list must not be empty."
    assert len(in_ssrd) > 0, message

    message = "T2M file list must not be empty."
    assert len(in_t2m) > 0, message

    message = "SSRD and T2M file lists should have same length."
    assert len(in_ssrd) == len(in_t2m), message

    message = "SSRD and T2M file lists should have same length."
    assert len(in_ssrd) == len(in_t2m), message

    if len(out_filenames) > 1:
        message = "Input and output list should have same size."
        assert len(out_filenames) == len(in_t2m), message

        message = "out_filenames elements should end in .nc."
        assert all(file.endswith(".nc") for file in out_filenames), message

    elif (len(out_filenames) == 1) & (len(in_ssrd) > 1):
        message = """"
        out_filenames seems to have too few elements or that 'default' option
        was not properly defined.
        """
        assert out_filenames[0] == "default", message

    elif (len(out_filenames) == 1) & (len(in_ssrd) == 1):
        if out_filenames[0] != "default":
            message = "out_filenames element should end in .nc."
            assert out_filenames[0].endswith(".nc"), message


def test_for_nans(var: np.ndarray, var_name: str) -> None:
    """
    Asserts that weather input data contains no nans.

    Parameters
    ----------
    var : np.ndarray
        Input data to be checked.
    var_name : str
        IDs variable under check.

    Returns
    -------
    None

    """
    message = "{} has nan values.".format(var_name)
    assert np.isnan(var).sum() == 0, message


def test_downscaling_rate(dt_orig: Union[int, dict], dt_downscale: int) -> None:
    """
    Asserts that intended downscaling is proportional to original time data.

    Parameters
    ----------
    dt_orig : Union[int,dict]
        Original time resolution, in minutes. Dict when ssrd and t2m have
        different values.
    dt_downscale : int
        Time resolution for which data is downscaled to, in minutes.

    Returns
    -------
    None

    """
    if isinstance(dt_orig, int):
        message = "dt_orig and dt_downscale are not evenly divisible."
        assert dt_orig % dt_downscale == 0, message
    elif isinstance(dt_orig, dict):
        message = "dt_orig and dt_downscale are not evenly divisible."
        assert dt_orig["ssrd"] % dt_downscale == 0, message
        assert dt_orig["t2m"] % dt_downscale == 0, message


def test_n_processes(n_procs: int) -> None:
    """

    Asserts that n_procs is a positive integer.

    Parameters
    ----------
    n_procs : int
        Number of processes (>1 means parallel computation).

    Returns
    -------
    None

    """
    message = "n_proc must be a positive integer."
    assert n_procs > 0, message


def test_pv_params(pv_params: dict) -> None:
    """
    Asserts that PV parameters are valid and if values provided are feasible.

    Parameters
    ----------
    pv_params : dict
        Customizable PV parameters.

    Returns
    -------
    None

    """
    # checks pv_param type
    message = "If not None, pv_params must be dict."
    assert isinstance(pv_params, dict), message

    # checks keys
    accepted_keys = ["tilt", "azim", "tracking", "thermal_coeff"]
    invalid_keys = pv_params.keys() - accepted_keys

    if invalid_keys:
        print(f"Only {accepted_keys} are accepted as PV parameters.")
        print("Invalid keys found in pv_param:", invalid_keys)

    # checks module tilt
    if "tilt" in pv_params:
        tilt = pv_params["tilt"]

        message = "module tilt should be int, float, or np.ndarray."
        assert isinstance(tilt, (int, float, np.ndarray)), message

        message = "module tilt must be between 0 and 90 (degrees)."
        assert np.all(tilt >= 0) & np.all(tilt <= 90), message

    # checks module azimuth
    if "azim" in pv_params:
        azim = pv_params["azim"]

        message = "module azimuth should be int, float, or np.ndarray."
        assert isinstance(azim, (int, float, np.ndarray)), message

        message = "module azim must be between 0 and 360 (degrees)."
        assert np.all(azim >= 0) & np.all(azim <= 360), message

    if {"tilt", "azim"}.issubset(pv_params):
        if isinstance(tilt, np.ndarray) & isinstance(azim, np.ndarray):
            if (tilt.ndims > 1) & (tilt.azim > 1):
                message = """
                if both non-single values, tilt and azimuth need same shape.
                """
                assert tilt.shape == azim.shape, message

    # checks tracking
    if "tracking" in pv_params:
        track = pv_params["tracking"]

        message = "tracking parameter must be int."
        assert isinstance(track, int), message

        message = "tracking must be either 0, 1, or 2."
        assert np.all(azim >= 0) & np.all(azim <= 2), message

    # checks Ross coefficient
    if "th_coeff" in pv_params:
        th_coeff = pv_params["th_coeff"]

        message = "thermal coeff. should be int or float."
        assert isinstance(th_coeff, (int, float)), message

        message = "thermal coefficient must be between 0 and 80 °C/kW/m2."
        assert (th_coeff >= 0) & (th_coeff <= 80), message


def test_pv_params2(
    pv_params: dict,
    ref_shape: tuple,
) -> None:
    """
    Asserts that tilt and azim in pv_params have suitable shape.

    Parameters
    ----------
    pv_params : dict
        Customizable PV parameters.
    ref_shape : np.ndarray
        Array shape from weather input data.

    Returns
    -------
    None

    """
    for param in ["tilt", "azim"]:
        if pv_params[param].size > 1:
            message = """
            if non-single valued, {} need same shape as weather inputs.
            """.format(param)
            assert pv_params[param].shape == ref_shape, message
