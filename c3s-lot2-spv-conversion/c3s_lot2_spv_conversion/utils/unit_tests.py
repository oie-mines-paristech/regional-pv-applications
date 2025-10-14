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


def test_pv_typology(pv_typology: Union[str, int]) -> None:
    """
    Asserts that the user defines a valid PV typology.

    Parameters
    ----------
    pv_typology : Union[str,int]
        IDs PV typology being computed.

    Returns
    -------
    None

    """
    message = "pv_typology must be either string or int."
    assert isinstance(pv_typology, (int, str)), message

    if isinstance(pv_typology, int):
        message = """If int, pv_typology must be between 60-63.
        60: rooftop_industrial, 61: rooftop_residential,
        62: utility_fix, 63: utility_track_1axis."""
        assert (pv_typology > 59) & (pv_typology < 64), message

    if isinstance(pv_typology, str):
        message = """If str, pv_typology must be either a
        rooftop_industrial, rooftop_residential, utility_fix, or
        utility_track_1axis."""
        ok_types = [
            "rooftop_industrial",
            "60",
            "rooftop_residential",
            "61",
            "utility_fix",
            "62",
            "utility_track_1axis",
            "63",
        ]
        assert pv_typology in ok_types, message.replace("\n", " ")


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


def test_downscaling_rate(dt_orig: int, dt_downscale: int) -> None:
    """
    Asserts that intended downscaling is proportional to original time data.

    Parameters
    ----------
    dt_orig : int
        Original time resolution, in minutes.
    dt_downscale : int
        Time resolution for which data is downscaled to, in minutes.

    Returns
    -------
    None

    """
    message = "dt_orig and dt_downscale are not evenly divisible."
    assert dt_orig % dt_downscale == 0, message


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
