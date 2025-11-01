from os import path
from typing import Optional, Tuple

import numpy as np
from regional_pv.core import spv_workflow

from c3s_lot1_spv_conversion.utils.read_functions import (
    read_support_inputs,
    read_weather_inputs,
)
from c3s_lot1_spv_conversion.utils.support_functions import (
    prep_output_file,
)
from c3s_lot1_spv_conversion.utils.unit_tests import (
    test_downscaling_rate,
    test_input_paths,
    test_n_processes,
    test_pv_params,
    test_pv_params2,
)

# declaring global variables for mypy checking
n_p: int
dt_or: int
dt_ds: int
print_prog: bool
pv_tilt: Optional[np.ndarray]
pv_azim: Optional[np.ndarray]
th_coef: float
SSRD: np.ndarray
T2M: np.ndarray
meta: dict


def spv_calcs(ix: int) -> Tuple[int, tuple, np.ndarray]:
    """
    Prepare PV calculations, parallelization-friendly.

    Parameters
    ----------
    ix : int
        1D index of weather data grid. Calculations are done pixel-by-pixel.

    Returns
    -------
    ix : int
        Used to track calculation progress (if desired) when doing parallel
        calculations.
    index : tuple
        Needed to store output when doing parallel calculations.
    out : np.ndarray
        Photovoltaic capacity factor for given pixel.

    """
    # calls global variables initially defined in compute_spv()
    # numper of processes, original and downscaled time resolution,
    # boolean to print computed pixels
    global n_p, dt_or, dt_ds, print_prog
    # module tilt/azimuth & thermal coefficient
    global pv_tilt, pv_azim, th_coef
    # Surface solar radiation downwelling, air temperature at 2-m, metadata
    global SSRD, T2M, meta

    # prints computed pixel to keep track of calculation progression
    if print_prog:
        print(ix)

    # convert to 2D index
    index = np.unravel_index(ix, SSRD.shape[1:])

    # extracts information for pixel being calculated
    meta_ix = [meta["time"], meta["vLon"][index[1]], meta["vLat"][index[0]]]

    SSRD_ix = SSRD[:, index[0], index[1]].reshape(-1, 1)
    T2M_ix = T2M[:, index[0], index[1]].reshape(-1, 1)

    if isinstance(pv_tilt, np.ndarray) and isinstance(pv_azim, np.ndarray):
        if pv_tilt.ndim == 2:
            pv_tilt_ix = pv_tilt[index[0], index[1]]
        else:
            pv_tilt_ix = pv_tilt
        if pv_azim.ndim == 2:
            pv_azim_ix = pv_azim[index[0], index[1]]
        else:
            pv_azim_ix = pv_azim
    else:
        # tracking case, tilt and azim are defined within regional-pv
        # package, based on sun position and tracking setup
        pv_tilt_ix = None
        pv_azim_ix = None

    # modelling chain to obtain photovoltaic capacity factors
    out = spv_workflow(
        "Fixed",
        SSRD_ix,
        T2M_ix,
        meta_ix,
        pv_azim_ix,
        pv_tilt_ix,
        None,
        th_coef,
        dt_or,
        dt_ds,
    )

    return ix, index, out


def compute_spv(
    in_ssrd_path: str,
    in_t2m_path: str,
    in_excl_mask_path: Optional[str],
    DataStream: str,
    dt_orig: int = 60,
    dt_downscale: int = 15,
    pv_params: Optional[dict] = None,
    print_progress: bool = False,
    n_procs: int = 1,
    demo_mode: bool = False,
    to_save: bool = False,
    out_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read input files and proceed with conversion to PV capacity factor.

    Parameters
    ----------
    in_ssrd_path: str
        Path to .nc file containing surface solar radiation downwelling.
    in_t2m_path: str
        Path to .nc file containing 2-m air temperature.
    in_excl_mask_path : Optional[str]
        Path to exclusion mask .nc, filtering pixels in the calculations.
    DataStream: str
        IDs data as historical (HIST), projection (PROJ), seasonal (SEAS).
    dt_orig: int, optional
        Data's original time resolution, in minutes. Default: 60.
    dt_downscale: int, optional
        Time resolution to which data is downscaled, in minutes, to better
            account for variation in angle of incidence (then, reaggregated
            to hourly). Default: 15.
    pv_params: dict, optional
        Customizable PV parameters. Default: None.
    print_progress: bool
        Print computed pixels ID to keep track of progress. Default: False.
    n_procs: int, optional
        Number of processes (>1 means parallel computation). Default: 1.
    demo mode: bool, optional
        If True, assumes directories for example input data. Default: False.
    to_save: bool, optional
        If True, output is saved in a .nc file. Default: False.
    out_path: Optional[str]
        Path for expected output .nc file if to_save is True. Default: None.

    Returns
    -------
    out_all : np.ndarray
        Resulting 3D numpy array containing photovoltaic capacity factor.
    XX : np.ndarray
        Meshgrid on longitude, for plotting.
    YY : np.ndarray
        Meshgrid on latitude, for plotting.

    """
    # global variables
    global n_p, dt_or, dt_ds, print_prog
    global pv_tilt, pv_azim, th_coef
    global SSRD, T2M, meta

    # checks if n_procs is > 0
    test_n_processes(n_procs)

    # checks if dt_orig is multiple of dt_downscale
    # (downscale factor should be integer)
    test_downscaling_rate(dt_orig, dt_downscale)

    test_input_paths(in_ssrd_path, in_t2m_path)

    # if outputs are to be saved, checks path provided
    if to_save:
        msg = "out_path variable must be str"
        assert isinstance(out_path, str), msg

        msg = "out_path does not point to a .nc file"
        assert out_path.endswith(".nc"), msg

        msg = "out_path points to a non-existing directory"
        directory = path.dirname(out_path)
        assert path.exists(directory), msg

    # needs this renaming since a parameter cannot be a global variable
    n_p = n_procs
    dt_or = dt_orig
    dt_ds = dt_downscale
    print_prog = print_progress

    if pv_params:  # not None
        # checks for user defined inputs
        test_pv_params(pv_params)
        th_coef = pv_params.get("thermal_coeff", 21)
    else:
        # Ross coefficient in °C per kW/m2
        # Default value for free standing installations
        # Skoplaki (2008). doi: 10.1016/j.solmat.2008.05.016
        th_coef = 21

    if in_excl_mask_path == "default":
        import importlib.resources as pkg_resources
        import json

        from c3s_lot1_spv_conversion import anci

        with pkg_resources.path(anci, "default_mask_name.json") as template:
            with open(template, "r") as f:
                mask_file_name = json.load(f)[f"excl_mask_path_{DataStream}"]

            with pkg_resources.path(anci, mask_file_name) as temp:
                in_excl_mask_path = str(temp)

    # reads weather data, namely ssrd and t2m
    # meta variables are extracted from ssrd (vLat,vLon,time)
    SSRD, T2M, meta = read_weather_inputs(in_ssrd_path, in_t2m_path)

    # read supporting data
    # pv_tilt and pv_azim are equal to values in pv_params if existing
    pv_tilt, pv_azim, excl_mask = read_support_inputs(
        SSRD.shape, in_excl_mask_path, pv_params
    )

    if pv_params:
        # if non-single value, checks for same shape as climate data
        test_pv_params2(pv_params, SSRD.shape)

    # ID indices where it is useful to calculate SPV, in 1d format
    # (i.e. skips nighttime period and excluded areas)
    if excl_mask:  # if an exclusion mask is provided
        ok_index = (excl_mask == 0) & (SSRD.sum(axis=0) > 0)
    else:
        ok_index = SSRD.sum(axis=0) > 0
    ok_index = np.ravel_multi_index(np.where(ok_index), SSRD.shape)

    # array where to store spv output
    out_all = np.zeros_like(SSRD)
    if n_procs == 1:
        for ix in ok_index.tolist():
            # first output is ix, only needed in parallel computation
            _, index, out = spv_calcs(ix)

            # stores output
            out_all[:, index[0], index[1]] = out

    # parallel computation
    # for the moment, only for for POSIX systems (e.g. Linux)
    elif n_procs > 1:
        import multiprocessing as mp

        try:
            mp.set_start_method("fork")
        except ValueError:
            msg = """The parallel computations failed. One possible
            reason is that, for the moment, this is only possible for
            POSIX systems (Linux, Mac)."""
            raise Exception(msg)
        except RuntimeError:  # has been set before (in a previous run)
            pass
        with mp.Pool(n_procs) as pool:
            for ix, index, out in pool.imap_unordered(
                spv_calcs, ok_index.tolist(), chunksize=64
            ):
                # stores ouptut
                out_all[:, index[0], index[1]] = out

            pool.close()
            pool.join()

    # meshgrid, can be useful for plotting
    XX, YY = np.meshgrid(meta["vLon"], meta["vLat"])

    if to_save:
        if isinstance(out_path, str):
            # creates and encodes output .nc file with PV capacity factor
            prep_output_file(
                out_path,
                DataStream,
                out_all,
                meta["vLon"],
                meta["vLat"],
                meta["time"],
            )

    return out_all, XX, YY
