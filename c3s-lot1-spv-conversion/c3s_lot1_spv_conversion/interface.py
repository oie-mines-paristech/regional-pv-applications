from os import path
from typing import Optional, Tuple, Union

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
dt_or: Union[int, dict]
dt_ds: int
print_prog: bool
pv_param: dict
SSRD: np.ndarray
T2M: np.ndarray
meta: dict
pv_tilt_ix: Optional[Union[int, float]]
pv_azim_ix: Optional[Union[int, float]]


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
    global pv_param
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

    if pv_param.get("tracking", 0) > 0:
        pv_tilt_ix = None
        pv_azim_ix = None
    else:
        if isinstance(pv_param["tilt"], (int, float)):
            pv_tilt_ix = pv_param["tilt"]
        elif isinstance(pv_param["tilt"], np.ndarray):
            if pv_param["tilt"].ndim == 2:
                pv_tilt_ix = pv_param["tilt"][index[0], index[1]].item()
            else:
                pv_tilt_ix = pv_param["tilt"].item()

        if isinstance(pv_param["azim"], (int, float)):
            pv_azim_ix = pv_param["azim"]
        elif isinstance(pv_param["azim"], np.ndarray):
            if pv_param["azim"].ndim == 2:
                pv_azim_ix = pv_param["azim"][index[0], index[1]].item()
            else:
                pv_azim_ix = pv_param["azim"].item()

    # modelling chain to obtain photovoltaic capacity factors
    out = spv_workflow(
        ssrd=SSRD_ix,
        t2m=T2M_ix,
        meta=meta_ix,  # lat, lon, time
        azim=pv_azim_ix,
        tilt=pv_tilt_ix,
        w_orient=None,
        k=pv_param["th_coeff"],
        tracking=pv_param["tracking"],
        dt_orig=dt_or,
        dt_downscale=dt_ds,
    )

    return ix, index, out


def compute_spv(
    in_ssrd_path: str,
    in_t2m_path: str,
    in_excl_mask_path: Optional[str],
    DataStream: str,
    dt_orig: Optional[Union[int, dict]] = None,
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
    dt_orig: Union[int,dict], optional
        Data's original time resolution, in minutes. Dictionary if different
        values for ssrd and t2m. Default: 60.
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
    global pv_param
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
    dt_ds = dt_downscale
    print_prog = print_progress

    if dt_orig:  # if not None
        # user-defined
        dt_or = dt_orig
    else:
        # default values
        if DataStream != "SEAS":
            dt_or = 60
        else:
            dt_or = {"ssrd": 24 * 60, "t2m": 6 * 60}

    if pv_params:  # not None
        # checks for user defined inputs
        test_pv_params(pv_params)

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

    # if seasonal, adjust default tilt and azim from 0.25 to 1deg resolution
    if DataStream == "SEAS":
        if not pv_params or "tilt" not in pv_params:
            if isinstance(pv_tilt, np.ndarray):
                dims = pv_tilt[1:, :].shape
                pv_tilt = (
                    pv_tilt[1:, :]
                    .reshape(dims[0] // 4, 4, dims[1] // 4, 4)
                    .mean(axis=(1, 3))
                )
            if isinstance(pv_azim, np.ndarray):
                dims = pv_azim[1:, :].shape
                pv_azim = (
                    pv_azim[1:, :]
                    .reshape(dims[0] // 4, 4, dims[1] // 4, 4)
                    .mean(axis=(1, 3))
                )

    if pv_params:
        # if non-single value, checks for same shape as climate data
        test_pv_params2(pv_params, SSRD.shape)

        # if was not provided from the start, assume default values
        if "tilt" not in pv_params:
            pv_params["tilt"] = pv_tilt
        if "azim" not in pv_params:
            pv_params["azim"] = pv_azim
    else:
        # default values
        pv_params = {
            "tilt": pv_tilt,
            "azim": pv_azim,
            "tracking": 0,
            "th_coeff": 21,
        }

    if isinstance(pv_params, dict):
        pv_param = pv_params

    # ID indices where it is useful to calculate SPV, in 1d format
    # (i.e. skips nighttime period and excluded areas)
    if isinstance(excl_mask, np.ndarray):  # if an exclusion mask is provided
        ok_index = (excl_mask == 0) & (SSRD.sum(axis=0) > 0)
    else:
        ok_index = SSRD.sum(axis=0) > 0
    ok_index = np.ravel_multi_index(np.where(ok_index), SSRD.shape[1:])

    # array where to store spv output
    # if data is daily or coarser
    if isinstance(dt_or, dict) and dt_or["ssrd"] >= 24 * 60:
        # adjust for different final resolution (hourly)
        time_f = dt_or["ssrd"] // 60
        out_all = np.zeros((SSRD.shape[0] * time_f, *SSRD.shape[1:]))

        # create hourly timestamps for final output
        hour_offsets = np.arange(time_f).astype("timedelta64[h]")
        meta["time2"] = (meta["time"][:, None] + hour_offsets).flatten()
    else:
        out_all = np.zeros_like(SSRD)
        meta["time2"] = meta["time"]

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

    del (SSRD, T2M)
    if to_save:
        if isinstance(out_path, str):
            # creates and encodes output .nc file with PV capacity factor
            prep_output_file(
                out_path,
                DataStream,
                out_all,
                meta["vLon"],
                meta["vLat"],
                meta["time2"],
            )

    return out_all, XX, YY
