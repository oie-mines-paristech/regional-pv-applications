import json
from os import path
from typing import Optional, Tuple

import numpy as np
import xarray as xr


def get_demo_paths() -> Tuple[str, str, str]:
    """
    Get paths for demo files contained in the package directory.

    Returns
    -------
    Tuple[list, list,str]
        List of ssrd and t2m demos files, and spv output path.

    """
    script_path = path.dirname(path.abspath(__file__))
    ssrd_dir = path.join(path.split(script_path)[0], "demo_data", "ssrd")
    t2m_dir = path.join(path.split(script_path)[0], "demo_data", "t2m")
    out_dir = path.join(path.split(script_path)[0], "demo_data", "spv")

    return ssrd_dir, t2m_dir, out_dir


def read_support_inputs(
    meta: dict,
    ref_shape: tuple,
    in_exclMask_path: Optional[str] = "default",
    pv_params: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read support inputs from auxiliary files.

    Parameters
    ----------
    meta: dict
        Describes lat/lon borders of weather data.
    ref_shape: np.ndarray
        Shape of weather data, to check for consistency.
    in_exclMask_path : str
        Path to exclusion mask being considered.
    pv_params: Optional[dict]
        User-defined custom parameters. Default: None.

    Returns
    -------
    pv_tilt : np.ndarray
        PV module tilt.
    pv_azim : np.ndarray
        PV module azimuth.
    excl_mask : Optional[np.ndarray]
        Exclusion mask, IDing pixels to be ignored during calculations.
        None corresponds to assuming all pixels.
    """
    if pv_params is None:  # if pv_params is not provided
        pv_tilt, pv_azim = read_POA_params(meta)
    elif pv_params is not None:  # if pv_params is provided
        keys = pv_params.keys()
        if ("azim" not in keys) & ("tilt" not in keys):
            pv_tilt, pv_azim = read_POA_params(meta)
        elif ("azim" in keys) & ("tilt" in keys):
            pv_tilt = pv_params["tilt"]
            pv_azim = pv_params["azim"]
        elif "azim" in keys:
            # reads module tilt/azimuth parameters used for regional PV modelling
            pv_tilt, _ = read_POA_params(meta)
            pv_azim = pv_params["azim"]
        elif "tilt" in keys:
            pv_tilt = pv_params["tilt"]
            # reads module tilt/azimuth parameters used for regional PV modelling
            _, pv_azim = read_POA_params(meta)

    # exclusion mask identifying locations where calculations are not done
    # if no path is provided, no filtering is done
    if in_exclMask_path:
        excl_mask = read_exclusMask(meta, in_exclMask_path)
    else:
        excl_mask = np.zeros(ref_shape[1:])

    return pv_tilt, pv_azim, excl_mask


def read_POA_params(
    meta: dict,
    pv_params: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read module tilt and azimuth data, which define plane-of-array (POA).

    Parameters
    ----------
    meta: dict
        Describes lat/lon borders of weather data.
    pv_params: Optional[dict]
        User-defined custom parameters. Default: None.

    Returns
    -------
    pv_tilt : np.ndarray
        PV module tilt.
    pv_azim : np.ndarray
        PV module azimuth.

    """
    # checks if any default data is needed
    check = 0
    if pv_params:
        # checks if pv_params is provided, if so copies dict keys
        keys = list(pv_params.keys())

        if "tilt" in keys:
            pv_tilt = pv_params["tilt"]
        if "azim" in keys:
            pv_azim = pv_params["azim"]

        if ("tilt" not in keys) | ("azim" not in keys):
            check = 1
    else:
        check = 1

    if check == 1:
        script_dir = path.dirname(path.abspath(__file__))
        file_dir = path.split(script_dir)[0]
        file = "C3S_Lot1_PECD_PVprm_v6.nc"

        keys = ["dummy"]

        with xr.open_dataset(path.join(file_dir, "anci", file)) as nc_data:
            # compares borders of grids (if weather data is subset,
            # exclusion mask needs to be subset)
            a = nc_data["longitude"].values.min() - meta["vLon"].min()
            b = nc_data["longitude"].values.max() - meta["vLon"].max()
            c = nc_data["latitude"].values.min() - meta["vLat"].min()
            d = nc_data["latitude"].values.max() - meta["vLat"].max()

            if abs(a) + abs(b) + abs(c) + abs(d) > 0:
                lon_slice = slice(meta["vLon"].min(), meta["vLon"].max())
                lat_slice = slice(meta["vLat"].max(), meta["vLat"].min())
                nc_data = nc_data.sel(longitude=lon_slice, latitude=lat_slice)

            if "tilt" not in keys:
                # module tilt is assumed as 75% of optimal value for each pixel
                pv_tilt = nc_data["OptTilt"].values.clip(0, 40) * 0.75

            if "azim" not in keys:
                # south for north hemisphere, north otherwise
                pv_azim = nc_data["pv_azim"].values

    return pv_tilt, pv_azim


def read_exclusMask(
    meta: dict,
    in_exclMask_path: str,
    prm_excl: str = "PVmask",
) -> np.ndarray:
    """
    Read exclusion mask .nc file.

    Parameters
    ----------
    meta: dict
        Describes lat/lon borders of weather data.
    in_exclMask_path : str
        Path to exclusion mask .nc file.
    prm_excl: str
        Exclusion mask parameter within .nc file. Default: prm_excl

    Returns
    -------
    exclus_mask : np.ndarray
        Exclusion mask, identifying areas which are deemed highly
        unlikely to have PV installed.

    """
    if in_exclMask_path == "default":
        script_dir = path.dirname(path.abspath(__file__))

        file_dir = path.split(script_dir)[0]
        with open(
            path.join(file_dir, "anci", "default_mask_name.json"), "r"
        ) as json_file:
            file = json.load(json_file)["excl_mask"]

        in_exclMask_path = path.join(file_dir, "anci", file)

    with xr.open_dataset(in_exclMask_path) as nc_data:
        lon_slice = slice(meta["vLon"].min(), meta["vLon"].max())
        lat_slice = slice(meta["vLat"].max(), meta["vLat"].min())

        nc_data = nc_data.sel(longitude=lon_slice, latitude=lat_slice)
        exclus_mask = nc_data[prm_excl].values

    return exclus_mask


def read_weather_inputs(
    in_ssrd_path: str, in_t2m_path: str
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Read weather inputs from nc. files.

    Parameters
    ----------
    in_ssrd_path : str
        Path to ssrd .nc file.
    in_t2m_path : str
        Path to t2m .nc file.

    Returns
    -------
    ssrd : np.ndarray
        Surface solar radiation downwelling.
    t2m : np.ndarray
        Air temperature at 2-m height.
    meta : dict
        Metadata describing the weather data (time, latitude, longitude).

    """
    # read solar radiation data and metadata
    ssrd, meta = read_ssrd_and_meta(in_ssrd_path)

    # read air temperature at 2-meter height
    t2m = read_t2m(in_t2m_path)

    return ssrd, t2m, meta


def read_ssrd_and_meta(
    in_ssrd_path: str,
    prm_ssrd: str = "ssrd",
) -> Tuple[np.ndarray, dict]:
    """
    Read ssrd data from .nc, assumes data is in W.m-2.

    Parameters
    ----------
    in_ssrd_path : str
        Path to ssrd .nc file.
    prm_ssrd : str, optional
        Parameter identifying surface solar radiation downwelling field
        in .nc field. Default: 'ssrd'.

    Returns
    -------
    ssrd : np.ndarray
        Surface solar radiation downwelling data.
    meta : dict
        Metadata describing ssrd (time, latitude, longitude). Is assumed to
        equally represent t2m.

    """
    with xr.open_dataset(in_ssrd_path) as nc_data:
        ssrd = nc_data[prm_ssrd].values
        # TODO: include check on root_grp[prm_ssrd].units

        meta = {}
        meta["vLon"] = nc_data["longitude"].values
        meta["vLat"] = nc_data["latitude"].values
        meta["time"] = nc_data["time"].values

    # saturate ssrd below threshold to 0
    # (reduces number of calculations since night is ignored)
    thresh = 1
    ssrd = np.where(ssrd < thresh, 0, ssrd)

    return ssrd, meta


def read_t2m(
    in_t2m_path,
    prm_t2m: str = "t2m",
    t2m_unit: str = "K",
) -> np.ndarray:
    """
    Read air temperature at 2-m height from .nc file, assumes data is in K.

    Parameters
    ----------
    in_t2m_path : TYPE
        Path to t2m .nc file.
    prm_t2m : str, optional
        Parameter identifying 2 metre temperature field in .nc field.
        The default is 't2m'.
    t2m_unit : str, optional
        Air temperature units. Default: 'K'.

    Returns
    -------
    t2m : np.ndarray
        Air temperature 2-m height in K.

    """
    print(in_t2m_path)
    with xr.open_dataset(in_t2m_path) as nc_data:
        t2m = nc_data[prm_t2m].values

    if t2m_unit == "K":
        # convert Kelvin to Celsius
        t2m = t2m - 273.15

    return t2m
