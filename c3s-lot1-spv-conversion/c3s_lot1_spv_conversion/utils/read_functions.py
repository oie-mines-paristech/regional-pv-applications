import json
from os import path
from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr


def read_support_inputs(
    ref_shape: tuple,
    in_exclMask_path: Optional[str] = "default",
    pv_params: Optional[dict] = None,
) -> Tuple[
    Optional[Union[float, int, np.ndarray]],
    Optional[Union[float, int, np.ndarray]],
    Optional[np.ndarray],
]:
    """
    Read support inputs from auxiliary files.

    Parameters
    ----------
    ref_shape: np.ndarray
        Shape of weather data, to check for consistency.
    in_exclMask_path : str
        Path to exclusion mask being considered.
    pv_params: Optional[dict]
        User-defined custom parameters. Default: None.

    Returns
    -------
    pv_tilt : Union[float, int, np.ndarray]
        PV module tilt.
    pv_azim : Union[float, int, np.ndarray]
        PV module azimuth.
    excl_mask : Optional[np.ndarray]
        Exclusion mask, IDing pixels to be ignored during calculations.
        None corresponds to assuming all pixels.
    """
    if pv_params and pv_params.get("tracking", 0) > 0:
        # tracking is user-defined
        pv_tilt = None
        pv_azim = None
    elif isinstance(pv_params, dict):
        # checks if module tilt anda azimuth are provided by user
        has_tilt = "tilt" in pv_params
        has_azim = "azim" in pv_params

        if has_tilt and has_azim:
            pv_tilt = pv_params["tilt"]
            pv_azim = pv_params["azim"]
        elif has_azim:
            # fetches default tilt
            pv_tilt, _ = read_POA_params()
            pv_azim = pv_params["azim"]
        elif has_tilt:
            # fetches default azim
            _, pv_azim = read_POA_params()
            pv_tilt = pv_params["tilt"]
        else:
            # fetches default tilt and azim
            pv_tilt, pv_azim = read_POA_params()
    else:
        # fetches default tilt and azim
        pv_tilt, pv_azim = read_POA_params()

    # exclusion mask identifying locations where calculations are not done
    if in_exclMask_path:
        excl_mask = read_exclusMask(in_exclMask_path)
    else:  # if no path is provided, no filtering is done
        excl_mask = None

    return pv_tilt, pv_azim, excl_mask


def read_POA_params(
    pv_params: Optional[dict] = None,
) -> Tuple[Union[int, float, np.ndarray], Union[int, float, np.ndarray]]:
    """
    Read module tilt and azimuth data, which define plane-of-array (POA).

    Parameters
    ----------
    pv_params: Optional[dict]
        User-defined custom parameters. Default: None.

    Returns
    -------
    pv_tilt : Union[float, int, np.ndarray]
        PV module tilt.
    pv_azim : Union[float, int, np.ndarray]
        PV module azimuth.

    """
    # checks if there is user-defined inputs
    defaults_needed = []
    if (pv_params is None) or ("tilt" not in pv_params):
        defaults_needed.append("tilt")
    else:
        pv_tilt = pv_params["tilt"]

    if (pv_params is None) or ("azim" not in pv_params):
        defaults_needed.append("azim")
    else:
        pv_azim = pv_params["azim"]

    # if user does not specify parameters (i.e., use default)
    if defaults_needed:
        script_dir = path.dirname(path.abspath(__file__))
        file_dir = path.split(script_dir)[0]
        file = "C3S_Lot1_PECD_PVprm_v6.nc"

        with xr.open_dataset(path.join(file_dir, "anci", file)) as nc_data:
            if "tilt" in defaults_needed:
                # module tilt is assumed as 75% of optimal value for each pixel
                pv_tilt = nc_data["OptTilt"].values.clip(0, 40) * 0.75

            if "azim" in defaults_needed:
                # south for north hemisphere, north otherwise
                pv_azim = nc_data["pv_azim"].values

    return pv_tilt, pv_azim


def read_exclusMask(
    in_exclMask_path: str,
    prm_excl: str = "PVmask",
) -> np.ndarray:
    """
    Read exclusion mask .nc file.

    Parameters
    ----------
    in_exclMask_path : str
        Path to exclusion mask .nc file.
    prm_excl: str
        Exclusion mask parameter within .nc file. Default: "PVmask"

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
    in_t2m_path: str,
    prm_t2m: str = "t2m",
    t2m_unit: str = "K",
) -> np.ndarray:
    """
    Read air temperature at 2-m height from .nc file, assumes data is in K.

    Parameters
    ----------
    in_t2m_path : str
        Path to t2m .nc file.
    prm_t2m : str, optional
        Parameter identifying 2 metre temperature field in .nc field.
        Default: 't2m'.
    t2m_unit : str, optional
        Air temperature units. Default: 'K'.

    Returns
    -------
    t2m : np.ndarray
        Air temperature 2-m height in K.

    """
    with xr.open_dataset(in_t2m_path) as nc_data:
        t2m = nc_data[prm_t2m].values

    if t2m_unit == "K":
        # convert Kelvin to Celsius
        t2m = t2m - 273.15

    return t2m
