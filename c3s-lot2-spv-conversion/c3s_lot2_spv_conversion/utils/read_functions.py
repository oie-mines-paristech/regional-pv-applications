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
    pv_type: str,
    meta: dict,
    ref_shape: tuple,
    in_exclMask_path: Optional[str] = "default",
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    float,
    np.ndarray,
]:
    """
    Read support inputs from auxiliary files.

    Parameters
    ----------
    pv_type: str
        IDs PV typology being computed.
    meta: dict
        Describes lat/lon borders of weather data.
    in_exclMask_path : Optional[str] = None
        Path to exclusion mask being considered.
        Default is None, which corresponds to considering no mask.

    Returns
    -------
    pv_tilt : Optional[np.ndarray]
        PV module tilt. None for tracking.
    pv_azim : Optional[np.ndarray]
        PV module azimuth. None for tracking.
    w: Optional[np.ndarray]
        Weights to combine various orientations (tilt, azimuth).
        None for tracking or single-orientation.
    k: float
        PV thermal (Ross) coefficient.
    excl_mask : np.ndarray
        Exclusion mask, IDing pixels to be ignored during calculations.
    """
    pv_tilt, pv_azim, w = read_POA_params(pv_type, meta)
    k = set_thermal_coef(pv_type)

    # exclusion mask identifying locations where calculations are not done
    # if no path is provided, no filtering is done
    if in_exclMask_path:
        excl_mask = read_exclusMask(meta, in_exclMask_path)
    else:
        excl_mask = np.zeros(ref_shape[1:])

    return pv_tilt, pv_azim, w, k, excl_mask


def read_POA_params(
    pv_type: str,
    meta: dict,
    pv_params: Optional[dict] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Read module tilt and azimuth data, which define plane-of-array (POA).

    Parameters
    ----------
    pv_type : str
        IDs PV typology being computed.
    meta : dict
        Describes lat/lon borders of weather data.
    pv_params : Optional[dict], optional
        DESCRIPTION. The default is None.

    Returns
    -------
    pv_tilt : Optional[np.ndarray]
        PV module tilt.
        None if tracking typology.
    pv_azim : Optional[np.ndarray]
        PV module azimuth.
        None if tracking typology.
    w_orient: Optional[np.ndarray]
        Weights for combining multiple module orientations.
        None if single orientation or tracking typology.
    """
    if pv_type in ["rooftop_residential", "utility_fix"]:
        script_dir = path.dirname(path.abspath(__file__))
        file_dir = path.split(script_dir)[0]
        file = "C3S_PECD_PVprm_v4.nc"
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

    if pv_type == "rooftop_residential":
        pv_azim = nc_data["pv_azim"][:].data
        pv_tilt = nc_data["pv_ResiTilt"][:].data
        w_orient = None

    elif pv_type == "rooftop_industrial":
        pv_azim = np.array([180, 90, 270])
        pv_tilt = np.array([10, 10, 10])
        w_orient = np.array([0.5, 0.25, 0.25])

    elif pv_type == "utility_fix":
        pv_tilt = nc_data["pv_OptTilt"][:].data * 0.75
        pv_azim = nc_data["pv_azim"][:].data
        w_orient = None

    elif pv_type == "utility_track_1axis":
        pv_azim = None
        pv_tilt = None
        w_orient = None

    return pv_tilt, pv_azim, w_orient


def set_thermal_coef(
    pv_type: str,
) -> float:
    """
    Define Ross coeficient based on PV typology (i.e., type of mounting).

    Parameters
    ----------
    pv_type : Union[str,float]
        IDs PV typology being computed.

    Returns
    -------
    k: float
        PV thermal (Ross) coefficient.

    """
    # Ross coefficient in °C per kW/m2
    if "rooftop" in pv_type:  # includes both residential and industrial
        # rooftop, contiguous to surface (less convective cooling)
        # Skoplaki (2008). doi: 10.1016/j.solmat.2008.05.016
        k = 34  # 0.034 °C per W/m2
    elif "utility" in pv_type:
        # free standing, room for air circulation (more convective cooling)
        # Skoplaki (2008). doi: 10.1016/j.solmat.2008.05.016
        k = 21  # 0.021 °C per W/m2

    return k


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
        Parameter identifying exclusion mask in nc file. Default: "PVmask"

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
        The default is 't2m'.
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
