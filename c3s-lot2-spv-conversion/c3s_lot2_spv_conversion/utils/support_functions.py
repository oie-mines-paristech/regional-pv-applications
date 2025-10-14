import datetime
import importlib.resources as pkg_resources
import json

import numpy as np
import xarray as xr

from c3s_lot2_spv_conversion import anci


def read_output_metadata(DataStream: str) -> dict:
    """
    Read JSON file with metadata template and customize some fields.

    Parameters
    ----------
    DataStream : str
        Historical (HIST) / Seasonal (SEAS) / Projection (PROJ).
    template_path : str, optional
        Metadata template for output .nc file.
        The default is './anci/output_metadata.json'.

    Returns
    -------
    metadata_dict : dict
        Resulting dictionary with complete metadata.

    """
    with pkg_resources.path(anci, "output_metadata.json") as json_path:
        with open(json_path, "r") as json_file:
            metadata_dict = json.load(json_file)

            metadata_dict["Creation_date"] = str(datetime.datetime.now())

            if DataStream == "HIST":
                metadata_dict["Forecast_type"] = "Reanalysis"
                metadata_dict["ClimateData"] = "ERA5"
            elif DataStream == "PROJ":
                metadata_dict["Forecast_type"] = "Climate projection"

    return metadata_dict


def prep_output_file(
    out_path: str,
    DataStream: str,
    spv_cf: np.ndarray,
    vLon: np.ndarray,
    vLat: np.ndarray,
    time_: np.ndarray,
):
    """
    Create and encode .nc output file.

    Parameters
    ----------
    out_path : str
        Path to which output file must be saved.
    DataStream : str
        Data stream of the used input weather data.
    spv_cf : np.ndarray
        PV capacity factor.
    vLon : np.ndarray
        Longitude values of the considered grid.
    vLat : np.ndarray
        Latitude values of the considered grid.
    time_ : list
        Timestamps.

    Returns
    -------
    None.

    """
    # Convert time to NetCDF format
    time_units = "hours since 1900-01-01 00:00:00"
    time_ref = np.datetime64("1900-01-01 00:00:00")
    h_delta = np.timedelta64(3600, "s")

    time_encoded = ((time_ - time_ref) / h_delta).astype(int)

    meta_dict = read_output_metadata(DataStream)

    # Create xarray Dataset
    spv_cf_meta = {
        "long_name": "Regional solar PV capacity factor",
        "units": "kW/kW_installed",
    }

    lon_meta = {"long_name": "longitude", "units": "degrees_east"}
    lat_meta = {"long_name": "latitude", "units": "degrees_north"}
    time_meta = {"long_name": "time", "units": time_units}
    time_meta["calendar"] = "gregorian"

    ds = xr.Dataset(
        {
            "spv_cf": (
                ["time", "latitude", "longitude"],
                spv_cf,
                spv_cf_meta,
            )
        },
        coords={
            "longitude": (
                "longitude",
                vLon,
                lon_meta,
            ),
            "latitude": (
                "latitude",
                vLat,
                lat_meta,
            ),
            "time": (
                "time",
                time_encoded,
                time_meta,
            ),
        },
        attrs=meta_dict,
    )

    # Save to NetCDF
    #  ValueError: invalid format for scipy.io.netcdf backend: 'NETCDF4'
    ds.to_netcdf(
        out_path,
        format="NETCDF4",
        encoding={
            "spv_cf": {
                "zlib": True,
                "least_significant_digit": 3,
            }
        },
    )
