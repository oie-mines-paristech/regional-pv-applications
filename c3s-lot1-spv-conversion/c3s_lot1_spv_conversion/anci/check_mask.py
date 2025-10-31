import xarray as xr

# Replace with your actual file path
ds1 = xr.open_dataset("ANCI_SPVL-mask_C3S2LOT1_025d_v1.00.nc")

# Display basic info about the dataset
print(ds1)

print("-------------------------------")

# Replace with your actual file path
ds2 = xr.open_dataset("ANCI_SPVL-mask_C3S2LOT1_100d_v1.00.nc")

# Display basic info about the dataset
print(ds2)

print("-------------------------------")

ds0 = xr.open_dataset(
    "H_PVma_InCS_Armi_---_0000m_Glob_025d_S202211120000_E202211120000_INV_MAP_NA-_NA-_NA-_NA-_NA_NA---_NA---_MT00-.nc"
)
print(ds0)
