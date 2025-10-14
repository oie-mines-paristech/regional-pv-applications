# c3s-lot2-spv-conversion

Code to produce solar photovoltaic capacity factors within PECDv4.2.

## Background

Between 2022-2025, Rodrigo Amaro e Silva (ULisbon & Mines Paris PSL) and Yves-Marie
Saint-Drenan (Mines Paris PSL) led the development of the photovoltaic modelling chain
comprised within the versions 4.x of the [Pan-European Climate Database (PECD)](https://climate.copernicus.eu/powering-europe-through-climate-uncertainty),
funded by Copernicus Climate Change Services (C3S).

This consisted in the integration of the [regional-pv](https://github.com/ramaroesilva/regional-pv/tree/main)
package according to the PECD needs.

## Workflow for users

Clone the repository and define `regional-pv-applications\c3s-lot2-spv-conversion`
as your current directory.

For best experience create a new conda environment (e.g. `c3s-lot2-spv-conversion`)
with Python 3.12:

```
conda create -n c3s-lot2-spv-conversion -c conda-forge python=3.12
conda activate c3s-lot2-spv-conversion
conda env update -f environment.yml
```

Then, within a Python script, import the package and run `interface`:

```python
# path to c3s-lot2-spv-conversion folder
repo_path = "..."  # to be completed by user
sys.path.append(base_path)

import c3s_lot2_spv_conversion

# check `notebooks` folder for an actual example
out = c3s_lot2_spv_conversion.interface.compute_spv(...)
```

## License

```
Copyright 2025, ARMINES.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
