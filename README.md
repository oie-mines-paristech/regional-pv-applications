# c3s-lot1-spv-conversion

Code to produce solar photovoltaic capacity factors within the C3S Energy
global service.

## Background

Between 2022-2025, Rodrigo Amaro e Silva (ULisbon & Mines Paris PSL) and Yves-Marie
Saint-Drenan (Mines Paris PSL) led the development of the photovoltaic modelling chain
comprised within the [global extension of the existing C3S Energy service](https://climate.copernicus.eu/operational-service-energy-sector),
funded by Copernicus Climate Change Services (C3S).

## Workflow for users

1. Clone the repository either through GitHub interface or through git running

`git clone https://github.com/oie-mines-paristech/regional-pv-applications.git`

2. In Anaconda, define `regional-pv-applications\c3s-lot1-spv-conversion` as your current
   directory.

1. Change git branch running `git checkout c3s_global`

For best experience create a new conda environment (e.g. `c3s-lot1-spv-conversion`)
with Python 3.12:

```
conda create -n c3s-lot1-spv-conversion -c conda-forge python=3.12
conda activate c3s-lot1-spv-conversion
conda env update -f environment.yml
pip install . --no-deps
```

Then, within a Python script, import the package and run `interface`:

```python
import c3s_lot1_spv_conversion

# check `notebooks` folder for an actual example
out = c3s_lot1_spv_conversion.interface.compute_spv(...)
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
