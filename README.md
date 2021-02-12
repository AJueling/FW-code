# Code for the Ocean Science publication [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4537845.svg)](https://doi.org/10.5281/zenodo.4537845)
### The Atlantic's Freshwater Budget under Climate Change in the Community Earth System Model with Strongly Eddying Oceans
### (doi:10.5194/os-2020-76)
#### by André Jüling, Xun Zhang, Daniele Castellana, Anna S. von der Heydt, and Henk A. Dijkstra

## Code Structure

This repository holds the files creating the figures of the publication. Original model files and many derived files are not included due to size limitation (single month CESM output files are 56 GB in the high resolution setup). There is more functionality in this code than what is used for the figures alone, but dependencies should be clear from the code.

There is an `environment.yml` file that can be used to create the conda environment used.

For questions regarding the code please contact me: a.juling@uu.nl

```
FW-code
│   README.md
│   LICENSE
│
└───doc               (documentation)
|   │   enviroment.yml              conda environment used (see https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
│
└───src
    │
    │   Fig{1-10}.ipynb             jupyter notebooks with Figures from paper
    │   GM_eddy_fluxes.ipynb        notebook to create supplementary GM eddy flux figure
    │
    │   aa_derivation_fields.py     creating derived fields
    │   constants.py                defining reused physical constants
    │   curly_bracket.py            plots curly brackets
    │   filters.py                  filters for time series
    │   FW_budget.py                calculate freshwater budget terms
    │   FW_plots.py                 functions to create some of the plots
    │   FW_transport.py             calculate freshwater transport terms
    │   grid.py                     deals with CESM grid
    │   maps.py                     creates maps
    │   mimic_alpha.py              auxiliary plotting function to fake transparancy in .eps
    │   MOC.py                      calculate AMOC
    │   obs_cesm_maps.py            world maps of observations and biases
    │   paths.py                    collection of file paths and names
    │   plotting.py                 auxiliary plotting functions
    │   read_binary.py              functions to read CESM binary output
    │   regions.py                  define world ocean regions
    │   timeseries.py               function to loop over CESM files
    │   xr_DataArrays.py            xarray auxiliary functions
    │   xr_integrate.py             xarray integration functions
    │   xr_regression.py            xarray regression functions
```
