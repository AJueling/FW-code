import os
import xgcm
import dask
import numpy as np
import xarray as xr
import pop_tools
import tqdm.notebook as tqdm_notebook
import matplotlib.pyplot as plt

from grid import create_tdepth, find_array_idx
from paths import file_ex_ocn_ctrl, path_prace
from xr_DataArrays import xr_DZ, xr_DXU, xr_DZ_xgcm
from xhistogram.xarray import histogram


def calculate_AMOC_sigma_z(domain, ds, fn=None):
    """ calculate the AMOC in depth and density space """
    assert domain in ['ocn', 'ocn_low']
    for q in ['PD', 'VVEL', 'DXT', 'DYT', 'DXU', 'DYU', 'REGION_MASK']:  assert q in ds

    (grid, ds_) = pop_tools.to_xgcm_grid_dataset(ds)
    ds_['DZU'] = xr_DZ_xgcm(domain=domain, grid='U')

    metrics = {
        ('X'): ['DXT', 'DXU'],  # X distances
        ('Y'): ['DYT', 'DYU'],  # Y distances
        ('Z'): ['DZU'],  # Z distances
    }
    coords = {
        'X': {'center':'nlon_t', 'right':'nlon_u'},
        'Y': {'center':'nlat_t', 'right':'nlat_u'},
        'Z': {'center':'z_t', 'left':'z_w_top', 'right':'z_w_bot'}
    }
    grid = xgcm.Grid(ds_, metrics=metrics, coords=coords)

    print('merged annual datasets do not convert to U/T-lat/lons')
    if 'nlat' in ds_.VVEL.dims:
        rn = {'nlat':'nlat_u', 'nlon':'nlon_u'}
        ac = {'nlat_u':ds_.nlat_u, 'nlon_u':ds_.nlon_u}
        ds_['VVEL'] = ds_.VVEL.rename(rn).assign_coords()
    if 'nlat' in ds_.PD.dims:
        rn = {'nlat':'nlat_t', 'nlon':'nlon_t'}
        ac = {'nlat_t':ds_.nlat_t, 'nlon_t':ds_.nlon_t}
        ds_['PD'] = ds_.PD.rename(rn).assign_coords(ac)

    print('interpolating density to UU point')
    ds_['PD'] = grid.interp(grid.interp(ds_['PD'], 'X'), 'Y')
    
    print('interpolating REGION_MASK to UU point')
    fn_MASK = f'{path_prace}/MOC/AMOC_MASK_uu_{domain}.nc'
    if os.path.exists(fn_MASK):  
        AMOC_MASK_uu = xr.open_dataarray(fn_MASK)
    else:
        MASK_uu = grid.interp(grid.interp(ds_.REGION_MASK, 'Y'), 'X')
        AMOC_MASK_uu = xr.DataArray(np.in1d(MASK_uu, [-12,6,7,8,9,11,12]).reshape(MASK_uu.shape),
                                dims=MASK_uu.dims, coords=MASK_uu.coords)
        AMOC_MASK_uu.to_netcdf(fn_MASK)

    print('AMOC(y,z);  [cm^3/s] -> [Sv]')
    AMOC_yz = (grid.integrate(grid.cumint(ds_.VVEL.where(AMOC_MASK_uu),'Z',boundary='fill'), 'X')/1e12)
#     AMOC_yz = (ds_.VVEL*ds_.DZU*ds_.DXU).where(AMOC_MASK_uu).sum('nlon_u').cumsum('z_t')/1e12
    AMOC_yz = AMOC_yz.rename({'z_w_top':'z_t'}).assign_coords({'z_t':ds.z_t})
    AMOC_yz.name = 'AMOC(y,z)'

    print('AMOC(sigma_0,z);  [cm^3/s] -> [Sv]')
    if int(ds_.PD.isel(z_t=0).mean().values)==0:
        PD, PDbins = ds_.PD*1000, np.arange(-10,7,.05)
    if int(ds_.PD.isel(z_t=0).mean().values)==1:
        PD, PDbins = (ds_.PD-1)*1000, np.arange(5,33,.05)

    print('histogram')
    weights = ds_.VVEL.where(AMOC_MASK_uu)*ds_.DZU*ds_.DXU/1e12
#     ds_.PD.isel(z_t=0).plot()
    AMOC_sz = histogram(PD, bins=[PDbins], dim=['z_t'],
                         weights=weights).sum('nlon_u', skipna=True).cumsum('PD_bin').T
    AMOC_sz.name = 'AMOC(y,PD)'

    # output to file
    if fn is not None:  xr.merge([AMOC_yz,AMOC_sz]).to_netcdf(fn)
    return AMOC_yz, AMOC_sz


def calculate_MOC(ds, DXU, DZU, MASK):
    """ Atlantic Meridional Overturning circulation 
    
    input:
    ds   .. xr Dataset of CESM output
    
    output:
    MOC .. 2D xr DataArray
    """
    assert 'VVEL' in ds
    MOC = (ds.VVEL*DXU*DZU).where(MASK).sum(dim='nlon')/1e2  # [m^3/s]
    for k in np.arange(1,42):
        MOC[k,:] += MOC[k-1,:]
    
    return MOC


def approx_lats(domain):
    """ array of approx. latitudes for ocean """
    assert domain in ['ocn', 'ocn_low']
    if domain=='ocn':
        ds = xr.open_dataset(file_ex_ocn_ctrl, decode_times=False)
        lats = ds.TLAT[:,900].copy()
        lats[:120] = -78
    elif domain=='ocn_low':
        ds = xr.open_dataset(file_ex_ocn_lpd, decode_times=False)
        lats = ds.TLAT[:,].copy()
    return lats



def AMOC_max(AMOC):
    """ AMOC maximum at 26 deg N, 1000 m """
    lats    = approx_lats('ocn')
    tdepths = create_tdepth('ocn')
    j26   = find_array_idx(lats, 26)       # 1451
    z1000 = find_array_idx(tdepths, 1000)  #   21
    
    return AMOC.isel({'z_t':z1000, 'nlat':j26})



def plot_AMOC(AMOC):
    lats    = approx_lats('ocn')
    tdepths = create_tdepth('ocn')
    j26   = find_array_idx(lats, 26)
    z1000 = find_array_idx(tdepths, 1000)
    return