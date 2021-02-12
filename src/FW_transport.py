""" Freshwater/Salt Transport
- freshwater relative to 35 g/kg
- taking partial bottom cells of high resolution POP grid into account

if called directly, this scripts calculates:

(1) the total salt/volume/freshwater flux through Bering Strait and Strait of Gibraltar
->  f'{path_prace}/Mov/FW_Med_Bering_{run}_{y:04d}.nc'

(2) the Atlantic freshwater and salt meridional transport terms
    (overturning, azonal, total, and eddy) as a function of latitude
->  f'{path_prace}/Mov/FW_SALT_fluxes_{run}_{y:04d}.nc'
"""
import os
import gc
import sys
import tqdm
import xgcm
# import dask
import numpy as np
import xarray as xr
import warnings
import pop_tools

warnings.filterwarnings('ignore')

from paths import file_ex_ocn_ctrl, file_ex_ocn_lpd, path_prace, path_results,\
                  path_ocn_ctrl, spinup, path_ocn_rcp, rcpstr, \
                  CESM_filename, path_ocn_lpd, lpdstr, path_ocn_lr1, lr1str
from datetime import datetime
from constants import rho_sw
from timeseries import IterateOutputCESM
from xr_DataArrays import xr_DZ, xr_DZ_xgcm

metrics = {
    ('X'): ['DXT', 'DXU', 'HTN', 'HUS'],  # X distances
    ('Y'): ['DYT', 'DYU', 'HTE', 'HUW'],  # Y distances
    ('Z'): ['DZT', 'DZU'],                # Z distances
    ('X', 'Y'): ['TAREA', 'UAREA']        # Areas
    }
coords = {
    'X': {'center':'nlon_t', 'right':'nlon_u'},
    'Y': {'center':'nlat_t', 'right':'nlat_u'},
    'Z': {'center':'z_t', 'left':'z_w_top', 'right':'z_w_bot'}
    }
sections_high = {
    'Bering': dict(nlon=slice(2929,2947), nlat=1979),
    '24S'   : dict(nlon=slice(500,1300), nlat=934),
    'Med'   : dict(nlon=1042, nlat=slice(1555,1559))
    }
sections_low = {
    'Bering': dict(nlon=slice(198,202), nlat=333),
    '24S'   : dict(nlon=np.arange(-10, 60), nlat=105), # across array boundary
    'Med'   : dict(nlon=32,nlat=slice(290,295))
    }

def add_attrs(da, name, units, long_name):
    """ gives name to xr.DataArray `da` and adds attributes `units` and `long_name` """
    da.name = name
    da.attrs = {'units':units, 'long_name':long_name}
    return da

def calc_section_transport(ds):
    """ calculate the transport across Bering Strait and Gibraltar sections
    section  predetermined (nlon,nlat) dictionary
    """
    for q in ['SALT', 'VVEL', 'UVEL', 'VNS', 'UES', 'DXT', 'DYT', 'DXU', 'DYU', 'TAREA', 'DZT', 'DZU']:
        assert q in ds
        
    if ds.TAREA.shape==(2400, 3600):  section = sections_high
    elif ds.TAREA.shape==(384, 320):  section = sections_low
    
    S0 = 35.  # [g/kg]

    def transports(name):
        ds_  = ds.isel(section[name])
        if name in ['Bering', '24S']:  # zonal section
            DT = ds_.DXT
            DU = ds_.DXU
            VELS = ds_.VNS
            VEL = ds_.VVEL
            dims = ['z_t','nlon']
            if name=='Bering':  name2, name3 = 'BS', 'Bering Strait'
            elif name=='24S':   name2, name3 = '24S', '24S section'
        elif name=='Med':              # meridional section
            DT = ds_.DYT
            DU = ds_.DYU
            VELS = ds_.UES
            VEL = ds_.UVEL
            dims = ['z_t', 'nlat']
            name2, name3 = 'Med', 'Strait of Gibraltar'

        s = (ds_.SALT*DT*ds_.DZT).sum(dims)/(DT*ds_.DZT).where(ds_.SALT>0).sum(dims)  # [1]
        S = (VELS*ds_.TAREA*ds_.DZT).sum(dims)/1e4  # [cm^2*m/s] -> [kg_{Salt}/s]
        V = (VEL*DU*ds_.DZU).sum(dims)/1e4          # [cm^2*m/s] -> [m^3/s]
        F = -1/S0*(S - S0*V)                        # [m^3/s]

        s = add_attrs(s, f's_{name2}', '[1]'  , f'{name3} average salinity')
        S = add_attrs(S, f'S_{name2}', 'kg/s' , f'{name3} salt transport')
        V = add_attrs(V, f'V_{name2}', 'm^3/s', f'{name3} volume transport')
        F = add_attrs(F, f'F_{name2}', 'm^3/s', f'{name3} freshwater transport')
        if name in ['Med', '24S']:
            ov = (VEL*DU*ds_.DZU).sum(dims[1]).cumsum('z_t')/1e4
            ov = add_attrs(ov, f'ov_{name2}', 'm^3/s', f'{name3} volume overturning')
            return xr.merge([s, S, V, F, ov])
        else:
            return xr.merge([s, S, V, F])

    ds_BS = transports('Bering')
    ds_24S = transports('24S')
    ds_Med = transports('Med')

    ds = xr.merge([ds_BS, ds_24S, ds_Med], compat='override')
    ds.attrs = {'calculation':'FW_transport.py:calc_section_transport()',
                'notes':'freshwater calculation relative to salinity 35'}
    if 'time' not in ds:  ds = ds.expand_dims('time'); print('expanded time dimension')
    return ds


def calc_transports_pbc(grid, ds, MASK_tt, MASK_uu, S0=35.0):
    """ calculate the merid. FW and SALT transport terms with xgcm grid including PBCs """
    # SALT
    SALT = grid.interp(grid.interp(ds.SALT.where(ds.SALT>0).where(MASK_tt), 'Y'), 'X')  # SALT interpolated to UU grid
    VVEL = (ds.VVEL*ds.DZU/ds.dz).where(MASK_uu).where(ds.VVEL<1000)  # reduced velocity due to PBCs

    # non-PBC-weighted mean here, as salinity differences between neighbouring cells are assumed very small
    SALT_zm = SALT.mean('nlon_u')
    
    # VVEL terms:  weighted by cell thickness
    Vbar = (VVEL*ds.DXU*ds.DZU).sum(['nlon_u','z_t'])  # barotropic volume transport [cm^2 m s^-1]
    AREA_section = (ds.DXU*ds.DZU).where(VVEL<1000).sum(['nlon_u','z_t'])  # [cm m]
    VVEL_bar = Vbar/AREA_section  # barotropic velocity [cm/s]
    Vbar = Vbar/1e10  # [cm^2 m s^-1] -> [Sv]
    VVEL_xint = (VVEL*ds.DXU).sum('nlon_u')  # zonal integral velocity (y,z) [cm^2/s]
    VVEL_xint_s = ((VVEL-VVEL_bar)*ds.DXU).sum('nlon_u')  # zonal integral of pure overturning velocity (y,z) [cm^2/s]
    VVEL_zm = (VVEL*ds.DXU).sum('nlon_u')/ds.DXU.where(VVEL<1000).sum('nlon_u')  # zonal mean velocity (y,z) [cm/s]
    
    
    SALT_prime = (SALT - SALT_zm)  # azonal salt component (x,y,z) [g/kg]
    VVEL_prime = (VVEL - VVEL_zm)  # azonal velocity comp. (x,y,z) [cm/s]
    
    # TRANSPORT terms
    # can integrate now vertically with dz, as transport terms are reduced by PBC_frac
    Fov = ( -1/S0*(VVEL_xint*(SALT_zm - S0)*ds.dz).sum(dim='z_t'))/1e12  # 1 Sv = 1e12 cm^3/s
    Sov = (VVEL_xint*SALT_zm*ds.dz).sum(dim='z_t')*rho_sw/1e9            # 1 kg/s = rho_w * 1e-9 g/kg cm^3/s
    
    Fovs = ( -1/S0*(VVEL_xint_s*(SALT_zm - S0)*ds.dz).sum(dim='z_t'))/1e12
    Sovs = (VVEL_xint_s*SALT_zm*ds.dz).sum(dim='z_t')*rho_sw/1e9

    vSp = (ds.DXU*ds.dz*VVEL_prime*SALT_prime).sum(dim=['nlon_u','z_t'])  # product of primed velocity and salinity [cm^3/s * g/kg]
    Saz = vSp*rho_sw/1e9
    Faz = -vSp/1e12/S0
    
    # total salt transport on (nlon_t, nlat_u)-gridpoints (see /doc/calculation_VNS.md)
    rn, ac = {'nlat_t':'nlat_u'}, {'nlat_u':ds['nlat_u'].values}
    # if len(ds.nlon_t)==384:  # LR-CESM
    #     TVOL_tu = (ds.DZT*ds.TAREA).rename(rn).assign_coords(ac)  # volume T-cell shifted northward to (nlon_t, nlat_u)
    #     MASK_tu = MASK_tt.rename(rn).assign_coords(ac)
    #     # MASK_tu = grid.interp(MASK_tt,'Y')
    #     SALT_tu = ds.SALT.rename(rn).assign_coords(ac)
    #     # SALT_tu = grid.interp(ds.SALT,'Y')
    #     St = rho_sw/1e9*(ds.VNS*TVOL_tu).where(MASK_tu).sum(dim=['z_t','nlon_t'])
    #     Sm = rho_sw/1e9*(grid.interp(ds.DXU*ds.VVEL, 'X')*grid.interp(ds.DZT,'Y')*SALT_tu).where(MASK_tu).sum(dim=['nlon_t','z_t'])
    #     # Sm = rho_sw/1e9*(ds.VVEL*SALT*ds.DXU*ds.DZU).sum(dim=['nlon_u','z_t'])
    #     Se = St - Sm
    # elif len(ds.nlon_t)==3600:  # HR-CESM
    TVOL_tu = (ds.DZT*ds.TAREA).rename(rn).assign_coords(ac)  # volume T-cell shifted northward to (nlon_u, nlat_t)
    St = rho_sw/1e9*(ds.VNS*TVOL_tu).where(grid.interp(MASK_tt,'Y')).sum(dim=['z_t','nlon_t'])  
    Sm = rho_sw/1e9*(ds.VVEL*SALT*ds.DXU*ds.DZU).sum(dim=['nlon_u','z_t'])
    Se = St - Sm

    Vbar     = add_attrs(Vbar    , 'Vbar'    , 'Sv'  , 'barotropic volume transport')
    Fov      = add_attrs(Fov     , 'Fov'     , 'Sv'  , 'freshwater transport overturning component')
    Fovs     = add_attrs(Fovs    , 'Fovs'    , 'Sv'  , 'freshwater transport overturning component w/o barotropic flow')
    Faz      = add_attrs(Faz     , 'Faz'     , 'Sv'  , 'freshwater transport azonal component')
    Sov      = add_attrs(Sov     , 'Sov'     , 'kg/s', 'salt transport overturning component')
    Sovs     = add_attrs(Sovs    , 'Sovs'    , 'kg/s', 'salt transport overturning component w/o barotropic flow')
    Saz      = add_attrs(Saz     , 'Saz'     , 'kg/s', 'salt transport azonal component')
    Se       = add_attrs(Se      , 'Se'      , 'kg/s', 'salt transport eddy component')
    Sm       = add_attrs(Sm      , 'Sm'      , 'kg/s', 'salt transport mean component')
    St       = add_attrs(St      , 'St'      , 'kg/s', 'salt transport total component')
    VVEL_bar = add_attrs(VVEL_bar, 'VVEL_bar', 'cm/s', 'barotropic velocity')
    VVEL_zm  = add_attrs(VVEL_zm , 'VVEL_zm' , 'cm/s', 'zonal mean velocity')
    SALT_zm  = add_attrs(SALT_zm , 'SALT_zm' , '[1]' , 'zonal mean salinity')

    ds = xr.merge([SALT_zm, VVEL_bar, VVEL_zm, Vbar, Fov, Fovs, Faz, Sov, Sovs, Saz, Se, Sm, St])
    return ds


def FW_SALT_flux_dataset(run):
    """ combines annual files inot single xr dataset """
    if run=='ctrl':  yy = np.arange(1,301)
    elif run=='lpd':  yy = np.arange(154,601)
    elif run in ['rcp', 'lr1']:  yy = np.arange(2000,2101)

    fn = f'{path_prace}/Mov/FW_SALT_fluxes_{run}.nc'
    if os.path.exists(fn):
        ds = xr.open_dataset(fn, decode_times=False)
    else:
        ds_list = []
        for y in yy:
            ds_ = xr.open_dataset(f'{path_prace}/Mov/FW_SALT_fluxes_{run}_{y:04d}.nc')
            ds_list.append(ds_.copy())
        ds = xr.concat(ds_list, dim='time')
        ds.to_netcdf(fn)
    return ds

def mf_list(name, y1, y2):
    """ list of all yearly files in decade starting in year Y """
    return [name+f'{y:04d}-{m:02d}.nc' for y in range(y1,y2) for m in range(1,13)]

def geometry_file(domain):
    """ returns xr.Dataset with geometry fields """
    assert domain in ['ocn', 'ocn_low']
    fn = f'{path_prace}/Mov/ds_geo_{domain}.nc'
    if os.path.exists(fn):
        ds_geo = xr.open_dataset(fn)
    else:
        print(f'creating geometry file:  {fn}')
        if domain=='ocn':        fe = file_ex_ocn_ctrl
        elif domain=='ocn_low':  fe = file_ex_ocn_lpd
        geometrics = ['TAREA', 'UAREA', 'dz', 'DXT', 'DXU', 'HTN', 'HUS', 'DYT', 'DYU', 'HTE', 'HUW', 'REGION_MASK']
        ds_geo = xr.open_dataset(fe, decode_times=False)[geometrics].drop(['TLONG','TLAT','ULONG','ULAT']).squeeze()
        DZT = xr_DZ(domain)
        DZU = xr_DZ(domain, grid='U')
        DZT.name, DZU.name = 'DZT', 'DZU'
        ds_geo = xr.merge([ds_geo, DZT, DZU])
        ds_geo.to_netcdf(fn)
    return ds_geo

def calc_sections(ds, fns):
    ds_sect = calc_section_transport(ds)
    if fns is None:  return ds_sect
    else:  ds_sect.to_netcdf(fns); print(f'created file:  {fns}')
    
def calc_transports(ds, domain, fnt):
    print('prepare grid')
    DZTx = xr_DZ_xgcm(domain=domain, grid='T')
    DZUx = xr_DZ_xgcm(domain=domain, grid='U')
    (grid, ds_) = pop_tools.to_xgcm_grid_dataset(ds)#, metrics=metrics, coords=coords)
#     print(grid)
#     print(ds_)
    ds_['DZT'] = DZTx
    ds_['DZU'] = DZUx
#     print(ds_)
    
#     grid = xgcm.Grid(ds_)
    print('prepare mask')
    MASK_tt = ds_.REGION_MASK
    Atl_MASK_tt = xr.DataArray(data=np.in1d(MASK_tt, [6,8,9,12]).reshape(MASK_tt.shape),
                               dims=MASK_tt.dims,
                               coords=MASK_tt.coords)
    print('prepare renaming')
    rn = {'nlat_t':'nlat_u', 'nlon_t':'nlon_u'}
    ac = {'nlat_u':ds_['nlat_u'].values, 'nlon_u':ds_['nlon_u'].values}
    Atl_MASK_uu = Atl_MASK_tt.rename(rn).assign_coords(ac)
    print('actual calculation')
    ds_trans = calc_transports_pbc(grid=grid, ds=ds_, MASK_tt=Atl_MASK_tt, MASK_uu=Atl_MASK_uu)
    print('calculation done')
    if fnt is None:  return ds_trans
    else:  ds_trans.to_netcdf(fnt);  print(f'created file:  {fnt}')
    

def main(run, ys=None, ye=None):
    """
    6:30 for lpd transports only 
    21 min for 30 years of monthly lpd  
    input: {run}
    run .. file name

    output:
    ds .. dataset containing FW/SALT transport terms
          saved to f'{path_prace}/Mov/FW_SALT_fluxes_{run}_{y}.nc'
    """
    print(datetime.now())
    if run in ['ctrl', 'rcp']:
        if run=='ctrl':
            s1, s2 = path_ocn_ctrl, spinup
#             if ys is None and ye is None:  ys, ye = 200, 230 #230
        elif run=='rcp':
            s1, s2 = path_ocn_rcp, rcpstr
#             if ys is None and ye is None:  ys, ye = 2070, 2100 #2000, 2101
        domain = 'ocn'
    elif run in ['lpd', 'lr1']:
        if run=='lpd':
            s1, s2 = path_ocn_lpd, lpdstr
            if ys is None and ye is None:  ys, ye = 500, 530
        elif run=='lr1':
            s1, s2 = path_ocn_lr1, lr1str
            if ys is None and ye is None:  ys, ye = 2000, 2101
        domain = 'ocn_low'
    else:
        raise ValueError(f'`run`={run} not implemented')
    ds_geo = geometry_file(domain)
    
    mfkw = dict(decode_times=False, combine='nested', concat_dim='time')
    q = ['SALT', 'VNS', 'UES', 'VVEL', 'UVEL']
    if domain=='ocn_low':  # can concatenate all files
        fns = f'{path_prace}/Mov/FW_Med_Bering_{run}_{ys:04d}-{ye:04d}_monthly.nc'
        fnt = f'{path_prace}/Mov/FW_SALT_fluxes_{run}_{ys:04d}-{ye:04d}_monthly_alt.nc'
        if os.path.exists(fns) and os.path.exists(fnt):
            print(f'files exists:\n{fns}\n{fnt}')
            pass
        else:
            ds_mf = xr.open_mfdataset(mf_list(f'{s1}/{s2}.pop.h.', y1=ys, y2=ye), **mfkw)[q]
            ds = xr.merge([ds_mf, ds_geo])

            # section transports
            if os.path.exists(fns):  print(f'file exists:\n  {fns}'); pass
            else:                    print(f'create file:\n  {fns}'); calc_sections(ds, fns)

            # transport terms
            if os.path.exists(fnt):  print(f'file exists:\n  {fnt}'); pass 
            else:                    print(f'create file:\n  {fnt}'); calc_transports(ds, domain, fnt)

    elif domain=='ocn':    
        for (y,m,f) in IterateOutputCESM(run=run, domain=domain, tavg='monthly'):
            if m==1: print(y)
            if y<ys or y>ye: continue
            fns = f'{path_prace}/Mov/FW_Med_Bering_monthly_{run}_{y:04d}-{m:02d}.nc'
            fnt = f'{path_prace}/Mov/FW_SALT_fluxes_monthly_{run}_{y:04d}-{m:02d}_alt.nc'
            ds_ = xr.open_dataset(f, decode_times=False)[q]
            ds = xr.merge([ds_, ds_geo])
            del(ds_)
            if os.path.exists(fns) and os.path.exists(fnt):
                print(f'files exists:\n{fns}\n{fnt}')
                pass
            else:
                if os.path.exists(fns):  print(f'file exists:\n  {fns}'); pass
                else:                    print(f'create file:\n  {fns}'); calc_sections(ds, fns)

                # transport terms
                if os.path.exists(fnt):  print(f'file exists:\n  {fnt}'); pass
                else:                    print(f'create file:\n  {fnt}'); calc_transports(ds, domain, fnt)
            gc.collect()
      
      
        # loop only through pentades because of memory limits
#         dy = 1
#         y_list = np.arange(ys, ye, dy)
#         for y in tqdm.tqdm(y_list):
#             fns = f'{path_prace}/Mov/FW_Med_Bering_{run}_{y:04d}-{y+dy:04d}_monthly.nc'
#             fnt = f'{path_prace}/Mov/FW_SALT_fluxes_{run}_{y:04d}-{y+dy:04d}_monthly.nc'
#             if os.path.exists(fns) and os.path.exists(fnt):
#                 print(f'files exists:\n{fns}\n{fnt}')
#                 pass
#             else:
#                 ds_mf = xr.open_mfdataset(mf_list(f'{s1}/{s2}.pop.h.', y1=ys, y2=ye), **mfkw)[q]
#                 ds = xr.merge([ds_mf, ds_geo])

#                 # section transports
#                 if os.path.exists(fns):  print(f'file exists:\n  {fns}'); pass
#                 else:                    print(f'create file:\n  {fns}'); calc_sections(ds, fns)

#                 # transport terms
#                 if os.path.exists(fnt):  print(f'file exists:\n  {fnt}'); pass
#                 else:                    print(f'create file:\n  {fnt}'); calc_transports(ds, domain, fnt)
                    
    print(datetime.now())
    return 


if __name__=='__main__':
    print(sys.argv)
    run = sys.argv[1]
    if len(sys.argv)>2:
        ys = int(sys.argv[2])
        ye = int(sys.argv[3])
    else:
        ys, ye = None, None
    main(run=run, ys=ys, ye=ye)

    
    

"""
dt = dict(decode_times=False, combine='nested', concat_dim='time')
da_VNS  = xr.open_mfdataset(dyn(f'{path_prace}/{run}/ocn_yrly_VNS_'      , y), **dt)
da_UES  = xr.open_mfdataset(dyn(f'{path_prace}/{run}/ocn_yrly_UES_'      , y), **dt)
ds_VEL  = xr.open_mfdataset(dyn(f'{path_prace}/{run}/ocn_yrly_UVEL_VVEL_', y), **dt)
# some UES files have no name
# if run=='ctrl':  da_UES  = xr.open_dataarray(f'{path_prace}/{run}/ocn_yrly_UES_{y:04d}.nc' , decode_times=False)
# if e:            da_UES  = xr.open_dataarray(f'{path_prace}/{run}/ocn_yrly_UES_{y:04d}.nc' , decode_times=False)
# some files were created without setting `decode_times` to `False`
# this creates a `time` variable which messes with xr.merge
# if 'time' in da_VNS:   da_VNS  = da_VNS .drop('time')
# if 'time' in da_UES:   da_UES  = da_UES .drop('time')



da_SALT = xr.open_dataset(f'{path_prace}/{run}/ocn_yrly_SALT_{y:04d}.nc', decode_times=False)
da_VNS  = xr.open_dataset(f'{path_prace}/{run}/ocn_yrly_VNS_{y:04d}.nc' , decode_times=False)
da_VVEL = xr.open_dataset(f'{path_prace}/{run}/ocn_yrly_UVEL_VVEL_{y:04d}.nc', decode_times=False).VVEL
# some files were created without setting `decode_times` to `False`
# this creates a `time` variable which messes with xr.merge
if 'time' in da_SALT:  da_SALT = da_SALT.drop('time') 
if 'time' in da_VNS:  da_VNS = da_VNS.drop('time') 

ds = xr.merge([da_SALT, da_VNS, da_VVEL, ds_geo])
ds.VNS .attrs['grid_loc'] = 3121
ds.VVEL.attrs['grid_loc'] = 3221
ds.SALT.attrs['grid_loc'] = 3111

"""