import os
import numpy as np
import xarray as xr

from tqdm import tqdm, tqdm_notebook
from paths import CESM_filename, path_prace, file_ex_ocn_ctrl, file_ex_ocn_lpd
from timeseries import IterateOutputCESM
from xr_DataArrays import depth_lat_lon_names, xr_DZ, xr_DXU
from xr_regression import xr_quadtrend, xr_linear_trends_2D
from grid import shift_ocn_low

class DeriveField(object):
    """ functions to generate netcdf files derived from CESM output / obs. """
    
    def __init__(self, run):
        assert run in ['ctrl', 'rcp', 'hq', 'lpd', 'lc1', 'lpi', 'lr1', 'lr2', 'ld', 'had', 'lq']
        self.run = run
        if run in ['ctrl', 'rcp', 'hq']:
            self.domain = 'ocn'
        elif run in ['lpd', 'lpi', 'lc1', 'lr1', 'lr2', 'ld', 'lq']:
            self.domain = 'ocn_low'
        elif run=='had':
            self.domain = 'ocn_had'
        print(f'self.domain = {self.domain} for {self.run} run, some functions may require setting different domain')
    
    
    def multifile_name(self):
        return
    
    
    def yrly_avg_nc(self, domain, fields, test=False, years=None):
        """ creates yearly average file from monthly
        input:    fields .. list of field names
        output:   [writes netCDF file]
        (takes approx. 2 min for high res ocean data for two 3D fields)
        (takes approx. 4 sec for lower res atm data for one 2D and one 3D field)
        """
        assert domain in ['ocn', 'ocn_rect', 'atm', 'ice']
        assert self.run in ['ctrl', 'rcp', 'hq', 'lpd', 'lc1', 'lpi', 'lr1', 'lr2', 'ld', 'lq']

        print(f'yearly averaging of {self.run} {domain}')
        for field in fields:  print(f'   {field}')

        name = ''
        n_fields = len(fields)
        for i, field in enumerate(fields):
            name += field
            if i<n_fields-1:
                name += '_'
                
        ffield = fields[0]
        
        cond = False
        if years is None:
            cond = True
        elif np.all(years<501):
            cond = True

        if self.run=='lpd' and domain=='atm' and cond:
            # there are no monthly files for the lpd run
            # but for consistency, I create the same files as for the runs with monthly output only
            for y, m, s in IterateOutputCESM(domain=domain, run=self.run, tavg='yrly'):
                new_filename = CESM_filename(domain=domain, run=self.run, y=y, m=0, name=name)
                ds = xr.open_dataset(s, decode_times=False)
                dim = len(np.shape(ds[ffield]))
                if   dim==3:  ds_out = (ds[ffield][0,:,:]).to_dataset()
                elif dim==4:  ds_out = (ds[ffield][0,:,:,:]).to_dataset()
                if len(fields)>1:
                    for field in fields[1:]:
                        dim = len(np.shape(ds[field]))
                        if   dim==3:  ds_out[field] = ds[field][0,:,:]
                        elif dim==4:  ds_out[field] = ds[field][0,:,:,:]
                print(y, new_filename)
                ds_out.to_netcdf(path=new_filename, mode='w')
            return
        
        else:                
            first_year = IterateOutputCESM(domain=domain, run=self.run, tavg='monthly').year

            for y, m, s in IterateOutputCESM(domain=domain, run=self.run, tavg='monthly'):
                if years is not None and y not in years:  continue
                if m==1:
                    new_filename = CESM_filename(domain=domain, run=self.run, y=y, m=0, name=name)
                if os.path.exists(new_filename):
                    continue
                ds = xr.open_dataset(s, decode_times=False)

                if m==1:  # create new xr Dataset
                    dim = len(np.shape(ds[ffield]))
                    if domain in ['atm', 'ocn', 'ice']:
                        if dim==3:  # 2D field
                            ds_out = (ds[ffield][0,:,:]/12).to_dataset()
                        elif dim==4:  # 3D
                            ds_out = (ds[ffield][0,:,:,:]/12).to_dataset()
                    elif domain=='ocn_rect':
                        if dim==2:  # 2D
                            ds_out = (ds[ffield][:,:]/12).to_dataset()
                        elif dim==3:  # 3D
                            ds_out = (ds[ffield][:,:,:]/12).to_dataset()
                    for field in fields[1:]:  # add rest of fields
                        dim = len(np.shape(ds[field]))
                        if domain in ['atm', 'ocn', 'ice']:
                            if dim==3:
                                ds_out[field] = ds[field][0,:,:]/12
                            elif dim==4:
                                ds_out[field] = ds[field][0,:,:,:]/12
                        elif domain=='ocn_rect':
                            if dim==2:
                                ds_out[field] = ds[field][:,:]/12
                            elif dim==3:
                                ds_out[field] = ds[field][:,:,:]/12

                else:  # add subsequent monthly values
                    for field in fields:
                        dim = len(np.shape(ds[field]))
                        if domain in ['atm', 'ocn', 'ice']:
                            if   dim==3:  ds_out[field][:,:]   += ds[field][0,:,:]/12
                            elif dim==4:  ds_out[field][:,:,:] += ds[field][0,:,:,:]/12
                        elif domain=='ocn_rect':
                            if   dim==2:  ds_out[field][:,:]   += ds[field][:,:]/12
                            elif dim==3:  ds_out[field][:,:,:] += ds[field][:,:,:]/12

                if m==12:  # write to new file
                    if 'TLONG' in ds_out:  ds_out.drop('TLONG')
                    if 'TLAT'  in ds_out:  ds_out.drop('TLAT')
                    if 'ULONG' in ds_out:  ds_out.drop('ULONG')
                    if 'ULAT'  in ds_out:  ds_out.drop('ULAT')
                    print(y, new_filename)
                    ds_out.to_netcdf(path=new_filename, mode='w')

                if test==True and y==first_year+2:  break
            print('done!')
            return
    
    
    def make_SST_yrly_data_file(self):
        """ generates annual SST .nc file (t,lat,lon)
        ca. 4:30 min for ctrl/rcp, 1:25 for lpi
        """
        for i, (y,m,s) in enumerate(IterateOutputCESM('ocn', self.run, 'yrly', name='TEMP_PD')):
            print(y)
            da = xr.open_dataset(s, decode_times=False).TEMP[0,:,:]
            da = da.drop(['z_t', 'ULONG', 'ULAT'])
            da['TLAT' ] = da['TLAT' ].round(decimals=2)
            da['TLONG'] = da['TLONG'].round(decimals=2)
            del da.encoding["contiguous"]
            ds = t2ds(da=da, name='SST', t=int(round(da.time.item())))
            ds.to_netcdf(path=f'{path_prace}/SST/SST_yrly_{self.run}_{y}.nc', mode='w')

        combined = xr.open_mfdataset(f'{path_prace}/SST/SST_yrly_{self.run}_*.nc',
                                     concat_dim='time',
                                     autoclose=True,
                                     coords='minimal')
        combined.to_netcdf(f'{path_prace}/SST/SST_yrly_{self.run}.nc')
        # remove extra netCDF files
        return
    
    
    def make_pwqd_TEMP_files(self):
        """ quadratically detrends annually averaged TEMP field at each point for selected 250 year segments of CTRL or LPD simulations
        pwqd : `point wise quadratically detrended`
        """
        if self.run=='ctrl':
            path   = f'{path_prace}/ctrl_rect'
            interp = '.interp900x602'
            mf_fn  = f'{path}/TEMP_PD_yrly_*.interp900x602.nc'
            trange = np.arange(50,300)
            km     = 42
            z      = 'depth_t'
        elif self.run=='lpd':
            path   = f'{path_prace}/lpd'
            interp = ''
            mf_fn  = f'{path}/ocn_yrly_TEMP_PD_*.nc'
            trange = np.arange(0,250)
            km     = 60
            z      = 'z_t'

        # concatenate yearly files
        yrly_TEMP_file = f'{path}/TEMP_yrly{interp}.nc'
        try:
#             assert 1==0
            assert os.path.exists(yrly_TEMP_file)
        except:
            print('making yrly TEMP file')
            da = xr.open_mfdataset(mf_fn, concat_dim='time').TEMP
            da = da.isel(time=trange)
            da.assign_coords(time=da.time.values).to_netcdf(yrly_TEMP_file)
            da.close()
        
        # calculating detrended TEMP field for each vertical level b/c of memory limitations
        for k in tqdm(range(km)):
            fn = f'{path}/TEMP_yrly_pwqd_{k:02d}{interp}.nc'
            try:
#                 assert 1==0
                assert os.path.exists(fn)
            except:
                da_k = xr.open_dataarray(yrly_TEMP_file, decode_times=False).isel({z:k})
                da_pwqd_k = da_k - xr_quadtrend(da_k)
                da_pwqd_k.to_netcdf(fn)
                da_pwqd_k.close()
        # concatenating 
        print(f'{path}/TEMP_yrly_pwqd_*{interp}.nc')
        da_pwqd = xr.open_mfdataset(f'{path}/TEMP_yrly_pwqd_*{interp}.nc',
                                    concat_dim=['depth_t'], chunks={'time':1})
        if self.run=='ctrl':
            da_pwqd = da_pwqd.assign_coords(time=np.arange(51,301))
        elif self.run=='lpd':
            da_pwqd = da_pwqd.assign_coords(time=np.arange(154,404))

#         da = xr.open_dataarray(yrly_TEMP_file, decode_times=False)
#         da_pwqd = da - xr_quadtrend(da)

        # writing out files for individual years
        print(da_pwqd.time)
        for i, y in tqdm(enumerate(da_pwqd.time)):  # 9 mins for ctrl
            da_pwqd.isel(time=i).to_netcdf(f'{path}/TEMP_pwqd_yrly_{int(y.values):04d}{interp}.nc')
        
        return
    
    
    def make_TEMP_SALT_transport_section(self, section_dict, years=None):
        """ zonal or meridional sections along grid lines
        {'nlat':j_34, 'nlon':slice(i_SA,i_CGH)}
        """
        print(self.run)
        assert self.domain in ['ocn', 'ocn_low'], 'only implemented for ocn and ocn_low grids'
        if years is None:  years = np.arange(2000,2101)
        dss = {}
        DZT = xr_DZ(domain=self.domain)
        DZT.name = 'DZT'
        if self.run in ['ctrl', 'rcp']:   fn = file_ex_ocn_ctrl
        elif self.run in ['lpd', 'lr1']:  fn = file_ex_ocn_lpd
        geometry = xr.open_dataset(fn, decode_times=False).drop('time')[['DYT', 'DYU', 'DXT', 'DXU', 'z_t', 'dz']]

        for k, y in enumerate(years):
#             if k==2:  break
            print(y)
            fn_UVEL_VVEL = f'{path_prace}/{self.run}/ocn_yrly_UVEL_VVEL_{y:04d}.nc'
            fn_SALT_VNS_VNE = f'{path_prace}/{self.run}/ocn_yrly_SALT_VNS_UES_{y:04d}.nc'
            fn_TEMP = f'{path_prace}/{self.run}/ocn_yrly_TEMP_{y:04d}.nc'
            fn_VNT_UET = f'{path_prace}/{self.run}/ocn_yrly_VNT_UET_{y:04d}.nc'
            
            UVEL_VVEL = xr.open_dataset(fn_UVEL_VVEL, decode_times=False).drop('time')
            SALT_VNS_UES = xr.open_dataset(fn_SALT_VNS_VNE, decode_times=False).drop('time')
            TEMP = xr.open_dataset(fn_TEMP, decode_times=False).drop('time')
            VNT_UET = xr.open_dataset(fn_VNT_UET, decode_times=False).drop('time')
            
            ds = xr.merge([UVEL_VVEL, SALT_VNS_UES, TEMP, VNT_UET, geometry, DZT],compat='override')
            ds = ds.assign_coords(time=365*y+31).expand_dims('time')  # 31.1. of each year
            
            for key in section_dict.keys():
                fn =  f'{path_prace}/{self.run}/section_{key}_{self.run}_{y:04d}.nc'
                if y>years[0] and os.path.exists(fn):  continue
                else:
                    (i,j) = section_dict[key]
                    ds_ = ds

                    if type(i)==int and type(j)==tuple: 
                        sel_dict = {'nlat':slice(j[0],j[1]), 'nlon':i}
                        if k==0:
                            lon1 = f'{ds_.TLONG.sel(nlon=i,nlat=j[0]).values:6.1f}'
                            lon2 = f'{ds_.TLONG.sel(nlon=i,nlat=j[1]).values:6.1f}'
                            lat1 = f'{ds_.TLAT.sel(nlon=i,nlat=j[0]).values:6.1f}'
                            lat2 = f'{ds_.TLAT.sel(nlon=i,nlat=j[1]).values:6.1f}'
                            print(f'{key:10} merid section: {str(sel_dict):50},{lon1}/{lon2}E,{lat1}N-{lat2}N')
                        list1 = ['UVEL', 'SALT', 'TEMP', 'UES', 'UET', 'DZT', 'DYT', 'DYU', 'z_t', 'dz']
                    elif type(i)==tuple and type(j)==int:
                        if i[1]<i[0]:
                            ds_ = shift_ocn_low(ds_)
                            if i[0]>160:
                                i = (i[0]-320, i[1])

                        sel_dict = {'nlat':j, 'nlon':slice(i[0],i[1])}
                        if k==0:
                            lon1 = f'{ds_.TLONG.sel(nlon=i[0],nlat=j).values:6.1f}'
                            lon2 = f'{ds_.TLONG.sel(nlon=i[1], nlat=j).values:6.1f}'
                            lat1 = f'{ds_.TLAT.sel(nlon=i[0],nlat=j).values:6.1f}'
                            lat2 = f'{ds_.TLAT.sel(nlon=i[1],nlat=j).values:6.1f}'
                            print(f'{key:10} zonal section: {str(sel_dict):50},{lon1}E-{lon2}E,{lat1}/{lat2}N')
                        list1 = ['VVEL', 'SALT', 'TEMP', 'VNS', 'VNT', 'DZT', 'DXT', 'DXU', 'z_t', 'dz']
                    else: raise ValueError('one of i/j needs to be length 2 tuple of ints and the other an int')
                    ds_ = ds_[list1].sel(sel_dict)

                    if k==0:
                        TLAT, TLONG = ds_.TLAT, ds_.TLONG
                    else:
                        ds_['TLAT'], ds_['TLONG'] = TLAT, TLONG 
                    ds_.to_netcdf(fn)
        return
    
    
    def combine_sections(self, sections, years):
        """ """
        for key in tqdm_notebook(sections):
            sects = []
            for y in tqdm_notebook(years):
                fn_ =  f'{path_prace}/{self.run}/section_{key}_{self.run}_{y:04d}.nc'
                da = xr.open_dataset(fn_, decode_times=False)
                if np.shape(np.array(da.coords['nlon'].values))==():  da = da.drop('nlon')
                if np.shape(np.array(da.coords['nlat'].values))==():  da = da.drop('nlat')
                sects.append(da)
            print(len(sects))
            ds = xr.concat(sects, dim='time')#, compat='override')
            ds.to_netcdf(f'{path_prace}/{self.run}/section_{key}_{self.run}.nc')
        return

    
    def make_section_trend_file(self, sections):
        """ calculate trends"""
        for key in tqdm_notebook(sections):
            fn = f'{path_prace}/{self.run}/section_{key}_trend_{self.run}.nc'
            if os.path.exists(fn):  continue
            ds = xr.open_dataset(f'{path_prace}/{self.run}/section_{key}_{self.run}.nc')
            if 'VVEL' in ds:
                ls = ['VVEL', 'SALT', 'TEMP', 'VNS', 'VNT']
                dim_names = ['nlon', 'z_t']
            else:
                ls = ['UVEL', 'SALT', 'TEMP', 'UES', 'UET']
                dim_names = ['nlat', 'z_t']

            for i, l in tqdm_notebook(enumerate(ls)):
                da = xr_linear_trends_2D(ds[l], dim_names=dim_names, with_nans=True)
                da.name = l+'_trend'
                if i==0:
                    ds_ = da.to_dataset()
                else:
                    ds_ = xr.merge([ds_, da])
            ds_.to_netcdf(fn)
        return
    
    
    def make_GMST_with_trends_file(self):
        """ builds a timeseries of the GMST and saves it to a netCDF
    
        input:
        run    .. (str) ctrl or cp

        output:
        ds_new .. xr Dataset containing GMST and T_zonal
        """
        domain = 'atm'
        tavg   = 'yrly'
        name   = 'T_T850_U_V'

        if run in ['ctrl', 'rcp']:   AREA = xr_AREA('atm')
        elif run=='lpi':             AREA = xr_AREA('atm_f19')
        elif run=='lpd':             AREA = xr_AREA('atm_f09')

        AREA_lat   = AREA.sum(dim='lon')
        AREA_total = AREA.sum(dim=('lat','lon'))


        if run in ['lpd']:  name = None

        ny   = len(IterateOutputCESM(domain=domain, run=run, tavg=tavg, name=name))
        first_yr = IterateOutputCESM(domain=domain, run=run, tavg=tavg, name=name).year
        iterator = IterateOutputCESM(domain=domain, run=run, tavg=tavg, name=name)
        years    = (np.arange(ny) + first_yr)*365  # this is consistent with CESM output

        for i, (y, m, file) in enumerate(iterator):
            print(y)
            assert os.path.exists(file)
            if run in ['ctrl', 'rcp', 'lpi']:
                da = xr.open_dataset(file, decode_times=False)['T'][-1,:,:]
            elif run in ['lpd']:
                da = xr.open_dataset(file, decode_times=False)['T'][0,-1,:,:]

            if i==0:  # create new xr Dataset
                lats = da.lat.values
                ds_new = xr.Dataset()
                ds_new['GMST']    = xr.DataArray(data=np.zeros((ny)),
                                                 coords={'time': years},
                                                 dims=('time'))
                ds_new['T_zonal'] = xr.DataArray(data=np.zeros((ny, len(lats))), 
                                                 coords={'time': years, 'lat': lats},
                                                 dims=('time', 'lat'))

            ds_new['GMST'][i]      = (da*AREA).sum(dim=('lat','lon'))/AREA_total
            ds_new['T_zonal'][i,:] = (da*AREA).sum(dim='lon')/AREA_lat

        # [K] to [degC]
        for field in ['GMST', 'T_zonal']:  
            ds_new[field] = ds_new[field] + abs_zero

        # rolling linear trends [degC/yr]
        ds_new = rolling_lin_trends(ds=ds_new, ny=ny, years=years)

        # fits
        lfit = np.polyfit(np.arange(ny), ds_new.GMST, 1)
        qfit = np.polyfit(np.arange(ny), ds_new.GMST, 2)

        ds_new[f'lin_fit']  = xr.DataArray(data=np.empty((len(ds_new['GMST']))),
                                           coords={'time': years},
                                           dims=('time'),
                                           attrs={'lin_fit_params':lfit})
        ds_new[f'quad_fit'] = xr.DataArray(data=np.empty((len(ds_new['GMST']))),
                                           coords={'time': years},
                                           dims=('time'),
                                           attrs={'quad_fit_params':qfit})

        for t in range(ny):
            ds_new[f'lin_fit'][t]  =                lfit[0]*t + lfit[1]
            ds_new[f'quad_fit'][t] = qfit[0]*t**2 + qfit[1]*t + qfit[2]

        ds_new.to_netcdf(path=f'{path_results}/GMST/GMST_{run}.nc', mode='w')

        return ds_new
        
        
    def make_AHC_file(self):
        """ calculate total atmospheric heat content [Joule] """
        def atm_heat_content(ds, dp, AREA):
            assert 'T' in ds
            assert 'lat' in ds.coords
            assert 'lat' in AREA.coords
            assert 'lon' in ds.coords
            assert 'lon' in AREA.coords
            assert 'lev' in ds.coords
            assert 'lev' in dp.coords
            return (AREA*dp*ds['T']*cp_air).sum()
    
    
    def make_OHC_file(self):
        return
    
    
    def make_MOC_file(self):
        # # ca 20 sec per file
        # # 50 min for ctrl
        # # 26 min for rcp
        if selfrun in ['ctrl', 'rcp']:
            DXU  = xr_DXU(self.domain)            # [m]
            DZU  = xr_DZ(self.domain, grid='U')   # [m]
            # MASK = Atlantic_mask('ocn')   # Atlantic
            # MASK = boolean_mask('ocn', 2) # Pacific
            MASK = boolean_mask(self.domain, 0)   # Global OCean
            for i, (y,m,s) in enumerate(IterateOutputCESM(domain=self.domain,
                                                          run=self.run,
                                                          tavg='yrly',
                                                          name='UVEL_VVEL')):
                # ca. 20 sec per year
                print(i, y, s)
                ds = xr.open_dataset(s, decode_times=False)
                MOC = calculate_MOC(ds=ds, DXU=DXU, DZU=DZU, MASK=MASK)
                if i==0:
                    MOC_out = MOC.copy()
                else:
                    MOC_out = xr.concat([MOC_out, MOC], dim='time')
            #     if y==202: break
            MOC_out.to_netcdf(f'{path_results}/MOC/GMOC_{self.run}.nc')
        
        elif self.run in ['lpd', 'lpi']:
            ds = xr.open_mfdataset()
        
        return
    
    def make_CICE_file(self):
        return
    
    def make_freshwater_35S_file(self):
        return
    
    def make_SST_index_file(self):
        return