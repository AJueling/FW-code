""" plotting freshwater/salt fluxes

color scheme: 
fluxes/convergences   C0 (ov); C1 (az); C2 (eddy); C3 (total), C8 (Med/BS)
SFWF                  C4 (P+E+R); C6 (P+E); C7 (R)
mixing                C5
d/dt(SALT)            C9

colors (tab20) + 0 (HR-CESM, tab20(14)) or 1 (LR-CESM, tab20(15))
fluxes/convergences    0 (ov); 2 (az); 4 (eddy); 6 (total), 16 (Med/BS)
SFWF                   8 (P+E+R); 12 (P+E); tab20b(14) (R)
mixing                10
d/dt(SALT)            18

# term                 where is the data stored                         to improve
# d/dt(SALT)           f'{path_results}/SALT/SALT_integrals_ctrl.nc'    use proper DZT and nlats as boundaries
# SFWF                                                                  add other terms, use proper nlats as boundaries
# fluxes/convergences                                                   include Baltic Sea
"""

#region: imports
import numpy as np
import pickle
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc_file('rc_file_paper')
from paths import path_prace, path_results
from paths import file_ex_ocn_ctrl, file_ex_ocn_lpd, file_RMASK_ocn, file_RMASK_ocn_low
from FW_budget import load_obj, lat_bands
from mimic_alpha import make_color_transparent as mct
from curly_bracket import curlyBrace
from FW_transport import sections_high, sections_low
from xr_regression import ocn_field_regression, xr_linear_trend, xr_regression_with_stats
#endregion


#region: auxiliary functions
S0 = 35.
lat_bounds = ['90N', '60N', '45N', '10N', '10S', '34S']
nabla, Delta, mix, deg = r'$\nabla$', r'$\Delta$', r'$_{mix}$', r'$\!^\circ\!$'
ov, az, ed, to, BSMed, half = r'$_{ov}$', r'$_{az}$', r'$_{eddy}$', r'$_{total}$', r'$_{BS/Med}$', r'$\frac{1}{2}\times$'
xrkw = {'decode_times':False}                 # xr.open_* keywords
vbkw = dict(y=0, width=.1, align='edge')      # vertical bar keywords
hbkw = dict(left=0, height=.1, align='edge')  # horizontal bar keywords

dsh = xr.open_dataset(file_ex_ocn_ctrl, decode_times=False)
dsl = xr.open_dataset(file_ex_ocn_lpd , decode_times=False)

RMASK_ocn = xr.open_dataarray(file_RMASK_ocn)
RMASK_low = xr.open_dataarray(file_RMASK_ocn_low)
Atl_MASK_ocn = xr.DataArray(np.in1d(RMASK_ocn, [6,8,9]).reshape(RMASK_ocn.shape),
                            dims=RMASK_ocn.dims, coords=RMASK_ocn.coords)
Atl_MASK_low = xr.DataArray(np.in1d(RMASK_low, [6,8,9]).reshape(RMASK_low.shape),
                            dims=RMASK_low.dims, coords=RMASK_low.coords)


def nlat_at(sim, lat):
    """ returns nlat of high or low res grid for certain latitudes """
    assert sim in ['HIGH', 'LOW']
    if type(lat)==str:
        assert lat in lat_bounds
        lat_str = lat
        if lat_str[-1]=='N':  lat = float(lat_str[:-1])
        if lat_str[-1]=='S':  lat = -float(lat_str[:-1])
    elif type(lat)==float or type(lat)==int:
        if lat<0: lat_str = str(-lat)+'S'
        else:     lat_str = str(lat)+'N'
        assert lat_str in lat_bounds
    else:
        print(f'lat {lat}, type is {type(lat)}')
    
    sdict = load_obj(f'{path_results}/sections/section_dict_{sim.lower()}')
    nlat = sdict[lat_str]['nlat']
    if sim=='LOW' and lat==-34:  nlat = 85
    if sim=='LOW' and lat==60:   nlat = 347
    if sim=='HIGH' and lat==-34:   nlat = 811
    if sim=='HIGH' and lat==60:   nlat = 1866
    return nlat

def get_ddt_SALT(sim, latS, latN):
    """ calculate d(S)/dt term in [kg/s] 
    as the difference between the last and first years

    `make_SALT_vol_int_dict(run)` in `FW_budget.py` makes
    f'{path_results}/SALT/SALT_integrals_{run}.nc'
    """
    if sim=='HIGH':   run, rcp = 'ctrl', 'rcp'
    elif sim=='LOW':  run, rcp = 'lpd' , 'lr1'
    # files created in `FW_budget.py` `make_SALT_vol)_int`
    salt_ctl = xr.open_dataset(f'{path_results}/SALT/SALT_integrals_{run}.nc', **xrkw)
    salt_rcp = xr.open_dataset(f'{path_results}/SALT/SALT_integrals_{rcp}.nc', **xrkw)
    assert len(salt_ctl.time)==101  # dataset contains only integrals for 101 years concurrent to rcp
    if latS==-34 and latN==60:
        for i, (latS_, latN_) in enumerate([(-34,-10),(-10,10),(10,45),(45,60)]):
            n1 = f'SALT_0-1000m_timeseries_{latS_}N_{latN_}N'
            n2 = f'SALT_below_1000m_timeseries_{latS_}N_{latN_}N'
            if i==0:
                salt_ctl_ = (salt_ctl[n1] + salt_ctl[n2])
                salt_rcp_ = (salt_rcp[n1] + salt_rcp[n2])
            else:
                salt_ctl_ += (salt_ctl[n1] + salt_ctl[n2])
                salt_rcp_ += (salt_rcp[n1] + salt_rcp[n2])
        ddtS_ctl = (salt_ctl_.isel(time=30)-salt_ctl_.isel(time=0)).values/30/365/24/3600
        ddtS_rcp = (salt_rcp_.isel(time=-1)-salt_rcp_.isel(time=0)).values/101/365/24/3600
    else:

        n1 = f'SALT_0-1000m_timeseries_{latS}N_{latN}N'
        n2 = f'SALT_below_1000m_timeseries_{latS}N_{latN}N'
        # multiply by 100 because saved as g_S/kg_W * m^3 = 1e-2 kg_S
        salt_ctl_ = (salt_ctl[n1] + salt_ctl[n2])
        salt_rcp_ = (salt_rcp[n1] + salt_rcp[n2])
        ddtS_ctl = (salt_ctl_.isel(time=30)-salt_ctl_.isel(time=0)).values/30/365/24/3600
        ddtS_rcp = (salt_rcp_.isel(time=-1)-salt_rcp_.isel(time=0)).values/101/365/24/3600
    return ddtS_ctl, ddtS_rcp

def get_SFWF(sim, quant, latS, latN):
    """ reads the integrated surface freshwater fluxes between latS and latN from pickled object """
    assert sim in ['HIGH', 'LOW'] and  quant in ['SALT', 'FW']
    assert latS in [-34, -10, 10, 45, 60]
    assert latN in [-10, 10, 45, 60, 90]

    d = load_obj(f'{path_results}/SFWF/Atlantic_SFWF_integrals_{sim}_{latS}N_{latN}N')
    # if quant=='SALT':  fac = -.00347e9  # 1e9 kg/s/Sv * salinity_factor of POP2 \approx -S0/1e6*.1?
    if quant=='SALT':  fac = -35e6  # 1e9 kg/s/Sv * salinity_factor of POP2 \approx -S0/1e6*.1?
    elif quant=='FW':  fac = 1.
    mean, trend = {}, {}
    
    mean['PE']   = (d[f'Pmi_Sv']+d[f'Emi_Sv'])*fac
    mean['R']    = d[f'Rmi_Sv']*fac
    mean['SFWF'] = d[f'Tmi_Sv']*fac

    trend['PE']  = (d[f'Pti_Sv']+d[f'Eti_Sv'])*fac
    trend['P']   = d[f'Pti_Sv']*fac
    trend['E']   = d[f'Eti_Sv']*fac
    trend['R']   = d[f'Rti_Sv']*fac
    trend['SFWF'] = d[f'Tti_Sv']*fac
    return mean, trend

def get_fluxes(sim, quant, lat=None, latS=None, latN=None):
    """ fluxes at `lat` and neg. flux convergences between latS and latN"""
    assert sim in ['HIGH', 'LOW'] and  quant in ['SALT', 'FW']
    assert latS in [-34, -10, 10, 45, None]
    assert latN in [-10, 10, 45, 60, None]
    if sim=='HIGH':   run, rcp = 'ctrl', 'rcp'
    elif sim=='LOW':  run, rcp = 'lpd', 'lr1'

    dso = xr.open_dataset(f'{path_prace}/Mov/FW_SALT_fluxes_{run}.nc', **xrkw)
    dst = xr.open_dataset(f'{path_prace}/Mov/FW_SALT_fluxes_{rcp}.nc', **xrkw)
    if lat is not None:   nlat = nlat_at(sim, lat)
    if latS is not None:  nlat_S, nlat_N =  nlat_at(sim, latS), nlat_at(sim, latN)
    # select 30 year mean of fluxes fro branch-off year 200/500
    if sim=='HIGH':   dso = dso.isel(time=slice(200,230)).mean('time')
    elif sim=='LOW':  dso = dso.isel(time=slice(346,376)).mean('time')

    fluxes, fac = {}, 365*100
    if quant=='SALT' and lat is not None:
        fluxes['ov']  = dso.Sov.isel(nlat_u=nlat)
        fluxes['az']  = dso.Saz.isel(nlat_u=nlat)
        fluxes['ed']  = dso.Se.isel(nlat_u=nlat)
        fluxes['to']  = dso.St.isel(nlat_u=nlat)
        
        fluxes['tov'] = fac*xr_linear_trend(dst.Sov.isel(nlat_u=nlat))
        fluxes['taz'] = fac*xr_linear_trend(dst.Saz.isel(nlat_u=nlat))
        fluxes['ted'] = fac*xr_linear_trend(dst.Se.isel(nlat_u=nlat))
        fluxes['tto'] = fac*xr_linear_trend(dst.St.isel(nlat_u=nlat))

    elif quant=='FW' and lat is not None:
        fluxes['ov']  = dso.Fov.isel(nlat_u=nlat)
        fluxes['az']  = dso.Faz.isel(nlat_u=nlat)
        fluxes['ed']  = -dso.Se.isel(nlat_u=nlat)/35e6  # [kg/s] -> [Sv]
        fluxes['to']  = (dso.Fov+dso.Faz+fluxes['ed']).isel(nlat_u=nlat)

        fluxes['tov'] = fac*xr_linear_trend(dst.Fov.isel(nlat_u=nlat))
        fluxes['taz'] = fac*xr_linear_trend(dst.Faz.isel(nlat_u=nlat))
        fluxes['ted'] = -fac*xr_linear_trend(dst.Se.isel(nlat_u=nlat)/35e6)
        fluxes['tto'] = fac*xr_linear_trend((dst.Fov+dst.Faz-dst.Se/35e6).isel(nlat_u=nlat))

    conv = {}
    if quant=='SALT' and latS is not None:
        conv['ov']  = dso.Sov.isel(nlat_u=nlat_S) - dso.Sov.isel(nlat_u=nlat_N)
        conv['az']  = dso.Saz.isel(nlat_u=nlat_S) - dso.Saz.isel(nlat_u=nlat_N)
        conv['ed']  = dso.Se.isel(nlat_u=nlat_S)  - dso.Se.isel(nlat_u=nlat_N)
        conv['to']  = dso.St.isel(nlat_u=nlat_S)  - dso.St.isel(nlat_u=nlat_N)
        
        conv['tov'] = fac*xr_linear_trend(dst.Sov.isel(nlat_u=nlat_S) - dst.Sov.isel(nlat_u=nlat_N))
        conv['taz'] = fac*xr_linear_trend(dst.Saz.isel(nlat_u=nlat_S) - dst.Saz.isel(nlat_u=nlat_N))
        conv['ted'] = fac*xr_linear_trend(dst.Se.isel(nlat_u=nlat_S)  - dst.Se.isel(nlat_u=nlat_N))
        conv['tto'] = fac*xr_linear_trend(dst.St.isel(nlat_u=nlat_S)  - dst.St.isel(nlat_u=nlat_N))
        
    elif quant=='FW' and latS is not None:
        conv['ov']  = dso.Fov.isel(nlat_u=nlat_S) - dso.Fov.isel(nlat_u=nlat_N)
        conv['az']  = dso.Faz.isel(nlat_u=nlat_S) - dso.Faz.isel(nlat_u=nlat_N)
        conv['ed']  = -(dso.Se.isel(nlat_u=nlat_S) - dso.Se.isel(nlat_u=nlat_N))/35e6
        conv['to']  = (dso.Fov+dso.Faz-dso.Se/35e6).isel(nlat_u=nlat_S) - \
                      (dso.Fov+dso.Faz-dso.Se/35e6).isel(nlat_u=nlat_N)

        conv['tov'] = fac*xr_linear_trend(dst.Fov.isel(nlat_u=nlat_S) - dst.Fov.isel(nlat_u=nlat_N))
        conv['taz'] = fac*xr_linear_trend(dst.Faz.isel(nlat_u=nlat_S) - dst.Faz.isel(nlat_u=nlat_N))
        conv['ted'] = -(fac*xr_linear_trend(dst.Se.isel(nlat_u=nlat_S)  - dst.Se.isel(nlat_u=nlat_N))/35e6)
        conv['tto'] = fac*xr_linear_trend((dst.Fov+dst.Faz-dst.Se/35e6).isel(nlat_u=nlat_S) - \
                                          (dst.Fov+dst.Faz-dst.Se/35e6).isel(nlat_u=nlat_N))
    return fluxes, conv

def get_BS_Med(sim):
    """ Bering Strait, Strait of Gibraltar, & 24S transports:
    mean of control and trend of rcp simulations
    """
    # if sim=='HIGH':  ctl, rcp = 'ctrl_020', 'rcp_209'
    # elif sim=='LOW':  ctl, rcp = 'lpd', 'lr1'
    # def add_dim(da):
    #     return da.expand_dims('time')
    # ds_ctl = xr.open_mfdataset(f'{path_prace}/Mov/FW_Med_Bering_{ctl}*.nc', combine='by_coords', concat_dim='time', decode_times=False, preprocess=add_dim)
    # ds_rcp = xr.open_mfdataset(f'{path_prace}/Mov/FW_Med_Bering_{rcp}*.nc', combine='by_coords', concat_dim='time', decode_times=False, preprocess=add_dim)

    if sim=='HIGH':
        kw = dict(combine='by_coords', concat_dim='time', decode_times=False)
        ds_ctl = xr.open_mfdataset(f'{path_prace}/Mov/FW_Med_Bering_monthly_ctrl_020*.nc', **kw).mean('time').compute()
        ds_rcp = xr.open_mfdataset(f'{path_prace}/Mov/FW_Med_Bering_monthly_rcp_209*.nc', **kw).mean('time').compute()
    elif sim=='LOW':
        kw = dict(decode_times=False)
        ds_ctl = xr.open_dataset(f'{path_prace}/Mov/FW_Med_Bering_lpd_0500-0530_monthly.nc', **kw).mean('time')
        ds_rcp = xr.open_dataset(f'{path_prace}/Mov/FW_Med_Bering_lr1_2000-2101_monthly.nc', **kw).isel(time=slice(-12*10,None)).mean('time')
    ds_trend = ds_rcp-ds_ctl
    # ds_trend = ds_rcp
    return ds_ctl, ds_trend

def draw_region_labels(fig, names=True):
    for i, lb in enumerate([f'34{deg}S',f'10{deg}S',f'10{deg}N',f'45{deg}N',f'60{deg}N','Bering\nStrait',f'34{deg}S',f'60{deg}N']):
        name = ['SA-STG', 'ITCZ', 'NA-STG', 'NA-SPG', 'Arctic', None, 'Atlantic', None][i]
        fig.text(3/32+i*1/8,.84, lb, fontsize=7, rotation=90, va='bottom', ha='center', color='grey')
        if names:  fig.text(5/32+i*1/8,.92, name, va='center', ha='center', fontsize=7, color='k')
    return

def make_Med_mask(length, sections):
    mask = np.ones((length))
    nlat1 = sections['Med']['nlat'].start
    nlat2 = sections['Med']['nlat'].stop
    for i in np.arange(nlat1, nlat2):
        mask[i] = 0
    return mask

def mask_Med(lat, da, sections, mask2=None):
    mask = np.ones((len(lat)))
    nlat1 = sections['Med']['nlat'].start
    nlat2 = sections['Med']['nlat'].stop
    for i in np.arange(nlat1, nlat2):
        mask[i] = 0
    masked = np.isfinite(da.where(mask))
    if mask2 is None:
        return lat.values[masked], da.values[masked]
    elif mask2 is not None:
        return lat.values[masked], da[masked].where(mask2[masked])

def lons_lats_from_sdict(s, lat, MASK):
    """ returns lons and lats from sdict entry """
    if MASK.shape==(2400,3600):  sim = 'HIGH'
    elif MASK.shape==(384,320):  sim = 'LOW'
    TLAT, TLONG = MASK.TLAT, MASK.TLONG
    if type(s['nlon'])==slice and type(s['nlat'])==int:  # E-W section
        nlon1, nlon2, nlat = s['nlon'].start, s['nlon'].stop, nlat_at(sim=sim, lat=lat)
        print(lat,nlat)
        lats, lons = [], []
        if nlon1>nlon2:
            nlons = np.concatenate([np.arange(nlon1,len(TLONG[0,:])), np.arange(0, nlon2)])
        else:
            nlons = np.arange(nlon1, nlon2)
        for nlon in nlons:
            lons.append(float(TLONG[nlat, nlon].values))
            lats.append(float(TLAT [nlat, nlon].values))
        for i in range(len(lons)):
            if lons[i]>180:  lons[i] -= 360
    return lons, lats

def Atl_lats(sim):
    if sim=='HIGH':
        lats = dsh.TLAT.where(Atl_MASK_ocn).mean(dim='nlon', skipna=True)
        n = 2400
    elif sim=='LOW':
        lats = (dsl.TLAT.where(Atl_MASK_low).mean(dim='nlon', skipna=True)+\
                dsl.TLAT.where(Atl_MASK_low).max(dim='nlon', skipna=True))/2
        n = 384
    return lats.fillna(np.linspace(-100,90,n))

def draw_scales(f, left, vlim, hlim):
    """ darwing scales in mean and trend plots"""
    assert left in [True,False]
    # print(hlim,vlim)

    if left:
        xv, xh = 11.75, 9.5
    else:
        xv, xh = 13.25, 13.5

    if vlim in [.25,.3]:     vc = .2/vlim; vt = [-.2,0,.2]
    elif vlim in [1.1,1.9]:  vc = 1/vlim; vt = [-1,0,1]
    # else: print('no vertical scale correction')
    
    if hlim in [.7,.9]:  hc = .5/hlim; ht = [0,.5]
    elif hlim==.2:       hc = .1/hlim; ht = [0,.1]
    # else: print('no horizontal scale correction')

    ax0 = f.add_axes([xv/16,.75-.25*vc,0,.25*vc])
    ax0.set_yticks(vt)
    ax0.set_ylim((0,vlim*vc))
    ax0.invert_yaxis()

    ax1 = f.add_axes([xv/16,.2,0,.25*vc])
    ax1.set_yticks(vt)
    ax1.set_ylim((0,vlim*vc))
    
    for ax in [ax0, ax1]:
        ax.set_xticks([])
        # ax.set_yticklabels(ax.get_yticks(),(rotation=90)
        ax.tick_params(axis='y', labelrotation = 90)
        if left:
            ax.yaxis.tick_right()
            for sp in ['top','left','bottom']:  ax.spines[sp].set_visible(False)
        else:
            for sp in ['top','right','bottom']:  ax.spines[sp].set_visible(False)

    ax2 = f.add_axes([xh/16,.1,1/8*hc,.1])
    ax2.set_xticks(ht)
    ax2.set_xlim((0,hlim*hc))
    ax2.set_yticks([])
    for sp in ['left','top','right']:  ax2.spines[sp].set_visible(False)
    
    for ax in [ax0, ax1, ax2]:  ax.patch.set_alpha(0)
    return

def draw_labels(f, units):
    ax0 = f.add_axes([12.498/16,.475,0,0])
    ax1 = f.add_axes([13.25/16,.1,0,0])
    for ax in [ax0,ax1]:
        for sp in ['top','left','bottom','right']:  ax.spines[sp].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    if units=='Sv':
        t1, t2, ha = 'fluxes/convergences  [Sv]', 'transports  [Sv]','center'
    elif units=='Sv/century':
        t1, t2, ha = 'flux/conv. trends  [Sv/100yr]', 'tr. trends  [Sv/100yr]', 'right'
    ax0.text(0,0,t1,fontsize=7,va='center',ha=ha,rotation=90)
    ax1.text(0,0,t2,fontsize=7,va='top',ha='right')
    return

#endregion


def FW_summary_plot(quant, empty=False):  # Fig. 9a
    """ plots main budget terms for certain regions
    d/dt = conv + SFWF + diffusion
    """
    assert quant in ['FW', 'SALT']
    if quant=='FW':      q, Q0, Q1 = 0, 'F', 'W'
    elif quant=='SALT':  q, Q0, Q1 = 1, 'S', 'S'

    # f = plt.figure(figsize=(8,2))
    f = plt.figure(figsize=(6.4,1.8))

    for i, (latS, latN) in enumerate(lat_bands):
        #region: define axes
        if i==0:  # Atlantic 34S-60N
            ax = f.add_axes([13.5/16,.05,1/8,.78])
            for s in ['top', 'bottom']:  ax.spines[s].set_visible(False)
            ax.xaxis.set_ticks([])
            ax.axhline(c='k', lw=.5)
            # ax.set_ylabel(f'{quant} fluxes in [{["Sv","kg/s"][q]}]')
            vert_lim = [(-1.2,1.2),(-40,40)][q]
            ax.set_xlim((-.25,1.2))
            ax.set_ylim(vert_lim)
            ax.set_yticks([np.arange(-1,1.2,.5),np.arange(-40,42,20)][q])

        else:
            ax = f.add_axes([(2*i+1)/16-1.5/16,.05,1/8,.78])
            for s in ['top', 'bottom']:  ax.spines[s].set_visible(False)
            if i<5:  ax.spines['right'].set_visible(False)
            ax.xaxis.set_ticks([])
            ax.axhline(c='k', lw=.5)
            if i==1:  ax.set_ylabel(f'{["freshwater", "salt"][q]} budget  [{["Sv","kt/s"][q]}]')
            else:     ax.yaxis.set_ticklabels([])
            vert_lim = [(-.9,.9),(-32,32)][q]
            ax.set_xlim((-.25,1.2))
            ax.set_ylim(vert_lim)
            ax.set_yticks([np.arange(-.8,1,.4),np.arange(-30,32,15)][q])
        #endregion
        
        for s, sim in enumerate(['HIGH', 'LOW']):  
            if empty:  continue

            #region: d/dt
            if quant=='SALT':  fac = 1e-6
            elif quant=='FW':  fac = -1e-6/S0
            ddtS_ctl, ddtS_rcp = get_ddt_SALT(sim=sim, latS=latS, latN=latN)
            label = [r'$\Delta \bar{W}$', r'$\Delta \bar{S}$'][q]
            bdt = ax.bar(x=s/10, height=ddtS_ctl*fac, **vbkw, color=plt.cm.tab20(18+s), label=label)
            ax.arrow(x=.05+s/10, y=0, dy=ddtS_rcp*fac, dx=0)
            #endregion
            
            #region: surface fluxes
            sfwf_mean, sfwf_trend = get_SFWF(sim=sim, quant=quant, latS=latS, latN=latN)
            bsf = ax.bar(x=.25+s/10 , height=sfwf_mean['SFWF']/[1,1e6][q], **vbkw, color=plt.cm.tab20(8+s), label=[r'$F_{surf}$', r'$F^S_{surf}$'][q])
            ax.arrow(x=.3+s/10, y=sfwf_mean['SFWF']/[1,1e6][q], dy=sfwf_trend['SFWF']/[1,1e6][q], dx=0)
            #endregion

            #region: meridional flux convergences incl. BS/Med
            inflow, inflow_trend = get_BS_Med(sim)
            if latS==60:  # Arctic
                fluxes, conv = get_fluxes(sim=sim, quant=quant, lat=latS, latS=None, latN=None)
                print(f'BS  {quant}:', inflow[f'{Q0}_BS'].values/1e6)
                print(f'60N {quant}:', fluxes['to'].values/[1,1e6][q])
                conv['to'] = fluxes['to']/[1,1e6][q] + inflow[f'{Q0}_BS'].values/1e6  # [m^3/s] -> [Sv], [kg/s] -> [kt/s]
            else:
                fluxes, conv = get_fluxes(sim=sim, quant=quant, lat=latS, latS=latS, latN=latN)
                conv['to'] = conv['to']/[1,1e6][q]
                if latS<35 and latN>35:  # Med
                    conv['to'] -= inflow[f'{Q0}_Med'].values/1e6  # [m^3/s] -> [Sv], [kg/s] -> [kt/s]
                ax.arrow(x=.55+s/10, y=conv['to'], dx=0, dy=conv['tto']/[1,1e6][q])
            labeln = [r'$F_{\nabla}$', r'$F^S_{\nabla}$'][q]
            bct = ax.bar(x=.5+s/10, height=conv['to'], **vbkw, color=plt.cm.tab20(6+s), label=labeln)
            #endregion
            
            #region: mixing term
            diffusion = ddtS_ctl*fac - conv['to'] - sfwf_mean['SFWF']/[1,1e6][q]
            bdi = ax.bar(x=.75+s/10, height=diffusion, **vbkw, color=mct('C5', 1-.3*s), label=[r'$F_{mix}$', r'$F^S_{mix}$'][q])
            #endregion
            
    #region: numbering, region labels
    f.text(.01,.92,'(a)')
    draw_region_labels(f)  
    #endregion

    # plt.savefig(f'{path_results}/Mov/{["FW","SALT"][q]}_region_budget_total.eps')
    # plt.savefig(f'{path_results}/FW-paper/{["FW","SALT"][q]}_region_budget_total.eps')
    if q==0 and empty==False:
        plt.savefig(f'{path_results}/FW-paper/Fig9a.eps')
    return


def FW_region_plot(quant, empty=False):  # Fig. 9b
    """ plots SFWF, merid. transport and its convergence for specific regions
    quant .. Fw or SALT
    """
    assert quant in ['FW', 'SALT']
    if quant=='FW':      q, Q, Q1 = 0, 'F', 'W'
    elif quant=='SALT':  q, Q, Q1 = 1, 'S', 'S'
    lat_bounds = ['90N', '60N', '45N', '10N', '10S', '34S']

    # f = plt.figure(figsize=(8,3))
    f = plt.figure(figsize=(6.4,2.5))
    # draw_region_labels(f)

    dd = {}
    for i, (latS, latN) in enumerate(lat_bands):
        #region: define axes
        if i==0:  # Atlantic 34S-60N
            ax = f.add_axes([13.5/16,.2,1/8,.55])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

            axb = f.add_axes([13.5/16,-.05,1/8 ,.5 ])
            axt = f.add_axes([13.5/16,.5  ,1/8 ,.5 ])
            axl = f.add_axes([12  /16,.2  ,3/16,.55])
            axr = f.add_axes([14  /16,.2  ,3/16,.55])
            vert_lim = [(-1.9,1.9),(-8e7,8e7)][q]
            hor_lim = [(-.9,.9),(-3e8,3e8)][q]

            draw_scales(f=f,left=False,vlim=vert_lim[1],hlim=hor_lim[1])
            draw_labels(f,'Sv')

        else:
            ax = f.add_axes([(2*i+1)/16-1.5/16,.2,1/8,.55])
            # if i==5:  ax.spines['right'].set_visible(False)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

            axb = f.add_axes([(2*i+1)/16-1.5/16,-.05,1/8 ,.5 ])
            axt = f.add_axes([(2*i+1)/16-1.5/16,.5  ,1/8 ,.5 ])
            axl = f.add_axes([(2*i+1)/16-3/16  ,.2  ,3/16,.55])
            axr = f.add_axes([(2*i+1)/16-1/16  ,.2  ,3/16,.55])

            vert_lim = [(-1.1,1.1),(-4.5e7,4.5e7)][q]
            hor_lim = [(-.7,.7),(-1.2e8,1.2e8)][q]

            if i==1:
                draw_scales(f=f,left=True,vlim=vert_lim[1],hlim=hor_lim[1])

        for ax in [axl, axr]:
            ax.set_xlim(hor_lim)
            ax.set_ylim((-.35,1.3))

        for ax in [axb, axt, axl, axr]:
            # ax.patch.set_alpha(0)
            ax.axis('off')
            ax.xaxis.set_ticks([])

        for ax in [axt, axb]:
            ax.set_ylim(vert_lim)
            ax.set_xlim((-.25,1.45))
        #endregion

        #region:  plotting
        for s, sim in enumerate(['HIGH', 'LOW']):
            if empty:  continue
            nlat_ = nlat_at(sim, latS)
            run, rcp = ['ctrl','lpd'][s], ['rcp', 'lr1'][s]

            #region: d/dt  [axb]
            if quant=='SALT':  fac = 1
            elif quant=='FW':  fac = -1e-6/S0
            ddtS_ctl, ddtS_rcp = get_ddt_SALT(sim=sim, latS=latS, latN=latN)
            label = [r'$\Delta \bar{W}$', r'$\Delta \bar{S}$'][q]
            bdt = axt.bar(x=.75+s/10, height=ddtS_ctl*fac, **vbkw, color=plt.cm.tab20(18+s), label=label)
            axt.arrow(x=.8+s/10, y=0, dx=0, dy=ddtS_rcp*fac)
            #endregion

            #region: meridional flux (convergences) means/trends, incl. BS/Med  [axl, axr, axb]
            inflow, inflow_trend = get_BS_Med(sim)
            if latS==60:  # BS in Arctic
                fluxes, conv = get_fluxes(sim=sim, quant=quant, lat=latS, latS=None, latN=None)
                conv['ov'], conv['az'], conv['ed'] = 0, 0, 0
                conv['to'] = fluxes['to'] + inflow[f'{Q}_BS'].values/[1e6,1][q]  # [m^3/s] -> [Sv]
                conv['tov'], conv['taz'], conv['ted'], conv['tto'] = 0, 0, 0, fluxes['tto']
                
                labelb = [r'$F_{BS/Med}$',r'$F^S_{BS/Med}$'][q]
                axr.set_xlim(hor_lim)
                bi = axr.barh(y=.375+s/10, width=-inflow[f'{Q}_BS'].values/[1e6,1][q],\
                              **hbkw, color=mct('C8', 1-.3*s), label=labelb, zorder=1)
                axr.arrow(x=-inflow[f'{Q}_BS'].values/[1e6,1][q], y=.425+s/10, dx=-inflow_trend[f'{Q}_BS'].values/[1e6,1][q], dy=0)
            else:
                fluxes, conv = get_fluxes(sim=sim, quant=quant, lat=latS, latS=latS, latN=latN)
                if latS<35 and latN>35:  # Med
                    conv['to'] -= inflow[f'{Q}_Med'].values/[1e6,1][q]  # [m^3/s] -> [Sv] 
                    bi = axb.bar(x=1+s/10, height=-inflow[f'{Q}_Med'].values/[1e6,2][q],\
                                 **vbkw, color=mct('C8', 1-.3*s))
                    axb.arrow(y=-inflow[f'{Q}_Med'].values/[1e6,1][q], x=1.05+s/10, dy=-inflow_trend[f'{Q}_Med'].values/[1e6,1][q], dx=0)

            def hbar(ax, y, w, c, l=None):
                return ax.barh(y=y, width=w, color=c, label=l, **hbkw)

            bov = hbar(ax=axl, y=.5 +s/10, w=fluxes['ov'], c=plt.cm.tab20(0+s))
            baz = hbar(ax=axl, y=.25+s/10, w=fluxes['az'], c=plt.cm.tab20(2+s))
            bed = hbar(ax=axl, y=0  +s/10, w=fluxes['ed'], c=plt.cm.tab20(4+s))
            bto = hbar(ax=axl, y=.75+s/10, w=fluxes['to'], c=plt.cm.tab20(6+s))

            axl.arrow(x=fluxes['ov'], y=.55+s/10, dx=fluxes['tov'], dy=0)
            axl.arrow(x=fluxes['az'], y=.3 +s/10, dx=fluxes['taz'], dy=0)
            axl.arrow(x=fluxes['ed'], y=.05+s/10, dx=fluxes['ted'], dy=0)
            axl.arrow(x=fluxes['to'], y=.8 +s/10, dx=fluxes['tto'], dy=0, clip_on=False)
            
            if i==0:  # draw 60 North values in Atlantic box
                fluxes_, conv_ = get_fluxes(sim=sim, quant=quant, lat=latN, latS=latS, latN=latN)

                hbar(ax=axr, y=.5 +s/10, w=fluxes_['ov'], c=plt.cm.tab20(0+s))
                hbar(ax=axr, y=.25+s/10, w=fluxes_['az'], c=plt.cm.tab20(2+s))
                hbar(ax=axr, y=0  +s/10, w=fluxes_['ed'], c=plt.cm.tab20(4+s))
                hbar(ax=axr, y=.75+s/10, w=fluxes_['to'], c=plt.cm.tab20(6+s))

                axr.arrow(x=fluxes_['ov'], y=.55+s/10, dx=fluxes_['tov'], dy=0)
                axr.arrow(x=fluxes_['az'], y=.3 +s/10, dx=fluxes_['taz'], dy=0)
                axr.arrow(x=fluxes_['ed'], y=.05+s/10, dx=fluxes_['ted'], dy=0)
                axr.arrow(x=fluxes_['to'], y=.8 +s/10, dx=fluxes_['tto'], dy=0)

            axb.bar(x=0  +s/10, height=conv['to'], **vbkw, color=plt.cm.tab20(6+s))
            axb.bar(x=.25+s/10, height=conv['ov'], **vbkw, color=plt.cm.tab20(0+s))
            axb.bar(x=.5 +s/10, height=conv['az'], **vbkw, color=plt.cm.tab20(2+s))
            axb.bar(x=.75+s/10, height=conv['ed'], **vbkw, color=plt.cm.tab20(4+s))
        
            axb.arrow(x=.05+s/10, y=conv['to'], dx=0, dy=conv['tto'])
            axb.arrow(x=.3 +s/10, y=conv['ov'], dx=0, dy=conv['tov'])
            axb.arrow(x=.55+s/10, y=conv['az'], dx=0, dy=conv['taz'])
            axb.arrow(x=.8 +s/10, y=conv['ed'], dx=0, dy=conv['ted'])


            #endregion

            #region: surface fluxes [axt]
            sfwf_mean, sfwf_trend = get_SFWF(sim=sim, quant=quant, latS=latS, latN=latN)
            bsf = axt.bar(x=.0 +s/10, height=-sfwf_mean['SFWF'], **vbkw, color=plt.cm.tab20(8+s))
            br  = axt.bar(x=.25+s/10, height=-sfwf_mean['R']   , **vbkw, color=plt.cm.tab20b(14+s))
            bpe = axt.bar(x=.5 +s/10, height=-sfwf_mean['PE']  , **vbkw, color=plt.cm.tab20(12+s))

            axt.arrow(x=.05+s/10, y=-sfwf_mean['SFWF'], dx=0, dy=-sfwf_trend['SFWF'])
            axt.arrow(x=.3 +s/10, y=-sfwf_mean['R']   , dx=0, dy=-sfwf_trend['R']  )
            axt.arrow(x=.55+s/10 , y=-sfwf_mean['PE'] , dx=0, dy=-sfwf_trend['PE'] )
            # axt.arrow(x=.57+s/10, y=0                 , dx=0, dy=-sfwf_trend['P']  )
            # axt.arrow(x=.53+s/10, y=0                 , dx=0, dy=-sfwf_trend['E']  )
            #endregion

            #region: diffusion [axt]
            diffusion = ddtS_ctl*fac - sfwf_mean['SFWF'] - conv['to']
            bdi = axt.bar(x=1+s/10, height=diffusion, **vbkw, color=plt.cm.tab20(10+s))
            #endregion
        #endregion

    #region: legend, numbering, scales


    # ax = f.add_axes([10.5/16,.2+[0.01,.06][q],1/8,.55])
    # ax.axis('off')
    # l = ax.legend(handles=[bsf, br, bpe, bdt, bdi, bto, bov, baz, bed, bi], ncol=2, fontsize=7, loc='center', labelspacing=.5, columnspacing=.7, handlelength=1, handletextpad=.5, frameon=True)
    # l.set_zorder(0)
    f.text(.01,.93,'(b)')
    # draw_region_labels(f,names=False)  

    for i in range(3):
        ax = f.add_axes([.75/12,[.625,.325,.075][i],.02,[.25,.3,.25][i]])
        ax.axis("off")
        t = ['surface\nfluxes','meridional\ntransports','transport\nconvergences'][i]+'\n'
        curlyBrace(f, ax, p1=(0,0), p2=(0,1), k_r=.1, str_text=t, color='k', linewidth=.5, fontdict={'fontsize':7})
    #endregion

    plt.savefig(f'{path_results}/Mov/{["FW","SALT"][q]}_region_budget.eps')
    plt.savefig(f'{path_results}/FW-paper/{["FW","SALT"][q]}_region_budget.eps')
    if q==0 and empty==False:
        plt.savefig(f'{path_results}/FW-paper/Fig9b.eps')
    return


def FW_trend_plot(quant, empty=False):  # Fig. 9c
    """ plots SFWF, merid. transport and its convergence for specific regions
    quant .. Fw or SALT
    """
    assert quant in ['FW', 'SALT']
    if quant=='FW':      q, Q, Q1 = 0, 'F', 'W'
    elif quant=='SALT':  q, Q, Q1 = 1, 'S', 'S'
    lat_bounds = ['90N', '60N', '45N', '10N', '10S', '34S']

    f = plt.figure(figsize=(6.4,2.5))
    hap = dict(x=0, dy=0, width=.04, length_includes_head=True, head_width=.06, head_length=.02, ec=None, lw=0, clip_on=False) # horizontal arrow properties
    vap = dict(y=0, dx=0, width=.06, length_includes_head=True, head_width=.08, head_length=.03, ec=None, lw=0, clip_on=False) # vertical arrow properties

    for i, (latS, latN) in enumerate(lat_bands):
        #region: define axes
        if i==0:  # Atlantic 34S-60N
            ax = f.add_axes([13.5/16,.2,1/8,.55])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

            axb = f.add_axes([13.5/16,-.05,1/8 ,.5 ])
            axt = f.add_axes([13.5/16,.5  ,1/8 ,.5 ])
            axl = f.add_axes([12  /16,.2  ,3/16,.55])
            axr = f.add_axes([14  /16,.2  ,3/16,.55])
            vert_lim = [(-.3,.3),(-8e7,8e7)][q]
            hor_lim = [(-.2,.2),(-3e8,3e8)][q]
            draw_scales(f=f,left=False,vlim=vert_lim[1],hlim=hor_lim[1])
            draw_labels(f,'Sv/century')

        else:
            ax = f.add_axes([(2*i+1)/16-1.5/16,.2,1/8,.55])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

            axb = f.add_axes([(2*i+1)/16-1.5/16,-.05,1/8 ,.5 ])
            axt = f.add_axes([(2*i+1)/16-1.5/16,.5  ,1/8 ,.5 ])
            axl = f.add_axes([(2*i+1)/16-3/16  ,.2  ,3/16,.55])
            axr = f.add_axes([(2*i+1)/16-1/16  ,.2  ,3/16,.55])

            vert_lim = [(-.25,.25),(-2e7,2e7)][q]
            hor_lim = [(-.2,.2),(-.7e8,.7e8)][q]
            if i==1:
                draw_scales(f=f,left=True,vlim=vert_lim[1],hlim=hor_lim[1])
        for ax in [axl, axr]:
            ax.set_xlim(hor_lim)
            ax.set_ylim((-.35,1.3))

        for ax in [axb, axt, axl, axr]:
            # ax.patch.set_alpha(0)
            ax.axis('off')
            ax.xaxis.set_ticks([])

        for ax in [axt, axb]:
            ax.set_ylim(vert_lim)
            ax.set_xlim((-.25,1.45))

        #endregion
        #region:  plotting
        for s, sim in enumerate(['HIGH', 'LOW']):
            if empty:  continue
            nlat_ = nlat_at(sim, latS)
            run, rcp = ['ctrl','lpd'][s], ['rcp', 'lr1'][s]

            #region: d/dt  [axb]
            if quant=='SALT':  fac = 1
            elif quant=='FW':  fac = -1e-6/S0
            ddtS_ctl, ddtS_rcp = get_ddt_SALT(sim=sim, latS=latS, latN=latN)
            label = [r'$\Delta \bar{W}$', r'$\Delta \bar{S}$'][q]
            axt.arrow(x=.8+s/10, dy=ddtS_rcp*fac, **vap, color=plt.cm.tab20(18+s))
            #endregion

            #region: meridional flux (convergences) means/trends, incl. BS/Med  [axl, axr, axb]
            inflow, inflow_trend = get_BS_Med(sim)
            if latS==60:  # BS in Arctic
                fluxes, conv = get_fluxes(sim=sim, quant=quant, lat=latS, latS=None, latN=None)
                # conv['ov'], conv['az'], conv['ed'] = 0, 0, 0
                # conv['to'] = fluxes['to'] + inflow[f'{Q}_BS'].values/[1e6,1][q]  # [m^3/s] -> [Sv]
                conv['tov'], conv['taz'], conv['ted'], conv['tto'] = 0, 0, 0, fluxes['tto']
                
                axr.set_xlim(hor_lim)
                axr.arrow(y=.425+s/10, dx=-inflow_trend[f'{Q}_BS'].values/[1e6,1][q], color=plt.cm.tab20(16+s), **hap)

            else:
                fluxes, conv = get_fluxes(sim=sim, quant=quant, lat=latS, latS=latS, latN=latN)
                if latS<35 and latN>35:  # Med
                    axb.arrow(x=1.05+s/10, dy=-inflow_trend[f'{Q}_Med'].values/[1e6,1][q], color=plt.cm.tab20(16+s), **vap)

            axl.arrow(y=.55+s/10, dx=fluxes['tov'], color=plt.cm.tab20(0+s), **hap)
            axl.arrow(y=.3 +s/10, dx=fluxes['taz'], color=plt.cm.tab20(2+s), **hap)
            axl.arrow(y=.05+s/10, dx=fluxes['ted'], color=plt.cm.tab20(4+s), **hap)
            axl.arrow(y=.8 +s/10, dx=fluxes['tto'], color=plt.cm.tab20(6+s), **hap)
            
            if i==0:  # draw 60 North values in Atlantic box
                fluxes_, conv_ = get_fluxes(sim=sim, quant=quant, lat=latN, latS=latS, latN=latN)
                axr.arrow(y=.55+s/10, dx=fluxes_['tov'], color=plt.cm.tab20(0+s), **hap)
                axr.arrow(y=.3 +s/10, dx=fluxes_['taz'], color=plt.cm.tab20(2+s), **hap)
                axr.arrow(y=.05+s/10, dx=fluxes_['ted'], color=plt.cm.tab20(4+s), **hap)
                axr.arrow(y=.8 +s/10, dx=fluxes_['tto'], color=plt.cm.tab20(6+s), **hap)
        
            axb.arrow(x=.3 +s/10, dy=conv['tov'], color=plt.cm.tab20(0+s), **vap)
            axb.arrow(x=.55+s/10, dy=conv['taz'], color=plt.cm.tab20(2+s), **vap)
            axb.arrow(x=.8 +s/10, dy=conv['ted'], color=plt.cm.tab20(4+s), **vap)
            axb.arrow(x=.05+s/10, dy=conv['tto'], color=plt.cm.tab20(6+s), **vap)
            #endregion

            #region: surface fluxes [axt]
            sfwf_mean, sfwf_trend = get_SFWF(sim=sim, quant=quant, latS=latS, latN=latN)
            axt.arrow(x=.05+s/10, dy=-sfwf_trend['SFWF'], color=plt.cm.tab20(8+s)  , **vap)
            axt.arrow(x=.3 +s/10, dy=-sfwf_trend['R']   , color=plt.cm.tab20b(14+s), **vap)
            axt.arrow(x=.55+s/10, dy=-sfwf_trend['PE']  , color=plt.cm.tab20(12+s) , **vap)
            #endregion

        #endregion

    #region: legend, numbering, scales
    f.text(.01,.93,'(c)')
    #endregion
    if q==0 and empty==False:
        plt.savefig(f'{path_results}/FW-paper/Fig9c.eps')
    return


def FW_merid_fluxes_plot():  # Fig. 7
    """ creates Fig. 7 """
    f, ax = plt.subplots(2,2, figsize=(6.4,5), sharey='row', sharex='col', gridspec_kw={'height_ratios':[2,1]})
    ax[0,0].set_ylabel(r'northward freshwater transport $F$  [Sv]')
    ax[1,0].set_ylabel(r'$F$ linear trend  [Sv/100yr]')
    ov, az, eddy, tot, BS = r'$_{ov}$', r'$_{az}$', r'$_{eddy}$', r'$_{tot}$', r'$_{BS}$'
    for i, sim in enumerate(['HIGH','LOW']):
        ctl = ['ctrl', 'lpd'][i]
        rcp = ['rcp', 'lr1'][i]
        inflow, inflow_trend = get_BS_Med(sim)
        SALT_Arctic_mean, SALT_Arctic_trend = get_SFWF(sim=sim, quant='SALT', latS=60, latN=90)
        FW_Arctic_mean, FW_Arctic_trend = get_SFWF(sim=sim, quant='FW', latS=60, latN=90)
        section = [sections_high,sections_low][i]
        yrs = [slice(200,203), slice(500-154,530-154)][i]
        ax[0,i].set_title(['HR-CESM','LR-CESM'][i])
        for j in range(2):
            ax[j,i].axhline(0, c='k', lw=.7)#, zorder=1)
            ax[j,i].axvline(0, c='k', lw=.7)#, zorder=1)``
            ax[j,i].set_xlim((-34,60))
            ax[j,i].set_xticks([-34,-10,0,10,45,60])
            ax[j,i].grid(axis='x', lw=.5)
        lats = Atl_lats(sim)
        dso = xr.open_dataset(f'{path_prace}/Mov/FW_SALT_fluxes_{ctl}.nc', decode_times=False).isel(time=yrs)
        dst = xr.open_dataset(f'{path_prace}/Mov/FW_SALT_fluxes_{rcp}.nc', decode_times=False)
        
        Fov = dso.Fov.mean('time')
        Faz = dso.Faz.mean('time')
        Fe  = -dso.Se .mean('time')/35e6  # [kg/s] -> [Sv]
        Ft  = Fov+Faz+Fe
        
        # CTRL
        ax[0,i].plot(lats, Fov               , c='C0', label=r'$F_{ov}$')
        ax[0,i].plot(lats, Faz               , c='C1', label=r'$F_{az}$')
        
        ax[0,i].plot(*mask_Med(lats, Fe, section), c='C2', label=r'$F_{eddy}$')
        ax[0,i].plot(*mask_Med(lats, Ft, section), c='C3', label=r'$F_{total}$')

        # BS inflow + Arctic SFWF
        ax[0,i].plot([62,62], [0,-inflow[f'F_BS']/1e6],    lw=2, c='C8', label=r'$F_{BS}$'         , clip_on=False)
        ax[0,i].plot([61,61], [0,-FW_Arctic_mean['SFWF']], lw=2, c='C4', label=r'$F_{surf,Arctic}$', clip_on=False)
        ax[1,i].plot([62,62], [0,-inflow_trend[f'F_BS']/1e6], lw=2, c='C8', label=r'$F_{BS}$'         , clip_on=False)
        ax[1,i].plot([61,61], [0,-FW_Arctic_trend['SFWF']]  , lw=2, c='C4', label=r'$F_{surf,Arctic}$', clip_on=False)

        # RCP
        rn = {'dim_0':'nlat_u'}
        Fov_trend = xr_linear_trend(dst.Fov).rename(rn)*365*100
        Faz_trend = xr_linear_trend(dst.Faz).rename(rn)*365*100
        Fe_trend  = xr_linear_trend(-dst.Se/35e6).rename(rn)*365*100
        Ft_trend  = xr_linear_trend(dst.Fov+dst.Faz-dst.Se/35e6).rename(rn)*365*100
        
        rn = {'dim_0':'nlat_u'}
        Fov_stats = xr_regression_with_stats(dst.Fov)
        Faz_stats = xr_regression_with_stats(dst.Faz)
        Fe_stats  = xr_regression_with_stats(-dst.Se/35e6)
        Ft_stats  = xr_regression_with_stats(dst.Fov+dst.Faz-dst.Se/35e6)
        
        Fov_trend = Fov_stats.slope*365*100
        Faz_trend = Faz_stats.slope*365*100
        Fe_trend  = Fe_stats .slope*365*100
        Ft_trend  = Ft_stats .slope*365*100
        
        ax[0,i].plot(lats, Fov + Fov_trend                  , c='C0', ls='--', lw=.8)
        ax[0,i].plot(lats, Faz + Faz_trend                  , c='C1', ls='--', lw=.8)
        ax[0,i].plot(*mask_Med(lats, Fe + Fe_trend, section), c='C2', ls='--', lw=.8)
        ax[0,i].plot(*mask_Med(lats, Ft + Ft_trend, section), c='C3', ls='--', lw=.8)

        ax[1,i].plot(lats, Fov_trend                   , c='C0', lw=.5)
        ax[1,i].plot(lats, Faz_trend                   , c='C1', lw=.5)
        ax[1,i].plot(*mask_Med(lats, Fe_trend, section), c='C2', lw=.5)
        ax[1,i].plot(*mask_Med(lats, Ft_trend, section), c='C3', lw=.5)
        
        ax[1,i].plot(lats, Fov_trend.where(Fov_stats.p_value<0.05)        , c='C0', lw=1)
        ax[1,i].plot(lats, Faz_trend.where(Faz_stats.p_value<0.05)        , c='C1', lw=1)
        ax[1,i].plot(*mask_Med(lats, Fe_trend, section, mask2=xr.where(Fe_stats.p_value<0.05,1,0).values), c='C2', lw=1)
        ax[1,i].plot(*mask_Med(lats, Ft_trend, section, mask2=xr.where(Ft_stats.p_value<0.05,1,0).values), c='C3', lw=1)

        ax[1,i].set_xlabel(r'latitude $\theta$  [$\!^\circ\!$N]')
        ax[0,i].set_ylim((-.85,.65))
        ax[1,i].set_ylim((-.27,.12))
        ax[0,i].text(.01, .94, '('+['a','b'][i]+')', transform=ax[0,i].transAxes)
        ax[1,i].text(.01, .88, '('+['c','d'][i]+')', transform=ax[1,i].transAxes)
    l1 = ax[0,0].legend(loc=3, fontsize=8, columnspacing=.7, handlelength=1, frameon=False) #ncol=2
    ax[0,0].add_artist(l1)

    # legends CTRL + RCP
    l1, = ax[0,0].plot([],[], c='k', label=f'CTRL mean')#{[200,500][i]}-{[200,500][i]+29}')
    l2, = ax[0,0].plot([],[], c='k', ls='--', lw=.8, label=f'RCP 2100')
    ax[0,0].legend(handles=[l1,l2], fontsize=8, loc=1, columnspacing=.7, handlelength=1.3, frameon=False)
    # plt.savefig(f'{path_results}/Mov/FW_transports.eps', dpi=300)
    # plt.savefig(f'{path_results}/FW-paper/FW_transports.eps')
    plt.savefig(f'{path_results}/FW-paper/Fig7.eps')
    return


def FS_merid_fluxes_plot():
    """"""
    f, ax = plt.subplots(1,2, figsize=(6.4,3), sharey='row')
    ax[0].set_ylabel('northward S transport [kt/s]')
    ov, az, eddy, tot, BS = r'$_{ov}$', r'$_{az}$', r'$_{eddy}$', r'$_{tot}$', r'$_{BS}$'
    for i, sim in enumerate(['HIGH','LOW']):
        ctl = ['ctrl', 'lpd'][i]
        rcp = ['rcp', 'lr1'][i]
        inflow, inflow_trend = get_BS_Med(sim)
        SALT_Arctic_mean, SALT_Arctic_trend = get_SFWF(sim=sim, quant='SALT', latS=60, latN=90)
        mask_Med = make_Med_mask([2400, 384][i], [sections_high,sections_low][i])
        yrs = [slice(200,203), slice(500-154,530-154)][i]
        ax[i].set_title(['HR-CESM','LR-CESM'][i])
        ax[i].axhline(0, c='k', lw=.7)#, zorder=1)
        ax[i].axvline(0, c='k', lw=.7)#, zorder=1)
        ax[i].set_xlim((-34,60))
        ax[i].set_xticks([-34,-10,0,10,45,60])
        ax[i].grid()
        lats = Atl_lats(sim)
        dso = xr.open_dataset(f'{path_prace}/Mov/FW_SALT_fluxes_{ctl}.nc', decode_times=False).isel(time=yrs)
        dst = xr.open_dataset(f'{path_prace}/Mov/FW_SALT_fluxes_{rcp}.nc', decode_times=False)
        
        Sov = dso.Sov.mean('time')/1e6  # [kg/s] -> [kt/s]
        Saz = dso.Saz.mean('time')/1e6
        Se  = dso.Se .mean('time')/1e6
        St  = dso.St .mean('time')/1e6
        
        # CTRL        
        ax[i].plot(lats, Sov               , c='C0', label=r'$F^S_{ov}$')
        ax[i].plot(lats, Saz               , c='C1', label=r'$F^S_{az}$')
        ax[i].plot(lats, Se.where(mask_Med), c='C2', label=r'$F^S_{eddy}$')
        ax[i].plot(lats, St.where(mask_Med), c='C3', label=r'$F^S_{total}$')

        # BS inflow + Arctic SFWF
        ax[i].plot([62,62], [0,-inflow[f'S_BS']/1e6]         , lw=2, c='C8', label=r'$F^S_{BS}$', clip_on=False)
        ax[i].plot([61,61], [0,-SALT_Arctic_mean['SFWF']/1e6], lw=2, c='C4', label=r'$F^S_{surf,Arctic}$', clip_on=False)        
        # RCP
        rn = {'dim_0':'nlat_u'}        
        Sov_trend = Sov + xr_linear_trend(dst.Sov).rename(rn)*365*100/1e6  # [kg/s] -> [kt/s/100yr]
        Saz_trend = Saz + xr_linear_trend(dst.Saz).rename(rn)*365*100/1e6
        Se_trend  = Se  + xr_linear_trend(dst.Se ).rename(rn)*365*100/1e6
        St_trend  = St  + xr_linear_trend(dst.St ).rename(rn)*365*100/1e6
        
        ax[i].plot(lats, Sov_trend               , c='C0', ls='--', lw=.8)
        ax[i].plot(lats, Saz_trend               , c='C1', ls='--', lw=.8)
        ax[i].plot(lats, Se_trend.where(mask_Med), c='C2', ls='--', lw=.8)
        ax[i].plot(lats, St_trend.where(mask_Med), c='C3', ls='--', lw=.8)
        
        ax[i].set_xlabel(r'latitude $\theta$ [$\!^\circ\!$N]')
        ax[i].set_ylim((-92,33))
        ax[i].text(.01, .92, '('+['a','b'][i]+')', transform=ax[i].transAxes)
        l1 = ax[i].legend(ncol=3, columnspacing=.7, loc=4, fontsize=7, handlelength=.8)
        ax[i].add_artist(l1)
        
            # legends CTRL + RCP
        l, = ax[i].plot([],[], c='k', label=f'CTRL mean')#{[200,500][i]}-{[200,500][i]+29}')
        L = [l]
        l, = ax[i].plot([],[], c='k', ls='--', lw=.8, label=f'RCP year 2100')
        L.append(l)
        ax[i].legend(handles=L, fontsize=7, ncol=2, loc=1)
    plt.savefig(f'{path_results}/Mov/FS_transports', dpi=300)
    plt.savefig(f'{path_results}/FW-paper/FS_transports.eps')
    return
