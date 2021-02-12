import cartopy
import matplotlib
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

matplotlib.rc_file('rc_file_paper')

cPC = ccrs.PlateCarree()

def bias_maps(do, dh, dl, kw1, kw2=None, fn=None):
    """"""
    f = plt.figure(figsize=(6.4,3.8), constrained_layout=False)
    axo = f.add_axes([.01 ,.49,.485,.49], projection=ccrs.PlateCarree(central_longitude=-60))
    axh = f.add_axes([.01 ,  0,.485,.49], projection=ccrs.PlateCarree(central_longitude=-60))
    axl = f.add_axes([.505,  0,.485,.49], projection=ccrs.PlateCarree(central_longitude=-60))

    if kw2 is None:  # one colorbar only
        ncbars = 1
        axc = f.add_axes([.515,.735,.465,.04])
        if 'log' in kw1:
            if kw1['log']==True:
                kwo = kwc = dict(norm=matplotlib.colors.LogNorm(vmin=kw1['vmin'], vmax=kw1['vmax']), cmap=kw1['cmap'], transform=cPC)
        else:
            kwo = kwc = dict(vmin=kw1['vmin'], vmax=kw1['vmax'], cmap=kw1['cmap'], transform=cPC)
        kw2 = dict(lat='TLAT', lon='TLONG')

    else:  # two colorbars
        ncbars = 2
        axc1 = f.add_axes([.51,.9,.475,.04])
        axc2 = f.add_axes([.51,.695,.475,.04])
        kwo = dict(vmin=kw1['vmin'], vmax=kw1['vmax'], cmap=kw1['cmap'], transform=cPC)
        kwc = dict(vmin=kw2['vmin'], vmax=kw2['vmax'], cmap=kw2['cmap'], transform=cPC)
        if 'lat' not in kw2:
            kw2['lat'] = 'TLAT'
        if 'lon' not in kw2:
            kw2['lon'] = 'TLONG'

    imo = axo.pcolormesh(do[kw1['lon']], do[kw1['lat']], do, **kwo)
    imh = axh.pcolormesh(dh[kw2['lon']], dh[kw2['lat']], dh, **kwc)
    iml = axl.pcolormesh(dl[kw2['lon']], dl[kw2['lat']], dl, **kwc)


    if ncbars==1:
        cbar = plt.colorbar(imo, cax=axc, orientation='horizontal', label=kw1['label'])
        if 'ticks' in kw1:
            ticks = kw1['ticks']
            cbar.ax.minorticks_off()
            cbar.ax.get_xaxis().set_ticks(ticks)
            cbar.ax.get_xaxis().set_ticklabels(ticks)
    elif ncbars==2:
        cbar1 = plt.colorbar(imo, cax=axc1, orientation='horizontal', label=kw1['label'])
        cbar2 = plt.colorbar(imh, cax=axc2, orientation='horizontal', label=kw2['label'])

    for i, ax in enumerate([axo, axh, axl]):
        ax.add_feature(cartopy.feature.LAND, zorder=2, edgecolor='black', facecolor='grey', linewidth=.5)
        ax.text(.01,.91, '('+['a','b','c'][i]+')', transform=ax.transAxes, color='k',
                bbox=dict(ec='None', fc='w', alpha=0.5, pad=0.0))
        gl = ax.gridlines(crs=cPC, draw_labels=False, linewidth=.5)
        gl.ylocator = matplotlib.ticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
        ax.set_global()
        ax.set_title(['OBS','HR-CESM','LR-CESM'][i])

    if type(fn)==str:
        plt.savefig(fn, dpi=600)
    return