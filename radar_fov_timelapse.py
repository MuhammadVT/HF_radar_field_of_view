# load packages
from davitpy.pydarn.radar import network, radar
from davitpy.utils import plotUtils 
from davitpy.pydarn.plotting import overlayRadar, overlayFov 
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import pandas as pd
import numpy as np

def draw_axes(fig_size=None, hemi='both'):
    """Generates an empty figure suitable for the value given to hemi (hemisphere)

    Parameters
    ----------
    fig_size : tuple or None
        Figure size (height, width) in inches. Default to None. 
    hemi : str
        Stands for hemisphere. Acceptable values are "north", "south" and "both". 
        Default to "both".
    
    Returns
    -------
    Matplotlib Figure object
    
    """
    
    if hemi == 'both':
        # initial the parameters
        nrows=21; ncols=2
        hspace = 0.04; wspace = 0.04
        h_pad=1; w_pad=1

        # create a figure 
        fig = plt.figure(figsize=fig_size)
        axt = plt.subplot2grid((nrows,ncols), (0,0), colspan=ncols)
        axn = plt.subplot2grid((nrows,ncols), (1,0), colspan=1, rowspan=nrows-2)
        axs = plt.subplot2grid((nrows,ncols), (1,1), colspan=1, rowspan=nrows-2)
        axb = plt.subplot2grid((nrows,ncols), (nrows-1,0), colspan=ncols, rowspan=1)
        axes = np.array([axt, axn, axs, axb])
        plt.subplots_adjust(hspace=hspace, wspace=wspace)

    if hemi == 'north' or hemi == 'south':
        # initial the parameters
        nrows=21; ncols=1
        hspace = 0.04; wspace = 0.04
        h_pad=1; w_pad=1

        # create a figure 
        fig = plt.figure(figsize=fig_size)
        axt = plt.subplot2grid((nrows,ncols), (0,0), colspan=ncols)
        axm = plt.subplot2grid((nrows,ncols), (1,0), colspan=1, rowspan=nrows-2)
        axb = plt.subplot2grid((nrows,ncols), (nrows-1,0), colspan=ncols, rowspan=1)
        axes = np.array([axt, axm, axb])

    return fig, axes

def get_rad_by_year(stime, etime):
    """Fetch radars available within a perid between stime and etime.

    Parameters 
    ----------
    stime : datetime.datetime
        Start time
    etime : datetime.datetime
        End time

    Returns
    -------
    Pandas DataFrame
    
    """
    
    # initialize parameters
    codes_north = []
    nbeams_north = []
    stimes_north = []
    codes_south = []
    nbeams_south = []
    stimes_south = []

    # get all radars
    rads = network().radars

    # loop through the radars
    for rad in rads:
        rad_stime = rad.stTime
        rad_etime = rad.edTime
        code = rad.code
        code = str(code[0])

        # select radars that exsit between stime and etime
        if (rad_stime >= stime) and (rad_stime <= etime):

            # select only the active radars
            if (rad.status==1):

                # radars in northern hemisphere
                if rad.sites[0].geolat > 0:
                    codes_north.append(code)
                    nbeams_north.append(rad.sites[0].maxbeam)
                    stimes_north.append(rad_stime)

                # radars in southern hemisphere
                else:
                    codes_south.append(code)
                    nbeams_south.append(rad.sites[0].maxbeam)
                    stimes_south.append(rad_stime)

    # All of these radars will be in orange (midlatitude radars).
    rads_midlat_north = ['hok', 'hkw','adw','ade','cvw','cve','fhw','fhe','bks','wal']    
    rads_midlat_south = ['tig','unw', 'bpk']

    # These are the high-latitude superdarn radars plotted in blue.
    rads_highlat_north  = ['ksr','kod','pgr','sas','kap','gbr','pyk','han','sto']
    rads_highlat_south = ['ker','sye','sys','san','hal','sps','dce','zho']

    # These are the polar darn radars plotted in green.
    rads_polar_north = ['inv','rkn','cly', 'lyr'] 
    rads_polar_south = ['mcm']

    rads_midlat = rads_midlat_north + rads_midlat_south
    rads_highlat = rads_highlat_north + rads_highlat_south
    rads_polar = rads_polar_north + rads_polar_south

    # contruct pandas DataFrames
    # for northern hemisphere
    dfn = pd.DataFrame(index = stimes_north,
                       data=zip(codes_north, nbeams_north, ['N']*len(stimes_north),
                       ), columns=['code', 'nbeams', 'hemi'])

    # for southern hemisphere
    dfs = pd.DataFrame(index = stimes_south,
                       data=zip(codes_south, nbeams_south, ['S']*len(stimes_south),
                       ), columns=['code', 'nbeams', 'hemi'])

    # join the above two DataFrames
    df = dfn.append(dfs)
    if not df.empty:
        df.loc[:, 'region'] = np.nan
        codes_tmp = df.code.tolist()
        for cod in codes_tmp:
            if cod in rads_midlat:
                df.loc[df.code==cod, 'region'] = 'mid'
            if cod in rads_highlat:
                df.loc[df.code==cod, 'region'] = 'high'
            if cod in rads_polar:
                df.loc[df.code==cod, 'region'] = 'polar'

    return df 

def plot_fovs(stime, etime, stm2=None, etm=dt.datetime(2016, 1, 31),
              hemi='both', coords='mag', fovAlpha=0.7, 
              fpath='./full_time_lapse'):
    """
    Parameters
    ----------
    stime : datetime.datetime
    etime : datetime.datetime
    stm2 : datetime.datetime
    etm : datetime.datetime
    hemi : str
        Hemisphere(s). Valid inputs are "north", "south", "both"
    coords : str
        Coordinates used in the plot. Valid inputs are "mag", "geo", "mlt"
    fovAlpha : float
        Transparency of the radar fields of view.
        Valid inputs are between 0.0 to 1.0.
    fpath : str
        File path where the output plots will be saved
    """

    if hemi == 'both':
        fig, axes = draw_axes(fig_size=(15, 9), hemi=hemi)
        axt , axn, axs, axb = axes
    if hemi == 'north':
        fig, axes = draw_axes(fig_size=(11, 13), hemi=hemi)
        axt , axn, axb = axes
    if hemi == 'south':
        fig, axes = draw_axes(fig_size=(11, 13), hemi=hemi)
        axt , axs, axb = axes

    # plot the time lapse axes
    stime_tmp = stime
    if stm2 is not None:
        stime = stm2
    del_w = (etm.year - stime.year) + (etm.month - stime.month) / 12.0
    del_b = (etime.year - stime.year) + (etime.month - stime.month) / 12.0
    axt.add_patch(Rectangle((stime.year, 0), stime.year + del_w, 1, facecolor='white'))
    axt.add_patch(Rectangle((stime.year, 0), del_b, 1, facecolor='black', alpha=fovAlpha))
    #axt.set_xlabel(etime.strftime('%Y/%b'), fontsize=15, fontweight='bold', labelpad=5)
    axt.set_xlabel(etime.strftime('%Y / %b'), fontsize=17, labelpad=7)
    xxmin = stime.year+(stime.month-1)/12.
    xxmax = etm.year+(etm.month-1)/12.
    #xxmin = stime.year
    #xxmax = etm.year
    axt.set_xlim([xxmin, xxmax])
    axt.set_ylim([0, 1])
    axt.xaxis.set_tick_params(direction='in')
    axt.xaxis.set_ticks_position('top')
    axt.get_yaxis().set_visible(False)

    # set the xticks
    if (stime.year % 10) < 5:
        xstrt = stime.year + (5-(stime.year % 10))
    if (stime.year % 10) == 5 or (stime.year % 10) == 0:
        xstrt = stime.year
    if (stime.year % 10) > 5:
        xstrt = stime.year + (10-(stime.year % 10))

    if (etm.year % 10) < 5:
        xend = etm.year - (etm.year % 10)
    if (etm.year % 10) == 5 or (etm.year % 10) == 0:
        xend = etm.year
    if (etm.year % 10) > 5:
        xend = etm.year + (5-(etm.year % 10))
    xint = range(xstrt, xend+5, 5)
    axt.xaxis.set_ticks([])
    #axt.get_xaxis().get_major_formatter().set_scientific(False)
    axt.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axt.xaxis.set_ticks(xint)

    stime = stime_tmp

    # get radar three letter codes
    df = get_rad_by_year(stime, etime)
    df.dropna(inplace=True)
    if not df.empty:
        rads_north = df.loc[df.hemi=='N', :].code.tolist()
        rads_mid_north = df.loc[np.logical_and(df.hemi=='N', df.region=='mid')].code.tolist()
        rads_high_north = df.loc[np.logical_and(df.hemi=='N', df.region=='high')].code.tolist()
        rads_polar_north = df.loc[np.logical_and(df.hemi=='N', df.region=='polar')].code.tolist()
        rads_south = df.loc[df.hemi=='S', :].code.tolist()
        rads_mid_south = df.loc[np.logical_and(df.hemi=='S', df.region=='mid')].code.tolist()
        rads_high_south = df.loc[np.logical_and(df.hemi=='S', df.region=='high')].code.tolist()
        rads_polar_south = df.loc[np.logical_and(df.hemi=='S', df.region=='polar')].code.tolist()

    # colors
    midlat_color = 'darkorange' 
    highlat_color = 'dodgerblue'
    polar_color = 'springgreen' 
    msize = 15
    fsize = 25
    
    # overlay radar names and fovs
    # northern hemisphere
    if hemi == 'both' or hemi == 'north':
        plt.sca(axn)
        m1 = plotUtils.mapObj(boundinglat=30., gridLabels=True, coords=coords)
        m1.ax = axn
        if not df.empty:
            overlayRadar(m1, codes=rads_north, markerSize=msize, fontSize=fsize)
            # overlay mid-latitude radars.
            overlayFov(m1, codes=rads_mid_north, maxGate=75, fovColor=midlat_color, fovAlpha=fovAlpha)
            # overlay high-latitude radars.
            overlayFov(m1, codes=rads_high_north, maxGate=75, fovColor=highlat_color, fovAlpha=fovAlpha)
            # overlay polar radars
            overlayFov(m1, codes=rads_polar_north, maxGate=75, fovColor=polar_color, fovAlpha=fovAlpha)

        font_size = 5
        axn.xaxis.set_tick_params(labelsize=font_size)
        axn.yaxis.set_tick_params(labelsize=font_size)

    # southern hemisphere
    if hemi == 'both' or hemi == 'south':
        plt.sca(axs)
        m2 = plotUtils.mapObj(boundinglat=-30., gridLabels=True, coords=coords)
        m2.ax = axs
        if not df.empty:
            overlayRadar(m2, codes=rads_south, markerSize=msize, fontSize=fsize)

            # overlay midlatitude radars
            overlayFov(m2, codes=rads_mid_south, maxGate=75, fovColor=midlat_color, fovAlpha=fovAlpha)

            # overlay high-latitude radars
            overlayFov(m2, codes=rads_high_south, maxGate=75, fovColor=highlat_color, fovAlpha=fovAlpha)

            # overlay polar radars
            overlayFov(m2, codes=rads_polar_south, maxGate=75, fovColor=polar_color, fovAlpha=fovAlpha)

        font_size = 5
        axs.xaxis.set_tick_params(labelsize=font_size)
        axs.yaxis.set_tick_params(labelsize=font_size)


    # plot the legend axes
    strt = 0.1; dlw = 0.3; ww = 0.07; hh = 0.2
    axb.add_patch(Rectangle((strt, 0), ww, hh, facecolor=polar_color, alpha=fovAlpha))
    axb.add_patch(Rectangle((strt + dlw, 0), ww, hh, facecolor=highlat_color, alpha=fovAlpha))
    axb.add_patch(Rectangle((strt + 2*dlw, 0), ww, hh, facecolor=midlat_color, alpha=fovAlpha))
    axb.annotate('Polar Cap', xy=(strt+ww+0.02, hh/2.0), ha='left', va='center')
    axb.annotate('High-Latitude', xy=(strt+dlw+ww+0.02, hh/2.0), ha='left', va='center')
    axb.annotate('Mid-Latitude', xy=(strt+2*dlw+ww+0.02, hh/2.0), ha='left', va='center')
    axb.set_xlim([0, 1])
    axb.set_ylim([0, 0.2])
    axb.axis('off')
    #axb.get_yaxis().set_visible(False)
    #axb.get_xaxis().set_visible(False)

    # save the plot
    dpi = 200
    fig.savefig(fpath+'.png', dpi=dpi)
    #fig.savefig('/home/muhammad/Dropbox/full.png', dpi=dpi)
    #fig.show()
    fig.clf()
    plt.close()

def loop_fovs(stm=dt.datetime(1983, 1, 31), etm=dt.datetime(2016, 1, 31),
              stm2=None, hemi='both', coords='mag', fovAlpha=0.7):

    """
    Parameters
    ----------
    stm : datetime.datetime
    etm : datetime.datetime
    stm2 : datetime.datetime
    hemi : str
        Hemisphere(s). Valid inputs are "north", "south", "both"
    coords : str
        Coordinates used in the plot. Valid inputs are "mag", "geo", "mlt"
    fovAlpha : float
        Transparency of the radar fields of view.
        Valid inputs are between 0.0 to 1.0.
    """

     
    if stm2 is None:
        dts = pd.date_range(start=stm, end=etm, freq='6m')
    else:
        dts = pd.date_range(start=stm2, end=etm, freq='6m')
    if hemi == 'both':
        #fpath = './full_time_lapse/'
        #fpath = './time_lapse/'
        fpath = './tmp/'
    if hemi == 'north':
        fpath = './north_time_lapse/'
    if hemi == 'south':
        fpath = './south_time_lapse/'

    for dti in dts:
        plot_fovs(stm, dti, stm2=stm2, etm=etm, hemi=hemi, coords=coords,
                fovAlpha=fovAlpha,
                fpath=fpath + dti.strftime('%Y%b'))
                #fpath='/home/muhammad/Dropbox/full_time_lapse/' + dti.strftime('%Y%b'))

stm = dt.datetime(1983, 1, 31)
stm2 = dt.datetime(2003, 1, 31)   # to start the plotting time from 2003
#stm2 = None
etm = dt.datetime(2016, 1, 31)
#coords='mlt'
coords='mag'
hemi = 'both'
#hemi = 'north'
#hemi = 'south'
loop_fovs(stm=stm, etm=etm, hemi=hemi, stm2=stm2, coords=coords, fovAlpha=0.7)

