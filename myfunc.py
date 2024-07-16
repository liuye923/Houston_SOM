import xarray as xr
import pandas as pd
import numpy as np
#import sompy
import dask

import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
#from geocat.viz import cmaps as gvcmaps
import cmaps
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
#from geocat.viz import util as gvutil

#import global_variable as gv
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

class MyCalc(object):
    '''statistical functions'''
    def __init__(self):
        pass

    def two_sample_diff(self, xda1, xda2, dim='time', **kwargs):
        from scipy.stats import ttest_ind
        if isinstance(xda1, xr.Dataset): xda1 = xda1.to_array()
        if isinstance(xda2, xr.Dataset): xda2 = xda2.to_array()
#        da1, da2 = xr.broadcast(xda1, xda2)
        da1, da2 = (xda1, xda2)
        dim_idx = da1.dims.index(dim)
        t, p = ttest_ind(da1.data, da2.data, axis=dim_idx)
        dims   = [k for k in da1.dims if k != dim]
        coords = {k: v for k, v in da1.coords.items() if k != dim}
        p    = xr.DataArray(p, dims=dims, coords=coords)
        diff = da1.mean(dim) - da2.mean(dim)
        return diff, p, da1.mean(dim), da2.mean(dim)

    def calc_distribution(self, data, groupby, bins):
        groups = data.groupby(pd.cut(groupby, bins))
        return groups


############################ Functions for this project ########################
### Statistic functions ###
class MyFunc(MyCalc):
    def __init__(self):
        pass

    def var_register(self, var):
        var_dict = dict(z=dict(scale=1/9.8/10., offset=0., unit=r"dm"),
                        q=dict(scale=1000., offset=0., unit=r"$k\ kg{-1}$"),
                        u=dict(scale=1., offset=0., unit=r"$m\ s{-1}$"),
                        v=dict(scale=1., offset=0., unit=r"$m\ s{-1}$"),
                        w=dict(scale=1., offset=0., unit=r"$Pa\ s{-1}$"),
                        cape=dict(scale=1., offset=0., unit=r"$J\ kg{-1}$"),
                        cin=dict(scale=-1., offset=0., unit=r"$J\ kg{-1}$"),
                        t2=dict(scale=1., offset=-273.15, unit=r"$DegC$"),
#                        cin=dict(scale=0.0152594961218324, offset=499.992649931982, unit=r"$J\ kg{-1}$"),
                       )
        return var_dict[var]

    def select_month(self, *args, month=None, season=None):
        if (month is None) & (season is None): return args
        if season is not None:
            if season=="JJA": month=(6,7,8)
            if season=="MAM": month=(3,4,5)
            if season=="SON": month=(9,10,11)
            if season=="DJF": month=(12,1,2)
        out = []
        for v in args:
#            if isinstance(month, (tuple, list)):
#                out.append(v.sel(time=((v["time.month"]>=month[0])&(v["time.month"]<=month[1]))))
#            else:
#                out.append(v.sel(time=v["time.month"]==month))
            if isinstance(month, (tuple, list)):
                out.append(v.sel(time=v["time.month"].isin(month)))
            else:
                out.append(v.sel(time=v["time.month"]==month))
#        print(out)
        if len(out) == 1: return out[0]
        return out
myfunc = MyFunc()

### TRACKS ###
class MyTrack(object):
    '''functions deal with tracks'''
    def __init__(self):
        pass

    def create_outname(self, prefix, year):
        if not isinstance(year, (tuple, list)): year = (year, )
        def join_strings(in_tuple, d="-"):
            if len(in_tuple) == 1:
                return str(in_tuple[0])
            else:
                return d.join([str(v) for v in in_tuple])
        year = join_strings(year)
        return "_".join((prefix, year))

    def load_data_idc(self, outname, year, season, extent=None):
        fname = f"data/{outname}.nc"
        tracks =  xr.open_dataset(fname)["tracks"]
        tracks = tracks.swap_dims({"tracks":"time"})
        if season != 'Ann':
            tracks = myfunc.select_month(tracks, season=season)
        tracks = tracks.swap_dims({"time":"tracks"})
        if extent is not None: tracks = self.filter_location(tracks, extent)
        return tracks

    def calc_start_location(self, multi_year_events):
        tracks = []; lat = []; lon = []; time = []
        for events in multi_year_events:
            tracks.append(events["tracks"])
            lat.append(events["meanlat"].sel(times=0))
            lon.append(events["meanlon"].sel(times=0))
            time.append(events["datetimestring"].sel(times=0))
        tracks = xr.concat(tracks, dim="tracks")
        tracks.name  = "tracks"
        lat    = xr.concat(lat, dim="tracks")
        lon    = xr.concat(lon, dim="tracks")
        time   = xr.concat(time, dim="tracks")
        from datetime import datetime
        time   = pd.to_datetime([it.decode() for it in time.data], format="%Y-%m-%d_%H:%M:")
        print(time)
        tracks.coords["lat"]    = ("tracks", lat.data)
        tracks.coords["lon"]    = ("tracks", lon.data)
        tracks.coords["time"]   = ("tracks", time.values)
        return tracks

    def filter_location(self, data, rr):
        print(f"filter--location: {rr}")
        return data.loc[(data["lon"]>=rr[0]) & (data["lon"]<=rr[1]) &
                        (data["lat"]>=rr[2]) & (data["lat"]<=rr[3])]

    def load_feature_idc(self, year):
        multi_year_event = []
        for yr in np.arange(year[0], year[1]+1):
#            fname = f"data/feature_{yr}.nc"
            fname = f"data/feature_new_{yr}.nc"
            ds = xr.open_dataset(fname).squeeze()
            ds.coords["tracks"] = ds.coords["tracks"] + yr*100000 - 1
            multi_year_event.append(ds)
        ds = xr.concat(multi_year_event, dim="tracks")
        return ds

    def calc_occurrence(self, data, extent, delta=0.5):
        data.coords["lon"] = np.floor((data["lon"] - extent[0]) / delta).astype(int)
        data.coords["lat"] = np.floor((data["lat"] - extent[2]) / delta).astype(int)
        data.coords["z"] = data.coords["lon"] * 10000 + data.coords["lat"]
        freq = data.groupby("z").count()
        print(freq)

        grid = xr.DataArray(np.nan, coords=[("lat", np.arange((extent[3]-extent[2])/delta-1).astype(int)),
                                            ("lon", np.arange((extent[1]-extent[0])/delta-1).astype(int))],
                                      dims=["lat", "lon"])
        grid = grid.stack(xy=["lon", "lat"])
        grid.coords["z"] = grid.coords["lon"] * 10000 + grid.coords["lat"]
        print(grid)

        freq = freq.interp(z=grid["z"], method="nearest", kwargs={"fill_value": 0})
        del(freq["z"])
        freq = freq.unstack()
        freq = freq.transpose("lat", "lon")
        freq["lon"] = extent[0] + freq["lon"] * delta + delta
        freq["lat"] = extent[2] + freq["lat"] * delta + delta
        return freq

    def calc_point_to_map(self, data, extent, delta=0.5, fill_value=0):
        print(data)
        data.coords["lon"] = np.floor((data["lon"] - extent[0]) / delta).astype(int)
        data.coords["lat"] = np.floor((data["lat"] - extent[2]) / delta).astype(int)
        data.coords["z"] = data.coords["lon"] * 10000 + data.coords["lat"]
        print(data)
        print("aaaaa")
        grouped = data.groupby("z").mean()
        print(grouped)

        grid = xr.DataArray(np.nan, coords=[("lat", np.arange((extent[3]-extent[2])/delta+1).astype(int)),
                                            ("lon", np.arange((extent[1]-extent[0])/delta+1).astype(int))],
                                      dims=["lat", "lon"])
        grid = grid.stack(xy=["lon", "lat"])
        grid.coords["z"] = grid.coords["lon"] * 10000 + grid.coords["lat"]
        print(grid)

        grouped = grouped.interp(z=grid["z"], method="nearest", kwargs={"fill_value": fill_value})
        del(grouped["z"])
        grouped = grouped.unstack()
        grouped = grouped.transpose("lat", "lon")
        grouped["lon"] = extent[0] + grouped["lon"] * delta
        grouped["lat"] = extent[2] + grouped["lat"] * delta
        return grouped

    def calc_track_initial_time_freq(self, track):
        count = track.groupby("time.hour").count()
        freq  = count / track.count() * 100.
        return freq

    def calc_track_initial_month_freq(self, track):
        count = track.groupby("time.month").count()
        freq  = count / track.count() * 100.
        return freq

    def calc_track_duration_freq(self, track):
        count = track.groupby("lifetime").count()
        freq  = count / track.count() * 100.
        return freq

    def calc_track_rainrate_freq(self, track, vname):
        count = track.groupby(vname).count()
        freq  = count / track.count() * 100.
        return freq

    def calc_track_freq(self, track, vname):
        from scipy import stats 
        iht  = track[vname].data.flatten()
        iht  = iht[np.logical_not(np.isnan(iht))]
        kde  = stats.gaussian_kde(dataset=iht)
        return kde

    def load_data_prec(self, year, season, tracks, extent=None, **kwargs):
        fname   = "/glade/scratch/yeliu/Tracer/precp_??????.nc"
        data    = xr.open_mfdataset(fname, combine='by_coords')["precipitation_st4"].squeeze() 
#        data    = data.load()
        data = data.sel(time=data["time.year"].isin(np.arange(year[0], year[1]+1)))
        if season != 'Ann':
            data = myfunc.select_month(data, season=season)
        data = data.fillna(0)

        # filter data based on cells
        time = tracks["time"].dt.floor("H")
        time = np.unique(np.array(time))

        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            if extent is not None:
                print(data)
                print(extent[0], extent[1])
                data = data.sel(lon=slice(extent[0], extent[1]), 
                                lat=slice(extent[2], extent[3]))
            print(data)
            met = data.sel(time=time)
        return met, data

mytrack = MyTrack()

### Reanalysis ###
class MyERA5(object):
    '''functions for this project'''
    def __init__(self):
        pass

    def load_data_era5(self, var, vname, year, season, time=None, tracks=None, extent=None, **kwargs):
        data = self.read_era5_variable(var, vname, **kwargs)
        data = data.sel(time=data["time.year"].isin(np.arange(year[0], year[1]+1)))
        data = myfunc.select_month(data, season=season)

        # filter data based on cells
        if tracks is not None:
            time = tracks["time"].dt.floor("H")
            time = np.unique(np.array(time))
            
        if var == "cin":
            data.coords["longitude"] = data["longitude"].data + 360
            data = data.fillna(0.)
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            if extent is not None:
                print(data)
                print(extent[0]+360, extent[1]+360)
                data = data.sel(longitude=slice(extent[0]+360, extent[1]+360), 
                                latitude=slice(extent[2], extent[3]))
            print(data.time)
            print(time)
            if time is not None:
                met = data.sel(time=time)
                return met, data 
            else: 
                return data

    def read_era5_variable(self, var, vname, scale=1., offset=0., **kwargs):
        datadir = "../ERA5"
        if (level:=kwargs.get("level")) is not None:
            fname   = f"{var}.????.{level}.nc"
        else:
            fname   = f"{var}.????.nc"
        fname   = f"{datadir}/{fname}"
        print(f"loading data from: {fname}")
        data    = xr.open_mfdataset(fname, combine='by_coords')[vname].squeeze() * scale + offset
#        data    = data.load()
        data    = data[:,::-1,:]
        return data

    def calc_anomaly(self, met, clm):
        met_mean = met.mean("time")
        clm_mean = clm.mean("time")
        ano = met_mean  - clm_mean
        ano, pval = self.two_sample_diff(met, clm)
        return ano, pval, met_mean, clm_mean

myera5  = MyERA5()

### SOM ###
class MySOM(object):
    def __init__(self):
        pass
    def calc_anomaly(self, data):
        return data - data.mean("time")

    def calc_anomaly_monthly(self, data):
        return data.groupby("time.month") - data.groupby("time.month").mean("time")

    def convert_to_1D(self, *args):
        dat = []
        for v in args:
#            v = v - v.mean(dim=["longitude", "latitude"])
            v = (v-v.mean(dim='time'))/v.std(dim='time')
            dat.append(self._convert_to_1D(v))
        dat = xr.concat(dat, 'z').reset_index('z')
        dat.name = 'preSOM'
        dim = dat.shape
        dat.coords['z'] = np.arange(dim[1])
        return dat

    def _convert_to_1D(self, dat):
        lat = dat.latitude
        weight = np.sqrt(np.cos(1./180*3.14159*lat))
        datw = dat * weight
        dim  = datw.shape
        datw = datw.stack(z=['longitude','latitude'])
        datw = datw.dropna('z')
        return datw

    def read_bmu(self, fname_bmu, season, pcptype=None, flag_swap=False, flag_cluster=False):
        #...load bmu 
        bmu = xr.open_dataset(fname_bmu)["bmu"]
        if flag_swap:
            print("swaping bmu....")
            swap_dict = dict(
                MAM_idc=[1, 3, 2, 0],
                MAM_mcs=[0, 1, 3, 2],
                JJA_idc=[1, 0, 3, 2],
                JJA_mcs=[2, 3, 1, 0],
                SON_idc=[1, 3, 2, 0],
                SON_mcs=[0, 2, 3, 1],
                DJF_idc=[1, 3, 2, 0],
                DJF_mcs=[1, 0, 2, 3],
            )
            swap = swap_dict[f"{season}_{pcptype}"]
            for i in np.arange(4):
                bmu = xr.where(bmu==swap[i], i+1000, bmu)
            bmu = bmu - 1000

        if flag_cluster:
            print("combining clusters...")
            swap_dict = dict(
#                MAM_idc=[1, 3, 2, 0],
#                MAM_mcs=[0, 1, 3, 2],
                JJA_idc=[1, 0, 3, 2],
                JJA_mcs=[2, 3, 1, 0],
           )
            for ib, ic in enumerate(bmu.attrs["cluster"]):
                bmu = xr.where(bmu==ib, ic+1000, bmu)
            bmu = bmu - 1000
        return bmu

    def calc_cluster_mapsize(self, data, bmu, mapsize=None):
        data = data.assign_coords(bmu=("time", bmu.data))#["bmu"] = bmu.astype(np.float)
        clusters = []
        cluster_mean = []
        pct = []
        for ibmu in np.arange(mapsize[0] * mapsize[1]):
            cluster = data.where(data["bmu"]==ibmu, drop=True)
            if cluster.size==0:
                clusters.append(None)
                cluster_mean.append(None)
                pct.append(0.)
            else:
                clusters.append(cluster)
                cluster_mean.append(cluster.mean("time"))
                pct.append(cluster.shape[0] / data.shape[0] * 100.)
        self.ncluster = len(cluster_mean)
        return clusters, cluster_mean, pct

    def calc_bmu_to_cluster(self, bmu):
        cluster = bmu.attrs["cluster"]
        nbmu    = bmu.max().astype(int).item() + 1
        for i in range(nbmu):
            bmu = xr.where(bmu==i, cluster[i]+1000, bmu)
        bmu = bmu - 1000
        return bmu

    def calc_cluster_track_mapsize(self, tracks, bmu, mapsize=None):
        clusters = []
        time = tracks["time"].dt.floor("H")
        time_unique = np.unique(np.array(time))
        time_unique = np.intersect1d(time_unique, bmu.coords["time"].data)
        tracks = tracks.swap_dims({"tracks":"time"})
        for ibmu in np.arange(mapsize[0] * mapsize[1]):
            itime = time_unique[bmu==ibmu]
            flag = [True if it in itime else False for it in time.data]
            cluster = tracks.sel(time=flag)
            if cluster.size==0:
                clusters.append(None)
            else:
                cluster = cluster.swap_dims({"time":"tracks"})
                clusters.append(cluster)
        return clusters

    def bmu_filename(self, mapsize, factors, month, opt):
        fname = f"som/bmu_map{mapsize[0]}x{mapsize[1]}_fac{'+'.join(factors)}"
        if isinstance(month, str): 
            print(month)
            fname += f"{month}"
        else:
            if isinstance(month, (tuple, list)):
                fname += f"_{month[0]}-{month[1]}"
            else:
                fname += f"{month}"
        return fname+f"_{opt}.nc"

mysom   = MySOM()

### Reanalysis ###
class MyPlot(object):
    '''plot figures'''
    def __init__(self):
        pass
    def _colormap(self, cmap, clev, cend=None, cstart=None):
        '''extract colormap and levels'''
        if type(cmap)==str: cmap = getattr(cmaps, cmap)
        if cend is not None:
            cmap = cmap[:cend,:]
            cmap.N = cend
        if cstart is not None:
            cmap = cmap[cstart:,:]
            cmap.N = cmap.N - cstart
            print(cstart, cmap.N)
        norm = matplotlib.colors.BoundaryNorm(clev, cmap.N)
        return cmap, clev, norm

    def add_contourf(self, ax, lon, lat, data, cmap, clev, **kwargs):
        #...colormap
        cmap, clev, norm = self._colormap(cmap, clev, cend=kwargs.get("cend", None), cstart=kwargs.get("cstart", None))
        #...add contourf
        cplot = ax.contourf(lon.data, lat.data, data.data,
                    levels=clev, cmap=cmap,
                    norm=norm,
                    transform=ccrs.PlateCarree(),
                    )
        #...draw colorbar
        if kwargs.get("colorbar", False):
            cbar = plt.colorbar(cplot, ax=ax,
                    orientation="horizontal",
                    ticks=clev[1:-1:2],
                    drawedges=True,
                    extendrect=True,
#                    pad=0.08,
                    pad=0.15,
                    shrink=0.75,
                    aspect=30)
        return cplot

    def add_pcolor(self, ax, lon, lat, data, cmap, clev, **kwargs):
        #...colormap
        cmap, clev, norm = self._colormap(cmap, clev, cend=kwargs.get("cend", None))
        #...add contourf
        cplot = ax.pcolormesh(lon.data, lat.data, data.data,
                    cmap=cmap, norm=norm,
                    transform=ccrs.PlateCarree(),
                    )
        #...draw colorbar
        if kwargs.get("colorbar", False):
            cbar = plt.colorbar(cplot, ax=ax,
                    orientation="horizontal",
                    ticks=clev[1:-1:2],
                    drawedges=True,
                    extendrect=True,
                    pad=0.15,
                    shrink=0.75,
                    aspect=30)
        return cplot

    def add_contour(self, ax, lon, lat, data, clev, color="k", **kwargs):
        #...add contourf
        cplot = ax.contour(lon.data, lat.data, data.data,
                    levels=clev, colors=color,
                    linewidths=0.9,
                    transform=ccrs.PlateCarree(),
                    )
        return cplot

    def add_hatches(self, ax, lon, lat, data, clev, **kwargs):
        #...add contourf
        plt.rcParams['hatch.linewidth'] = kwargs.get("lw", 1)
        plt.rcParams['hatch.color'] = kwargs.get("lc", "k")
        cplot = ax.contourf(lon.data, lat.data, data.data,
                    levels=clev, colors="none",
                    hatches=["/"],
                    transform=ccrs.PlateCarree(),
                    )
        return cplot

    def add_scatter(self, ax, x, y, **kwargs):
        _kwargs = dict(s=2, marker='o', 
                       color='cyan', alpha=1.0, 
                       transform=ccrs.PlateCarree()) 
        _kwargs.update(kwargs)
        splot = ax.scatter(x, y, **_kwargs)
        return splot

    def add_vector(self, ax, lon, lat, u, v, **kwargs):
        #...add contourf
#        u = gvutil.set_vector_density(u, 0.017)
#        v = gvutil.set_vector_density(v, 0.017)
        u = u[::10,::10]
        v = v[::10,::10]
        lon, lat = u["longitude"], u["latitude"]

#        data_length = np.abs(np.sqrt(np.square(u) + np.square(v))).data
#        max_length  = np.max(data_length)
        max_length  = kwargs.get("max_length", 8)
        quiver_length = max_length
        disp_length = 5
        scale_factor = max_length/disp_length

#        print(max_length)
#        print("scale_factor", scale_factor)

        vplot = ax.quiver(lon, lat, u.data, v.data,
                       zorder=1,
                       pivot="middle",
                       units='xy', width=0.3,
                       scale=scale_factor, scale_units='xy')
        # Draw legend for vector plot
        #ax.add_patch(
        #    plt.Rectangle((150, -140),
        #                  30,
        #                  30,
        #                  facecolor='white',
        #                  edgecolor='black',
        #                  clip_on=False))
        if kwargs.get('flag_draw_qk', 1)==1:
            qk_location = kwargs.get("qk_location", (0.83, -0.16))
            qk = ax.quiverkey(vplot,
                              *qk_location,
                              quiver_length,
                              f"{quiver_length}"+r' $ms^{-1}$',
                              labelpos='W',
                              coordinates='axes',
                              color='black')
        return vplot

    def add_rectangle(self, ax, rect, **kwargs):
        rect_kwargs = dict(xy=(rect[0], rect[2]), width=rect[1]-rect[0], height=rect[3]-rect[2], 
                           ec="tab:purple", lw=2, 
                           fc="none", transform=ccrs.PlateCarree())
        rect_kwargs.update(kwargs)
        ax.add_patch(mpatches.Rectangle(**rect_kwargs))

    def add_axes(self, ax, **kwargs):
        # Set extent of maps created in the following cells:
#        axis_extent = gv.get_value("axis_extent", [-120, -60, 15, 50])
        axis_extent = kwargs.get("axis_extent", [-120, -60, 15, 50])
        #...add map features
        states = NaturalEarthFeature(category="cultural",
                                     scale="50m",
                                     facecolor="none",
                                     name="admin_1_states_provinces_shp")
        states_kwargs = dict(linewidth=0.5, edgecolor="tab:gray")
        states_kwargs.update(kwargs.get("states_kwargs", {}))
        ax.add_feature(states, **states_kwargs)
        coast_kwargs = dict(linewidth=0.8, edgecolor="tab:gray")
        coast_kwargs.update(kwargs.get("coast_kwargs", {}))
        ax.coastlines('50m', **coast_kwargs)
        ax.set_extent(axis_extent, crs=ccrs.PlateCarree())
        #...add strings
        if (ss:=kwargs.get("leftstring")) is not None:
            x, y, s, ss_dict = ss
            ss_kwargs = dict(fontsize=12, transform=ax.transAxes)
            ss_kwargs.update(ss_dict)
            ax.text(x, y, s, **ss_kwargs)
        if (ss:=kwargs.get("rightstring")) is not None:
            x, y, s, ss_dict = ss
            ss_kwargs = dict(fontsize=12, transform=ax.transAxes)
            ss_kwargs.update(ss_dict)
            ax.text(x, y, s, **ss_kwargs)

    def add_grid(self, ax, **kwargs):
        gl_kwargs = dict(crs=ccrs.PlateCarree(),
                          draw_labels=True,
                          dms=False,
                          x_inline=False,
                          y_inline=False,
                          linewidth=1,
                          color="black",
                          alpha=0.25)
        gl_kwargs.update({k:kwargs[k] for k in gl_kwargs if k in kwargs})
        # Draw gridlines
        gl = ax.gridlines(**gl_kwargs)
        
        # Manipulate latitude and longitude gridline numbers and spacing
        gl.top_labels      = kwargs.get("top_labels", False) 
        gl.bottom_labels   = kwargs.get("bottom_labels", True) 
        gl.left_labels     = kwargs.get("left_labels", True) 
        gl.right_labels    = kwargs.get("right_labels", False) 
        gl.xlocator = mticker.FixedLocator(kwargs.get("xticks", np.linspace(-180, 180, 13)))
        gl.ylocator = mticker.FixedLocator(kwargs.get("yticks", np.linspace(-90, 90, 7)))
        gl.xlabel_style = {"rotation": 0, "size": 10}
        gl.ylabel_style = {"rotation": 0, "size": 10}

myplot  = MyPlot()

