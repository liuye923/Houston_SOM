import os
from glob import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import pandas as pd
from myfunc import mysom


class mainfunc(object):
    def __init__(self, **kwargs):
        pass
    
    def find_merge_split_tracks(self, data, stats):
        start_time     = data['start_time']
        end_time       = data['end_time']
        start_split_tracknumber = data['start_split_tracknumber']
        end_merge_tracknumber   = data['end_merge_tracknumber']
        start_maxrange_flag     = data['start_maxrange_flag']
        end_maxrange_flag       = data['end_maxrange_flag']
        df = data['output']

        df['start_split_tracknumber'] = start_split_tracknumber
        df['end_merge_tracknumber']   = end_merge_tracknumber
        df['start_maxrange_flag']     = start_maxrange_flag
        df['end_maxrange_flag']       = end_maxrange_flag
        df['start_time']              = start_time
        df['end_time']                = end_time
        data.update(dict(output=df))
        
        # Get track start/end hour
        start_hour = start_time.dt.hour
        end_hour = end_time.dt.hour
        
        # Find tracks not end with merge or not start with split
        nonmerge = np.where(np.isnan(end_merge_tracknumber))[0]
        nonsplit = np.where(np.isnan(start_split_tracknumber))[0]
        ntracks_nonmerge = len(nonmerge)
        ntracks_nonsplit = len(nonsplit)
        print(f'Number of non-merge tracks: {ntracks_nonmerge}')
        print(f'Number of non-split tracks: {ntracks_nonsplit}')
        
        nonsplit_in = np.where(np.isnan(start_split_tracknumber) & (start_maxrange_flag == 1))[0]
        nonmerge_in = np.where(np.isnan(end_merge_tracknumber) & (end_maxrange_flag == 1))[0]
        ntracks_nonsplit_in = len(nonsplit_in)
        ntracks_nonmerge_in = len(nonmerge_in)
        print(f'Number of non-split tracks within max range: {ntracks_nonsplit_in}')
        print(f'Number of non-merge tracks within max range: {ntracks_nonmerge_in}')
        
        merge = np.where(end_merge_tracknumber > 0)[0]
        split = np.where(start_split_tracknumber > 0)[0]
        ntracks_merge = len(merge)
        ntracks_split = len(split)
        print(f'Number of merge tracks: {ntracks_merge}')
        print(f'Number of split tracks: {ntracks_split}')

        data.update(dict(
            start_hour=start_hour, 
            end_hour=end_hour,
            split=split,
            merge=merge,
            nonmerge=nonmerge,
            nonsplit=nonsplit,
            nonsplit_in=nonsplit_in,
            nonmerge_in=nonmerge_in,
        ))
        print(merge)
        print(split)
        print(nonmerge)
        print(nonsplit)
        return data
       
        
    def get_cell_start_end_location_status(self, data, stats):
        maxrange_flag = data['maxrange_flag']
        cell_lon      = data['cell_lon']
        cell_lat      = data['cell_lat']
        lifetime      = data['lifetime']
        time_res      = data['time_res']
        ntracks       = data['ntracks']
        tracks        = data['tracks']

        # Get track start values
        start_maxrange_flag = maxrange_flag.isel(times=0)
        start_lon = cell_lon.isel(times=0)
        start_lat = cell_lat.isel(times=0)
        print(max(start_lon), min(start_lon))

        # Get duration (lifetime counts) and maxrange_flag in Numpy arrays for speed
        duration = (lifetime.values / time_res).astype(int)
        rangeflag = maxrange_flag.values
        celllon = cell_lon.values
        celllat = cell_lat.values

        end_maxrange_flag = np.ones(ntracks, dtype=float)
        end_lon = np.full(ntracks, np.NaN, dtype=float)
        end_lat = np.full(ntracks, np.NaN, dtype=float)

        # Get track last valid value
        for ii in range(0, ntracks):
            # Get duration for the track
            iduration = duration[ii]
            # Get valid values for the track
            imaxrangeflag = rangeflag[ii,0:iduration]
            icell_lon = celllon[ii,0:iduration]
            icell_lat = celllat[ii,0:iduration]

            # Get last value from the track
            end_maxrange_flag[ii] = imaxrangeflag[-1]
            end_lon[ii] = icell_lon[-1]
            end_lat[ii] = icell_lat[-1]

        data.update(dict(
            start_lon=start_lon,
            start_lat=start_lat,
            end_lon=end_lon,
            end_lat=end_lat,
            duration=duration,
            end_maxrange_flag=end_maxrange_flag,
            start_maxrange_flag=start_maxrange_flag,
        ))
        print('creating dataframe')
        df = pd.DataFrame(
            np.array([tracks, start_lon, start_lat, end_lon, end_lat, duration, lifetime, end_maxrange_flag, start_maxrange_flag]).T,
            columns = ['tracks', 'start_lon', 'start_lat', 'end_lon', 'end_lat', 'duration', 'lifetime', 'end_maxrange_flag', 'start_maxrange_flag'],
            index = np.arange(ntracks)
        )
        data['output']  = df
        return data

    def get_cell_statistics(self, data, stats):
        data['time_res'] = stats.attrs['time_resolution_hour']
        data['pixel_radius'] = stats.attrs['pixel_radius_km']

        data['tracks'] = stats['tracks']
        data['times'] = stats['times']
        data['lifetime'] = stats['track_duration'] * data['time_res']
        data['start_time'] = stats['start_basetime']
        data['end_time'] = stats['end_basetime']
        data['end_merge_tracknumber'] = stats['end_merge_tracknumber']
        data['start_split_tracknumber'] = stats['start_split_tracknumber']
        data['cell_lon'] = stats['cell_meanlon']
        data['cell_lat'] = stats['cell_meanlat']
        data['cell_area'] = stats['cell_area']
        data['maxrange_flag'] = stats['maxrange_flag']
        
        data['maxdbz'] = stats['max_dbz']
        data['eth10'] = stats['maxETH_10dbz']
        data['eth20'] = stats['maxETH_20dbz']
        data['eth30'] = stats['maxETH_30dbz']
        data['eth40'] = stats['maxETH_40dbz']
        data['eth50'] = stats['maxETH_50dbz']
        # data['rainrate'] = stats['pf_ccrate'] + stats['pf_sfrate']
        return data


    def find_valid_tracks(self, data, stats):
        print(f"Number of tracks: {stats.dims['tracks']}")
        # Get cell initial location
        cell_lon0 = stats['cell_meanlon'].isel(times=0)
        cell_lat0 = stats['cell_meanlat'].isel(times=0)
        # Find tracks where initiation longitude is not NaN
        # These tracks are problematic and should be excluded
        ind = np.where((cell_lon0<-90)&(cell_lat0>20))[0]
        
        # Subset the tracks
        stats = stats.isel(tracks=ind).load()
        ntracks = stats.dims['tracks']
        print(f'Number of valid tracks: {ntracks}')
        data['ntracks'] = ntracks
        return stats

    def calc_distribution(self, data, groupby, bins):
        # groups = data.groupby(pd.cut(groupby, bins))
        groups = data.groupby(pd.cut(groupby, bins, labels=bins[1:]))
        return groups 
    
    def count_tracks_by_diurnal_timing(self, data, by='start_time'):
        column_by     = pd.to_datetime(data[by]).dt.hour.to_xarray().rename({'index':'tracks'})

        # Get initiation and time
        hour_bin = np.arange(-1, 24, 1)
        
        groups = self.calc_distribution(data, column_by, hour_bin)
        
        # hist_starthour, bins = np.histogram(start_hour, bins=hour_bin, range=(0,24), density=False)
        
        count = groups.count().iloc[:,0]
        count.index = count.index.astype(int)
        
        # Convert to local time (UTC-5)
        count = np.roll(count, -6) 

        # Convert to relative values
        count  = count / np.sum(count) * 100.
        
        return hour_bin[1:], count
    
    
    def mean_tracks_by_diurnal_timing(self, data, by='start_time'):
        header = data.columns
        column_by     = pd.to_datetime(data[by]).dt.hour.to_xarray().rename({'index':'tracks'})

        # Get initiation and time
        hour_bin = np.arange(-1, 24, 1)
        
        groups = self.calc_distribution(data, column_by, hour_bin)
        
        # hist_starthour, bins = np.histogram(start_hour, bins=hour_bin, range=(0,24), density=False)
        
        mean = groups.mean()
        mean.index = mean.index.astype(int)
            
        # Convert to local time (UTC-5)
        # mean = np.roll(mean, -5) 
        
        return hour_bin[1:], mean
  
    
    def calc_cell_cluster_som(self, data, season=None):
        start_time      = pd.to_datetime(data['start_time']).to_xarray().rename({'index':'tracks'})
        start_time.coords['tracks'] = ('tracks', data['tracks'])
        start_time_hour = start_time.dt.round('1H')
        bmu  = mysom.read_bmu(f'../MCS_IDC_uv_q_small/som/bmu_map2x2_facu+v+q{season}_idc.nc', season, flag_cluster=False, flag_swap=True, pcptype='idc')
        track_cluster = []
        for ibmu in np.arange(4):
            time_bmu  = bmu.where(bmu==ibmu, drop=True).time
            xtime  = np.intersect1d(time_bmu.data, start_time_hour.data)
            xstart_time_hour = start_time_hour.where(start_time_hour.isin(xtime), drop=True)
            xtrack = xstart_time_hour.tracks.data
            track_cluster.append(xtrack)
        return track_cluster
        
    def count_tracks_by_location(self, data):
        start_lon = data['start_lon']
        start_lat = data['start_lat']
        end_lon = data['end_lon']
        end_lat = data['end_lat']

        start_split_tracknumber = data['start_split_tracknumber']
        start_maxrange_flag = data['start_maxrange_flag']
        end_merge_tracknumber = data['end_merge_tracknumber']
        end_maxrange_flag = data['end_maxrange_flag']
        
        nonsplit_in = np.where(np.isnan(start_split_tracknumber) & (start_maxrange_flag == 1))[0]
        nonmerge_in = np.where(np.isnan(end_merge_tracknumber) & (end_maxrange_flag == 1))[0]
        merge = np.where(end_merge_tracknumber > 0)[0]
        split = np.where(start_split_tracknumber > 0)[0]

        buffer = 0.05
        minlon, maxlon = np.nanmin([start_lon.values, end_lon])-buffer, np.nanmax([start_lon.values, end_lon])+buffer
        minlat, maxlat = np.nanmin([start_lat.values, end_lat])-buffer, np.nanmax([start_lat.values, end_lat])+buffer
        # print(minlon, maxlon, minlat, maxlat)

        # make a cartesian grid to count the cells
        bins = [24, 24]
        # bins = [48, 48]
        ranges = [[minlon+buffer,maxlon-buffer], [minlat+buffer,maxlat-buffer]]

        hist2d_startloc, xbins, ybins = np.histogram2d(start_lon.values, start_lat.values, bins=bins, range=ranges)
        hist2d_endloc, xbins, ybins = np.histogram2d(end_lon, end_lat, bins=bins, range=ranges)

        hist2d_startloc = hist2d_startloc.transpose()# / 13
        hist2d_endloc = hist2d_endloc.transpose()# / 13
        
        hist2d_startloc_nonsplit, xbins, ybins = np.histogram2d(start_lon.values[nonsplit_in], start_lat.values[nonsplit_in], bins=bins, range=ranges)

        
        return xbins, ybins, hist2d_startloc, hist2d_endloc, hist2d_startloc_nonsplit
        
        
radar_util = mainfunc()
        
     


