import pandas as pd
import xarray as xr
import numpy as np
import dask

import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature

import global_variable as gv
from myfunc import myfunc, mysom, mytrack, myera5, myplot
 
class mainfunc(object):
    def __init__(self, var=None, vname=None, level=None, year=(2004,2017), season="JJA", **kwargs):
        track_extent  = gv.get_value("track_extent", [-98, -93, 28, 32])
        met_extent    = gv.get_value("met_extent", [-120, -70, 15, 50])
        plt_extent    = gv.get_value("plt_extent", [-110, -75, 25, 45])

        met  = myera5.load_data_era5(var, vname, year, season, 
            extent=met_extent, level=level, **myfunc.var_register(var))
        met = met.load()
        savename = var
        if level is not None: savename=f"{var}{level}"
        met.to_netcdf(f"met/{savename}_{season}.nc")


if __name__ == '__main__':
#    mainfunc(var="q", vname="Q", level=925, season="JJA")
#    mainfunc(var="z", vname="Z", level=500, season="JJA")
#    mainfunc(var="u", vname="U", level=925, season="JJA")
#    mainfunc(var="v", vname="V", level=925, season="JJA")
#    mainfunc(var="u", vname="U", level=200, season="JJA")
#    mainfunc(var="v", vname="V", level=200, season="JJA")
#    mainfunc(var="z", vname="Z", level=200, season="JJA")
#    mainfunc(var="w", vname="W", level=500, season="JJA")
#    mainfunc(var="t2", vname="VAR_2T", season="JJA")

#    mainfunc(var="q", vname="Q", level=925, season="MAM")
#    mainfunc(var="z", vname="Z", level=500, season="MAM")
#    mainfunc(var="u", vname="U", level=925, season="MAM")
#    mainfunc(var="v", vname="V", level=925, season="MAM")
#    mainfunc(var="u", vname="U", level=200, season="MAM")
#    mainfunc(var="v", vname="V", level=200, season="MAM")
#    mainfunc(var="z", vname="Z", level=200, season="MAM")
#    mainfunc(var="w", vname="W", level=500, season="MAM")
#    mainfunc(var="t2", vname="VAR_2T", season="MAM")

#    mainfunc(var="q", vname="Q", level=925, season="SON")
#    mainfunc(var="z", vname="Z", level=500, season="SON")
#    mainfunc(var="u", vname="U", level=925, season="SON")
#    mainfunc(var="v", vname="V", level=925, season="SON")
#    mainfunc(var="u", vname="U", level=200, season="SON")
#    mainfunc(var="v", vname="V", level=200, season="SON")
#    mainfunc(var="z", vname="Z", level=200, season="SON")
#    mainfunc(var="w", vname="W", level=500, season="SON")
#    mainfunc(var="t2", vname="VAR_2T", season="SON")

    mainfunc(var="q", vname="Q", level=925, season="DJF")
    mainfunc(var="z", vname="Z", level=500, season="DJF")
    mainfunc(var="u", vname="U", level=925, season="DJF")
    mainfunc(var="v", vname="V", level=925, season="DJF")
    mainfunc(var="u", vname="U", level=200, season="DJF")
    mainfunc(var="v", vname="V", level=200, season="DJF")
    mainfunc(var="z", vname="Z", level=200, season="DJF")
    mainfunc(var="w", vname="W", level=500, season="DJF")
    mainfunc(var="t2", vname="VAR_2T", season="DJF")
