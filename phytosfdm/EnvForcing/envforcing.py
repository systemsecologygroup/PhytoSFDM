#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import netcdf
import scipy.interpolate as intrp
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon


class ExtractEnvFor:
    """
    Class with methods to extract the environmental forcing,
    named: mixed layer depth (MLD), photosynthetic active radiation (PAR)
    sea surface temperature (SST) and concentration of nutrients below MLD (N0).
    The class uses three methods: "spatialave" computes an spatial average at the
    geographic location specify by the user and returns a monthly spatial average.
    "dailyinterp" extrapolate the monthly climatological data to a daily time steps.
    Last, "firstderivspl" calculates the first derivative for a daily interpolation
    using splines.

    Parameters
    ----------
    lat: is the Latitude with a range from -90 to 90 degrees (North negative).
    lon: is the Longitude with a range from -180 to 180 degrees (East positive).
    rangebb: is the spatial range of the bounding box in degrees.
    varname: is the variable names on the provided netcdf file. It could be either
        'mld' for Mixed Layer Depth,'par' for Photosynthetic Active Radiation, 'sst'
        for Sea Surface Temperature and 'n0x' for nutrient concentration below MLD.
    """
    
    def __init__(self, lat, lon, rangebb, varname):
        self.Lat = lat
        self.Lon = lon
        self.RangeBB = rangebb
        self.varname = varname
        self.fordir = os.path.split(os.path.realpath(__file__))[0]
        self.outForcing = self.spatialave()
           
    def spatialave(self):
        """
        Method to extract spatially averaged environmental forcing.

        Returns
        -------
        The spatial average of the respective environmental forcing per month.
        """

        if self.varname == 'mld':
            ncfile = netcdf.netcdf_file(self.fordir+'/mld_vd_WOA_Monterey_and_Levitus.nc', 'r')
            nclat = ncfile.variables['lat'].data.copy()
            nclon = ncfile.variables['lon'].data.copy()
            ncdat = ncfile.variables[self.varname].data.copy()
            ncfile.close()
            nclon_transform = np.hstack((nclon[nclon > 180]-360., nclon[nclon < 180]))
            ncdat_new = np.dstack((ncdat[:, :, 180:], ncdat[:, :, :180]))
            mskdat = np.ma.masked_less(ncdat_new, 0)
            longrid, latgrid = np.meshgrid(nclon_transform, nclat)
            selectarea = np.logical_and(longrid <= self.Lon+self.RangeBB, longrid >= self.Lon-self.RangeBB) *\
                         np.logical_and(latgrid <= self.Lat+self.RangeBB, latgrid >= self.Lat-self.RangeBB)
            outforcing = np.mean(mskdat[:, selectarea], axis=1)
            return np.append(outforcing, outforcing[0])
        elif self.varname == 'par':
            ncfile = netcdf.netcdf_file(self.fordir+'/par_MODIS_2002to2011.nc', 'r')
            nclat = ncfile.variables['lat'].data.copy()
            nclon = ncfile.variables['lon'].data.copy()
            ncdat = ncfile.variables[self.varname].data.copy()
            mskdat = np.ma.masked_less(ncdat, 0)
            ncfile.close()
            nclon_transform = np.hstack((nclon[nclon > 180]-360, nclon[nclon < 180]))
            longrid, latgrid = np.meshgrid(nclon_transform, nclat)
            selectarea = np.logical_and(longrid <= self.Lon+self.RangeBB, longrid >= self.Lon-self.RangeBB) *\
                         np.logical_and(latgrid <= self.Lat+self.RangeBB, latgrid >= self.Lat-self.RangeBB)
            outforcing = np.mean(mskdat[:, selectarea], axis=1)
            return np.append(outforcing, outforcing[0])
        elif self.varname == 'n0x':
            ncfile = netcdf.netcdf_file(self.fordir+'/n0x_WOA09.nc', 'r')
            nclat = ncfile.variables['lat'].data.copy()
            nclon = ncfile.variables['lon'].data.copy()
            ncdat = ncfile.variables[self.varname].data.copy()
            ncdepth = ncfile.variables['depth'].data.copy()
            ncfile.close()
            mlddat = ExtractEnvFor(self.Lat, self.Lon, self.RangeBB, 'mld')
            ncdat_mld = np.zeros((12, 180, 360))
            for i in range(0, 12):
                (depthindx,) = (ncdepth >= mlddat.outForcing[i]).nonzero()
                ncdat_mld[i, :, :] = ncdat[i, depthindx.min(), :, :]
            nclon_transform = np.hstack((nclon[nclon > 180]-360, nclon[nclon < 180]))
            ncdat_new = np.dstack((ncdat_mld[:, :, 180:], ncdat_mld[:, :, :180]))
            mskdat = np.ma.masked_less(ncdat_new, 0)
            longrid, latgrid = np.meshgrid(nclon_transform, nclat)
            selectarea = np.logical_and(longrid <= self.Lon+self.RangeBB, longrid >= self.Lon-self.RangeBB) *\
                         np.logical_and(latgrid <= self.Lat+self.RangeBB, latgrid >= self.Lat-self.RangeBB)
            outforcing = np.mean(mskdat[:, selectarea], axis=1)
            return np.append(outforcing, outforcing[0])
        elif self.varname == 'sst':
            ncfile = netcdf.netcdf_file(self.fordir+'/sst-t_an-WOA09.nc', 'r')
            nclat = ncfile.variables['lat'].data.copy()
            nclon = ncfile.variables['lon'].data.copy()
            ncdat = ncfile.variables[self.varname].data.copy()
            ncfile.close()
            nclon_transform = np.hstack((nclon[nclon > 180]-360., nclon[nclon < 180]))
            ncdat_new = np.dstack((ncdat[:, 0, :, 180:], ncdat[:, 0, :, :180]))
            mskdat = np.ma.masked_greater(ncdat_new, 99)
            longrid, latgrid = np.meshgrid(nclon_transform, nclat)
            selectarea = np.logical_and(longrid <= self.Lon+self.RangeBB, longrid >= self.Lon-self.RangeBB) *\
                         np.logical_and(latgrid <= self.Lat+self.RangeBB, latgrid >= self.Lat-self.RangeBB)
            outforcing = np.mean(mskdat[:, selectarea], axis=1)
            return np.append(outforcing, outforcing[0])
        else:
            return 'Please specify either mld, par, n0x or sst'

    def dailyinterp(self, time, kind='spline', k=3, s=None):
        """
        Method to interpolate from monthly to daily environmental data.

        Parameters
        ----------
        time: in days
        kind: the type of interpolation either linear, cubic, spline or
               piecewise polynomial
        k: Degree of the smoothing spline
        s: Positive smoothing factor used to choose the number of knots

        Returns
        -------
        The temporally interpolated environmental forcing.
        """
        
        tmonth = np.linspace(0., 12., 13.)
        newt = np.mod(time, 365.)*12./365.
        if kind == 'spline':
            outintp = intrp.UnivariateSpline(tmonth, self.outForcing, k=k, s=s)
            return outintp(newt)
        elif kind == 'PWPoly':
            outintp = intrp.PchipInterpolator(tmonth, self.outForcing)
            return outintp(newt)
        else:
            outintp = intrp.interp1d(tmonth, self.outForcing, kind=kind)
            return outintp(newt)    
    
    def firstderivspl(self, time, k=3, s=None):
        """
        Method to calculate the first derivative of an interpolated spline.

        Parameters
        ----------
        time: in days
        kind: the type of interpolation either linear, cubic, spline or
               piecewise polynomial
        k: Degree of the smoothing spline
        s: Positive smoothing factor used to choose the number of knots

        Returns
        -------
        The first derivative of the temporally interpolated environmental forcing spline.
        """
        
        tmonth = np.linspace(0., 365., 13.)
        newt = np.mod(time, 365.)
        outintp = intrp.UnivariateSpline(tmonth, self.outForcing, k=k, s=s)
        return outintp.derivative()(newt)

    def selected_area_plots(self):
        """
        Method to plot one of the averaged environmental
         forcing variable at the specified location.

        Parameters
        ----------


        Returns
        -------
        Plots of monthly averages and scatterplot of spatially average data for
        selected location

        """
        if self.varname == 'par':
            #Extract PAR data at location
            ncfile = netcdf.netcdf_file(self.fordir+'/par_MODIS_2002to2011.nc', 'r')
            nclat = ncfile.variables['lat'].data.copy()
            nclon = ncfile.variables['lon'].data.copy()
            ncdat = ncfile.variables[self.varname].data.copy()
            mskdat = np.ma.masked_less(ncdat, 0)
            ncfile.close()
            nclon_transform = np.hstack((nclon[nclon > 180]-360, nclon[nclon < 180]))
            longrid, latgrid = np.meshgrid(nclon_transform, nclat)
            selectarea = np.logical_and(longrid <= self.Lon+self.RangeBB, longrid >= self.Lon-self.RangeBB) *\
                         np.logical_and(latgrid <= self.Lat+self.RangeBB, latgrid >= self.Lat-self.RangeBB)
            ncdatlevs = np.arange(0, 61, 1)
            ncdatticks = [0, 15, 30, 45, 60]
            par_psfdm = ExtractEnvFor(self.Lat, self.Lon, self.RangeBB, 'par')
            var_interp = par_psfdm.dailyinterp(np.arange(0., 366., 1.0))
        if self.varname == 'mld':
            #Extract MLD data at location
            ncfile = netcdf.netcdf_file(self.fordir+'/mld_vd_WOA_Monterey_and_Levitus.nc', 'r')
            nclat = ncfile.variables['lat'].data.copy()
            nclon = ncfile.variables['lon'].data.copy()
            ncdat = ncfile.variables[self.varname].data.copy()
            ncfile.close()
            nclon_transform = np.hstack((nclon[nclon > 180]-360, nclon[nclon < 180]))
            ncdat_new = np.dstack((ncdat[:, :, 180:], ncdat[:, :, :180]))
            mskdat = np.ma.masked_less(ncdat_new, 0)
            longrid, latgrid = np.meshgrid(nclon_transform, nclat)
            selectarea = np.logical_and(longrid <= self.Lon+self.RangeBB, longrid >= self.Lon-self.RangeBB) *\
                         np.logical_and(latgrid <= self.Lat+self.RangeBB, latgrid >= self.Lat-self.RangeBB)
            ncdatlevs = np.arange(0, 501, 5)
            ncdatticks = [0, 250, 500]
            mld_psfdm = ExtractEnvFor(self.Lat, self.Lon, self.RangeBB, 'mld')
            var_interp = mld_psfdm.dailyinterp(np.arange(0., 366., 1.0), k=3)
        if self.varname == 'sst':
            #Extract SST data at location
            ncfile = netcdf.netcdf_file(self.fordir+'/sst-t_an-WOA09.nc', 'r')
            nclat = ncfile.variables['lat'].data.copy()
            nclon = ncfile.variables['lon'].data.copy()
            ncdat = ncfile.variables[self.varname].data.copy()
            ncfile.close()
            nclon_transform = np.hstack((nclon[nclon > 180]-360, nclon[nclon < 180]))
            ncdat_new = np.dstack((ncdat[:, 0, :, 180:], ncdat[:, 0, :, :180]))
            mskdat = np.ma.masked_greater(ncdat_new, 99)
            longrid, latgrid = np.meshgrid(nclon_transform, nclat)
            selectarea = np.logical_and(longrid <= self.Lon+self.RangeBB, longrid >= self.Lon-self.RangeBB) *\
                         np.logical_and(latgrid <= self.Lat+self.RangeBB, latgrid >= self.Lat-self.RangeBB)
            ncdatlevs = np.arange(-5, 41, 0.5)
            ncdatticks = [-5, 10, 25, 40]
            sst_psfdm = ExtractEnvFor(self.Lat, self.Lon, self.RangeBB, 'sst')
            var_interp = sst_psfdm.dailyinterp(np.arange(0., 366., 1.0), k=5)
        if self.varname == 'n0x':
            #Calculate average MLD at location
            mlddat = ExtractEnvFor(self.Lat, self.Lon, self.RangeBB, 'mld')
            #Extract N0x data at location
            ncfile = netcdf.netcdf_file(self.fordir+'/n0x_WOA09.nc', 'r')
            nclat = ncfile.variables['lat'].data.copy()
            nclon = ncfile.variables['lon'].data.copy()
            ncdat = ncfile.variables[self.varname].data.copy()
            ncdepth = ncfile.variables['depth'].data.copy()
            ncfile.close()
            ncdat_mld = np.zeros((12, 180, 360))
            for i in range(0, 12):
                (depthindx,) = (ncdepth >= mlddat.outForcing[i]).nonzero()
                ncdat_mld[i, :, :] = ncdat[i, depthindx.min(), :, :]
            nclon_transform = np.hstack((nclon[nclon > 180]-360, nclon[nclon < 180]))
            ncdat_new = np.dstack((ncdat_mld[:, :, 180:], ncdat_mld[:, :, :180]))
            mskdat = np.ma.masked_less(ncdat_new, 0)
            longrid, latgrid = np.meshgrid(nclon_transform,nclat)
            selectarea = np.logical_and(longrid <= self.Lon+self.RangeBB, longrid >= self.Lon-self.RangeBB) *\
                         np.logical_and(latgrid <= self.Lat+self.RangeBB, latgrid >= self.Lat-self.RangeBB)
            ncdatlevs = np.arange(0, 51, 0.5)
            ncdatticks = [0, 10, 20, 30, 40, 50]
            n0x_psfdm = ExtractEnvFor(self.Lat, self.Lon, self.RangeBB, 'n0x')
            var_interp = n0x_psfdm.dailyinterp(np.arange(0., 366., 1.0), k=5)

        #Plots
        #Global maps with location of the selected area overlay
        # of monthly averaged environmental forcing variable
        f13, axs13 = plt.subplots(3, 4, sharex='col', sharey='row')
        monthlist = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for idx, axs in enumerate(axs13.ravel()):
            mapbase = Basemap(resolution='c', projection='robin', lon_0=0, ax=axs)
            xx, yy = mapbase(longrid, latgrid)
            cs2 = mapbase.contourf(xx, yy, mskdat[idx, :, :], ncdatlevs, cmap=plt.cm.Spectral_r)
            mapbase.colorbar(cs2, 'bottom', ticks=ncdatticks)
            plons, plats = mapbase([self.Lon+self.RangeBB, self.Lon-self.RangeBB, self.Lon-self.RangeBB,
                                    self.Lon+self.RangeBB], [self.Lat+self.RangeBB, self.Lat+self.RangeBB,
                                                             self.Lat-self.RangeBB, self.Lat-self.RangeBB])
            mapbase.fillcontinents()
            xypoly = zip(plons, plats)
            poly = Polygon(xypoly, facecolor='none', edgecolor='black', lw=1.25)
            axs.add_patch(poly)
            axs.set_title(monthlist[idx])
        #Scatter plot of environmental forcing and interpolated data
        tdot = np.array([1., 32., 60., 91., 121., 152., 182., 213., 244., 274., 305., 335., 365.])
        plt.figure()
        plt.plot(np.arange(0., 366., 1.0), var_interp, c='black', lw=3)
        plt.plot(tdot, np.append(np.mean(mskdat[:, selectarea], axis=1), np.mean(mskdat[0, selectarea])), 'o')
        plt.ylabel(self.varname.upper())
        plt.xlabel('Month')
        plt.xticks(tdot, monthlist+['Jan'], rotation=30)

        plt.show()
