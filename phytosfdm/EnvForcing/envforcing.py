#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import netcdf
import scipy.interpolate as intrp
import os


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
        outforcing = np.array([])
        
        if self.varname == 'mld':
            ncfile = netcdf.netcdf_file(self.fordir+'/mld_vd_WOA_Monterey_and_Levitus.nc', 'r')
            nclat = ncfile.variables['lat'].data.copy()
            nclon = ncfile.variables['lon'].data.copy()
            (latindx,) = np.logical_and(nclat <= self.Lat+self.RangeBB, nclat >= self.Lat-self.RangeBB).nonzero()
            (lonindx,) = np.logical_and(nclon <= self.Lon+360+self.RangeBB, nclon >= self.Lon+360-self.RangeBB).nonzero()
            for i in range(12):
                ncdat = ncfile.variables[self.varname][i, latindx.min():latindx.max(), lonindx.min():lonindx.max()].copy()
                mskncdat = np.ma.masked_where(ncdat < 0, ncdat)
                outforcing = np.append(outforcing, np.mean(mskncdat))
            ncfile.close()
            return np.append(outforcing, outforcing[0])
        elif self.varname == 'par':
            ncfile = netcdf.netcdf_file(self.fordir+'/par_MODIS_2002to2011.nc', 'r')
            nclat = ncfile.variables['lat'].data.copy()
            nclon = ncfile.variables['lon'].data.copy()
            (latindx,) = np.logical_and(nclat <= self.Lat+self.RangeBB, nclat >= self.Lat-self.RangeBB).nonzero()
            (lonindx,) = np.logical_and(nclon <= self.Lon+180+self.RangeBB, nclon >= self.Lon+180-self.RangeBB).nonzero()
            for i in range(12):
                ncdat = ncfile.variables[self.varname][i, latindx.min():latindx.max(), lonindx.min():lonindx.max()].copy()
                mskncdat = np.ma.masked_where(ncdat < 0, ncdat)
                outforcing = np.append(outforcing, np.mean(mskncdat))
            ncfile.close()
            return np.append(outforcing, outforcing[0])
        elif self.varname == 'n0x':
            ncfile = netcdf.netcdf_file(self.fordir+'/n0x_WOA09.nc','r')
            nclat = ncfile.variables['lat'].data.copy()
            nclon = ncfile.variables['lon'].data.copy()
            (latindx,) = np.logical_and(nclat <= self.Lat+self.RangeBB, nclat >= self.Lat-self.RangeBB).nonzero()
            (lonindx,) = np.logical_and(nclon <= self.Lon+360+self.RangeBB, nclon >= self.Lon+360-self.RangeBB).nonzero()
            ncdepth = ncfile.variables['depth'].data.copy()
            mld = ExtractEnvFor(self.Lat, self.Lon, self.RangeBB, 'mld')
            for i in range(12):
                (depthindx,) = (ncdepth >= mld.outForcing[i]).nonzero()
                ncdat = ncfile.variables[self.varname][i, depthindx.min(), latindx.min():latindx.max(), lonindx.min():lonindx.max()].copy()
                mskncdat = np.ma.masked_where(ncdat < 0, ncdat)
                outforcing = np.append(outforcing, np.mean(mskncdat))
            ncfile.close()
            return np.append(outforcing, outforcing[0])
        elif self.varname == 'sst':
            ncfile = netcdf.netcdf_file(self.fordir+'/sst-t_an-WOA09.nc', 'r')
            nclat = ncfile.variables['lat'].data.copy()
            nclon = ncfile.variables['lon'].data.copy()
            (latindx,) = np.logical_and(nclat <= self.Lat+self.RangeBB, nclat >= self.Lat-self.RangeBB).nonzero()
            (lonindx,) = np.logical_and(nclon <= self.Lon+360+self.RangeBB, nclon >= self.Lon+360-self.RangeBB).nonzero()
            for i in range(12):
                ncdat = ncfile.variables[self.varname][i, 0, latindx.min():latindx.max(), lonindx.min():lonindx.max()].copy()
                mskncdat = np.ma.masked_where(ncdat > 100, ncdat)
                outforcing = np.append(outforcing, np.mean(mskncdat))
            ncfile.close() 
            return np.append(outforcing,outforcing[0])
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
