#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import odeint
import sympy

from phytosfdm.EnvForcing.envforcing import ExtractEnvFor


class SM:
    """
    Class with methods to quantify the size dynamics of phytoplankton
    communities in the ocean using a 0-dimensional NPZD model structure.

    Parameters
    ----------
    lat: is the Latitude with a range from -90 to 90 degrees (North positive).
    lon: is the Longitude with a range from -180 to 180 degrees (East positive).
    rangebb: is the spatial range of the bounding box in degrees.
    model: String specifying the size model variant to calculate, either 'Imm' for
        a model with immigration, 'TraitDif' for a model with trait diffusion
        mechanism, 'FixVar' for a model with a fix amount of variance or
        'UnsustVar' for a model that does not sustain variance and "FullModel"
        for a size model with a full spectrum of morphotypes.
    listparams: A list of tuples ('parameter_name',value) with the new values for
        for the specified parameter.
    defaultparams: a boolean to specify the use of default parameters or to
        modify the parameters suggested in ListParams.
    :assumimm: Assumptions on the immigrating community (P_imm). In assumption "P&L"
        the total biomass of P_imm is proportional to the biomass of the local
        community (P) and the mean size of P_imm is equal to the size of P. In
        assumption "L" only the size is equal between the two communities. In
        assumption "P" only the total biomass is proportional between the
        communities. Last if "None" P and L are different and independent from each
        other.

    Returns
    -------
    outvariables: is a ndarray that contains the results of all state variables.
        In addition, in the ndarray we included dummy variables that quantify
        the biomass fluxes between the state variables. Please refer to each model variant
        to identify each variable.
    """

    def __init__(self, lat, lon, rangebb, model, listparams=[], defaultparams=True, assumimm="P&L", kmld=3, smld=0,
                 kindmld="spline", kn0x=5, sn0x=None, kindn0x="spline", ksst=5, ssst=None, kindsst="spline", kpar=5,
                 spar=None, kindpar="spline"):
        self.Lat = lat
        self.Lon = lon
        self.RangeBB = rangebb
        self.Model = model
        self.ListParams = listparams
        self.defaultParams = defaultparams
        self.AssumImm = assumimm
        self.kmld = kmld
        self.smld = smld
        self.kindmld = kindmld
        self.kn0x = kn0x
        self.sn0x = sn0x
        self.kindn0x = kindn0x
        self.ksst = ksst
        self.ssst = ssst
        self.kindsst = kindsst
        self.kpar = kpar
        self.spar = spar
        self.kindpar = kindpar
        self.MLD = ExtractEnvFor(self.Lat, self.Lon, self.RangeBB, 'mld')
        self.N0X = ExtractEnvFor(self.Lat, self.Lon, self.RangeBB, 'n0x')
        self.SST = ExtractEnvFor(self.Lat, self.Lon, self.RangeBB, 'sst')
        self.PAR = ExtractEnvFor(self.Lat, self.Lon, self.RangeBB, 'par')
        self.Params = self.setup_params()
        self.initcond = self.initialconditions()
        self.timedays = np.arange(0., self.Params['timeyears']*365., 1.0)
        self.outvariables = self.modelrun()

    def initialconditions(self):
        """
        Method to assign the initial conditions depending on the type of model
        structure, i.e. full or aggregate model.
        """
        if self.Model == 'FullModel':
            return np.concatenate(([[self.Params['N0']], [self.Params['Z0']], [self.Params['D0']],
                                    [self.Params['P0']]*self.Params['NoMtype']]), 0)
        else:
            return np.concatenate(([[self.Params['N0']], [self.Params['Z0']], [self.Params['D0']], [self.Params['P0']],
                                    [self.Params['L_mean0']], [self.Params['L_var0']], [0]*29]), 0)

    def slns(self, ms, sv):
        """
        Method to log-transform the mean trait, based on the log-normal relationship,
        suggested by Bruggeman (2009).

        Parameters
        ----------
        ms: Mean trait or mean size
        sv: Trait variance or size variance
        """
        return np.log(ms / np.sqrt(sv / ms ** 2 + 1.))

    def vlnv(self, ms, sv):
        """
        Method to log-transform the trait variance, based on the log-normal
        relationship,suggested by Bruggeman (2009).

        Parameters
        ----------
        ms: Mean trait or mean size
        sv: Trait variance or size variance
        """
        return np.log(sv / ms ** 2 + 1.)

    def setup_params(self):
        """
        Method to assign either the default values to all parameters
        or to specify the new values according to the values provided
        on ListParams.
        """

        lnvar0 = self.vlnv(25.1, 500)
        lnmean0 = self.slns(25, 500)
        n0 = np.mean(self.N0X.outForcing[:12])

        default_parameters = {
            'kappa': 0.1,  	           # Diffusive mixing across thermocline (m*d^-1)
            'kw': 0.1,	               # Light attenuation constant (m^-1)
            'OptI': 30.,	           # Optimum irradiance (einstein*m^-2*d^-1)
            'muP': 1.5,	               # Phytoplankton max growth rate (d^-1)
            'moP': 0.05,	           # Phytoplankton mortality rate (d^-1)
            'muZ': 1.35,               # Zooplankton maximum grazing rate (d^-1)
            'moZ': 0.3,                # Zooplankton mortality
            'Kp': 0.1,                 # P half-saturation constant (mmol*m^-3)
            'deltaZ': 0.31,            # P assimilation coefficient (-)
            'deltaD': 0.1,	           # Mineralization rate (d^-1)
            'deltaI': 0.008,           # Immigration rate (mmol*m^-3 * d^-1)
            'nu': 0.008,               # Trait diffusivity parameter (-)
            'alphaG': -0.75,           # Slope of allometric grazer preference ([micro m ESD]^-1)
            'alphaU': 0.81,            # Slope of Kn allometric function (mmol * m^-3 * [micro m ESD]^-1),
                                       # modified from Litchman et. al 2007
            'betaU': 0.14275,          # Intercept of Kn allometric function (mmol * m^-3),
                                       # modified from Litchman et. al 2007
            'alphav': 1.17,            # Slope of Sinking allometric function (meters * d^-1 * [micro m ESD]^-1),
                                       # modified from Kiorboe 1993
            'betav': 0.01989,          # Intercept of Sinking allometric function (meters * d^-1),
                                       # modified from Kiorboe 1993
            'L_var0': lnvar0,          # Size variance of the immigrating community (Ln [micro m ESD]^2)
            'L_mean0': lnmean0,        # Mean size of the immigrating community (Ln micro m ESD)
            'N0': n0,                  # Initial Nutrient concentration (mmol*m^-3)
            'Z0': 0.1,                 # Initial Zooplankton concentration (mmol*m^-3)
            'D0': 0.01,                # Initial Detritus concentration (mmol*m^-3)
            'P0': 0.1,                 # Initial Phytoplankton concentration (mmol*m^-3)
            'timeyears': 10.0,         # Maximum running time of the model (years)
            'NoMtype': 10,             # Number of morphotypes for the full model
            'sizemin': 0.2,            # Minimum size of the phytoplankton community (micro m ESD)
            'sizemax': 50.             # Maximum size of the phytoplankton community (micro m ESD)
        }

        if self.defaultParams:
            return default_parameters

        else:
            for parnames, values in self.ListParams:
                default_parameters[parnames] = values
            return default_parameters

    def sizemodel_imm(self, x, t):
        """
        This size based model variant is based on Acevedo-Trejos et al. (2015) in Sci. Rep.

        Parameters
        ----------
        x: array with initial conditions for the state variables
        t: time
        """
        # Initialization of state variables and dummy variables to store the biomass fluxes
        N = x[0]
        Z = x[1]
        D = x[2]
        P = x[3]
        L = x[4]
        V = x[5]
        dxdt = np.zeros(35)

        # Edible phytoplankton
        Ped = P*(np.exp(L)**self.Params['alphaG']+0.5*V*self.Params['alphaG']**2*np.exp(L)**self.Params['alphaG'])

        # Gains of phytoplankton biomass
        NutrientUptake = N/(N+self.Params['betaU']*np.exp(L)**self.Params['alphaU'])
        LightHarvesting = 1./(self.Params['kw'] * self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld))\
                          * (-np.exp(1. - self.PAR.dailyinterp(t, kind=self.kindpar, k=self.kpar, s=self.spar)
                            / self.Params['OptI']) - (-np.exp((1. - (self.PAR.dailyinterp(t, kind=self.kindpar,
                            k=self.kpar, s=self.spar) * np.exp(-self.Params['kw'] * self.MLD.dailyinterp(t,
                            kind=self.kindmld, k=self.kmld, s=self.smld))) / self.Params['OptI']))))
        TemperatureDepGrowth = np.exp(0.063 * self.SST.dailyinterp(t, kind=self.kindsst, k=self.ksst, s=self.ssst))
        Gains = self.Params['muP'] * NutrientUptake * LightHarvesting * TemperatureDepGrowth

        # Losses of phytoplankton biomass
        Grazing = self.Params['muZ']*Z*np.exp(L)**self.Params['alphaG']/(Ped+self.Params['Kp'])
        Sinking = (self.Params['betav']*np.exp(L)**self.Params['alphav'])\
                  /self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        OtherPMortalities = self.Params['moP']
        Mixing = (self.Params['kappa'] + max(self.MLD.firstderivspl(t, k=self.kmld, s=self.smld), 0.)) \
                 / self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        Losses = Grazing + Sinking + OtherPMortalities + Mixing

        # Other Processes
        ZooGrowth = self.Params['deltaZ'] * Grazing * P
        ZooMortality = self.Params['moZ'] * Z**2
        ZooMixing = Z * self.MLD.firstderivspl(t, k=self.kmld, s=self.smld) \
                    / self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        UnassimilatedProduction = (1.-self.Params['deltaZ']) * Grazing * P
        Mineralization = self.Params['deltaD']*D
        DetritusMixing = D * (self.Params['kappa'] + max(self.MLD.firstderivspl(t, k=self.kmld, s=self.smld), 0.)) \
                         / self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        NMixing = Mixing * (self.N0X.dailyinterp(t, kind=self.kindn0x, k=self.kn0x, s=self.sn0x) - N)

        # Derivatives for the growth components of phytoplankton with respect to the trait
        tt0 = self.Params['muP']*LightHarvesting*TemperatureDepGrowth
        tt2 = self.Params['muZ']*Z
        d1r0 = -N*tt0*self.Params['betaU']*self.Params['alphaU']*np.exp(L)**self.Params['alphaU']\
               /(N + self.Params['betaU']*np.exp(L)**self.Params['alphaU'])**2
        d1r2 = tt2*self.Params['alphaG']*np.exp(L)**self.Params['alphaG']/(Ped + self.Params['Kp'])
        d1r3 = self.Params['betav']*self.Params['alphav']*np.exp(L)**self.Params['alphav']\
               /self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        d2r0 = N*tt0*self.Params['betaU']*self.Params['alphaU']**2\
               *(2*self.Params['betaU']*np.exp(L)**self.Params['alphaU']
                 /(N + self.Params['betaU']*np.exp(L)**self.Params['alphaU']) - 1)\
               *np.exp(L)**self.Params['alphaU']/(N + self.Params['betaU']*np.exp(L)**self.Params['alphaU'])**2
        d2r2 = tt2*self.Params['alphaG']**2*np.exp(L)**self.Params['alphaG']/(Ped + self.Params['Kp'])
        d2r3 = self.Params['betav']*self.Params['alphav']**2*np.exp(L)**self.Params['alphav']\
               /self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        d1 = d1r0-d1r2-d1r3
        d2 = d2r0-d2r2-d2r3

        # Corrections of higher order moments 
        E = 0.5*V*d2
        EN = 0.5*V*d2r0
        EZ = self.Params['deltaZ'] * 0.5*V*d2r2 * P
        EZD = (1.-self.Params['deltaZ']) * 0.5*V*d2r2 * P

        # Skewness and kurtosis according to normal distribution
        m3 = 0.
        m4 = 3.

        # Assumptions on the immigrating community
        if self.AssumImm == 'P&L':
            # Default assumptions as used by Acevedo-Trejos et al. (2015) in Sci.Rep.
            self.Params['L_mean0'] = L
            Immigration = self.Params['deltaI'] * P
        elif self.AssumImm == 'L':
            self.Params['L_mean0'] = L
            Immigration = self.Params['deltaI']
        elif self.AssumImm == 'P':
            Immigration = self.Params['deltaI'] * P
        elif self.AssumImm == 'None':
            Immigration = self.Params['deltaI']

        # state variables
        dxdt[0] = -P * (Gains + EN) + Mineralization + NMixing  # Nutrients
        dxdt[1] = ZooGrowth + EZ - ZooMortality - ZooMixing  # Zooplankton
        dxdt[2] = UnassimilatedProduction + P * OtherPMortalities + EZD + ZooMortality - Mineralization - DetritusMixing  # Detritus
        dxdt[3] = P * (Gains - Losses + E) + Immigration  # Phytoplankton
        dxdt[4] = V*d1 + m3*0.5*d2 + Immigration/P*(self.Params['L_mean0']-L)  # Mean Size
        dxdt[5] = 0.5*(m4-1.)*V**2*d2 + Immigration/P*((self.Params['L_var0']-V) + (self.Params['L_mean0']-L)**2)  # Size Variance

        # Biomass Fluxes
        dxdt[6] = P * Gains  # gross growth
        dxdt[7] = NutrientUptake  # Nutrient Uptake
        dxdt[8] = LightHarvesting  # Light Harvesting
        dxdt[9] = TemperatureDepGrowth  # Phytoplankton temperature dependency
        dxdt[10] = P * Grazing  # Grazing
        dxdt[11] = P * Sinking  # Phytoplankton Sinking
        dxdt[12] = P * OtherPMortalities  # Other P mortalities
        dxdt[13] = P * Mixing  # Phytoplankton Mixing
        dxdt[14] = ZooGrowth  # Zooplankton Growth
        dxdt[15] = ZooMortality  # Zooplankton Predator Mortality
        dxdt[16] = ZooMixing  # Zooplankton Mixing
        dxdt[17] = UnassimilatedProduction  # Unassimilated Production
        dxdt[18] = Mineralization  # Mineralization
        dxdt[19] = DetritusMixing  # Detritus Mixing
        dxdt[20] = NMixing  # Nutrients Mixing
        dxdt[21] = Immigration  # Immigration
        dxdt[22] = P*E  # P Higher order correction
        dxdt[23] = P*EN  # N Higher order correction
        dxdt[24] = EZ  # Grazing Higher order correction
        dxdt[25] = EZD  # Unassimilated Production Higher order correction
        dxdt[26] = V*d1  # Changes in mean size
        dxdt[27] = Immigration/P*(self.Params['L_mean0']-L)  # Immigrating mean size
        dxdt[28] = 0.5*(m4-1.)*V**2*d2  # Changes in size variance
        dxdt[29] = Immigration/P*((self.Params['L_var0']-V) + (self.Params['L_mean0']-L)**2)  # Immigrating size variance
        dxdt[30] = self.Params['betaU']*np.exp(L)**self.Params['alphaU']  # N Half Saturation
        dxdt[31] = d2  # second derivative with respect to trait
        dxdt[32] = d2r0  # second derivative of Nutrient Uptake with respect to the trait
        dxdt[33] = d2r2  # second derivative of Grazing with respect to the trait
        dxdt[34] = d2r3  # second derivative of Sinking with respect to the trait

        return dxdt

    def sizemodel_traitdif(self, x, t):
        """
        This size based model variant is based on Acevedo-Trejos et al. (2015) in Sci. Rep.
        but with a trait diffusion mechanism to sustain size variance as
        suggested by Merico et al. (2014) in Frontiers in Ecology and Evolution.

        Parameters
        ----------
        x: array with initial conditions for the state variables
        t: time
        """
        # Initialization of state variables and dummy variables to store the biomass fluxes
        N = x[0]
        Z = x[1]
        D = x[2]
        P = x[3]
        L = x[4]
        V = x[5]
        dxdt = np.zeros(35)

        # Edible phytoplankton
        Ped = P*(np.exp(L)**self.Params['alphaG']+0.5*V*self.Params['alphaG']**2*np.exp(L)**self.Params['alphaG'])

        # Gains of phytoplankton biomass
        NutrientUptake = N/(N+self.Params['betaU']*np.exp(L)**self.Params['alphaU'])
        LightHarvesting = 1./(self.Params['kw'] * self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld))\
                          * (-np.exp(1. - self.PAR.dailyinterp(t, kind=self.kindpar, k=self.kpar, s=self.spar)
                            / self.Params['OptI']) - (-np.exp((1. - (self.PAR.dailyinterp(t, kind=self.kindpar,
                            k=self.kpar, s=self.spar) * np.exp(-self.Params['kw'] * self.MLD.dailyinterp(t,
                            kind=self.kindmld, k=self.kmld, s=self.smld))) / self.Params['OptI']))))
        TemperatureDepGrowth = np.exp(0.063 * self.SST.dailyinterp(t, kind=self.kindsst, k=self.ksst, s=self.ssst))
        Gains = self.Params['muP'] * NutrientUptake * LightHarvesting * TemperatureDepGrowth

        # Losses of phytoplankton biomass
        Grazing = self.Params['muZ']*Z*np.exp(L)**self.Params['alphaG']/(Ped+self.Params['Kp'])
        Sinking = (self.Params['betav']*np.exp(L)**self.Params['alphav'])\
                  /self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        OtherPMortalities = self.Params['moP']
        Mixing = (self.Params['kappa'] + max(self.MLD.firstderivspl(t, k=self.kmld, s=self.smld), 0.)) \
                 / self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        Losses = Grazing + Sinking + OtherPMortalities + Mixing

        # Other Processes
        ZooGrowth = self.Params['deltaZ'] * Grazing * P
        ZooMortality = self.Params['moZ'] * Z**2
        ZooMixing = Z * self.MLD.firstderivspl(t, k=self.kmld, s=self.smld) \
                    / self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        UnassimilatedProduction = (1.-self.Params['deltaZ']) * Grazing * P
        Mineralization = self.Params['deltaD']*D
        DetritusMixing = D * (self.Params['kappa'] + max(self.MLD.firstderivspl(t, k=self.kmld, s=self.smld), 0.)) \
                         / self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        NMixing = Mixing * (self.N0X.dailyinterp(t, kind=self.kindn0x, k=self.kn0x, s=self.sn0x) - N)

        # Derivatives for the growth components of phytoplankton with respect to the trait
        tt0 = self.Params['muP']*LightHarvesting*TemperatureDepGrowth
        tt2 = self.Params['muZ']*Z
        d1r0 = -N*tt0*self.Params['betaU']*self.Params['alphaU']*np.exp(L)**self.Params['alphaU']\
               /(N + self.Params['betaU']*np.exp(L)**self.Params['alphaU'])**2
        d1r2 = tt2*self.Params['alphaG']*np.exp(L)**self.Params['alphaG']/(Ped + self.Params['Kp'])
        d1r3 = self.Params['betav']*self.Params['alphav']*np.exp(L)**self.Params['alphav']\
               /self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        d2r0 = N*tt0*self.Params['betaU']*self.Params['alphaU']**2*(2*self.Params['betaU']
                *np.exp(L)**self.Params['alphaU']/(N + self.Params['betaU']*np.exp(L)**self.Params['alphaU'])
                - 1)*np.exp(L)**self.Params['alphaU']/(N + self.Params['betaU']*np.exp(L)**self.Params['alphaU'])**2
        d2r2 = tt2*self.Params['alphaG']**2*np.exp(L)**self.Params['alphaG']/(Ped + self.Params['Kp'])
        d2r3 = self.Params['betav']*self.Params['alphav']**2*np.exp(L)**self.Params['alphav']\
               /self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        d3r0 = N*self.Params['betaU']*self.Params['alphaU']**3*tt0*(-6*self.Params['betaU']**2
                *np.exp(L)**(2*self.Params['alphaU'])/(N + self.Params['betaU']*np.exp(L)**self.Params['alphaU'])**2
                + 6*self.Params['betaU']*np.exp(L)**self.Params['alphaU']
                /(N + self.Params['betaU']*np.exp(L)**self.Params['alphaU']) - 1)\
               *np.exp(L)**self.Params['alphaU']/(N + self.Params['betaU']*np.exp(L)**self.Params['alphaU'])**2
        d4r0 = N*self.Params['betaU']*self.Params['alphaU']**4*tt0*(24*self.Params['betaU']**3
                *np.exp(L)**(3*self.Params['alphaU'])/(N + self.Params['betaU']*np.exp(L)**self.Params['alphaU'])**3
                - 36*self.Params['betaU']**2*np.exp(L)**(2*self.Params['alphaU'])/(N + self.Params['betaU']
                *np.exp(L)**self.Params['alphaU'])**2 + 14*self.Params['betaU']*np.exp(L)**self.Params['alphaU']
                /(N + self.Params['betaU']*np.exp(L)**self.Params['alphaU']) - 1)*np.exp(L)**self.Params['alphaU']\
               /(N + self.Params['betaU']*np.exp(L)**self.Params['alphaU'])**2
        d1 = d1r0-d1r2-d1r3
        d2 = d2r0-d2r2-d2r3

        # Corrections of higher order moments 
        E = 0.5*V*d2
        EN = 0.5*V*d2r0
        EZ = self.Params['deltaZ'] * 0.5*V*d2r2 * P
        EZD = (1.-self.Params['deltaZ']) * 0.5*V*d2r2 * P

        # Skewness and kurtosis according to normal distribution
        m3 = 0.
        m4 = 3.

        # state variables
        dxdt[0] = -P * (Gains + EN) + Mineralization + NMixing  # Nutrients
        dxdt[1] = ZooGrowth + EZ - ZooMortality - ZooMixing  # Zooplankton
        dxdt[2] = UnassimilatedProduction + P * OtherPMortalities + EZD + ZooMortality - Mineralization - DetritusMixing  # Detritus
        dxdt[3] = P * (Gains - Losses + E + 0.5*self.Params['nu']*(V*d4r0-3.*d2r0))  # Phytoplankton
        dxdt[4] = V*d1 + m3*0.5*d2 + self.Params['nu']*(V*d3r0-3.*d1r0)  # Mean Size
        dxdt[5] = 0.5*(m4-1.)*V**2*d2 + self.Params['nu']*(V**2*d4r0-5.*V*d2r0+2.*Gains)  # Size Variance

        # Biomass Fluxes
        dxdt[6] = P * Gains  # gross growth
        dxdt[7] = NutrientUptake  # Nutrient Uptake
        dxdt[8] = LightHarvesting  # Light Harvesting
        dxdt[9] = TemperatureDepGrowth  # Phytoplankton Temperature Dependency
        dxdt[10] = P * Grazing  # Grazing
        dxdt[11] = P * Sinking  # Phytoplankton Sinking
        dxdt[12] = P * OtherPMortalities  # Other P mortalities
        dxdt[13] = P * Mixing  # Phytoplankton Mixing
        dxdt[14] = ZooGrowth  # Zooplankton Growth
        dxdt[15] = ZooMortality  # Zooplankton Predator Mortality
        dxdt[16] = ZooMixing  # Zooplankton Mixing
        dxdt[17] = UnassimilatedProduction  # Unassimilated Production
        dxdt[18] = Mineralization  # Mineralization
        dxdt[19] = DetritusMixing  # Detritus Mixing
        dxdt[20] = NMixing  # Nutrients Mixing
        dxdt[21] = 0.5*self.Params['nu']*(V*d4r0-3.*d2r0)  # Trait Diffusion into P
        dxdt[22] = P*E  # P Higher order correction
        dxdt[23] = P*EN  # N Higher order correction
        dxdt[24] = EZ  # Grazing Higher order correction
        dxdt[25] = EZD  # Unassimilated Production Higher order correction
        dxdt[26] = V*d1  # Changes in mean size
        dxdt[27] = self.Params['nu']*(V*d3r0-3.*d1r0)  # Trait Diffusion into S
        dxdt[28] = 0.5*(m4-1.)*V**2*d2  # Changes in size variance
        dxdt[29] = self.Params['nu']*(V**2*d4r0-5.*V*d2r0+2.*Gains)  # Trait Diffusion into V
        dxdt[30] = self.Params['betaU']*np.exp(L)**self.Params['alphaU']  # N Half Saturation
        dxdt[31] = d2  # second derivative with respect to trait
        dxdt[32] = d2r0  # second derivative of Nutrient Uptake with respect to the trait
        dxdt[33] = d2r2  # second derivative of Grazing with respect to the trait
        dxdt[34] = d2r3  # second derivative of Sinking with respect to the trait

        return dxdt

    def sizemodel_fixvar(self, x, t):
        """
        This size based model variant is based on Acevedo-Trejos et al. (2015) in Sci. Rep.
        but with a fix size variance.

        Parameters
        ----------
        x: array with initial conditions for the state variables
        t: time
        """
        # Initialization of state variables and dummy variables to store the biomass fluxes
        N = x[0]
        Z = x[1]
        D = x[2]
        P = x[3]
        L = x[4]
        V = x[5]
        dxdt = np.zeros(35)

        # Edible phytoplankton
        Ped =P*(np.exp(L)**self.Params['alphaG']+0.5*V*self.Params['alphaG']**2*np.exp(L)**self.Params['alphaG'])

        # Gains of phytoplankton biomass
        NutrientUptake = N/(N+self.Params['betaU']*np.exp(L)**self.Params['alphaU'])
        LightHarvesting = 1./(self.Params['kw'] * self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld))\
                          * (-np.exp(1. - self.PAR.dailyinterp(t, kind=self.kindpar, k=self.kpar, s=self.spar)
                            / self.Params['OptI']) - (-np.exp((1. - (self.PAR.dailyinterp(t, kind=self.kindpar,
                            k=self.kpar, s=self.spar) * np.exp(-self.Params['kw'] * self.MLD.dailyinterp(t,
                            kind=self.kindmld, k=self.kmld, s=self.smld))) / self.Params['OptI']))))
        TemperatureDepGrowth = np.exp(0.063 * self.SST.dailyinterp(t, kind=self.kindsst, k=self.ksst, s=self.ssst))
        Gains = self.Params['muP'] * NutrientUptake * LightHarvesting * TemperatureDepGrowth

        # Losses of phytoplankton biomass
        Grazing = self.Params['muZ']*Z*np.exp(L)**self.Params['alphaG']/(Ped+self.Params['Kp'])
        Sinking = (self.Params['betav']*np.exp(L)**self.Params['alphav'])\
                  /self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        OtherPMortalities = self.Params['moP']
        Mixing = (self.Params['kappa'] + max(self.MLD.firstderivspl(t, k=self.kmld, s=self.smld), 0.)) \
                 / self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        Losses = Grazing + Sinking + OtherPMortalities + Mixing

        # Other Processes
        ZooGrowth = self.Params['deltaZ'] * Grazing * P
        ZooMortality = self.Params['moZ'] * Z**2
        ZooMixing = Z * self.MLD.firstderivspl(t, k=self.kmld, s=self.smld) \
                    / self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        UnassimilatedProduction = (1.-self.Params['deltaZ']) * Grazing * P
        Mineralization = self.Params['deltaD']*D
        DetritusMixing = D * (self.Params['kappa'] + max(self.MLD.firstderivspl(t, k=self.kmld, s=self.smld), 0.)) \
                         / self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        NMixing = Mixing * (self.N0X.dailyinterp(t, kind=self.kindn0x, k=self.kn0x, s=self.sn0x) - N)

        # Derivatives for the growth components of phytoplankton with respect to the trait
        tt0 = self.Params['muP']*LightHarvesting*TemperatureDepGrowth
        tt2 = self.Params['muZ']*Z
        d1r0 = -N*tt0*self.Params['betaU']*self.Params['alphaU']*np.exp(L)**self.Params['alphaU']\
               /(N + self.Params['betaU']*np.exp(L)**self.Params['alphaU'])**2
        d1r2 = tt2*self.Params['alphaG']*np.exp(L)**self.Params['alphaG']/(Ped + self.Params['Kp'])
        d1r3 = self.Params['betav']*self.Params['alphav']*np.exp(L)**self.Params['alphav']\
               /self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        d2r0 = N*tt0*self.Params['betaU']*self.Params['alphaU']**2*(2*self.Params['betaU']
                *np.exp(L)**self.Params['alphaU']/(N + self.Params['betaU']*np.exp(L)**self.Params['alphaU'])
                - 1)*np.exp(L)**self.Params['alphaU']/(N + self.Params['betaU']*np.exp(L)**self.Params['alphaU'])**2
        d2r2 = tt2*self.Params['alphaG']**2*np.exp(L)**self.Params['alphaG']/(Ped + self.Params['Kp'])
        d2r3 = self.Params['betav']*self.Params['alphav']**2*np.exp(L)**self.Params['alphav']\
               /self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        d1 = d1r0-d1r2-d1r3
        d2 = d2r0-d2r2-d2r3

        # Corrections of higher order moments 
        E = 0  # 0.5*V*d2
        EN = 0  # 0.5*V*d2r0
        EZ = 0  # self.Params['deltaZ'] * 0.5*V*d2r2 * P
        EZD = 0  # (1.-self.Params['deltaZ']) * 0.5*V*d2r2 * P

        # state variables
        dxdt[0] = -P * (Gains + EN) + Mineralization + NMixing  # Nutrients
        dxdt[1] = ZooGrowth + EZ - ZooMortality - ZooMixing  # Zooplankton
        dxdt[2] = UnassimilatedProduction + P * OtherPMortalities + EZD + ZooMortality - Mineralization - DetritusMixing  # Detritus
        dxdt[3] = P * (Gains - Losses + E)  # Phytoplankton
        dxdt[4] = V*d1  # Mean Size
        dxdt[5] = 0.0  # Size Variance

        # Biomass Fluxes
        dxdt[6] = P * Gains  # gross growth
        dxdt[7] = NutrientUptake  # Nutrient Uptake
        dxdt[8] = LightHarvesting  # Light Harvesting
        dxdt[9] = TemperatureDepGrowth  # Phytoplankton Temperature Dependency
        dxdt[10] = P * Grazing  # Grazing
        dxdt[11] = P * Sinking  # Phytoplankton Sinking
        dxdt[12] = P * OtherPMortalities  # Other P mortalities
        dxdt[13] = P * Mixing  # Phytoplankton Mixing
        dxdt[14] = ZooGrowth  # Zooplankton Growth
        dxdt[15] = ZooMortality  # Zooplankton Predator Mortality
        dxdt[16] = ZooMixing  # Zooplankton Mixing
        dxdt[17] = UnassimilatedProduction  # Unassimilated Production
        dxdt[18] = Mineralization  # Mineralization
        dxdt[19] = DetritusMixing  # Detritus Mixing
        dxdt[20] = NMixing  # Nutrients Mixing
        dxdt[21] = 0.  # No addition into P
        dxdt[22] = P*E  # P Higher order correction
        dxdt[23] = P*EN  # N Higher order correction
        dxdt[24] = EZ  # Grazing Higher order correction
        dxdt[25] = EZD  # Unassimilated Production Higher order correction
        dxdt[26] = V*d1  # Changes in mean size
        dxdt[27] = 0.  # No addition into S
        dxdt[28] = 0.  # Changes in size variance
        dxdt[29] = 0.  # No addition into V
        dxdt[30] = self.Params['betaU']*np.exp(L)**self.Params['alphaU']  # N Half Saturation
        dxdt[31] = 0.  # second derivative with respect to trait
        dxdt[32] = 0.  # second derivative of Nutrient Uptake with respect to the trait
        dxdt[33] = 0.  # second derivative of Grazing with respect to the trait
        dxdt[34] = 0.  # second derivative of Sinking with respect to the trait

        return dxdt

    def sizemodel_unvar(self, x, t):
        """
        This size based model variant is based on Acevedo-Trejos et al. (2015) in Sci. Rep.
        but without any mechanism to sustain variance.

        Parameters
        ----------
        x: array with initial conditions for the state variables
        t: time
        """
        # Initialization of state variables and dummy variables to store the biomass fluxes
        N = x[0]
        Z = x[1]
        D = x[2]
        P = x[3]
        L = x[4]
        V = x[5]
        dxdt = np.zeros(35)

        # Edible phytoplankton
        Ped = P*(np.exp(L)**self.Params['alphaG']+0.5*V*self.Params['alphaG']**2*np.exp(L)**self.Params['alphaG'])

        # Gains of phytoplankton biomass
        NutrientUptake = N/(N+self.Params['betaU']*np.exp(L)**self.Params['alphaU'])
        LightHarvesting = 1./(self.Params['kw'] * self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld))\
                          * (-np.exp(1. - self.PAR.dailyinterp(t, kind=self.kindpar, k=self.kpar, s=self.spar)
                            / self.Params['OptI']) - (-np.exp((1. - (self.PAR.dailyinterp(t, kind=self.kindpar,
                            k=self.kpar, s=self.spar) * np.exp(-self.Params['kw'] * self.MLD.dailyinterp(t,
                            kind=self.kindmld, k=self.kmld, s=self.smld))) / self.Params['OptI']))))
        TemperatureDepGrowth = np.exp(0.063 * self.SST.dailyinterp(t, kind=self.kindsst, k=self.ksst, s=self.ssst))
        Gains = self.Params['muP'] * NutrientUptake * LightHarvesting * TemperatureDepGrowth

        # Losses of phytoplankton biomass
        Grazing = self.Params['muZ']*Z*np.exp(L)**self.Params['alphaG']/(Ped+self.Params['Kp'])
        Sinking = (self.Params['betav']*np.exp(L)**self.Params['alphav'])\
                  /self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        OtherPMortalities = self.Params['moP']
        Mixing = (self.Params['kappa'] + max(self.MLD.firstderivspl(t, k=self.kmld, s=self.smld), 0.)) \
                 / self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        Losses = Grazing + Sinking + OtherPMortalities + Mixing

        # Other Processes
        ZooGrowth = self.Params['deltaZ'] * Grazing * P
        ZooMortality = self.Params['moZ'] * Z**2
        ZooMixing = Z * self.MLD.firstderivspl(t, k=self.kmld, s=self.smld) \
                    / self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        UnassimilatedProduction = (1.-self.Params['deltaZ']) * Grazing * P
        Mineralization = self.Params['deltaD']*D
        DetritusMixing = D * (self.Params['kappa'] + max(self.MLD.firstderivspl(t, k=self.kmld, s=self.smld), 0.)) \
                         / self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        NMixing = Mixing * (self.N0X.dailyinterp(t, kind=self.kindn0x, k=self.kn0x, s=self.sn0x) - N)

        # Derivatives for the growth components of phytoplankton with respect to the trait
        tt0 = self.Params['muP']*LightHarvesting*TemperatureDepGrowth
        tt2 = self.Params['muZ']*Z
        d1r0 = -N*tt0*self.Params['betaU']*self.Params['alphaU']*np.exp(L)**self.Params['alphaU']\
               /(N + self.Params['betaU']*np.exp(L)**self.Params['alphaU'])**2
        d1r2 = tt2*self.Params['alphaG']*np.exp(L)**self.Params['alphaG']/(Ped + self.Params['Kp'])
        d1r3 = self.Params['betav']*self.Params['alphav']*np.exp(L)**self.Params['alphav']\
               /self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        d2r0 = N*tt0*self.Params['betaU']*self.Params['alphaU']**2\
               *(2*self.Params['betaU']*np.exp(L)**self.Params['alphaU']
                 /(N + self.Params['betaU']*np.exp(L)**self.Params['alphaU']) - 1)\
               *np.exp(L)**self.Params['alphaU']/(N + self.Params['betaU']*np.exp(L)**self.Params['alphaU'])**2
        d2r2 = tt2*self.Params['alphaG']**2*np.exp(L)**self.Params['alphaG']/(Ped + self.Params['Kp'])
        d2r3 = self.Params['betav']*self.Params['alphav']**2*np.exp(L)**self.Params['alphav']\
               /self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        d1 = d1r0-d1r2-d1r3
        d2 = d2r0-d2r2-d2r3

        # Corrections of higher order moments 
        E = 0.5*V*d2
        EN = 0.5*V*d2r0
        EZ = self.Params['deltaZ'] * 0.5*V*d2r2 * P
        EZD = (1.-self.Params['deltaZ']) * 0.5*V*d2r2 * P

        # Skewness and kurtosis according to normal distribution
        m3 = 0.
        m4 = 3.

        # state variables
        dxdt[0] = -P * (Gains + EN) + Mineralization + NMixing  # Nutrients
        dxdt[1] = ZooGrowth + EZ - ZooMortality - ZooMixing  # Zooplankton
        dxdt[2] = UnassimilatedProduction + P * OtherPMortalities + EZD + ZooMortality - Mineralization - DetritusMixing  # Detritus
        dxdt[3] = P*(Gains - Losses + E)  # Phytoplankton
        dxdt[4] = V*d1 + m3*0.5*d2  # Mean Size
        dxdt[5] = 0.5*(m4-1.)*V**2*d2  # Size Variance

        # Biomass Fluxes
        dxdt[6] = P * Gains  # gross growth
        dxdt[7] = NutrientUptake  # Nutrient Uptake
        dxdt[8] = LightHarvesting  # Light Harvesting
        dxdt[9] = TemperatureDepGrowth  # Phytoplankton Temperature Dependency
        dxdt[10] = P * Grazing  # Grazing
        dxdt[11] = P * Sinking  # Phytoplankton Sinking
        dxdt[12] = P * OtherPMortalities  # Other P mortalities
        dxdt[13] = P * Mixing  # Phytoplankton Mixing
        dxdt[14] = ZooGrowth  # Zooplankton Growth
        dxdt[15] = ZooMortality  # Zooplankton Predator Mortality
        dxdt[16] = ZooMixing  # Zooplankton Mixing
        dxdt[17] = UnassimilatedProduction  # Unassimilated Production
        dxdt[18] = Mineralization  # Mineralization
        dxdt[19] = DetritusMixing  # Detritus Mixing
        dxdt[20] = NMixing  # Nutrients Mixing
        dxdt[21] = 0.  # No addition into P
        dxdt[22] = P*E  # P Higher order correction
        dxdt[23] = P*EN  # N Higher order correction
        dxdt[24] = EZ  # Grazing Higher order correction
        dxdt[25] = EZD  # Pellets Production Higher order correction
        dxdt[26] = V*d1  # Changes in mean size
        dxdt[27] = 0. # No addition into S
        dxdt[28] = 0.5*(m4-1.)*V**2*d2  # Changes in size variance
        dxdt[29] = 0.  # No addition into V
        dxdt[30] = self.Params['betaU']*np.exp(L)**self.Params['alphaU']  # N Half Saturation
        dxdt[31] = d2  # second derivative with respect to trait
        dxdt[32] = d2r0  # second derivative of Nutrient Uptake with respect to the trait
        dxdt[33] = d2r2  # second derivative of Grazing with respect to the trait
        dxdt[34] = d2r3  # second derivative of Sinking with respect to the trait

        return dxdt

    def fullmodel(self, x, t):
        """
        This size based model variant calculates structural changes of a community
        composed by n number of morphologically different phytoplankton.

        Parameters
        ----------
        x: array with initial conditions for the state variables
        t: time
        """
        N = x[0]
        Z = x[1]
        D = x[2]
        Ps = x[3:]
        dxdt = np.zeros(len(x))
        sizerange = np.linspace(np.log(self.Params['sizemin']), np.log(self.Params['sizemax']),
                                num=self.Params['NoMtype'])

        # Edible phytoplankton
        Ped = np.sum(Ps*np.exp(sizerange)**self.Params['alphaG'])

        # Non-P Processes
        Mixing = (self.Params['kappa'] + max(self.MLD.firstderivspl(t, k=self.kmld, s=self.smld), 0.)) \
                 / self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        ZooMortality = self.Params['moZ'] * Z**2
        ZooMixing = Z * self.MLD.firstderivspl(t, k=self.kmld, s=self.smld) \
                    / self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        Mineralization = self.Params['deltaD']*D
        DetritusMixing = D * (self.Params['kappa'] + max(self.MLD.firstderivspl(t, k=self.kmld, s=self.smld), 0.)) \
                         / self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
        NMixing = Mixing * (self.N0X.dailyinterp(t, kind=self.kindn0x, k=self.kn0x, s=self.sn0x) - N)

        dxdt[0] = Mineralization + NMixing
        dxdt[1] = - ZooMortality - ZooMixing
        dxdt[2] = ZooMortality - Mineralization - DetritusMixing
        for i in range(len(Ps)):
            S = sizerange[i]
            P = Ps[i]

            # Gains of phytoplankton biomass
            NutrientUptake = N/(N+self.Params['betaU']*np.exp(S)**self.Params['alphaU'])
            LightHarvesting = 1. / (self.Params['kw'] * self.MLD.dailyinterp(t, kind=self.kindmld,
                                                                             k=self.kmld, s=self.smld)) \
                              * (-np.exp(1. - self.PAR.dailyinterp(t, kind=self.kindpar, k=self.kpar, s=self.spar)
                                / self.Params['OptI']) - (-np.exp((1. - (self.PAR.dailyinterp(t, kind=self.kindpar,
                                k=self.kpar, s=self.spar) * np.exp(-self.Params['kw'] * self.MLD.dailyinterp(t,
                                kind=self.kindmld, k=self.kmld, s=self.smld))) / self.Params['OptI']))))
            TemperatureDepGrowth = np.exp(0.063 * self.SST.dailyinterp(t, kind=self.kindsst, k=self.ksst, s=self.ssst))
            Gains = self.Params['muP'] * NutrientUptake * LightHarvesting * TemperatureDepGrowth
            # Losses of phytoplankton biomass
            Grazing = self.Params['muZ']*Z*np.exp(S)**self.Params['alphaG']/(Ped+self.Params['Kp'])
            Sinking = (self.Params['betav']*np.exp(S)**self.Params['alphav'])\
                      /self.MLD.dailyinterp(t, kind=self.kindmld, k=self.kmld, s=self.smld)
            OtherPMortalities = self.Params['moP']
            Losses = Grazing + Sinking + OtherPMortalities + Mixing
            # Other Processes
            ZooGrowth = self.Params['deltaZ']* Grazing * P
            UnassimilatedProduction = (1.-self.Params['deltaZ']) * Grazing * P

            dxdt[0] = dxdt[0] - P*Gains
            dxdt[1] = dxdt[1] + ZooGrowth
            dxdt[2] = dxdt[2] + UnassimilatedProduction + P*OtherPMortalities
            dxdt[3+i] = P*(Gains - Losses)

        return dxdt

    def modelrun(self):
        """
        Method to integrate a specific size-based model variant.
        The integration it is done with the module "integrate.odeint",
        from the library scipy.
        """
        try:
            if self.Model == "Imm":
                outarray = odeint(self.sizemodel_imm, self.initcond, self.timedays)
                return outarray
            elif self.Model == "TraitDif":
                outarray = odeint(self.sizemodel_traitdif, self.initcond, self.timedays)
                return outarray
            elif self.Model == "FixVar":
                outarray = odeint(self.sizemodel_fixvar, self.initcond, self.timedays)
                return outarray
            elif self.Model == "UnsustVar":
                outarray = odeint(self.sizemodel_unvar, self.initcond, self.timedays)
                return outarray
            elif self.Model == "FullModel":
                outarray = odeint(self.fullmodel, self.initcond, self.timedays, rtol=1e-12, atol=1e-12)
                return outarray
            raise Exception("InputError:", "it is not a valid size model variant. Please specify one model variant "
                                           "from: Imm, TraitDif, Fixvar, UnsustVar or FullModel.\n")
        except Exception as expt:
            print expt.args[0], self.Model, expt.args[1]
            raise

    def get_derivatives(self):
        """
        Symbolic solutions for the derivatives of the phytoplankton growth terms with
        respect to the trait using library sympy.
        """
        # variables definition
        tt0, N, a, S, b, Ped, c, d, tt2, e, f, M, lnS = sympy.symbols('tt0,N,a,S,b,Ped,c,d,tt2,e,f,M,lnS')
        S = sympy.exp(lnS)
        trait = lnS

        # derivatives of growth terms with respect to the trait
        NutrientUptake = tt0*N/(N+a*S**b)
        print 'NutrientUptake d=', sympy.diff(NutrientUptake, trait)
        print 'NutrientUptake d^2=', sympy.diff(NutrientUptake, trait, 2)
        print 'NutrientUptake d^3=', sympy.diff(NutrientUptake, trait, 3)
        print 'NutrientUptake d^4=', sympy.diff(NutrientUptake, trait, 4)

        Grazing = S**c/(Ped+d)*tt2
        print 'Grazing d=', sympy.diff(Grazing, trait)
        print 'Grazing d^2=', sympy.diff(Grazing, trait, 2)

        Sinking = e*S**f/M
        print 'Sinking d=', sympy.diff(Sinking, trait)
        print 'Sinking d^2=', sympy.diff(Sinking, trait, 2)

        PED = S**c
        print 'Edible P d^2=', sympy.diff(PED, trait, 2)
