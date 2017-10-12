#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

from phytosfdm.EnvForcing.envforcing import ExtractEnvFor
from phytosfdm.SizeModels.sizemodels import SM

__version__ = "1.0.4"


def environemntalforcing(lat, lon, rbb):
    tos = time.time()
    print '\nComputation Time of Environmental Forcing'
    # Extract environmental forcing using the class ExtractEnvFor in the module envforcing
    MLD = ExtractEnvFor(lat, lon, rbb, 'mld')
    tos1 = time.time()
    print '    MLD=%4.3f seconds' % (tos1 - tos)
    PAR = ExtractEnvFor(lat, lon, rbb, 'par')
    tos2 = time.time()
    print '    PAR=%4.3f seconds' % (tos2 - tos1)
    SST = ExtractEnvFor(lat, lon, rbb, 'sst')
    tos3 = time.time()
    print '    SST=%4.3f seconds' % (tos3 - tos2)
    N0X = ExtractEnvFor(lat, lon, rbb, 'n0x')
    tos4 = time.time()
    print '    NOX=%4.3f seconds' % (tos4 - tos3)

    nt = np.arange(0., 366., 1.0)

    # Figure NO.1
    f1, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex='col', sharey='row')
    # MLD
    ax1.plot(nt, MLD.dailyinterp(nt), c='black', lw=3)
    ax1.set_ylabel('MLD (m)', multialignment='center', fontsize=12)
    ax1.set_xlim(0, 365)
    ax1.invert_yaxis()
    # PAR
    ax2.plot(nt, PAR.dailyinterp(nt), c='black', lw=3)
    ax2.set_ylabel('PAR \n' '(Ein / m^2 / d^1)', multialignment='center', fontsize=12)
    # SST
    ax3.plot(nt, SST.dailyinterp(nt, k=5), c='black', lw=3)
    ax3.set_ylabel('SST \n' '(degrees C)', multialignment='center', fontsize=12)
    # N0
    ax4.plot(nt, N0X.dailyinterp(nt, k=5), c='black', lw=3)
    ax4.set_ylabel('N0 \n' '(mmol N / m^3)', multialignment='center', fontsize=12)
    ax4.set_xlabel('Time (days)', fontsize=14)
    # defaultparams=False,listparams=[("timeyears",5),("muP",1.6), ('moZ', 0.02),('muZ', 1.35),('muC', 0.)]


def modelplot(lat, lon, rbb):
    sm_imm = SM(lat, lon, rbb, 'Imm')
    sm_imm_Zx2 = SM(lat, lon, rbb, 'Imm_Zx2')

    sm_imm_Zx2_1 = SM(lat, lon, rbb, "Imm_Zx2", defaultparams=False,
                      listparams=[("alphaG", -0.77), ('moC', 0.01), ('muC', .6), ('muH', 1.35), ('moH', 0.01)])

    sm_imm_Zx2_2 = SM(lat, lon, rbb, "Imm_Zx2", defaultparams=False,
                      listparams=[("alphaG", -0.8), ('moC', 0.01), ('muC', .6), ('muH', 1.35), ('moH', 0.01)])

    sm_imm_Zx2_3 = SM(lat, lon, rbb, "Imm_Zx2", defaultparams=False,
                      listparams=[("alphaG", -0.73), ('moC', 0.01), ('muC', .6), ('muH', 1.6), ('moH', 0.01)])

    nt = np.arange(0., 365., 1.0)
    runtime = int(sm_imm.Params['timeyears'])

    colors = ['#00C90D', '#01939A', 'black', 'grey', '#E2680B']
    alphas = [1., 0.8, 0.6, 0.4]
    lws = [1, 2.5, 4, 5.5]

    # artist for legends
    ImmArtist = plt.Line2D((0, 1), (0, 0), c=colors[1], alpha=alphas[0], lw=lws[1])
    Imm_Zx2Artist = plt.Line2D((0, 1), (0, 0), c=colors[0], alpha=alphas[0], lw=lws[0])
    Imm_Zx2Artist1 = plt.Line2D((0, 1), (0, 0), c=colors[2], alpha=alphas[0], lw=lws[0])
    Imm_Zx2Artist2 = plt.Line2D((0, 1), (0, 0), c=colors[3], alpha=alphas[0], lw=lws[0])
    Imm_Zx2Artist3 = plt.Line2D((0, 1), (0, 0), c=colors[4], alpha=alphas[0], lw=lws[0])

    # Figure NO.4
    f4, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex='col')
    # N
    ax1.plot(nt, sm_imm.outvariables[365 * (runtime - 1):365 * runtime, 0], c=colors[1], alpha=alphas[0], lw=lws[1])
    ax1.plot(nt, sm_imm_Zx2.outvariables[365 * (runtime - 1):365 * runtime, 0], c=colors[0], alpha=alphas[0], lw=lws[0])

    ax1.plot(nt, sm_imm_Zx2_1.outvariables[365 * (runtime - 1):365 * runtime, 0], c=colors[2], alpha=alphas[0],
             lw=lws[0])
    ax1.plot(nt, sm_imm_Zx2_2.outvariables[365 * (runtime - 1):365 * runtime, 0], c=colors[3], alpha=alphas[0],
             lw=lws[0])
    ax1.plot(nt, sm_imm_Zx2_3.outvariables[365 * (runtime - 1):365 * runtime, 0], c=colors[4], alpha=alphas[0],
             lw=lws[0])

    ax1.set_ylabel('Nutrients \n' '(mmol N / m^3)', multialignment='center', fontsize=10)
    # ax1.set_ylim(0, 12)
    ax1.tick_params('y', labelsize=10)
    ax1.set_xlim(0, 365)
    # P
    ax2.plot(nt, sm_imm.outvariables[365 * (runtime - 1):365 * runtime, 3], c=colors[1], alpha=alphas[0], lw=lws[1])
    ax2.plot(nt, sm_imm_Zx2.outvariables[365 * (runtime - 1):365 * runtime, 3], c=colors[0], alpha=alphas[0], lw=lws[0])

    ax2.plot(nt, sm_imm_Zx2_1.outvariables[365 * (runtime - 1):365 * runtime, 3], c=colors[2], alpha=alphas[0],
             lw=lws[0])
    ax2.plot(nt, sm_imm_Zx2_2.outvariables[365 * (runtime - 1):365 * runtime, 3], c=colors[3], alpha=alphas[0],
             lw=lws[0])
    ax2.plot(nt, sm_imm_Zx2_3.outvariables[365 * (runtime - 1):365 * runtime, 3], c=colors[4], alpha=alphas[0],
             lw=lws[0])

    ax2.set_ylabel('Phytoplankton \n' '(mmol N / m^3)', multialignment='center', fontsize=10)
    ax2.tick_params('y', labelsize=10)
    # Z + H
    ax3.plot(nt, sm_imm.outvariables[365 * (runtime - 1):365 * runtime, 1], c=colors[1], alpha=alphas[0], lw=lws[1])
    ax3.plot(nt, sm_imm_Zx2.outvariables[365 * (runtime - 1):365 * runtime, 1], c=colors[0], alpha=alphas[0], lw=lws[0])

    ax3.plot(nt, sm_imm_Zx2_1.outvariables[365 * (runtime - 1):365 * runtime, 1], c=colors[2], alpha=alphas[0],
             lw=lws[0])
    ax3.plot(nt, sm_imm_Zx2_2.outvariables[365 * (runtime - 1):365 * runtime, 1], c=colors[3], alpha=alphas[0],
             lw=lws[0])
    ax3.plot(nt, sm_imm_Zx2_3.outvariables[365 * (runtime - 1):365 * runtime, 1], c=colors[4], alpha=alphas[0],
             lw=lws[0])

    ax3.set_ylabel('Zoo+Herbsplankton \n' '(mmol N / m^3)', multialignment='center', fontsize=10)
    ax3.tick_params('y', labelsize=10)
    # D
    ax4.plot(nt, sm_imm.outvariables[365 * (runtime - 1):365 * runtime, 2], c=colors[1], alpha=alphas[0], lw=lws[1])
    ax4.plot(nt, sm_imm_Zx2.outvariables[365 * (runtime - 1):365 * runtime, 2], c=colors[0], alpha=alphas[0], lw=lws[0])

    ax4.plot(nt, sm_imm_Zx2_1.outvariables[365 * (runtime - 1):365 * runtime, 2], c=colors[2], alpha=alphas[0],
             lw=lws[0])
    ax4.plot(nt, sm_imm_Zx2_2.outvariables[365 * (runtime - 1):365 * runtime, 2], c=colors[3], alpha=alphas[0],
             lw=lws[0])
    ax4.plot(nt, sm_imm_Zx2_3.outvariables[365 * (runtime - 1):365 * runtime, 2], c=colors[4], alpha=alphas[0],
             lw=lws[0])

    ax4.set_ylabel('Detritus \n' '(mmol N / m^3)', multialignment='center', fontsize=10)
    ax4.tick_params('y', labelsize=10)
    ax4.set_xlabel(r'Time (days)', fontsize=14)
    # C TEST
    ax5.plot(nt, sm_imm_Zx2.outvariables[365 * (runtime - 1):365 * runtime, 35], c=colors[0], alpha=alphas[0],
             lw=lws[0])

    ax5.plot(nt, sm_imm_Zx2_1.outvariables[365 * (runtime - 1):365 * runtime, 35], c=colors[2], alpha=alphas[0],
             lw=lws[0])
    ax5.plot(nt, sm_imm_Zx2_2.outvariables[365 * (runtime - 1):365 * runtime, 35], c=colors[3], alpha=alphas[0],
             lw=lws[0])
    ax5.plot(nt, sm_imm_Zx2_3.outvariables[365 * (runtime - 1):365 * runtime, 35], c=colors[4], alpha=alphas[0],
             lw=lws[0])

    # ax5.set_ylim(-1, 5)
    ax5.set_ylabel('Carnivores \n' '(mmol N / m^3)', multialignment='center', fontsize=10)
    ax5.tick_params('y', labelsize=10)
    f4.legend([ImmArtist, Imm_Zx2Artist, Imm_Zx2Artist1, Imm_Zx2Artist2, Imm_Zx2Artist3],
              ['Immigration', 'Imm_Zx2', 'Imm_Zx2Artist1', 'Imm_Zx2Artist2', 'Imm_Zx2Artist3'], ncol=5,
              prop={'size': 10}, loc='upper center')

    # Figure NO.5
    f5, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col')
    # P
    ax1.plot(nt, sm_imm.outvariables[365 * (runtime - 1):365 * runtime, 3], c=colors[1], alpha=alphas[0], lw=lws[1])
    ax1.plot(nt, sm_imm_Zx2.outvariables[365 * (runtime - 1):365 * runtime, 3], c=colors[0], alpha=alphas[0], lw=lws[0])

    ax1.plot(nt, sm_imm_Zx2_1.outvariables[365 * (runtime - 1):365 * runtime, 3], c=colors[2], alpha=alphas[0],
             lw=lws[0])
    ax1.plot(nt, sm_imm_Zx2_2.outvariables[365 * (runtime - 1):365 * runtime, 3], c=colors[3], alpha=alphas[0],
             lw=lws[0])
    ax1.plot(nt, sm_imm_Zx2_3.outvariables[365 * (runtime - 1):365 * runtime, 3], c=colors[4], alpha=alphas[0],
             lw=lws[0])

    ax1.set_ylabel('Phytoplankton \n' '(mmol N / m^3)', multialignment='center', fontsize=12)
    ax1.tick_params('y', labelsize=12)
    ax1.set_xlim(0, 365)
    # S
    ax2.plot(nt, (sm_imm.outvariables[365 * (runtime - 1):365 * runtime, 4]), c=colors[1], alpha=alphas[0], lw=lws[1])
    ax2.plot(nt, (sm_imm_Zx2.outvariables[365 * (runtime - 1):365 * runtime, 4]), c=colors[0], alpha=alphas[0],
             lw=lws[0])

    ax2.plot(nt, (sm_imm_Zx2_1.outvariables[365 * (runtime - 1):365 * runtime, 4]), c=colors[2], alpha=alphas[0],
             lw=lws[0])
    ax2.plot(nt, (sm_imm_Zx2_2.outvariables[365 * (runtime - 1):365 * runtime, 4]), c=colors[3], alpha=alphas[0],
             lw=lws[0])
    ax2.plot(nt, (sm_imm_Zx2_3.outvariables[365 * (runtime - 1):365 * runtime, 4]), c=colors[4], alpha=alphas[0],
             lw=lws[0])

    ax2.set_ylabel('Mean Size \n' '(Ln microm ESD)', multialignment='center', fontsize=12)

    # ax2.set_ylim(0, 4)
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.tick_params('y', labelsize=12)
    # V
    ax3.plot(nt, (sm_imm.outvariables[365 * (runtime - 1):365 * runtime, 5]), c=colors[1], alpha=alphas[0], lw=lws[1])
    ax3.plot(nt, (sm_imm_Zx2.outvariables[365 * (runtime - 1):365 * runtime, 5]), c=colors[0], alpha=alphas[0],
             lw=lws[0])

    ax3.plot(nt, (sm_imm_Zx2_1.outvariables[365 * (runtime - 1):365 * runtime, 5]), c=colors[2], alpha=alphas[0],
             lw=lws[0])
    ax3.plot(nt, (sm_imm_Zx2_2.outvariables[365 * (runtime - 1):365 * runtime, 5]), c=colors[3], alpha=alphas[0],
             lw=lws[0])
    ax3.plot(nt, (sm_imm_Zx2_3.outvariables[365 * (runtime - 1):365 * runtime, 5]), c=colors[4], alpha=alphas[0],
             lw=lws[0])

    ax3.set_ylabel('Size Variance \n' '([Ln microm ESD]^2)', multialignment='center', fontsize=12)
    ax3.set_xlabel(r'Time (days)', fontsize=12)
    ax3.tick_params('both', labelsize=12)
    ax3.set_ylim(0, 0.8)
    # Legend
    f5.legend([ImmArtist, Imm_Zx2Artist, Imm_Zx2Artist1, Imm_Zx2Artist2, Imm_Zx2Artist3],
              ['Immigration', 'Imm_Zx2', 'Imm_Zx2Artist1', 'Imm_Zx2Artist2', 'Imm_Zx2Artist3'], ncol=5,
              prop={'size': 10}, loc='upper center')


def main_Zx2():
    print 'PhytoSFDM version:%s, Copyright (C) 2016 Esteban Acevedo-Trejos' % __version__
    print 'PhytoSFDM comes with ABSOLUTELY NO WARRANTY; for details see LICENSE.'
    print 'This is a free software, and you are welcome to redistributed it'
    print 'under certain conditions; see LICENSE for details.'

    # TODO An interactive user i/o

    # Geographical coordinates of modelled region.
    Latitude = 45.  # 90 to -90 degrees, North positive
    Longitude = -19.1  # -180 to 180 degrees, East positive
    RangeBoundingBox = 5.  # In degrees

    print '\nCarnivore Model variants calculated at test location: %.2f°N %.2f°W' % (Latitude, abs(Longitude))

    environemntalforcing(Latitude, Longitude, RangeBoundingBox)

    modelplot(Latitude, Longitude, RangeBoundingBox)

    plt.show()