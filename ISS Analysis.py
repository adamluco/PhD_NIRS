# ------------------------------------------------------------ #
# This script Analyzes ISS muscle protocol data
#
# Revision History
# Developer:            Version:   Date:        Notes:
# Adam A. Lucero        2.0.0      12/02/2017   PhD Script
# ------------------------------------------------------------ #
# ------------------- #  Import Statements
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import ISS
import pickle
# ------------------- #  Define files
filepath = 'C:\\Users\\adaml\\Dropbox\\Adam\\Massey\\Research\\Reproducibility study\\Data\\ISS REP\\ND\\'
filename = '01AL-REP-1.txt'
# ------------------- #  Editable variables
# Define sample rate (Hz)
Hz = 50
# Define number of stages
numstages = 3
# Define rolling average window (in ms)
rolWin = 600 * Hz//1000
# ------------------- #  Data import
data = pd.read_table(filepath + filename, header=None, skiprows=[0, 1, 2], low_memory=False, usecols=[0,1,2,3,4,5,6,7,8])
data.columns = ([['T']*3 + ['SlopeAC']*2 + ['SlopeDC']*2 + ['SlopePh']*2,
                                ['raw time', 'time', 'marker'] + ['wl1', 'wl2']*3])
# ------------------- # Experimental set up
# Calculate constant Hb signals for AC and DC slopes
data = ISS.restScat(data, rolWin)
# Select stage segments
stagePts = ISS.datSeg(data, 'Hb_csDC', numstages*2, instr=1)
# Create stage dfs
stages = ISS.stage_dfs(data, numstages, stagePts, seg=2)
# ------------------- # Rest
# Define number of VOs and AOs
numVO, numAO = 4, 2
# Select occlusions
restPts = ISS.datSeg(stages[0], 'Hb_csDC', numVO, numAO, exP=4, instr=2)
# Creates list of occlusions
r_occs = ISS.rOccSeg(stages[0], restPts, Hz)
# Averages perfusion segments and creates df
r_perf = ISS.r_perfusion(stages[0], restPts, rolWin)
# Pick occlusion scattering window to average
r_occHb, r_scats = ISS.pickOccScat(stages[0], restPts[:-4], numVO, numAO, r_occs, rolWin)
# Select and analyze VO slope analysis segments
r_voSlps = ISS.pickVoccslp(r_occHb, 'rest', 'Hb_DC', Hz, numVO=numVO)
# Calculate HbDif and cHbDif sigs
r_occHb = ISS.hbdif_sigs(r_occHb, numVO, numAO)
# Select and analyze AO slope analysis segments
r_aoSlps = ISS.pickAoccslp(r_occHb, 'rest', 'Hb_DC', 'HHb', Hz, numVO, numAO)
# ------------------- # Exercise 1
# Define number of VOs and AOs
numVO, numAO = 4, 2
# Define occlusions
ex1pts = ISS.datSeg(stages[1], 'Hb_csDC', numVO, numAO, instr=3)
# Create occlusion and perfusion segments
ex1Occs, ex1PerfSeg = ISS.occSeg(stages[1], ex1pts, Hz)
# Calculate cap data for cs
ex1Cap_cs = ISS.caprange_calc(ex1PerfSeg, rolWin, 'ex1', Hz, sigs=1, mpd=3.8, delmax=[], delmin=[])
# Calculate cap data for rolscat
ex1Cap_rol = ISS.caprange_calc(ex1PerfSeg, rolWin, 'ex1', Hz, sigs=2, mpd=3.8, delmax=[1], delmin=[])
# Combine cap data
ex1Cap = pd.concat([ex1Cap_cs, ex1Cap_rol], axis=1)
# Pick occlusion scattering window to average
ex1_occHb, ex1_scats = ISS.pickOccScat(stages[1], ex1pts, numVO, numAO, ex1Occs, rolWin)
# Select and analyze VO slope analysis segments
ex1_voSlps = ISS.pickVoccslp(ex1_occHb, 'ex1', 'Hb_DC', Hz, numVO=numVO)
# Calculate HbDif and cHbDif sigs
ex1_occHb = ISS.hbdif_sigs(ex1_occHb, numVO, numAO)
# Select and analyze AO slope analysis segments
ex1_aoSlps = ISS.pickAoccslp(ex1_occHb, 'ex1', 'Hb_DC', 'HbDif', Hz, numVO, numAO)
# ------------------- # Exercise 2
# Define number of VOs and AOs
numVO, numAO = 4, 2
# Define occlusions
ex2pts = ISS.datSeg(stages[2], 'Hb_csDC', numVO, numAO, instr=3)
# Create occlusion and perfusion segments
ex2Occs, ex2PerfSeg = ISS.occSeg(stages[2], ex2pts, Hz)
# Calculate cap data for cs
ex2Cap_cs = ISS.caprange_calc(ex2PerfSeg, rolWin, 'ex2', Hz, sigs=1, mpd=3.8, delmax=[], delmin=[])
# Calculate cap data for rolscat
ex2Cap_rol = ISS.caprange_calc(ex2PerfSeg, rolWin, 'ex2', Hz, sigs=2, mpd=3.8, delmax=[], delmin=[])
# Combine cap data
ex2Cap = pd.concat([ex2Cap_cs, ex2Cap_rol], axis=1)
# Pick occlusion scattering window to average
ex2_occHb, ex2_scats = ISS.pickOccScat(stages[2], ex2pts, numVO, numAO, ex2Occs, rolWin)
# Select and analyze VO slope analysis segments
ex2_voSlps = ISS.pickVoccslp(ex2_occHb, 'ex2', 'Hb_DC', Hz, numVO=numVO)
# Calculate HbDif and cHbDif sigs
ex2_occHb = ISS.hbdif_sigs(ex2_occHb, numVO, numAO)
# Select and analyze AO slope analysis segments
ex2_aoSlps = ISS.pickAoccslp(ex2_occHb, 'ex2', 'Hb_DC', 'HbDif', Hz, numVO, numAO)
# ------------------- #  Plot data and define mito segs
# Define number of MI tests
numMI = 2
# Select MI test segments
MIpts = ISS.datSeg(data.ix[stagePts[-1]:], 'Hb_csDC', numMI*2, instr=4)
# Places segments into Data Frames
MI = ISS.stage_dfs(data, numMI, MIpts, seg=2)
# Define number of AO in MI test, use 0 for indefinite AOs
numAO = 0
# ------------------- ## MIa occlusion selection
# Put occlusions into individual dfs
MIa_dfs = ISS.mito_occ(MI[0], 'Hb_csDC', numAO)
# Calculates and plots slope values
slps_a, scat_a = ISS.mitoslopes(MIa_dfs, rolWin, 'MIa', Hz)
# ------------------- ## MIb occlusion selection
# Put occlusions into individual dfs
MIb_dfs = ISS.mito_occ(MI[1], 'Hb_csDC', numAO)
# Calculates and plots slope values
slps_b, scat_b = ISS.mitoslopes(MIb_dfs, rolWin, 'MIb', Hz)
# ------------------- #


# slps_b = bishop.mitoslopes(MI[1], mark_b, act_sig)  # Calculates slope values
# slps_b = slps_b.drop(0)  # Drop a slope value if necessary
# fitParams_b, fitcovs_b = bishop.curvefit(slps_b, filename[:-4], '_MIb')  # Fits and plots curves
# ------------------- # Save to excel
writer = pd.ExcelWriter(filename[:-4]+'.xlsx', engine='xlsxwriter') # Write to excel
perf = [r_perf, ex1Cap, ex2Cap] # Perfusion data
perf = pd.concat(perf)
perf.to_excel(writer, sheet_name='Perfusion')
# mBF data
mBF = [r_voSlps, ex1_voSlps, ex2_voSlps]
mBF = pd.concat(mBF)
mBF.to_excel(writer, sheet_name='mBF')
# mVO2 data
mVO2 = [r_aoSlps, ex1_aoSlps, ex2_aoSlps]
mVO2 = pd.concat(mVO2)
mVO2.to_excel(writer, sheet_name='mVO2')
# mito data
mito = [slps_a, slps_b]
mito = pd.concat(mito)
mito.to_excel(writer, sheet_name='mito')
# Mito no Params
# mito = [slps_a, slps_b]
# mito = pd.concat(mito, keys=['MIa','MIb'])
mito.to_excel(writer, sheet_name='mito')
# Close and Save
writer.save()
# ----------------------------fin---------------------------- #
%reset

import imp
imp.reload(ISS)
