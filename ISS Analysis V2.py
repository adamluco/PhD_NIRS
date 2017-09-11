# ------------------------------------------------------------ #
# This script Analyzes ISS muscle protocol data
#
# Revision History
# Developer:            Version:   Date:        Notes:
# Adam A. Lucero        2.0.0      12/02/2017   PhD Script
# Adam A. Lucero        2.1.0      12/02/2017   Bug fixes and perfusion fix
# ------------------------------------------------------------ #
# ------------------- #  Import Statements
import pandas as pd
import ISS
# ------------------- #  Define files
filepath = 'C:\\Users\\adamluco\\Dropbox\\adam\\Massey\\Research\\Reproducibility study\\Data\\ISS REP\\ND\\'
filename = '01-1.txt'
# ------------------- #  Editable variables
# Define sample rate (Hz)
Hz = 50
# Define number of stages
numstages = 3
# Define rolling average window (in ms)
rolWin = 500 * Hz//1000
# ------------------- #  Data import
data = pd.read_table(filepath + filename, header=None, skiprows=[0, 1, 2], low_memory=False, usecols=[0,1,2,3,4,5,6,7,8])
data.columns = ([['T']*3 + ['SlopeAC']*2 + ['SlopeDC']*2 + ['SlopePh']*2,
                                ['raw time', 'time', 'marker'] + ['wl1', 'wl2']*3])
# ------------------- # Experimental set up
# Calculate constant Hb signals for AC and DC slopes
data, r_scat = ISS.restScat(data, rolWin)
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
r_voSlps, r_voScat = ISS.pickVoccslp(r_occs, 'rest', 'Hb_csDC', Hz, rolWin, numVO=numVO)
# Select and analyze AO slope analysis segments
r_aoSlps, r_aoScat = ISS.pickAoccslp(r_occs, 'rest', 'Hb_csDC', 'HHb', Hz, rolWin, numVO, numAO)
# ------------------- # Exercise 1
# Define number of VOs and AOs
numVO, numAO = 5, 2
# Define occlusions
ex1pts = ISS.datSeg(stages[1], 'Hb_csDC', numVO, numAO, instr=3)
# Create occlusion and perfusion segments
ex1Occs, ex1PerfSeg = ISS.occSeg(stages[1], ex1pts, Hz)
# Calculate cap data for cs
ex1Cap_cs, ex1PerfPts = ISS.caprange_calc(ex1PerfSeg, 'ex1', Hz, mpd=3.8, delmax=[ ], delmin=[ ])
# Calculate cap data for rolscat
ex1Cap_rol = ISS.caprange_calc_rol(ex1PerfSeg, rolWin, 'ex1', Hz, ex1PerfPts, mpd=3.8, delmax=[ ], delmin=[5 ])
# Combine cap data
ex1Cap = pd.concat([ex1Cap_cs, ex1Cap_rol], axis=1)
# Select and analyze VO slope analysis segments
ex1_voSlps, ex1_voScat = ISS.pickVoccslp(ex1Occs, 'ex1', 'Hb_csDC', Hz, rolWin, numVO=numVO)
# Select and analyze AO slope analysis segments
ex1_aoSlps, ex1_aoScat = ISS.pickAoccslp(ex1Occs, 'ex1', 'Hb_csDC', 'HHb', Hz, rolWin, numVO, numAO)
# ------------------- # Exercise 2
# Define number of VOs and AOs
numVO, numAO =3, 2
# Define occlusions
ex2pts = ISS.datSeg(stages[2], 'Hb_csDC', numVO, numAO, instr=3)
# Create occlusion and perfusion segments
ex2Occs, ex2PerfSeg = ISS.occSeg(stages[2], ex2pts, Hz)
# Calculate cap data for cs
ex2Cap_cs, ex2PerfPts = ISS.caprange_calc(ex2PerfSeg, 'ex2', Hz, mpd=3.8, delmax=[ ], delmin=[3,5 ])
# Calculate cap data for rolscat
ex2Cap_rol = ISS.caprange_calc_rol(ex2PerfSeg, rolWin, 'ex2', Hz, ex2PerfPts, mpd=3, delmax=[  11], delmin=[ 0,10])
# Combine cap dataee
ex2Cap = pd.concat([ex2Cap_cs, ex2Cap_rol], axis=1)
# Select and analyze VO slope analysis segments
ex2_voSlps, ex2_voScat = ISS.pickVoccslp(ex2Occs, 'ex2', 'Hb_csDC', Hz, rolWin, numVO=numVO)
# Select and analyze AO slope analysis segments
ex2_aoSlps, ex2_aoScat = ISS.pickAoccslp(ex2Occs, 'ex2', 'Hb_csDC', 'HHb', Hz, rolWin, numVO, numAO)
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
MIa_pts = ISS.mito_occ(MI[0], 'Hb_csDC', numAO)
# Calculates and plots slope values
slps_a, scat_a = ISS.mitoslopes(MI[0], MIa_pts, 'MIa', Hz, rolWin)
ISS.plot_mi(slps_a, 'MIa')
# ------------------- ## MIb occlusion selection
# Put occlusions into individual dfs
MIb_pts = ISS.mito_occ(MI[1], 'Hb_csDC', numAO)
# Calculates and plots slope values
slps_b, scat_b = ISS.mitoslopes(MI[1], MIb_pts, 'MIb', Hz, rolWin)
ISS.plot_mi(slps_b, 'MIb')
# ------------------- #
# Curve fit slopes for MIa
popt_a, pcov_a = [None]*4, [None]*4
popt_a[0], pcov_a[0] = ISS.curvefit_srce(slps_a, 'Hb_csAC', filename[:-4], 'MIa')
popt_a[1], pcov_a[1] = ISS.curvefit_srce(slps_a, 'Hb_csDC', filename[:-4], 'MIa')
popt_a[2], pcov_a[2] = ISS.curvefit_srce(slps_a, 'Hb_AC', filename[:-4], 'MIa')
popt_a[3], pcov_a[3] = ISS.curvefit_srce(slps_a, 'Hb_DC', filename[:-4], 'MIa')
# Curve fit slopes for MIb
popt_b, pcov_b = [None]*4, [None]*4
popt_b[0], pcov_b[0] = ISS.curvefit_srce(slps_b, 'Hb_csAC', filename[:-4], 'MIb')
popt_b[1], pcov_b[1] = ISS.curvefit_srce(slps_b, 'Hb_csDC', filename[:-4], 'MIb')
popt_b[2], pcov_b[2] = ISS.curvefit_srce(slps_b, 'Hb_AC', filename[:-4], 'MIb')
popt_b[3], pcov_b[3] = ISS.curvefit_srce(slps_b, 'Hb_DC', filename[:-4], 'MIb')
# ------------------- # Save to excel
# Write to excel
writer = pd.ExcelWriter(filename[:-4]+'.xlsx', engine='xlsxwriter')
# Perfusion data
perf = [r_perf, ex1Cap, ex2Cap]
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
# slopes data
slopes = [slps_a, slps_b]
slopes = pd.concat(slopes)
slopes.to_excel(writer, sheet_name='slopes')
# mito data
mito_a = pd.concat(popt_a)
mito_b = pd.concat(popt_b)
mito = pd.concat([mito_a, mito_b])
mito.to_excel(writer, sheet_name='mito')
fitCov_a = pd.concat(pcov_a)
fitCov_b = pd.concat(pcov_b)
fitCov = pd.concat([fitCov_a, fitCov_b])
fitCov.to_excel(writer, sheet_name='mito', startcol=7)
# Scaterring coefficients
scattering = [r_scat, r_voScat, r_aoScat, ex1_voScat, ex1_aoScat, ex2_voScat, ex2_aoScat, scat_a, scat_b]
scattering = pd.concat(scattering)
scattering.to_excel(writer, sheet_name='scat')
# Close and Save
writer.save()
# ----------------------------fin---------------------------- #
%reset

import imp
imp.reload(ISS)
