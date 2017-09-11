# ------------------------------------------------------------ #
# This script Analyzes perfusion, cap range, mBF, mVO2, and  mitochondrial index (MI) tests
#
# Revision History
# Developer:            Version:   Date:        Notes:
# Adam A. Lucero        1.0.0      12/02/2017   PhD Script
# ------------------------------------------------------------ #
# ------------------- #  Import Statements
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import pylab
from scipy import stats
from scipy.optimize import curve_fit
import bishop
import pickle
# ------------------- #  Define files
filepath = 'C:\\Users\\adaml\\Dropbox\\Adam\\Massey\\Research\\Reproducibility study\\Data\\Artinis REP\\T2D\\'
filename = '01NB-D-REP-3.txt'
# ------------------- #  Editable variables
st_row = 51
cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
columns = ['Sample', 't1_O2Hb', 't1_HHb', 't1_tHb', 't2_O2Hb', 't2_HHb', 't2_tHb',
             't3_O2Hb', 't3_HHb', 't3_tHb', 'TSI', 'event']
Hz = 10 # Define sample rate
numstages = 3
sig = ['O2Hb', 'HHb', 'tHb'] # Define signals
tx = ['t1', 't2', 't3'] # Define transmitters
act_sig = 'cHbDif' # Define active signal
# ------------------- #  Data import and edit
data = bishop.datImp(filepath + filename, columns, st_row, cols) # Import data
data['time'] = data['Sample']/Hz # Create time column
data = bishop.levels(data) # Reindex and create column levels
data = bishop.hbdif(data) # Calculate and add HbDif columns
data = bishop.chb(data) # Correct signals for BV movement and add to DataFrame
# ------------------- # Stage selection
datPoints = bishop.datSeg(data, numstages*2, Hz, instr=0, reset=True) # Plots all data for stage definition
stages = bishop.stage_dfs(data, numstages, datPoints) # Creates list of stages
# ------------------- # Rest
restOccs = bishop.datSeg(stages[0], 6, Hz, instr=2) # Define occlusions
r_occs = bishop.rOccSeg(stages[0], restOccs, Hz) # Creates list of occlusions and calculates perfusion
r_voslps = bishop.pickVoccslp(r_occs, 'rest') # Select VO slope analysis segments
r_aoslps = bishop.pickAoccslp(r_occs, 'rest') # Select AO slope analysis segments
r_perf = bishop.restperf(data) # select resting perfusion segment
# ------------------- # WL 1
WL1Occs = bishop.datSeg(stages[1], 6, Hz, instr=2) # Define occlusions
WL1_occs, WL1_perfSeg = bishop.occSeg(stages[1], WL1Occs, Hz) # Put occlusion segments into a list
WL1_cap = bishop.caprange(WL1_perfSeg, 'tHb', 't2', 'WL1', Hz, mpd=3.6, delmax=[], delmin=[]) # Calculate tHb params
WL1_voslps = bishop.pickVoccslp(WL1_occs, 'WL1') # Select VO slope analysis segments
WL1_aoslps = bishop.pickAoccslp(WL1_occs, 'WL1') # Select AO slope analysis segments
# ------------------- # WL 2
WL2Occs = bishop.datSeg(stages[2], 6, Hz, instr=2) # Define occlusions
WL2_occs, WL2_perfSeg = bishop.occSeg(stages[2], WL2Occs, Hz) # Put occlusion segments into a list
WL2_cap = bishop.caprange(WL2_perfSeg, 'tHb', 't2', 'WL2', Hz, mpd=3.6, delmax=[], delmin=[]) # Calculate tHb params
WL2_voslps = bishop.pickVoccslp(WL2_occs, 'WL2') # Select VO slope analysis segments
WL2_aoslps = bishop.pickAoccslp(WL2_occs, 'WL2') # Select AO slope analysis segments
# ------------------- #  Plot data and define mito segs
points = bishop.mito_segs(data, 'cHbDif', datPoints[-1]) # Define MI segments
MI = bishop.stage_dfs(data, 2, points) # Puts MI tests into list
# ------------------- #  Define mito timing
occMarks = np.array([ 0, 0.5, 15, 16.5, 30, 31.5, 46, 48, 64, 66.5, 82, 85, 100.5, 105, 121.5, 126, 144, 148.5, 166,
                  171.5, 188, 192.5, 211, 217, 251, 257, 292, 298]) * Hz # Diabetic AO marks
occMarks = np.array([ 0, 0.5, 12, 13, 24, 25.5, 36, 38, 53, 55.5, 69, 72, 85.5, 90, 106.5, 111, 127.5, 132, 148.5,
                  153, 169.5, 174, 191, 197, 230.5, 237, 270.5, 277]) * Hz # Healthy AO marks
#[ 0, 0.5, 12, 13.5, 24, 25.5, 37, 39, 53, 55.5, 69, 72, 85.5, 90, 106.5, 111, 127.5, 132, 148.5,
                 # 153, 169.5, 174, 191, 197, 231, 237, 271, 277]
# ------------------- #  MIa
start_a = bishop.aostart(MI[0], act_sig, Hz=Hz) # Defines 1st AO inflation
mark_a = bishop.occmarks(MI[0], occMarks, start_a, act_sig) # Plots MI with slope analysis segments
## mark_a[27] = 29670 # Change analysis window if necessary
# ------------------- ## Manual occlusion selection
numAO = 14
plt.plot(MI[0]['cHbDif'])
mark_a = bishop.pickpoint(MI[0]['cHbDif'], numAO*2)
# ------------------- #
slps_a = bishop.mitoslopes(MI[0], mark_a, act_sig) # Calculates slope values
# slps_a = slps_a.drop(0) # Drop a slope value if necessary
fitParams_a, fitcovs_a = bishop.curvefit(slps_a, filename[:-4], '_MIa') # Fits and plots curves
# ------------------- #  MIb
start_b = bishop.aostart(MI[1], act_sig, Hz=Hz) # Defines 1st AO inflation
mark_b = bishop.occmarks(MI[1], occMarks, start_b, act_sig) # plots MI with slope analysis windows
## mark_b[] =  # Change analysis window if necessary
# ------------------- ## Manual occlusion selection
numAO = 14
plt.plot(MI[1]['cHbDif'])
mark_b = bishop.pickpoint(MI[1]['cHbDif'], numAO*2)
# ------------------- #
slps_b = bishop.mitoslopes(MI[1], mark_b, act_sig)  # Calculates slope values
# slps_b = slps_b.drop(0)  # Drop a slope value if necessary
fitParams_b, fitcovs_b = bishop.curvefit(slps_b, filename[:-4], '_MIb')   # Fits and plots curves
# ------------------- # Save to excel
writer = pd.ExcelWriter(filename[:-4]+'.xlsx', engine='xlsxwriter') # Write to excel
perf = [r_perf, WL1_cap, WL2_cap] # Perfusion data
perf = pd.concat(perf, keys=['rest','WL1','WL2'])
perf.to_excel(writer, sheet_name='Perfusion')
# mBF data
mBF = [r_voslps, WL1_voslps, WL2_voslps]
mBF = pd.concat(mBF, keys=['rest','WL1','WL2'])
mBF.to_excel(writer, sheet_name='mBF')
# mVO2 data
mVO2 = [r_aoslps, WL1_aoslps, WL2_aoslps]
mVO2 = pd.concat(mVO2, keys=['rest','WL1','WL2'])
mVO2.to_excel(writer, sheet_name='mVO2')
# mito data
mito = [slps_a, slps_b, fitParams_a, fitParams_b]
mito = pd.concat(mito, keys=['MIa','MIb','fit_a','fit_b'])
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
imp.reload(bishop)
