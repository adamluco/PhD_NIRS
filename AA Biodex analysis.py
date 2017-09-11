# ------------------------------------------------------------ #
# This script Analyzes perfusion, cap range, mBF and mVO2 for AA study
#
# Revision History
# Developer:            Version:   Date:        Notes:
# Adam A. Lucero        1.0.0      25/04/2017   PhD Script
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
filepath = 'C:\\Users\\adaml\\Dropbox\\Adam\\Massey\\Research\\AA Study\\Data\\AA study data files\\Ex BF\\'
filename = '03-2.txt'
# ------------------- #  Editable variables
st_row = 51
cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
columns = ['Sample', 't1_O2Hb', 't1_HHb', 't1_tHb', 't2_O2Hb', 't2_HHb', 't2_tHb',
             't3_O2Hb', 't3_HHb', 't3_tHb', 'event']
Hz = 10 # Define sample rate
numstages = 2
sig = ['O2Hb', 'HHb', 'tHb'] # Define signals
tx = ['t1', 't2', 't3'] # Define transmitters
num_s, num_t = len(sig), len(tx) # Number of signals and transmitters
# ------------------- #  Data import and edit
data = bishop.datImp(filepath + filename, columns, st_row, cols) # Import data
data['time'] = data['Sample']/Hz # Create time column
data = bishop.levels(data, sig, tx) # Reindex and create column levels
# ------------------- # Stage selection
datPoints = bishop.datSeg(data, numstages*2, Hz, instr=0, reset=True) # Plots all data for stage definition
stages = bishop.stage_dfs(data, numstages, datPoints) # Creates list of stages
# ------------------- # Rest
numVO = 5
restOccs = bishop.datSeg(stages[0], numVO, Hz, instr=2) # Define occlusions
r_occs = bishop.rOccSeg(stages[0], restOccs, Hz) # Creates list of occlusions and calculates perfusion
r_voslps = bishop.pickVoccslp(r_occs, 'rest', numVO=numVO) # Select VO slope analysis segments
r_perf = bishop.restperf(data) # select resting perfusion segment
# ------------------- # WL 1
numVO = 3
WL1Occs = bishop.datSeg(stages[1], numVO, Hz, instr=2) # Define occlusions
WL1_occs, WL1_perfSeg = bishop.occSeg(stages[1], WL1Occs, Hz) # Put occlusion segments into a list
WL1_cap = bishop.caprange(WL1_perfSeg, 'tHb', 't2', 'WL1', Hz, mpd=1.5, delmax=[], delmin=[]) # Calculate tHb params
WL1_voslps = bishop.pickVoccslp(WL1_occs, 'WL1', numVO=numVO) # Select VO slope analysis segments
# ------------------- # Save to excel
writer = pd.ExcelWriter(filename[:-4]+'.xlsx', engine='xlsxwriter') # Write to excel
perf = [r_perf, WL1_cap] # Perfusion data
perf = pd.concat(perf, keys=['rest','WL1','WL2'])
perf.to_excel(writer, sheet_name='Perfusion')
# mBF data
mBF = [r_voslps, WL1_voslps]
mBF = pd.concat(mBF, keys=['rest','WL1','WL2'])
mBF.to_excel(writer, sheet_name='mBF')
# Close and Save
writer.save()
# ----------------------------fin---------------------------- #
%reset