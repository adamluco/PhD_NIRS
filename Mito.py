# ------------------------------------------------------------ #
# This script Analyzes mitochondrial index (MI) tests
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
from scipy import stats
from scipy.optimize import curve_fit
import bishop
import pickle
# ------------------- #  Define files
filepath = 'C:\\Users\\adaml\\Dropbox\\Adam\\Massey\\Research\\AA Study\\Data\\AA study data files\\MI\\'
filename = '26-2.txt'
# ------------------- #  Editable variables
st_row = 51
cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
columns = ['Sample', 't1_O2Hb', 't1_HHb', 't1_tHb', 't2_O2Hb', 't2_HHb', 't2_tHb',
             't3_O2Hb', 't3_HHb', 't3_tHb', 'event']
Hz = 10 # Define sample rate
numstages = 2
sig = ['O2Hb', 'HHb', 'tHb'] # Define signals
tx = ['t1', 't2', 't3'] # Define transmitters
act_sig = 'cHbDif' # Define active signal
# ------------------- #  Data import and edit
data = bishop.datImp(filepath + filename, columns, st_row, cols) # Import data
data['time'] = data['Sample']/Hz # Create time column
data = bishop.levels(data, sig, tx) # Reindex and create column levels
data = bishop.hbdif(data, tx) # Calculate and add HbDif columns
data = bishop.chb(data, tx) # Correct signals for BV movement and add to DataFrame
# ------------------- #  Plot data and define mito segs
# data = bishop.isc_norm(data, tx)
plt.figure(1, figsize=(14,10))
plt.plot(data['HbDif']) # Plot HbDif signal
points = bishop.pickpoint(data['HbDif'], 4) # Select st/end of MIa and MIb segments
MI = bishop.stage_dfs(data, 2, points) # Puts MI tests into list
# ------------------- #  Plot MIa test and apply occlusion time line
occMarks = np.array([ 0, 0.5, 15, 16.5, 30, 31.5, 46, 48, 64, 66.5, 82, 85, 100.5, 105, 121.5, 126, 144, 148.5, 166,
                  171.5, 188, 192.5, 211, 217, 251, 257, 292, 298]) * Hz # Diabetic AO marks
occMarks = np.array([ 0, 0.5, 12, 13, 24, 25.5, 36, 38, 53, 55.5, 69, 72, 85.5, 90, 106.5, 111, 127.5, 132, 148.5,
                  153, 169.5, 174, 191, 197, 230.5, 237, 270.5, 277]) * Hz # Healthy AO marks
#[ 0, 0.5, 12, 13.5, 24, 25.5, 37, 39, 53, 55.5, 69, 72, 85.5, 90, 106.5, 111, 127.5, 132, 148.5,
                 # 153, 169.5, 174, 191, 197, 231, 237, 271, 277]
# ------------------- # MIb
start_a = bishop.aostart(MI[0], act_sig, Hz=Hz) # Defines 1st AO inflation
mark_a = bishop.occmarks(MI[0], occMarks, start_a, act_sig) # Plots MI with slope analysis segments
## mark_a[29] = 11928 # Change analysis window if necessary
## mark_a = np.append(mark_a, 11928) # Add occlusion
# ------------------- ## Manual occlusion selection
numAO = 0
plt.plot(MI[0]['cHbDif'])
mark_a = bishop.pickpoint(MI[0]['cHbDif'], numAO*2)
# ------------------- #
slps_a = bishop.mitoslopes(MI[0], mark_a, act_sig) # Calculates slope values
fitParams_a, fitcovs_a = bishop.curvefit(slps_a, filename[:-4], '_MIa') # Fits and plots curves
# ------------------- #  MIb
start_b = bishop.aostart(MI[1], act_sig, Hz=Hz) # Defines 1st AO inflation
mark_b = bishop.occmarks(MI[1], occMarks, start_b, act_sig) # plots MI with slope analysis windows
## mark_b[] =  # Change analysis window if necessary
## mark_b = np.append(mark_b, 17153) # Add occlusion
# ------------------- ## Manual occlusion selection
numAO = 0
plt.plot(MI[1]['cHbDif'])
mark_b = bishop.pickpoint(MI[1]['cHbDif'], numAO*2)
# ------------------- #
act_sig = 'cHbDif'
slps_b = bishop.mitoslopes(MI[1], mark_b, act_sig)  # Calculates slope values
fitParams_b, fitcovs_b = bishop.curvefit(slps_b, filename[:-4], '_MIb')   # Fits and plots curves
# ------------------- # Save to excel
writer = pd.ExcelWriter(filename[:-4]+'.xlsx', engine='xlsxwriter') # Write to excel
# mito data
mito = [slps_a, slps_b, fitParams_a, fitParams_b]
mito = pd.concat(mito, keys=['MIa','MIb','fit_a','fit_b'])
mito.to_excel(writer, sheet_name='mito')
# Close and Save
writer.save()
# ----------------------------fin---------------------------- #
%reset

import imp
imp.reload(bishop)