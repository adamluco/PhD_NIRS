# ------------------------------------------------------------ #
# This script reads all signals from up to 2 Artinis probes as well as
# LabChart data for the analysis of mBF & mVO2 during resting and exercise stages
#
# Revision History
# Developer:            Version:   Date:        Notes:
# Adam A. Lucero        1.0.0      12/02/2017
# ------------------------------------------------------------ #
# Import required libraries and modules
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
# ------------------------------------------------------------ #
# Data File Paths, Parameters, and Variables
LC_path = 'C:/Users/adaml/Dropbox/Adam/Python/FBF trials/AL_FBF_tr1_LC.txt'
Art_path = 'C:/Users/adaml/Dropbox/Adam/Python/FBF trials/AL_FBF-tr1_Art.txt'
# ------------------------------------------------------------ #
# Editable Variables
Hz = 50
probe1 = 'FCR'
probe2 = 'FDP'
numstages = 4
# ------------------------------------------------------------ #
# Read in LC data and create DataFrame to PortaSync 1
def lcdata(lc_path):
    LC_names = ['time', 'ECG', 'Period', 'HR', 'HandGrip', 'Occ', 'PS']
    LC_skip = ['NaN', 'Interval=', 'xcelDateTime=', 'TimeFormat=', 'DateFormat=', 'ChannelTitle=', 'Range=']
    LC_data = pd.read_table(lc_path, header=None, names=LC_names, na_values=LC_skip, skip_blank_lines=True,
                        low_memory=False)
    LC_data = DataFrame(LC_data)
    LC_data = LC_data.drop(['Period', 'HR'], axis=1)
    LC_data = LC_data.dropna()
    LC_data = DataFrame(LC_data, dtype=float)
    def normalize(y):
        for y in range(len(y)):
        norm = (y - min(y))/(max(y)-min(y))
        return norm
    LC_mark = LC_data.PS.index[LC_data.PS > 1]
    st_time = LC_data.ix[LC_mark[0]].time
    LC_data[LC_data['time'] < st_time] = None
    LC_data = LC_data.dropna()
    return LC_data

LC_data = lcdata(LC_path)

# Read in Artinis data and create DataFrame to PortaSync 1
def artdata(art_path):
    Art_start = 67
    Art_names = ['Sample', 'p1t1_O2Hb', 'p1t1_HHb', 'p1t1_tHb',
                 'p1t2_O2Hb', 'p1t2_HHb', 'p1t2_tHb', 'p1t3_O2Hb', 'p1t3_HHb', 'p1t3_tHb',
                 'p2t1_O2Hb', 'p2t1_HHb', 'p2t1_tHb', 'p2t2_O2Hb', 'p2t2_HHb', 'p2t2_tHb',
                 'p2t3_O2Hb', 'p2t3_HHb', 'p2t3_tHb', 'PS_0', 'PS_1', 'event']
    Art_data = pd.read_table(art_path, header=None, names=Art_names, skiprows=np.arange(Art_start),
                             skip_blank_lines=True, low_memory=False)
    Art_data = DataFrame(Art_data, dtype=float)
    Art_data = Art_data.drop('event', axis=1)
    Art_data = Art_data.dropna()
    Art_mark = Art_data.PS_0.index[Art_data.PS_0 > 0.03]
    Art_data[Art_data['Sample'] < Art_mark[0]] = None
    Art_data = Art_data.dropna()
    return Art_data

Art_data = artdata(Art_path)

# Time align and combine into single DataFrame
def timealign(lc_data, art_data, hz):
    lc_data.time = lc_data.time - lc_data.time[lc_data.index[0]]  # LC
    lc_data.time = lc_data.time.round(2)
    lc = lc_data.set_index('time')
    art_data['time'] = art_data.Sample/hz  #Art
    art_data.Sample = art_data.Sample - art_data.Sample[art_data.index[0]]
    art_data.time = art_data.time - art_data.time[art_data.index[0]]
    art_data.time = art_data.time.round(2)
    art = art_data.set_index('time')
    data = lc.join(art) # LC and Art combined DataFrame
    return data

data = timealign(LC_data, Art_data, Hz)

# Calculate HbDif
def hbdif(data):
    data['p1t1_HbDif'] = data.p1t1_O2Hb - data.p1t1_HHb
    data['p1t2_HbDif'] = data.p1t2_O2Hb - data.p1t2_HHb
    data['p1t3_HbDif'] = data.p1t3_O2Hb - data.p1t3_HHb
    data['p2t1_HbDif'] = data.p2t1_O2Hb - data.p2t1_HHb
    data['p2t2_HbDif'] = data.p2t2_O2Hb - data.p2t2_HHb
    data['p2t3_HbDif'] = data.p2t3_O2Hb - data.p2t3_HHb
    return data

data = hbdif(data)

# Plot experimental data

def expplot(data):
    curr_PS = data.PS.copy()
    curr_Occ = data.Occ.copy()
    tempmin = min([min(data.p1t2_HHb), min(data.p2t2_HHb)])
    tempmax = max([max(data.p1t2_tHb), max(data.p2t2_tHb)])
    curr_PS[curr_PS > 1] = tempmax + (0.10 * tempmax)
    curr_PS[curr_PS < 1] = tempmin - (0.10 * tempmin)
    curr_Occ[curr_Occ > 1] = tempmax + (0.05 * tempmax)
    curr_Occ[curr_Occ < 1] = tempmin - (0.05 * tempmin)
    plt.figure(1, figsize=(14, 8))
    plt.plot(data.p1t2_tHb, 'g-', label=probe1+'tHb')
    plt.plot(data.p2t2_tHb, 'C2-', label=probe2+'tHb')
    plt.plot(data.p1t2_HHb, 'b-', label=probe1+'HHb')
    plt.plot(data.p2t2_HHb, 'C0-', label=probe2+'HHb')
    plt.plot(curr_PS, 'C1-', label='PS')
    plt.plot(curr_Occ, 'C6-', label='Occ')
    plt.legend(loc='best')
    plt.title('Entire Test')
    return()

expplot = expplot(data)
# Stage plot
def stageplot(df, y1, y2, num, instruct, probe1=probe1, probe2=probe2):
    curr_PS = df.PS.copy()
    curr_Occ = df.Occ.copy()
    tempmin = min(min(y1), min(y2))
    tempmax = max(max(y1), max(y2))
    curr_PS[curr_PS > 1] = tempmax + (0.05 * tempmax)
    curr_PS[curr_PS < 1] = tempmin - (0.05 * tempmin)
    curr_Occ[curr_Occ > 1] = tempmax + (0.02 * tempmax)
    curr_Occ[curr_Occ < 1] = tempmin - (0.02 * tempmin)
    plt.figure(2, figsize=(14, 8))
    plt.plot(df.Sample, y1, 'C2-', label=probe1+'tHb')
    plt.plot(df.Sample, y2, 'g-', label=probe2+'tHb')
    plt.plot(df.Sample, curr_PS, 'C1-', label='PS')
    plt.plot(df.Sample, curr_Occ, 'C6-', label='Occ')
    plt.title(instruct)
    print(instruct)
    stages = plt.ginput(num)
    plt.close()
    stages = np.array(stages)
    stages = stages[:, 0]
    stages = np.trunc(stages)
    stage_t = list()
    for x in range(len(stages)):
        y = data[data.Sample == stages[x]].index.tolist()
        stage_t.append(y)
    stage_t = sum(stage_t, [])
    stage_t = np.array(stage_t)
    stage_t = np.trunc(stage_t)
    return stage_t

instructions = 'Click st/end of rest, WL1, WL2, WL3'
stage_t = stageplot(data, data.p1t2_tHb, data.p2t2_tHb, numstages*2, instructions)

# Create stage DataFrames
rest = data.ix[stage_t[0]:stage_t[1]]
WL1 = data.ix[stage_t[2]:stage_t[3]]
WL2 = data.ix[stage_t[4]:stage_t[5]]
WL3 = data.ix[stage_t[6]:stage_t[7]]

# rest occlusions
instructions = 'Click just before beginning of each occlusion'
restoccs = stageplot(rest, rest.p1t1_tHb, rest.p2t2_tHb, 6, instructions)
vo1 =  rest.ix[restoccs[0]:restoccs[0]+17]
vo2 =  rest.ix[restoccs[1]:restoccs[1]+17]
vo3 =  rest.ix[restoccs[2]:restoccs[2]+17]
vo4 =  rest.ix[restoccs[3]:restoccs[3]+17]
ao1 =  rest.ix[restoccs[4]:restoccs[4]:17]
ao2 =  rest.ix[restoccs[5]:restoccs[5]:32]

# Plot Occlusions
def voplot(vo1, vo2, vo3, vo4, y1, y2, y3, y4):
    def markers(df, y):
        curr_Occ = df.Occ.copy()
        tempmin = min(y)
        tempmax = max(y)
        curr_Occ[curr_Occ > 0.5] = tempmax + (0.01 * tempmax)
        curr_Occ[curr_Occ < 0.5] = tempmin - (0.01 * tempmin)
        return curr_Occ
    plt.figure(figsize=(14, 10))
    plt.title('Click start of 4 x VOs')
    print('Click start of 4 x VOs')
    # VO1
    ax1 = plt.subplot2grid((6, 2), (0, 0), rowspan=2)
    vo1_Occ = markers(vo1, y1)  # resize markers
    plt.plot(vo1.Sample, y1, 'C2-')
    plt.plot(vo1.Sample, vo1_Occ, 'C6-')
    # VO2
    ax2 = plt.subplot2grid((6, 2), (0, 1), rowspan=2)
    vo2_Occ = markers(vo2, y2)
    plt.plot(vo2.Sample, y2, 'C2-')
    plt.plot(vo2.Sample, vo2_Occ, 'C6-')
    # ECG1
    ax3 = plt.subplot2grid((6, 2), (2, 0))
    ecg1_occ = markers(vo1, vo1.ECG)
    plt.plot(vo1.Sample, vo1.ECG, 'r')
    plt.plot(vo1.Sample, ecg1_occ, 'C6-')
    plt.plot(vo1.Sample, vo1.HR)
    # ECG2
    ax4 = plt.subplot2grid((6, 2), (2, 1))
    ecg2_occ = markers(vo2, vo2.ECG)
    plt.plot(vo2.Sample, vo2.ECG, 'r')
    plt.plot(vo2.Sample, ecg2_occ, 'C6-')
    # vo3
    ax5 = plt.subplot2grid((6, 2), (3, 0), rowspan=2)
    vo3_Occ = markers(vo3, y3)  # resize markers
    plt.plot(vo3.Sample, y3, 'C2-')
    plt.plot(vo3.Sample, vo3_Occ, 'C6-')
    # vo4
    ax6 = plt.subplot2grid((6, 2), (3, 1), rowspan=2)
    vo4_Occ = markers(vo4, y4)
    plt.plot(vo4.Sample, y4, 'C2-')
    plt.plot(vo4.Sample, vo4_Occ, 'C6-')
    # ECG3
    ax7 = plt.subplot2grid((6, 2), (5, 0))
    ecg3_occ = markers(vo3, vo3.ECG)
    plt.plot(vo3.Sample, vo3.ECG, 'r')
    plt.plot(vo3.Sample, ecg3_occ, 'C6-')
    # ECG4
    ax8 = plt.subplot2grid((6, 2), (5, 1))
    ecg4_occ = markers(vo4, vo4.ECG)
    plt.plot(vo4.Sample, vo4.ECG, 'r')
    plt.plot(vo4.Sample, ecg4_occ, 'C6-')
    plt.tight_layout()

voplot(vo1, vo2, vo3, vo4, vo1.p1t2_tHb, vo2.p1t2_tHb, vo3.p1t2_tHb, vo4.p1t2_tHb)


