# ------------------------------------------------------------ #
# This script imports REP data and performs stats
#
# Revision History
# Developer:            Version:   Date:        Notes:
# Adam A. Lucero        1.0.0      08/22/2017   PhD Script
#
# ------------------------------------------------------------ #
# ------------------------------------------------------------ # Import statements
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import ctypes
import pickle
# import stats
# ------------------------------------------------------------ # Data import

# Create list of all participants Perfusion DataFrames
list = ['01-', '02-', '03-', '04-', '05-', '06-', '07-', '08-', '09-', '10-']
pop = []
for l in ['ND', 'T2D']:
    perfDat, bfDat, voDat, miDat, scatDat = [], [], [], [], []
    fp = 'C:\\Users\\adamluco\\Dropbox\\adam\\Massey\\Research\\Reproducibility study\\Analysis\\ISS Analyzed\\'+ l + '\\'
    for j in list:
        for i in np.arange(3):
            x = fp + j + str(i+1) + '.xlsx'
            dat = []
            for k in ['Perfusion', 'mBF', 'mVO2', 'mito', 'scat']:
                q = pd.read_excel(x, k)
                dat.append(q)
            for m in np.arange(5):
                n = [perfDat, bfDat, voDat, miDat, scatDat]
                n[m].append(dat[m])
    pop.append(n)

# Perfusion Parameters
p = 0
perf, SaO2, max, min, range = [], [], [], [], []
for d in [0, 1]:
    dperf, dSaO2, dmax, dmin, drange = [], [], [], [], []
    for h in np.arange(4):
        r_perf, r_SaO2 = [], []
        ex1_perf, ex1_SaO2, ex1_max, ex1_min, ex1_range = [], [], [], [], []
        ex2_perf, ex2_SaO2, ex2_max, ex2_min, ex2_range = [], [], [], [], []
        for i in np.arange(30):
            # rest
            q = pop[d][p][i].iloc[0, h]
            w = pop[d][p][i].iloc[1, h]
            r_perf.append(q)
            r_SaO2.append(w)
            # ex 1
            q = pop[d][p][i].iloc[4, h]
            w = pop[d][p][i].iloc[5, h]
            e = pop[d][p][i].iloc[6, h]
            r = pop[d][p][i].iloc[7, h]
            s = pop[d][p][i].iloc[8, h]
            ex1_max.append(q)
            ex1_min.append(w)
            ex1_range.append(e)
            ex1_perf.append(r)
            ex1_SaO2.append(s)
            # ex 2
            q = pop[d][p][i].iloc[9, h]
            w = pop[d][p][i].iloc[10, h]
            e = pop[d][p][i].iloc[11, h]
            r = pop[d][p][i].iloc[12, h]
            s = pop[d][p][i].iloc[13, h]
            ex2_max.append(q)
            ex2_min.append(w)
            ex2_range.append(e)
            ex2_perf.append(r)
            ex2_SaO2.append(s)
        r_perf = DataFrame(np.reshape(r_perf, (10, 3)))
        r_SaO2 = DataFrame(np.reshape(r_SaO2, (10, 3)))
        ex1_max = DataFrame(np.reshape(ex1_max, (10, 3)))
        ex1_min = DataFrame(np.reshape(ex1_min, (10, 3)))
        ex1_range = DataFrame(np.reshape(ex1_range, (10, 3)))
        ex1_perf = DataFrame(np.reshape(ex1_perf, (10, 3)))
        ex1_SaO2 = DataFrame(np.reshape(ex1_SaO2, (10, 3)))
        ex2_max = DataFrame(np.reshape(ex2_max, (10,3)))
        ex2_min = DataFrame(np.reshape(ex2_min, (10,3)))
        ex2_range = DataFrame(np.reshape(ex2_range, (10,3)))
        ex2_perf = DataFrame(np.reshape(ex2_perf, (10,3)))
        ex2_SaO2 = DataFrame(np.reshape(ex2_SaO2, (10,3)))
        x = [ex1_max, ex2_max]
        y = [ex1_min, ex2_min]
        z = [ex1_range, ex2_range]
        j = [r_perf, ex1_perf, ex2_perf]
        k = [r_SaO2, ex1_SaO2, ex2_SaO2]
        dmax.append(x)
        dmin.append(y)
        drange.append(z)
        dperf.append(j)
        dSaO2.append(k)
    max.append(dmax)
    min.append(dmin)
    range.append(drange)
    perf.append(dperf)
    SaO2.append(dSaO2)

perfParams = [perf, SaO2, max, min, range]

# mBF Parameters
cc1, cc2, cc3, cc13 = [], [], [], []
p = 1
for d in [0, 1]:
    dcc1, dcc2, dcc3, dcc13 = [], [], [], []
    for h in [14, 15, 16, 17]:
        r_cc1, r_cc2, r_cc3, r_cc13 = [], [], [], []
        ex1_cc1, ex1_cc2, ex1_cc3, ex1_cc13 = [], [], [], []
        ex2_cc1, ex2_cc2, ex2_cc3, ex2_cc13 = [], [], [], []
        for i in np.arange(30):
            # rest
            q = pop[d][p][i].iloc[1, h]
            w = pop[d][p][i].iloc[2, h]
            e = pop[d][p][i].iloc[3, h]
            r = pop[d][p][i].iloc[4, h]
            r_cc1.append(q)
            r_cc2.append(w)
            r_cc3.append(e)
            r_cc13.append(r)
            # ex 1
            q = pop[d][p][i].iloc[5, h]
            w = pop[d][p][i].iloc[6, h]
            e = pop[d][p][i].iloc[7, h]
            r = pop[d][p][i].iloc[8, h]
            ex1_cc1.append(q)
            ex1_cc2.append(w)
            ex1_cc3.append(e)
            ex1_cc13.append(r)
            # ex 2
            q = pop[d][p][i].iloc[9, h]
            w = pop[d][p][i].iloc[10, h]
            e = pop[d][p][i].iloc[11, h]
            r = pop[d][p][i].iloc[12, h]
            ex2_cc1.append(q)
            ex2_cc2.append(w)
            ex2_cc3.append(e)
            ex2_cc13.append(r)
        r_cc1 = DataFrame(np.reshape(r_cc1, (10, 3)))
        r_cc2 = DataFrame(np.reshape(r_cc2, (10, 3)))
        r_cc3 = DataFrame(np.reshape(r_cc3, (10, 3)))
        r_cc13 = DataFrame(np.reshape(r_cc13, (10, 3)))
        ex1_cc1 = DataFrame(np.reshape(ex1_cc1, (10, 3)))
        ex1_cc2 = DataFrame(np.reshape(ex1_cc2, (10, 3)))
        ex1_cc3 = DataFrame(np.reshape(ex1_cc3, (10, 3)))
        ex1_cc13 = DataFrame(np.reshape(ex1_cc13, (10,3)))
        ex2_cc1 = DataFrame(np.reshape(ex2_cc1, (10,3)))
        ex2_cc2 = DataFrame(np.reshape(ex2_cc2, (10,3)))
        ex2_cc3 = DataFrame(np.reshape(ex2_cc3, (10,3)))
        ex2_cc13 = DataFrame(np.reshape(ex2_cc13, (10,3)))
        x = [r_cc1, ex1_cc1, ex2_cc1]
        y = [r_cc2, ex1_cc2, ex2_cc2]
        z = [r_cc3, ex1_cc3, ex2_cc3]
        j = [r_cc13, ex1_cc13, ex2_cc13]
        dcc1.append(x)
        dcc2.append(y)
        dcc3.append(z)
        dcc13.append(j)
    cc1.append(dcc1)
    cc2.append(dcc2)
    cc3.append(dcc3)
    cc13.append(dcc13)

mBF = [cc1, cc2, cc3, cc13]

# mVO2 Parameters
HHb, HbDif, cHHb, cHbDif = [], [], [], []
p = 2
for d in [0, 1]:
    dcc1, dcc2, dcc3, dcc13 = [], [], [], []
    for h in [14, 15, 16, 17]:
        r_cc1, r_cc2, r_cc3, r_cc13 = [], [], [], []
        ex1_cc1, ex1_cc2, ex1_cc3, ex1_cc13 = [], [], [], []
        ex2_cc1, ex2_cc2, ex2_cc3, ex2_cc13 = [], [], [], []
        for i in np.arange(30):
            # rest
            q = pop[d][p][i].iloc[1, h]
            w = pop[d][p][i].iloc[2, h]
            e = pop[d][p][i].iloc[3, h]
            r = pop[d][p][i].iloc[4, h]
            r_cc1.append(q)
            r_cc2.append(w)
            r_cc3.append(e)
            r_cc13.append(r)
            # ex 1
            q = pop[d][p][i].iloc[5, h]
            w = pop[d][p][i].iloc[6, h]
            e = pop[d][p][i].iloc[7, h]
            r = pop[d][p][i].iloc[8, h]
            ex1_cc1.append(q)
            ex1_cc2.append(w)
            ex1_cc3.append(e)
            ex1_cc13.append(r)
            # ex 2
            q = pop[d][p][i].iloc[9, h]
            w = pop[d][p][i].iloc[10, h]
            e = pop[d][p][i].iloc[11, h]
            r = pop[d][p][i].iloc[12, h]
            ex2_cc1.append(q)
            ex2_cc2.append(w)
            ex2_cc3.append(e)
            ex2_cc13.append(r)
        r_cc1 = DataFrame(np.reshape(r_cc1, (10, 3)))
        r_cc2 = DataFrame(np.reshape(r_cc2, (10, 3)))
        r_cc3 = DataFrame(np.reshape(r_cc3, (10, 3)))
        r_cc13 = DataFrame(np.reshape(r_cc13, (10, 3)))
        ex1_cc1 = DataFrame(np.reshape(ex1_cc1, (10, 3)))
        ex1_cc2 = DataFrame(np.reshape(ex1_cc2, (10, 3)))
        ex1_cc3 = DataFrame(np.reshape(ex1_cc3, (10, 3)))
        ex1_cc13 = DataFrame(np.reshape(ex1_cc13, (10,3)))
        ex2_cc1 = DataFrame(np.reshape(ex2_cc1, (10,3)))
        ex2_cc2 = DataFrame(np.reshape(ex2_cc2, (10,3)))
        ex2_cc3 = DataFrame(np.reshape(ex2_cc3, (10,3)))
        ex2_cc13 = DataFrame(np.reshape(ex2_cc13, (10,3)))
        x = [r_cc1, ex1_cc1, ex2_cc1]
        y = [r_cc2, ex1_cc2, ex2_cc2]
        z = [r_cc3, ex1_cc3, ex2_cc3]
        j = [r_cc13, ex1_cc13, ex2_cc13]
        dcc1.append(x)
        dcc2.append(y)
        dcc3.append(z)
        dcc13.append(j)
    HHb.append(dcc1)
    HbDif.append(dcc2)
    cHHb.append(dcc3)
    cHbDif.append(dcc13)

mVO2 = [HHb, HbDif, cHHb, cHbDif]

# Mito Parameters
kHHb, kHbDif, kcHHb, kcHbDif = [], [], [], []
p = 3
for d in [0, 1]:
    dHHb, dHbDif, dcHHb, dcHbDif = [], [], [], []
    for h in [0, 1, 2, 3]:
        HHb_a, HbDif_a, cHHb_a, cHbDif_a = [], [], [], []
        HHb_b, HbDif_b, cHHb_b, cHbDif_b = [], [], [], []
        for i in np.arange(30):
            # k_a
            q = pop[d][p][i].iloc[h, 2]
            w = pop[d][p][i].iloc[h + 4, 2]
            e = pop[d][p][i].iloc[h + 8, 2]
            r = pop[d][p][i].iloc[h + 12, 2]
            HHb_a.append(q)
            HbDif_a.append(w)
            cHHb_a.append(e)
            cHbDif_a.append(r)
            # k_b
            q = pop[d][p][i].iloc[h + 16, 2]
            w = pop[d][p][i].iloc[h + 20, 2]
            e = pop[d][p][i].iloc[h + 24, 2]
            r = pop[d][p][i].iloc[h + 28, 2]
            HHb_b.append(q)
            HbDif_b.append(w)
            cHHb_b.append(e)
            cHbDif_b.append(r)
        HHb_a = DataFrame(np.reshape(HHb_a, (10, 3)))
        HbDif_a = DataFrame(np.reshape(HbDif_a, (10, 3)))
        cHHb_a = DataFrame(np.reshape(r_cc3, (10, 3)))
        cHbDif_a = DataFrame(np.reshape(r_cc13, (10, 3)))
        HHb_b = DataFrame(np.reshape(HHb_b, (10, 3)))
        HbDif_b = DataFrame(np.reshape(HbDif_b, (10, 3)))
        cHHb_b = DataFrame(np.reshape(cHHb_b, (10, 3)))
        cHbDif_b = DataFrame(np.reshape(cHbDif_b, (10,3)))
        avg1 = (HHb_a + HHb_b) / 2
        avg2 = (HbDif_a + HbDif_b) / 2
        avg3 = (cHHb_a + cHHb_b) / 2
        avg4 = (cHbDif_a + cHbDif_b) / 2
        x = [HHb_a, HHb_b, avg1]
        y = [HbDif_a, HbDif_b, avg2]
        z = [cHHb_a, cHHb_b, avg3]
        j = [cHbDif_a, cHbDif_b, avg4]
        dHHb.append(x)
        dHbDif.append(y)
        dcHHb.append(z)
        dcHbDif.append(j)
    kHHb.append(dHHb)
    kHbDif.append(dHbDif)
    kcHHb.append(dcHHb)
    kcHbDif.append(dcHbDif)

mito = [kHHb, kHbDif, kcHHb, kHbDif]
# ------------------------------------------------------------ # Reproducibility Stats
# Averages
# param[var][con][sig][stage][day][participant]
# or
# param[var][con][sig][stage].iloc[participant, day]
def avg(param, var, con, sig, stage):
    dayMean, daySD = [], []
    grpMean, grpSD = [], []
    n, df = [], []
    for i in np.arange(3):
        x = np.average(param[var][con][sig][stage][i])
        y = np.std(param[var][con][sig][stage][i])
        z = len(param[var][con][sig][stage][i])
        dayMean.append(x)
        daySD.append(y)
        n.append(z)
        df.append(z-1)
    sumprod, SDprod = [], []
    for i in np.arange(len(dayMean)):
        x = dayMean[i] * n[i]
        y = daySD[i] * daySD[i] * df[i]
        sumprod.append(x)
        SDprod.append(y)
    grpMean = np.sum(sumprod)/np.sum(n)
    grpSD = np.sqrt(np.sum(SDprod)/np.sum(df))

    return dayMean, daySD, grpMean, grpSD

k_ND = avg(mito, 1, 0, 2, 2)
k_D = avg(mito, 1, 1, 2, 2)
k_ND = np.average(k_ND)
k_D = np.average(k_D)

param = mVO2
var = [0, 1, 2, 3]
con = [0, 1]
sig = [0, 1, 2, 3]
stage = [0, 1, 2]

mean, grp = [], [] # means[var][con][stage][sig]
for v in var:
    means1 = []
    grp1 = []
    for d in con:
        means2 = []
        grp2 = []
        for st in stage:
            means3 = []
            grp3 = []
            for s in sig:
                means4, grp4 = stats.rep_means(param[v][d][s][st])
                means3.append(means4)
                grp3.append(grp4)
            means2.append(means3)
            grp2.append(grp3)
        means1.append(means2)
        grp1.append(grp2)
    mean.append(means1)
    grp.append(grp1)

mBFMeans = grp.copy()
mVO2Means = grp.copy()



x1 = [0, 5, 15]
y1 = []
for j in sig:
    l = []
    for i in stage:
        mean = mBFMeans[0][1][i][j][0][0]
        l.append(mean)
    y1.append(l)

sigs = ['csAC', 'csDC', 'AC', 'DC']
for i in sig:
    plt.plot(x1, y1[i], label=sigs[i])
plt.legend(loc='lower right')

np.log(rep_df)*100

rep_df = mBF[0][0][0][1]
rely_tbl, raw_mean, raw_change, raw_mean_change_tbl = raw_rely(rep_df)
log_rely_tbl, btf_mean_tbl, log_change_tbl, tables = log_rely(rep_df)

# ----------------------------fin---------------------------- #
%reset

import imp
imp.reload(stats)