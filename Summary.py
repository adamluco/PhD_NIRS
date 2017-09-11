# Import statements
# ------------------------------------------------------------ # Import statements
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import ctypes
import pickle
import stat_fxs
# ------------------------------------------------------------ # Data import

# Create list of all participants Perfusion DataFrames
perfDat, bfDat, voDat, miDat = [], [], [], []
s1, s2, s3, s4 = 'Perfusion', 'mBF', 'mVO2', 'mito'
list = ['01-', '02-', '03-', '04-', '05-', '06-', '07-', '08-', '09-', '10-']
pop = []
for l in ['ND', 'T2D']:
    fp = 'C:\\Users\\adamluco\\Dropbox\\adam\\Massey\\Research\\Reproducibility study\\Analysis\\ISS Analyzed\\'+ l + '\\'
    for j in list:
        for i in np.arange(3):
            x = fp + j + str(i+1) + '.xlsx'
            dat = []
            for k in [s1, s2, s3, s4]:
                q = pd.read_excel(x, k)
                dat.append(q)
            for m in np.arange(4):
                n = [perfDat, bfDat, voDat, miDat]
                n[m].append(dat[m])
    pop.append(n)

# Perfusion Parameters
p = []
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
        r_perf = np.reshape(r_perf, (10, 3))
        r_SaO2 = np.reshape(r_SaO2, (10, 3))
        ex1_max = np.reshape(ex1_max, (10, 3))
        ex1_min = np.reshape(ex1_min, (10, 3))
        ex1_range = np.reshape(ex1_range, (10, 3))
        ex1_perf = np.reshape(ex1_perf, (10, 3))
        ex1_SaO2 = np.reshape(ex1_SaO2, (10, 3))
        ex2_max = np.reshape(ex2_max, (10,3))
        ex2_min = np.reshape(ex2_min, (10,3))
        ex2_range = np.reshape(ex2_range, (10,3))
        ex2_perf = np.reshape(ex2_perf, (10,3))
        ex2_SaO2 = np.reshape(ex2_SaO2, (10,3))
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
        r_cc1 = np.reshape(r_cc1, (10, 3))
        r_cc2 = np.reshape(r_cc2, (10, 3))
        r_cc3 = np.reshape(r_cc3, (10, 3))
        r_cc13 = np.reshape(r_cc13, (10, 3))
        ex1_cc1 = np.reshape(ex1_cc1, (10, 3))
        ex1_cc2 = np.reshape(ex1_cc2, (10, 3))
        ex1_cc3 = np.reshape(ex1_cc3, (10, 3))
        ex1_cc13 = np.reshape(ex1_cc13, (10,3))
        ex2_cc1 = np.reshape(ex2_cc1, (10,3))
        ex2_cc2 = np.reshape(ex2_cc2, (10,3))
        ex2_cc3 = np.reshape(ex2_cc3, (10,3))
        ex2_cc13 = np.reshape(ex2_cc13, (10,3))
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

mBFParams = [cc1, cc2, cc3, cc13]

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
        r_cc1 = np.reshape(r_cc1, (10, 3))
        r_cc2 = np.reshape(r_cc2, (10, 3))
        r_cc3 = np.reshape(r_cc3, (10, 3))
        r_cc13 = np.reshape(r_cc13, (10, 3))
        ex1_cc1 = np.reshape(ex1_cc1, (10, 3))
        ex1_cc2 = np.reshape(ex1_cc2, (10, 3))
        ex1_cc3 = np.reshape(ex1_cc3, (10, 3))
        ex1_cc13 = np.reshape(ex1_cc13, (10,3))
        ex2_cc1 = np.reshape(ex2_cc1, (10,3))
        ex2_cc2 = np.reshape(ex2_cc2, (10,3))
        ex2_cc3 = np.reshape(ex2_cc3, (10,3))
        ex2_cc13 = np.reshape(ex2_cc13, (10,3))
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
        HHb_a = np.reshape(HHb_a, (10, 3))
        HbDif_a = np.reshape(HbDif_a, (10, 3))
        cHHb_a = np.reshape(r_cc3, (10, 3))
        cHbDif_a = np.reshape(r_cc13, (10, 3))
        HHb_b = np.reshape(HHb_b, (10, 3))
        HbDif_b = np.reshape(HbDif_b, (10, 3))
        cHHb_b = np.reshape(cHHb_b, (10, 3))
        cHbDif_b = np.reshape(cHbDif_b, (10,3))
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


# ------------------------------------------------------------ # T2D
# Define filepath
fp = 'C:\\Users\\adamluco\\Dropbox\\adam\\Massey\\Research\\Reproducibility study\\Analysis\\ISS Analyzed\\T2D\\'

# Create list of all participants Perfusion DataFrames
part_d = []
perfusion_d = []
list = ['01-', '02-', '03-', '04-', '05-', '06-', '07-', '08-', '09-', '10-']
for j in list:
    for i in np.arange(3):
        x = fp + j + str(i+1) + '.xlsx'
        sheet = 'Perfusion'
        p = pd.read_excel(x, sheet)
        part_d.append(p)

# Create table of resting params
perfParams_d = ('perfusion', 'SaO2', 'max', 'min', 'range')

Hb_perf_d = []
Hb_SaO2_d = []
for h in np.arange(4):
    r_perf = []
    r_SaO2 = []
    for i in np.arange(30):
        perf = part_d[i].iloc[0, h]
        SaO2 = part_d[i].iloc[1, h]
        r_perf.append(perf)
        r_SaO2.append(SaO2)
    x = np.reshape(r_perf, (10,3))
    y = np.reshape(r_SaO2, (10,3))
    Hb_perf_d.append(x)
    Hb_SaO2_d.append(y)


# Create a table of ex 1 params
Hb_max_d, Hb_min_d, Hb_range_d = [], [], []
for h in np.arange(4):
    ex1_perf, ex1_SaO2, ex1_max, ex1_min, ex1_range = [], [], [], [], []
    for i in np.arange(30):
        max = part_d[i].iloc[4, h]
        min = part_d[i].iloc[5, h]
        range = part_d[i].iloc[6, h]
        perf = part_d[i].iloc[7, h]
        SaO2 = part_d[i].iloc[8, h]
        ex1_max.append(max)
        ex1_min.append(min)
        ex1_range.append(range)
        ex1_perf.append(perf)
        ex1_SaO2.append(SaO2)
    x = np.reshape(ex1_max, (10,3))
    y = np.reshape(ex1_min, (10,3))
    z = np.reshape(ex1_range, (10,3))
    j = np.reshape(ex1_perf, (10,3))
    k = np.reshape(ex1_SaO2, (10,3))
    Hb_max_d.append(x)
    Hb_min_d.append(y)
    Hb_range_d.append(z)
    Hb_perf_d.append(j)
    Hb_SaO2_d.append(k)

# Create tables of ex2 params
for h in np.arange(4):
    ex2_perf, ex2_SaO2, ex2_max, ex2_min, ex2_range = [], [], [], [], []
    for i in np.arange(30):
        max = part_d[i].iloc[9, h]
        min = part_d[i].iloc[10, h]
        range = part_d[i].iloc[11, h]
        perf = part_d[i].iloc[12, h]
        SaO2 = part_d[i].iloc[13, h]
        ex2_max.append(max)
        ex2_min.append(min)
        ex2_range.append(range)
        ex2_perf.append(perf)
        ex2_SaO2.append(SaO2)
    x = np.reshape(ex2_max, (10,3))
    y = np.reshape(ex2_min, (10,3))
    z = np.reshape(ex2_range, (10,3))
    j = np.reshape(ex2_perf, (10,3))
    k = np.reshape(ex2_SaO2, (10,3))
    Hb_max_d.append(x)
    Hb_min_d.append(y)
    Hb_range_d.append(z)
    Hb_perf_d.append(j)
    Hb_SaO2_d.append(k)
# ------------------------------------------------------------ # Organize data
# Make data tree
ND = [[[], Hb_max[0:4], Hb_max[4:8]], [[], Hb_min[0:4], Hb_min[4:8]],[[], Hb_range[0:4], Hb_range[4:8]],
      [Hb_perf[0:4], Hb_perf[4:8], Hb_perf[8:12]], [Hb_SaO2[0:4], Hb_SaO2[4:8], Hb_SaO2[8:12]]]
D = [[[], Hb_max_d[0:4], Hb_max_d[4:8]], [[], Hb_min_d[0:4], Hb_min_d[4:8]],[[], Hb_range_d[0:4], Hb_range_d[4:8]],
      [Hb_perf_d[0:4], Hb_perf_d[4:8], Hb_perf_d[8:12]], [Hb_SaO2_d[0:4], Hb_SaO2_d[4:8], Hb_SaO2_d[8:12]]]
Perf = [ND, D]

# Calculate averages and reproducibility data
pop = {0:'ND', 1:'D'} #2
param = {0:'max', 1:'min', 2:'range', 3:'perf', 4:'SaO2', 5:'r_tHb', 6:'r_SaO2'}
stage = {0:'rest', 2:'ex1', 3:'ex2'}
sig = {0:'Hb_csAC', 1:'Hb_csDC', 2:'Hb_AC', 3:'Hb_DC'}

def avg_all(pop, param, s):
    rest = []
    for i in np.arange(10):
        r_p = np.average(Perf[pop][param][0][s][i])
        rest.append(r_p)
    restPerf = np.average(rest)
    std_r = np.std(rest)

    ex1 = []
    for i in np.arange(10):
        ex1_p = np.average(Perf[pop][param][1][s][i])
        ex1.append(ex1_p)
    ex1Perf = np.average(ex1)
    std_ex1 = np.std(ex1)

    ex2 = []
    for i in np.arange(10):
        ex2_p = np.average(Perf[pop][param][2][s][i])
        ex2.append(ex2_p)
    ex2Perf = np.average(ex2)
    std_ex2 = np.std(ex2)
    avgParam = [[restPerf, ex1Perf, ex2Perf], [std_r, std_ex1, std_ex2]]

    return avgParam

def avg_all_nor(pop, param, s):

    ex1 = []
    for i in np.arange(10):
        ex1_p = np.average(Perf[pop][param][1][s][i])
        ex1.append(ex1_p)
    ex1Perf = np.average(ex1)
    std_ex1 = np.std(ex1)

    ex2 = []
    for i in np.arange(10):
        ex2_p = np.average(Perf[pop][param][2][s][i])
        ex2.append(ex2_p)
    ex2Perf = np.average(ex2)
    std_ex2 = np.std(ex2)
    avgParam = [[ex1Perf, ex2Perf], [std_ex1, std_ex2]]

    return avgParam

def avg_plot(ND_param, D_param, title, ylabel, ND='ND', T2D='T2D'):
    MVC = [0, 5, 15]
    plt.errorbar(MVC, ND_param[0], yerr=ND_param[1], capsize=5, marker='o', label=ND)
    plt.errorbar(MVC, D_param[0], yerr=D_param[1], capsize=5, marker='*', label=T2D)
    plt.xticks(MVC, [0, 5, 15])
    plt.legend(loc='best')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('% MVC')

def avg_plot_nor(ND_param, D_param, title, ylabel, ND='ND', T2D='T2D'):
    MVC = [5, 15]
    plt.errorbar(MVC, ND_param[0], yerr=ND_param[1], capsize=5, marker='o', label=ND)
    plt.errorbar(MVC, D_param[0], yerr=D_param[1], capsize=5, marker='*', label=T2D)
    plt.xticks(MVC, [5, 15])
    plt.legend(loc='best')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('% MVC')

csND_perf = avg_all(0, 3, 0)
csD_perf = avg_all(1, 3, 0)
avg_plot(csND_perf, csD_perf, 'Hb_csAC Perfusion', '[tHb] (uM)', 'AC_csND', 'AC_csT2D')

ND_perf = avg_all(0, 3, 2)
D_perf = avg_all(1, 3, 2)
avg_plot(ND_perf, D_perf, 'Hb_AC Perfusion', '[tHb] (uM)', 'AC_ND', 'AC_T2D')

dccsND_perf = avg_all(0, 3, 1)
dccsD_perf = avg_all(1, 3, 1)
avg_plot(dccsND_perf, dccsD_perf, 'Hb_csDC Perfusion', '[tHb] (uM)', 'DC_csND', 'DC_csT2D')

dcND_perf = avg_all(0, 3, 3)
dcD_perf = avg_all(1, 3, 3)
avg_plot(dcND_perf, dcD_perf, 'Hb_DC Perfusion', '[tHb] (uM)', 'DC_ND', 'DC_T2D')

ND_SaO2 = avg_all(0, 4, 0)
D_SaO2 = avg_all(1, 4, 0)
avg_plot(ND_SaO2, D_SaO2, 'AC_csSaO2', '%SaO2', 'ND', 'T2D')

ND_SaO2 = avg_all(0, 4, 2)
D_SaO2 = avg_all(1, 4, 2)
avg_plot(ND_SaO2, D_SaO2, 'AC_SaO2', '%SaO2', 'ND', 'T2D')

ND_range = avg_all_nor(0, 2, 0)
D_range = avg_all_nor(1, 2, 0)
avg_plot_nor(ND_range, D_range, 'cs[tHb] range', 'range (uM)')

ND_range = avg_all_nor(0, 2, 2)
D_range = avg_all_nor(1, 2, 2)
avg_plot_nor(ND_range, D_range, '[tHb] range', 'range (uM)')

# ------------------------------------------------------------ # Import healthy mBF and mVO2
# Define filepath
fp = 'C:\\Users\\adamluco\\Dropbox\\adam\\Massey\\Research\\Reproducibility study\\Analysis\\ISS Analyzed\\ND\\'

# Create list of all participants Perfusion DataFrames
bfDat = []
voDat = []
list = ['01-', '02-', '03-', '04-', '05-', '06-', '07-', '08-', '09-', '10-']
for j in list:
    for i in np.arange(3):
        x = fp + j + str(i+1) + '.xlsx'
        sheet1 = 'mBF'
        sheet2 = 'mVO2'
        p, l = pd.read_excel(x, sheet1), pd.read_excel(x, sheet2)
        dat.append(p)

# Create table of params
Hb_cc = []
for i in np.arange(30):
    Hb = []
    for h in [7, 8, 9, 10]:
        cc = []
        for j in np.arange(1, 5):
            r = dat[i].iloc[j, h]
            ex1 = dat[i].iloc[j+4, h]
            ex2 = dat[i].iloc[j+8, h]
            bf = [r, ex1, ex2]
            cc.append(bf)
        Hb.append(cc)
    Hb_cc.append(Hb)

