# ------------------------------------------------------------ #
""" Accompanying variable and functions for ISS analysis version 3.2 """
# Revision History
# Developer:            Version:   Date:        Notes:
# Adam A. Lucero        3.0.0      05/06/2017   LC and ISS data files
# Adam A. Lucero        3.1.0      05/06/2017   Bug fixes and perfusion fix
# Adam A. Lucero        3.1.1      05/06/2017   Occlusion marker fix
# ------------------------------------------------------------ # Import statements
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import ctypes
import pickle
# ------------------- # Variables
wavlen = {1:692, 2:834} # Wavelengths used (nm)
extHHb_wl1 = 4.5821 # extinction coefficient of HHb at 692 nm (cm^-1*mM^-1)
extO2Hb_wl1 = 0.9556 # extinction coefficient of O2Hb at 692 nm (cm^-1*mM^-1)
extH2O_wl1 = 1.01e-07 # extinction coefficient of H2O at 692 nm (cm^-1*mM^-1)
extHHb_wl2 = 1.7891 # extinction coefficient of HHb at 834 nm (cm^-1*mM^-1)
extO2Hb_wl2 = 2.3671 # extinction coefficient of O2Hb at 834 nm (cm^-1*mM^-1)
extH2O_wl2 = 6.06e-07 # extinction coefficient H2O at 834 nm (cm^-1*mM^-1)
corrCoefH2O = 55508.0 # H2O extinction conversion factor (mM)
percH2O = 0.7 #Percentage of H2O for absorption correction
v = 22600000000 # speed of light (cm/s)
F = 110000000 # Modulation frequency (Hz/2*pi)
HbSrce = ['Hb_csAC', 'Hb_csDC', 'Hb_AC', 'Hb_DC'] # Source of Hb signal
HbSig = ['HHb', 'HbDif', 'cHHb', 'cHbDif'] # Hb Signal for AO analysis
voName = {0: 'VO1', 1: 'VO2', 2: 'VO3', 3: 'VO4', 4: 'VO5', 5: 'VO6', 6: 'VO7', 7: 'VO8', 8: 'VO9', 9: 'V10'}
aoName = {0: 'AO1', 1: 'AO2', 2: 'AO3', 3: 'AO4', 4: 'AO5', 5: 'AO6', 6: 'AO7', 7: 'AO8', 8: 'A09', 9: 'A10',
          10: 'AO11', 11: 'AO12', 12: 'AO13', 13: 'AO14', 14: 'AO15', 15: 'AO16', 16: 'AO17', 17: 'AO18', 18: 'AO19',
          19: 'AO20'}
# ------------------- # Equation Functions
def radPhase(df):
    """ Takes phase signal in degrees/cm and converts to the absolute value in radians/cm """
    Sp = abs(df['SlopePh']) * (np.pi / 180)
    Sp.columns = [['radSlopePh'] * 2, ['wl1', 'wl2']]
    df = pd.concat([df, Sp], axis=1)
    return df

def scatMan_AC(df, Ph_slope, AC_slope):
    """ Calculates AC & Phase scattering for a given df """
    ua = ((np.pi * F) / v) * ((df[Ph_slope] / df[AC_slope]) - (df[AC_slope] / df[Ph_slope]))
    us = ((df[AC_slope] ** 2 - df[Ph_slope] ** 2))/(3 * ua) - ua
    us.columns = [['AC scat'] * 2, ['wl1', 'wl2']]
    return us

def scatMan_DC(df, Ph_slope, DC_slope):
    """ Calculates DC & Phase scattering for a given df """
    ua = -((np.pi * F) / v) * (df[DC_slope] / df[Ph_slope]) * \
            ((df[Ph_slope] ** 2 / df[DC_slope] ** 2) + 1) ** (-1 / 2)
    us = (df[DC_slope] ** 2/(3*ua)) - ua
    us.columns = [['DC scat'] * 2, ['wl1', 'wl2']]
    return us

def ua_ACscat(scat, Sac, Sp):
    ua = ((-scat + np.sqrt(scat ** 2 + 4 * ((Sac ** 2 - Sp **2)/3)))/2)
    ua['wl1'] = ua['wl1'] - (extH2O_wl1 * corrCoefH2O * percH2O)
    ua['wl2'] = ua['wl2'] - (extH2O_wl1 * corrCoefH2O * percH2O)
    return ua

def ua_DCscat(scat, Sdc):
    ua = (-scat + np.sqrt(scat ** 2 + 4 * ((Sdc ** 2)/3)))/2
    ua['wl1'] = ua['wl1'] - (extH2O_wl1 * corrCoefH2O * percH2O)
    ua['wl2'] = ua['wl2'] - (extH2O_wl1 * corrCoefH2O * percH2O)
    return ua

def Hb_sigs(uacn_wl1, uacn_wl2, title):
    O2Hb = 1000 * ((uacn_wl1 * extHHb_wl2) - (uacn_wl2 * extHHb_wl1))/((extO2Hb_wl1 * extHHb_wl2) - (extO2Hb_wl2 * extHHb_wl1))
    HHb =  1000 * ((uacn_wl2 * extO2Hb_wl1) - (uacn_wl1 * extO2Hb_wl2))/((extO2Hb_wl1 * extHHb_wl2) - (extO2Hb_wl2 * extHHb_wl1))
    tHb = O2Hb + HHb
    SaO2 = 100 * O2Hb/tHb
    Hb = pd.concat([O2Hb, HHb, tHb, SaO2], axis=1)
    Hb.columns = [[title]*4, ['O2Hb', 'HHb', 'tHb', 'SaO2']]
    return Hb

def hbdif(df, HbSrce):
    """Calculates HbDif and adds to DataFrame"""
    HbDif = df.xs('O2Hb', axis=1, level=1) - df.xs('HHb', axis=1, level=1)
    HbDif.columns = [ HbSrce, ['HbDif'] * len(HbSrce)]
    dat = [df, HbDif]
    df = pd.concat(dat, axis=1)
    return df

def chb(df, HbSrce):
    """Calculates corrected Hb signals"""
    beta = df.xs('O2Hb', axis=1, level=1) / df.xs('tHb', axis=1, level=1)
    beta.columns = [HbSrce, ['beta'] * len(HbSrce)]
    dat1 = [df, beta]
    df = pd.concat(dat1, axis=1)
    cO2Hb = df.xs('O2Hb', axis=1, level=1) - df.xs('tHb', axis=1, level=1) * (1 - df.xs('beta', axis=1, level=1))
    cHHb = df.xs('HHb', axis=1, level=1) - (df.xs('tHb', 1, 1) * df.xs('beta', 1, 1))
    cO2Hb.columns = [HbSrce, ['cO2Hb'] * len(HbSrce)]
    cHHb.columns = [HbSrce, ['cHHb'] * len(HbSrce)]
    dat2 = [df, cO2Hb, cHHb]
    df = pd.concat(dat2, axis=1)
    cHbDif = df.xs('cO2Hb', axis=1, level=1) - df.xs('cHHb', axis=1, level=1)
    cHbDif.columns = [HbSrce, ['cHbDif'] * len(HbSrce)]
    dat3 = [df, cHbDif]
    df = pd.concat(dat3, axis=1)
    return df

def fitFunc(t, A, B, k):
    """ Curve fitting function """
    return A - B*np.exp(-k*t)

# ------------------- # Plotting Functions
def pickpoint(df, num, reset=False):
    """
    Collects x-axis value from user clicks
    Parameters
    -----------
    df: DataFrame
    num: number of points to pick
    reset: if True, resets first and last points to equal df size if out of bounds.
            Default = False
    """
    points = plt.ginput(num, timeout=0)
    points = np.array(points)
    points = points[:, 0]
    points = np.trunc(points)
    if reset == True:
        if points[0] < 0:
            points[0] = 0
        else:
            points[0] = points[0]
        if points[-1] > len(df)-1:
            points[-1] = len(df)-1
        else:
            points[-1] = points[-1]
    else: pass
    plt.close()
    return points

def stage_dfs(df, num_stages, points, seg=1):
    """ Puts stage DataFrame segments into a list """
    stages = []
    if seg == 1:
        for i in range(num_stages):
            y = df.ix[points[i]:points[i+1]]
            stages.append(y)
        return stages
    elif seg == 2:
        for i in np.arange(1, num_stages+1):
            y = df.ix[points[(i*2-1)-1]:points[(i*2)-1]]
            stages.append(y)
        return stages

def antpoint(x, y, adj=0.10):
    """Annotate data with consecutive numbers"""
    i = x.tolist()
    j = y.tolist()
    n = list(range(len(y)))
    for k, HbSrcet in enumerate(n):
        plt.annotate(HbSrcet, (i[k], j[k]+adj))

def Mbox(title, text, style):
    """ Creates a message box """
    result = ctypes.windll.user32.MessageBoxW(0, text, title, style)
    return result

def hPlotter(df, ylist, label, title, ylabel, xlabel, colors=False):
    if colors == False:
        plt.plot(df[ylist])
    else:
        for i in range(len(ylist)):
            plt.plot(df[ylist[i]], colors[i])
    plt.legend(label, loc='best')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help.
    __author__ = "Marcos Duarte, https://github.com/demotu/BMC"
    __version__ = "1.0.4"
    __license__ = "MIT"
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                      % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.
    __author__ = "Marcos Duarte, https://github.com/demotu/BMC"
    __version__ = "1.0.4"
    __license__ = "MIT"


    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                              & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

def normalize(y):
    norm = (y - min(y)) / (max(y) - min(y))
    return norm

# ------------------- # LabChart fxs
def lc_dat(fp, fn, Hz, startmk=1, addpts=0):
    """ Read in LC data and create start """
    lc_names = ['time', 'ECG', 'Occlusion', 'Marker']
    lc_skip = ['NaN', 'Interval=', 'ExcelDateTime=', 'TimeFormat=', 'DateFormat=', 'ChannelTitle=', 'Range=']
    lc_dat = pd.read_table(fp + fn, header=None, na_values=lc_skip, names=lc_names, skip_blank_lines=True,
                            low_memory=False, skiprows=[0,1,2,3,4,5], usecols=[0,1,2,3])
    lc_dat = DataFrame(lc_dat)
    marker = lc_dat.dropna().index.values
    start = marker[startmk - 1] - addpts
    lc_dat.drop(['Marker'], axis=1, inplace=True)
    lc_dat.columns = [['LC']*3, ['time', 'ECG', 'Occlusion']]
    lc_dat = lc_dat.loc[start:]
    marker = marker - lc_dat.index.values[0]
    lc_dat.loc[:, ('LC', 'time')] = (lc_dat['LC', 'time'].values - lc_dat['LC', 'time'].iloc[0])
    #lc_ind = (lc_dat['LC', 'time'].values - lc_dat['LC', 'time'].iloc[0])*Hz
    #lc_ind = lc_ind.astype(int)
    #lc_dat.set_index((lc_dat.index.values - lc_dat.index.values[0]), inplace=True)
    return lc_dat, marker

def iss_dat(fp, fn, startmk=1):
    """ Read in ISS data and create start """
    iss_dat = pd.read_table(fp + fn, header=None, skiprows=[0, 1, 2], low_memory=False,
                         usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    iss_dat.columns = ([['LC'] * 3 + ['SlopeAC'] * 2 + ['SlopeDC'] * 2 + ['SlopePh'] * 2,
                     ['infl', 'defl', 'marker'] + ['wl1', 'wl2'] * 3])
    marker = iss_dat[iss_dat['LC', 'marker'] > 0].index.values
    iss_start = marker[startmk - 1]
    iss_dat.drop(['marker'], axis=1, level=1, inplace=True)
    iss_dat['LC', 'infl'] = 0
    iss_dat['LC', 'defl'] = 0
    #iss_dat = iss_dat.loc[start:]
    #iss_dat.set_index((iss_dat.index.values - iss_dat.index.values[0]), inplace=True)
    return iss_dat, marker, iss_start

def timealign(lc_dat, iss_dat, iss_start, mph=2.5):
    lc_dat.set_index((np.arange(iss_start, (iss_start + len(lc_dat)))), inplace=True)
    df = iss_dat.join(lc_dat).bfill()
    infl = detect_peaks(df['LC', 'Occlusion'], mph=mph, threshold=0.05, show=True)
    defl = detect_peaks(df['LC', 'Occlusion'], mph=0.3, threshold=0.05)
    defl = [x for x in defl if x not in infl]
    df.loc[infl, ('LC', 'infl')] = 1
    df.loc[defl, ('LC', 'defl')] = 1
    return df

def markers(df, dat, perc=0.1, onesig=False, x0=False):
    infl = df['LC', 'infl'].copy()
    defl = df['LC', 'defl'].copy()
    if onesig == True:
        mx = dat.max()
        mn = dat.min()
    else:
        mx = max(dat.max())
        mn = min(dat.min())
    infl.loc[infl == 1] = mx + (mx * perc)
    infl.loc[infl == 0] = mn - (mn * perc)
    defl.loc[defl == 1] = mx + (mx * perc)
    defl.loc[defl == 0] = mn - (mn * perc)
    if x0 == True:
        plt.plot(np.array(range(len(infl))), infl, ls='solid', color='magenta', label='inflate', linewidth=0.7)
        plt.plot(np.array(range(len(defl))), defl, ls='--', color='lightpink', label='deflate', linewidth=0.7)
    else:
        plt.plot(infl, ls='solid', color='magenta', label='inflate', linewidth=0.7)
        plt.plot(defl, ls='--', color='lightpink', label='deflate', linewidth=0.7)

def ecgnormnpeak(df, mpd):
    ecg = df['LC', 'ECG'].copy()
    ecg = normalize(ecg)
    mph = (max(ecg)-min(ecg))*0.5
    r_peak = detect_peaks(ecg, mph=mph, mpd=mpd)
    return ecg, r_peak

# ------------------- # Analysis fxs
def proc_scat(df, rolWin, proc=1):
    r_ACscat = scatMan_AC(df, 'radSlopePh', 'SlopeAC')
    r_DCscat = scatMan_DC(df, 'radSlopePh', 'SlopeDC')
    scat = pd.concat([r_ACscat, r_DCscat], axis=1)
    if proc == 1:
        scat = scat.mean()
    elif proc == 2:
        scat = scat.rolling(rolWin).mean().bfill()
    elif proc == 3:
        scat = scat
    return scat

def hbdif_sigs(df, numVO, numAO):
    """ Calculates HbDiff and corrected HbDiff signals """
    for i in range(numVO, numVO + numAO):
        df[i] = hbdif(df[i], HbSrce)
        df[i] = chb(df[i], HbSrce)
    return df

def hbsigcalc(df, scatdf, rolWin):
    """ Calculates Hb signals for AC and DC sources """
    corrAC_ua = ua_ACscat(scatdf['AC scat'], df['SlopeAC'], df['radSlopePh'])
    corrDC_ua = ua_DCscat(scatdf['DC scat'], df['SlopeDC'])
    Hb_AC = Hb_sigs(corrAC_ua['wl1'], corrAC_ua['wl2'], 'Hb_AC')
    Hb_AC = Hb_AC.rolling(rolWin).mean().bfill()
    Hb_DC = Hb_sigs(corrDC_ua['wl1'], corrDC_ua['wl2'], 'Hb_DC')
    Hb_DC = Hb_DC.rolling(rolWin).mean().bfill()
    hbdf = pd.concat([Hb_AC, Hb_DC], axis=1)
    return hbdf

def restScat(df, rolWin):
    """ User select resting data for resting us calculations """
    df = radPhase(df)
    instr = 'Select Resting Segment for Constant Scattering'
    fig = plt.figure(2, figsize=(14, 8))
    hPlotter(df, ['SlopeDC'], ['wl1', 'wl2'], instr, 'Slope DC', 'Sample')
    rest = pickpoint(df, 2)
    r_dat = stage_dfs(df, 1, rest)
    constScat = proc_scat(r_dat[0], rolWin, proc=1)
    hbdf = hbsigcalc(df, constScat, rolWin)
    hbdf.rename(columns={'Hb_AC':'Hb_csAC', 'Hb_DC':'Hb_csDC'}, inplace=True)
    df = pd.concat([df, hbdf], axis=1)
    scat = DataFrame(constScat, columns=[['rest'], ['rest']]).T
    return df, scat

def occ_name(occName, numOcc, numRepeat=1):
    occIndex = []
    for i in range(numOcc):
        x = [occName[i]] * numRepeat
        occIndex.extend(x)
    return occIndex

def datSeg(df, source, numVO, numAO=0, exP=0, instr=1, LC=False):
    """ Plots data for segment selection """
    if instr == 1:
        title = 'pick start/end of each stage'
    elif instr == 2:
        title = 'pick st of ' + str(numVO) + 'x VO and ' + str(numAO) + 'x AO and 2 x perf segs'
    elif instr == 3:
        title = 'pick st of ' + str(numVO) + 'x VO and ' + str(numAO) + 'x AO'
    elif instr == 4:
        title = 'pick start/end of each MI test'
    else: instr = instr
    fig = plt.figure(1, figsize=(14, 8))
    hPlotter(df[source], ['O2Hb', 'HHb', 'tHb'], ['O2Hb', 'HHb', 'tHb'], title, '[Hb] (uM)', 'Sample',
             colors=['r', 'b', 'lawngreen'])
    if LC == True:
        markers(df, df[source].loc[:,('O2Hb', 'HHb', 'tHb')])
    else: pass
    points = pickpoint(df['Hb_csAC'], (numVO + numAO + exP))
    return points

def rOccSeg(df, points, Hz, sec=17):
    """ Puts VOs into a list and calculates resting perfusion """
    rOccs = []
    for i in range(len(points) - 5):
        x = df.ix[points[i]:points[i] + sec * Hz]
        rOccs.append(x)
    y = df.ix[points[-5]:points[-5] + sec * Hz * 2]
    rOccs.append(y)
    return rOccs

def occSeg(df, points, Hz, sec=15):
    """ Puts VOs and perfusion segment into a list"""
    occs = []
    for i in range(len(points)):
        x = df.ix[points[i]:points[i] + sec * Hz]
        occs.append(x)
    perfSeg = df.ix[points[0]-45*Hz:points[0]]
    return occs, perfSeg

def perfusion(df, rolWin):
    """ Calculates perfusion for a given data segment """
    avgScat = proc_scat(df, rolWin, proc=2)
    hbdf = hbsigcalc(df, avgScat, rolWin)
    df = pd.concat([df, hbdf], axis=1)
    tHb = df.xs('tHb', axis=1, level=1).mean()
    SaO2 = df.xs('SaO2', axis=1, level=1).mean()
    perf = pd.concat([tHb, SaO2], axis=1)
    return perf

def r_perfusion(df, points, rolWin):
    """ Calculates resting perfusion """
    r = df.ix[points[-4]:points[-3]]
    bf = df.ix[points[-2]:points[-1]]
    r_p = perfusion(r, rolWin)
    bf_p = perfusion(bf, rolWin)
    perf = pd.concat([r_p, bf_p], axis=1)
    perf = DataFrame(perf)
    perf.columns = [['rest']*4,['perf', 'SaO2', 'r_t[Hb]', 'r_SaO2']]
    perf = perf.T
    print(perf)
    return perf

def caprange_calc(df, stage, Hz, mpd=3.8, delmax=[], delmin=[]):
    sigdat = pd.concat([df['Hb_csAC', 'tHb'], df['Hb_csDC', 'tHb'], df['Hb_csAC', 'SaO2'], df['Hb_csDC', 'SaO2']],
                       axis=1)
    cols = ['Hb_csAC', 'Hb_csDC']
    capDat = []
    for i in cols:
        cap = caprange(sigdat, i, Hz, mpd, delmax, delmin)
        capDat.append(cap)
    plt.title('Select 8x st/end between-kick perfusion segments for ' + str(cols))
    result = Mbox('Continue?', 'Are max and min points correct?', 4)
    if result == 7:
        plt.close()
    if result == 6:
        points = pickpoint(df, 16, reset=False)
        points = np.array(points, dtype=int)
        perfSegs = []
        for i in np.arange(1, 9):
            y = sigdat.ix[points[(i * 2 - 1) - 1]:points[(i * 2) - 1]]
            perfSegs.append(y)
        segs = pd.concat(perfSegs)
        perf = DataFrame.mean(segs)
        capDat[0].append(perf[0])
        capDat[0].append(perf[2])
        capDat[1].append(perf[1])
        capDat[1].append(perf[3])
        capdf = DataFrame(capDat, index=cols, columns=['Max', 'Min', 'Range', 'Perf', 'SaO2'])
        capdf.columns = ([[stage] * 5, ['Max', 'Min', 'Range', 'Perf', 'SaO2']])
        capdf = capdf.T
        print(capdf)
        return capdf, points

def caprange_calc_rol(df, rolWin, stage, Hz, points, mpd=3.8, delmax=[], delmin=[]):
    exPerfScat = proc_scat(df, rolWin, proc=2)
    exHb = hbsigcalc(df, exPerfScat, rolWin)
    exHb = pd.concat([df, exHb], axis=1)
    sigdat = pd.concat([exHb['Hb_AC', 'tHb'], exHb['Hb_DC', 'tHb'], exHb['Hb_AC', 'SaO2'], exHb['Hb_DC', 'SaO2']],
                       axis=1)
    cols = ['Hb_AC', 'Hb_DC']
    capDat = []
    for i in cols:
        cap = caprange(sigdat, i, Hz, mpd, delmax, delmin)
        capDat.append(cap)
    plt.title('Are Max and Mins correct?')
    result = Mbox('Continue?', 'Are max and min points correct?', 4)
    if result == 7:
        plt.close()
    if result == 6:
        plt.close()
        perfSegs = []
        for i in np.arange(1, 9):
            y = sigdat.ix[points[(i * 2 - 1) - 1]:points[(i * 2) - 1]]
            perfSegs.append(y)
        segs = pd.concat(perfSegs)
        perf = DataFrame.mean(segs)
        capDat[0].append(perf[0])
        capDat[0].append(perf[2])
        capDat[1].append(perf[1])
        capDat[1].append(perf[3])
        capdf = DataFrame(capDat, index=cols, columns=['Max', 'Min', 'Range', 'Perf', 'SaO2'])
        capdf.columns = ([[stage] * 5, ['Max', 'Min', 'Range', 'Perf', 'SaO2']])
        capdf = capdf.T
        print(capdf)
        return capdf

def caprange(df, source, Hz, mpd, delmax, delmin):
    """Detects maximum and minimum of signal and annotates on plot,
    calculates tHb range and averages exercise perfusion segments"""
    maxs = detect_peaks(df[source, 'tHb'], mpd=mpd*Hz)
    mins = detect_peaks(df[source, 'tHb'], mpd=mpd*Hz, valley=True)
    maxz = np.delete(maxs, delmax)
    minz = np.delete(mins, delmin)
    max = df.iloc[maxz][source, 'tHb'].mean()
    min = df.iloc[minz][source, 'tHb'].mean()
    range = max - min
    cap = [max, min, range]
    plt.figure(4, figsize=(14, 10))
    plt.plot(df[source, 'tHb'])
    plt.plot(df[source, 'tHb'].iloc[maxz], ls='None', marker='D', color='b')
    plt.plot(df[source, 'tHb'].iloc[minz], ls='None', marker='o', color='r')
    antpoint(df.iloc[maxs].index.values, df[source, 'tHb'].iloc[maxs])
    antpoint(df.iloc[mins].index.values, df[source, 'tHb'].iloc[mins])
    plt.show(block=False)
    return cap

def pickOccScat(df, points, numVO, numAO, occDflist, rolWin):
    """ Calculates scattering for user selected segments """
    occScatRol = proc_scat(df, rolWin, proc=2)
    y = occScatRol['DC scat', 'wl1'].iloc[0] + (occScatRol['DC scat', 'wl2'].iloc[0] - occScatRol['DC scat', 'wl1'].iloc[0])/2
    a = np.empty(len(points))
    a.fill(y)
    plt.figure(3, figsize=(14, 10))
    plt.plot(points, a, ls='', marker='+')
    title = 'Pick st of ' + str(numVO) + 'x VO and ' + str(numAO) + 'x AO'
    hPlotter(occScatRol, ['DC scat'], ['Chosen Occs', 'wl1', 'wl2'], title, 'ua (1/cm)', 'Sample')
    occSegs = pickpoint(df['Hb_csDC'], (numVO + numAO)*2)
    occScatdfs = stage_dfs(df, (numVO + numAO), occSegs, seg=2)
    scats = []
    for i in range(len(occScatdfs)):
        occScat = proc_scat(occScatdfs[i], rolWin, proc=1)
        scats.append(occScat)
    Hb_dfs = []
    for i in range(len(occScatdfs)):
        HbDf = hbsigcalc(occDflist[i], scats[i], rolWin)
        Hb_const = occDflist[i].drop(['SlopeAC', 'SlopeDC', 'SlopePh', 'radSlopePh'], axis=1, level=0)
        HbDf = pd.concat([Hb_const, HbDf], axis=1)
        Hb_dfs.append(HbDf)
    voIndex = occ_name(voName, numVO)
    aoIndex = occ_name(aoName, numAO)
    scats = DataFrame(scats, index=voIndex + aoIndex)
    return Hb_dfs, scats

def occ_scat(df, occ_df, numOcc, rolWin):
    scats = []
    Hb_dfs = []
    for i in range(numOcc):
        occScat = proc_scat(occ_df[i], rolWin)
        occHb = hbsigcalc(df[i], occScat, rolWin)
        Hb_const = df[i].drop(['SlopeAC', 'SlopeDC', 'SlopePh', 'radSlopePh'], axis=1, level=0)
        HbDf = pd.concat([Hb_const, occHb], axis=1)
        scats.append(occScat)
        Hb_dfs.append(HbDf)
    Hb_dfs = hbdif_sigs(Hb_dfs, 0, numOcc)
    return Hb_dfs, scats

def pickVoccslp(occlist, stage, source, Hz, rolWin, numVO=4, mpd=25, LC=False):
    """ User selects segments for VO and AO slope analysis """
    if LC == True:
        ecg, r_peak = [], []
        for i in range(numVO):
            e, r = ecgnormnpeak(occlist[i], mpd=mpd)
            ecg.append(e)
            r_peak.append(r)
        vo_plt = [(0, 0), (0, 1), (3, 0), (3, 1)]
        ecg_plt = [(2, 0), (2, 1), (5, 0), (5, 1)]
        plt.figure(figsize=(14, 10))
        for i in range(numVO):
            plt.subplot2grid((6, 2), vo_plt[i], rowspan=2)
            plt.plot(occlist[i][source, 'tHb'], 'g', label=['VO' + str(i+1)])
            markers(occlist[i], occlist[i][source, 'tHb'], perc= 0.001, onesig=True)
            plt.legend(loc='lower right')
        for i in range(numVO):
            plt.subplot2grid((6, 2), ecg_plt[i], rowspan=1)
            plt.plot(ecg[i], 'r')
            markers(occlist[i], ecg[i], perc= 0.03, onesig=True)
            plt.plot(ecg[i].iloc[r_peak[i]], ls='None', marker='+', color='b')
            antpoint(ecg[i].iloc[r_peak[i]].index, ecg[i].iloc[r_peak[i]], adj=0.05)
        plt.tight_layout()
        plt.show()
        plt.pause(0.001)
        rpks = input('Input first four r peaks for VOs')
        rpks = [x.strip() for x in rpks.split(',')]
        rpks = np.array(rpks, dtype=int)
        plt.close()
        z = [0, 4, 8, 12]
        ccs = []
        for i in range(numVO):
            for j in range(4):
                x = r_peak[i][rpks[z[i]+j]]
                ccs.append(x)
    else:
        plt.figure(3, figsize=(14, 10))
        for i in range(numVO):
            plt.plot(np.arange(len(occlist[i])), occlist[i][source, 'tHb'], label=['VO' + str(i+1)])
        plt.legend(loc='lower right')
        plt.title('Pick start of cardiac cycles 1-4 for ' + str(numVO) + 'x VO')
        ccs = pickpoint(occlist[0], numVO*4)
    ccs = np.array(ccs, dtype=int)
    ccslist = []
    HbOccDat = []
    vo = list(range(0, numVO*4, 4))
    for p in np.arange(numVO):
        j = vo[p]
        z = occlist[p].iloc[ccs[j] - 1:ccs[j + 3] - 1]
        HbOccDat.append(z)
    Hb_dfs, occScat = occ_scat(occlist[:numVO], HbOccDat, len(HbOccDat), rolWin)
    for p in np.arange(numVO):
        j = vo[p]
        HbDat = Hb_dfs[p].xs('tHb', axis=1, level=1)
        for i in np.arange(3):
            y = HbDat.iloc[ccs[j+i]-1:ccs[j+i+1]-1]
            for t in HbSrce:
                q, *b = stats.linregress(y.index.values / Hz, y[t].values)
                ccslist.append(q)
        z = HbDat.iloc[ccs[j]-1:ccs[j+3]-1]
        for u in HbSrce:
            w, *b = stats.linregress(z.index.values / Hz, z[u].values)
            ccslist.append(w)
    ccslist = np.reshape(ccslist, (numVO*4, len(HbSrce)))
    voIndex = occ_name(voName, numVO, 4)
    voslps = DataFrame(ccslist, index=[[stage] * numVO*4, voIndex, ['cc1', 'cc2', 'cc3', 'cc1:3'] * numVO],
                       columns=HbSrce)
    occScat = DataFrame(occScat, index=[[stage]*numVO, occ_name(voName, numVO)])
    print(voslps)
    return voslps, occScat

def pickAoccslp(occlist, stage, source, sig, Hz, rolWin, numVO=4, numAO=2, LC=False):
    """ Calculates linear regression through given signals for user selected data segments """
    plt.figure(2, figsize=(14, 10))
    for i in range(numVO, numVO + numAO):
        plt.plot(np.arange(len(occlist[i])), occlist[i][source, sig], label=['AO' + str(i-numVO+1)])
        if LC == True:
            markers(occlist[i], occlist[i][source, sig], onesig=True, x0=True, perc = 0.001)
        else:
            pass
    plt.legend(loc = 'lower right')
    plt.title('Pick st/end of ' + str(numAO) + 'x AO')
    aos = pickpoint(occlist[numVO], numAO*2)
    aos = np.array(aos, dtype=int)
    aolist = []
    HbOccDat = []
    for j in range(numAO):
        a = occlist[numVO + j].iloc[aos[((j+1) * 2 - 1)-1]:aos[((j+1) * 2)-1]]
        HbOccDat.append(a)
    Hb_dfs, occScat = occ_scat(occlist[numVO:numVO+numAO], HbOccDat, numAO, rolWin)
    for j in range(numAO):
        a = Hb_dfs[j].iloc[aos[((j+1) * 2 - 1)-1]:aos[((j+1) * 2)-1]]
        for p in HbSrce:
            HbDat = a.xs(p, axis=1, level=0)
            for s in HbSig:
                q, *b = stats.linregress(HbDat.index.values / Hz, HbDat[s])
                aolist.append(q)
    aolist = np.reshape(aolist, (numAO * len(HbSig), len(HbSrce)))
    aoslps = []
    for i in range(numAO):
        j = i*len(HbSrce)
        y = aolist[j:j+len(HbSig)].T
        aoslps.extend(y)
    aoIndex = occ_name(aoName, numAO, len(HbSig))
    aoslps = DataFrame(aoslps, index=[[stage] * numAO * len(HbSig), aoIndex, HbSig * numAO],
                                      columns=HbSrce)
    occScat = DataFrame(occScat, index=[[stage] * numAO, occ_name(aoName, numAO)])
    print(aoslps)
    return aoslps, occScat

def mito_occ(df, source, numAO, LC=False):
    plt.figure(3, figsize=(14, 10))
    colors = ['r', 'b']
    hPlotter(df[source], ['O2Hb', 'HHb'], ['O2Hb', 'HHb'], 'MI test', '[Hb] (uM)', 'Sample', colors=colors)
    plt.tight_layout()
    if LC == True:
        markers(df, df[source].loc[:,('O2Hb', 'HHb')])
    else:
        pass
    MIPts = pickpoint(df, numAO * 2)
    return MIPts

def mitoslopes(df, points, stage, Hz, rolWin):
    """ Calculates and plots MI occ slopes """
    numAO = len(points) // 2
    MI_dfs = [df] * numAO
    MI_scatDfs = stage_dfs(df, numAO, points, seg=2)
    Hb_dfs, MIscat = occ_scat(MI_dfs, MI_scatDfs, numAO, rolWin)
    ao_dfs = []
    for i in range(numAO):
        y = Hb_dfs[i].ix[points[((i+1)*2-1)-1]:points[((i+1)*2)-1]]
        ao_dfs.append(y)
    aolist = []
    time = []
    for i in HbSig:
        for k in range(numAO):
            t = ao_dfs[k].index.values / Hz
            x = t.mean()
            time.append(x)
            for j in HbSrce:
                q, *b = stats.linregress(t, ao_dfs[k][j, i])
                aolist.append(q)
    timeList = []
    for i in range(0, numAO * len(HbSig), numAO):
        x = time[i:i + numAO] - time[i]
        timeList.extend(x)
    timeList = np.round(timeList, 2)
    aolist = np.reshape(aolist, (numAO * len(HbSig), len(HbSrce)))
    aoIndex = occ_name(aoName, numAO)
    HbIndex = occ_name(HbSig, len(HbSig), numAO)
    aoslps = DataFrame(aolist, index=[HbIndex, timeList], columns=HbSrce)
    MIscat = DataFrame(MIscat, index=[[stage]*len(aoIndex), aoIndex])
    return aoslps, MIscat

def plot_mi(slps, test):
    """ Plots MI slopes for all Hb Sigs and all Hb sources """
    plt.figure(3, figsize=(14, 8))
    plt.suptitle(test + ' slopes')
    plt.xlabel('time (s)')
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(slps.loc[HbSig[0]], marker='o')
    ax1.legend(HbSrce, loc='upper right')
    plt.ylabel(HbSig[0])
    plt.xlabel('time (s)')
    antpoint(slps.loc[HbSig[0]].index.values, slps.loc[HbSig[0], HbSrce[3]].values)
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(slps.loc[HbSig[1]], marker='o')
    ax2.legend(HbSrce, loc='lower right')
    plt.ylabel(HbSig[1])
    plt.xlabel('time (s)')
    antpoint(slps.loc[HbSig[0]].index.values, slps.loc[HbSig[1], HbSrce[3]].values)
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(slps.loc[HbSig[2]], marker='o')
    ax3.legend(HbSrce, loc='upper right')
    plt.ylabel(HbSig[2])
    plt.xlabel('time (s)')
    antpoint(slps.loc[HbSig[0]].index.values, slps.loc[HbSig[2], HbSrce[3]].values)
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(slps.loc[HbSig[3]], marker='o')
    ax4.legend(HbSrce, loc='lower right')
    plt.ylabel(HbSig[3])
    plt.xlabel('time (s)')
    antpoint(slps.loc[HbSig[0]].index.values, slps.loc[HbSig[3], HbSrce[3]].values)

def curvefit(slps, sig, srce, fileName, test):
    """ Fits slope points to fitFunc and prints graph """
    time, slopes = slps.loc[sig].index.values, slps.loc[sig, srce].values
    fitParams, fitCovs = curve_fit(fitFunc, time, slopes, p0=np.random.rand(1, 3))
    A, B, k, tc = fitParams[0], fitParams[1], fitParams[2], 1/fitParams[2]
    y = fitFunc(tc, A, B, k)
    x = np.linspace(0, time[-1], 200)
    y_mod = fitFunc(x, A, B, k)
    plt.plot(time, slopes, marker='o', color='b', label=sig)
    plt.plot(x, y_mod, 'r--', label='$f(t) = %.3f$ - %.3f e^(-%.3f t)' % (A,B,k))
    plt.annotate('tc=' + str(round(tc, 2)) + '(s)', size=15, xy=(tc, y), xytext=(tc + 50, y),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3", facecolor='black'))
    plt.legend(loc='lower right')
    plt.title(test)
    plt.xlabel('time (s)')
    plt.ylabel('mVO2(slope)')
    plt.savefig(fileName + '_' + sig + '_' + srce + '_' + test + '.png')
    return fitParams, fitCovs

def curvefit_srce(slps, srce, fileName, test):
    """ Fits slope points to fitFunc and prints graph """
    time = slps.loc['HHb'].index.values
    HHb, HbDif, cHHb, cHbDif = slps.loc['HHb', srce].values, slps.loc['HbDif', srce].values, \
                               slps.loc['cHHb', srce].values, slps.loc['cHbDif', srce].values
    x = np.linspace(0, time[-1], 200)
    fP, SD = [], []
    A, B, k, tc = [], [], [], []
    y, y_mod = [], []
    err = float("inf")
    num_attempts = 100
    for i in [HHb, HbDif, cHHb, cHbDif]:
        popt, pcov = curve_fit(fitFunc, time, i, bounds=([-np.inf, -np.inf, 0.005], [np.inf, np.inf, 0.067]))
        A_var, B_var, k_var = popt
        tc_var = 1/k_var
        y_var = fitFunc(tc_var, A_var, B_var, k_var)
        y_mod_var = fitFunc(x, A_var, B_var, k_var)
        y_mod_var = fitFunc(x, A_var, B_var, k_var)
        fP.append([A_var, B_var, k_var, tc_var])
        SD.append(np.sqrt(np.diag(pcov)))
        A.append(A_var)
        B.append(B_var)
        k.append(k_var)
        tc.append(tc_var)
        y.append(y_var)
        y_mod.append(y_mod_var)
    # Figure
    fig = plt.figure(3, figsize=(12, 8))
    fig.suptitle(srce + ' ' + test)
    # Ax 1
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(time, HHb, marker='o', color='b', label='HHb')
    ax1.plot(x, y_mod[0], 'r--', label='$f(t) = %.3f$ - %.3f e^(-%.3f t)' % (A[0],B[0],k[0]))
    ax1.annotate('tc=' + str(round(tc[0], 2)) + '(s)', size=15, xy=(tc[0], y[0]), xytext=(tc[0] + 50, y[0]),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3", facecolor='black'))
    ax1.legend(loc='upper right')
    plt.xlabel('time (s)')
    plt.ylabel('HHb mVO2(slope)')
    # Ax 2
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(time, HbDif, marker='o', color='b', label='HbDif')
    ax2.plot(x, y_mod[1], 'r--', label='$f(t) = %.3f$ - %.3f e^(-%.3f t)' % (A[1],B[1],k[1]))
    ax2.annotate('tc=' + str(round(tc[1], 2)) + '(s)', size=15, xy=(tc[1], y[1]), xytext=(tc[1] + 50, y[1]),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3", facecolor='black'))
    ax2.legend(loc='lower right')
    plt.xlabel('time (s)')
    plt.ylabel('mVO2(slope)')
    # Ax 3
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(time, cHHb, marker='o', color='b', label='cHHb')
    ax3.plot(x, y_mod[2], 'r--', label='$f(t) = %.3f$ - %.3f e^(-%.3f t)' % (A[2],B[2],k[2]))
    ax3.annotate('tc=' + str(round(tc[2], 2)) + '(s)', size=15, xy=(tc[2], y[2]), xytext=(tc[2] + 50, y[2]),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3", facecolor='black'))
    ax3.legend(loc='upper right')
    plt.xlabel('time (s)')
    plt.ylabel('mVO2(slope)')
    # Ax 4
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(time, cHbDif, marker='o', color='b', label='cHbDif')
    ax4.plot(x, y_mod[3], 'r--', label='$f(t) = %.3f$ - %.3f e^(-%.3f t)' % (A[3],B[3],k[3]))
    ax4.annotate('tc=' + str(round(tc[3], 2)) + '(s)', size=15, xy=(tc[3], y[3]), xytext=(tc[3] + 50, y[3]),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3", facecolor='black'))
    ax4.legend(loc='lower right')
    plt.xlabel('time (s)')
    plt.ylabel('mVO2(slope)')
    plt.savefig(fileName + '_' + srce + '_' + test + '.png')
    fP = DataFrame(fP, index=[[test + '_' + srce]*len(HbSig), HbSig], columns=['A', 'B', 'k', 'tc'])
    SD = DataFrame(SD, index=[[test + '_' + 'SD']*len(HbSig), HbSig], columns=['A', 'B', 'k'])
    return fP, SD