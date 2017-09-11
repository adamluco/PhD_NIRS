import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import ctypes
import pickle

def datImp(filename, columns, st_row, cols):
    """ Import .csv file as table """
    data = pd.read_table(filename, header=None, names=columns, skiprows=np.arange(st_row), usecols=cols,
                         skip_blank_lines=True, low_memory=False)
    data = DataFrame(data)
    return data

def levels(df, sig, tx):
    """ Reindex Artinis DataFrame and create individual signal Data Frames """
    df = (df.drop(['event', 'Sample'], axis=1)
            .reindex(columns=['time', 'TSI', 't1_O2Hb', 't2_O2Hb', 't3_O2Hb', 't1_HHb', 't2_HHb', 't3_HHb',
                         't1_tHb', 't2_tHb', 't3_tHb']))
    signals = []
    for i in sig:
        y = [i]*len(tx)
        signals.extend(y)
    df.columns = [['a'] * 2 + signals,
                  ['time', 'TSI'] + tx * len(sig)]
    return df

def hbdif(df, tx):
    """Calculates HbDif and adds to DataFrame"""
    HbDif = df['O2Hb'] - df['HHb']
    HbDif.columns = [['HbDif'] * len(tx), tx]
    dat = [df, HbDif]
    df = pd.concat(dat, axis=1)
    return df

def chb(df, tx):
    """Calculates corrected Hb signals"""
    beta = df['O2Hb'] / df['tHb']
    beta.columns = [['beta'] * len(tx), tx]
    dat1 = [df, beta]
    df = pd.concat(dat1, axis=1)
    cO2Hb = df['O2Hb'] - df['tHb'] * (1 - df['beta'])
    cHHb = df['HHb'] - (df['tHb'] * df['beta'])
    cO2Hb.columns = [['cO2Hb'] * len(tx), tx]
    cHHb.columns = [['cHHb'] * len(tx), tx]
    dat2 = [df, cO2Hb, cHHb]
    df = pd.concat(dat2, axis=1)
    cHbDif = df['cO2Hb'] - df['cHHb']
    cHbDif.columns = [['cHbDif'] * len(tx), tx]
    dat3 = [df, cHbDif]
    df = pd.concat(dat3, axis=1)
    return df

def datSeg(df, num, Hz, instr=0, reset=False):
    """Plots all data t1-t3 for stage selection"""
    plt.figure(1, figsize=(14, 10))
    tHbColor = {0:'lightgreen'}
    plt.plot(df['tHb', 't1'], 'lightgreen')
    plt.plot(df['tHb', 't2'], 'lawngreen')
    plt.plot(df['tHb', 't3'], 'g-')
    plt.plot(df['O2Hb', 't1'], 'lightcoral')
    plt.plot(df['O2Hb', 't2'], 'r-')
    plt.plot(df['O2Hb', 't3'], 'firebrick')
    plt.plot(df['HHb', 't1'], 'lightskyblue')
    plt.plot(df['HHb', 't2'], 'b-')
    plt.plot(df['HHb', 't3'], 'navy')
    plt.xlabel('sample (' + str(Hz) + 'Hz)')
    plt.ylabel('[Hb] ($\mu$M)')
    if instr == 0:
        plt.title('pick st/end of each stage')
        print('pick st/end of each stage')
    elif instr == 1:
        plt.title('pick st of 4 x VO and 2 x AO, and st/end of perfusion segment')
        print('pick st of 4 x VO and 2 x AO, and st/end of perfusion segment')
    elif instr == 2:
        plt.title('pick st of 4 x VO and 2 x AO')
        print('pick st of 4 x VO and 2 x AO')
    points = pickpoint(df, num, reset=reset)
    plt.close()
    return points

def occplot(df, num, Hz):
    """ Plots t1-t3 tHb and HHb signals """
    plt.figure(2, figsize=(14, 10))
    plt.plot(df['tHb', 't1'], 'lightgreen')
    plt.plot(df['tHb', 't2'], 'lawngreen')
    plt.plot(df['tHb', 't3'], 'g-')
    plt.plot(df['HHb', 't1'], 'lightskyblue')
    plt.plot(df['HHb', 't2'], 'b-')
    plt.plot(df['HHb', 't1'], 'navy')

def rOccSeg(df, points, Hz, sec=17):
    """ Puts VOs into a list and calculates resting perfusion """
    rOccs = []
    for i in range(len(points) - 1):
        x = df.ix[points[i]:points[i] + sec * Hz]
        rOccs.append(x)
    y = df.ix[points[-1]:points[-1] + sec * Hz * 2]
    rOccs.append(y)
    return rOccs

def restperf(df):
    plt.figure(1, figsize=(14, 10))
    plt.plot(df['tHb', 't1'], 'lightgreen')
    plt.plot(df['tHb', 't2'], 'lawngreen')
    plt.plot(df['tHb', 't3'], 'g-')
    plt.plot(df['O2Hb', 't1'], 'lightcoral')
    plt.plot(df['O2Hb', 't2'], 'r-')
    plt.plot(df['O2Hb', 't3'], 'firebrick')
    plt.ylabel('[Hb] ($\mu$M)')
    plt.title('Select perfusion segment')
    points = pickpoint(df, 2, reset=False)
    plt.close()
    x = df.ix[points[0]:points[1]]
    plt.figure(1, figsize=(14, 10))
    plt.plot(x['tHb', 't1'], 'lightgreen')
    plt.plot(x['tHb', 't2'], 'lawngreen')
    plt.plot(x['tHb', 't3'], 'g-')
    plt.plot(x['O2Hb', 't1'], 'lightcoral')
    plt.plot(x['O2Hb', 't2'], 'r-')
    plt.plot(x['O2Hb', 't3'], 'firebrick')
    plt.ylabel('[Hb] ($\mu$M)')
    plt.title('Define perfusion segment')
    perfPoints = pickpoint(df, 2, reset=False)
    plt.close()
    perf = df.ix[perfPoints[0]:perfPoints[1]]
    frames = DataFrame.mean(perf['tHb'])
    rPerf = DataFrame(frames, columns=[['rest'], ['perf']]).transpose()
    print(rPerf)
    return rPerf


def occSeg(df, points, Hz, sec=15):
    """ Puts VOs and perfusion segment into a list"""
    occs = []
    for i in range(len(points)):
        x = df.ix[points[i]:points[i] + sec * Hz]
        occs.append(x)
    perfSeg = df.ix[points[0]-45*Hz:points[0]]
    return occs, perfSeg

def pickVoccslp(occlist, stage, tx, numVO=4):
    """ User selects segments for VO and AO slope analysis """
    plt.figure(3, figsize=(14, 10))
    for i in range(numVO):
        plt.plot(np.arange(len(occlist[i])), occlist[i]['tHb', 't2'], label=['VO' + str(i)])
    plt.legend(loc='lower right')
    plt.title('Pick start of cardiac cycles 1-4 for 4xVO')
    ccs = pickpoint(occlist[0], numVO*4)
    ccs = np.array(ccs, dtype=int)
    ccslist = []
    vo = list(range(0,numVO*4,4))
    for p in range(numVO):
        j = vo[p]
        for i in np.arange(3):
            y = occlist[p]['tHb'].iloc[ccs[j+i]-1:ccs[j+i+1]-1]
            for t in tx:
                q, *b = stats.linregress(y.index.values, y[t])
                ccslist.append(q)
        z = occlist[p]['tHb'].iloc[ccs[j]-1:ccs[j+3]-1]
        for u in tx:
            w, *b = stats.linregress(z.index.values, z[u])
            ccslist.append(w)
    voName = {0: 'VO1', 1: 'VO2', 2: 'VO3', 3: 'VO4', 4:'VO5', 5:'VO6', 6:'VO7', 7:'VO8', 8:'VO9', 9:'V10'}
    voIndex = []
    for i in range(numVO):
        x = [voName[i]] * 4
        voIndex.extend(x)
    voslps = DataFrame(ccslist, index=[[stage] * numVO*4, voIndex, ['cc1', 'cc2', 'cc3', 'cc1:3'] * numVO],
                       columns=tx)
    return voslps

def pickAoccslp(occlist, stage, tx, numVO=4, numAO=2):
    plt.figure(4, figsize=(14, 10))
    for i in range(numVO, numVO + numAO):
        plt.plot(np.arange(len(occlist[i])), occlist[i]['HbDif', 't2'])
    plt.legend('lower right')
    plt.title('Pick st/end of 2xAO')
    aos = pickpoint(occlist[numVO+1], numAO * 2)
    aos = np.array(aos, dtype=int)
    aolist = []
    for p in ['HHb', 'HbDif', 'cHHb', 'cHbDif']:
        for j in range(numAO):
            a = occlist[numVO + j][p].iloc[aos[((j+1) * 2 - 1)-1]:aos[((j+1) * 2)-1]]
            for t in tx:
                q, *b = stats.linregress(a.index.values, a[t])
                aolist.append(q)
    aoslps = DataFrame(aolist, index=[[stage] * 8, ['HHb']*2+['HbDif']*2+['cHHb']*2+['cHbDif']*2, ['AO1', 'AO2'] * 4],
                                      columns=tx)
    return aoslps

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
        Nirs._plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

def antpoint(x, y):
    """Annotate data with consecutive numbers"""
    i = x.tolist()
    j = y.tolist()
    n = list(range(len(y)))
    for k, txt in enumerate(n):
        plt.annotate(txt, (i[k], j[k]+0.10))

def Mbox(title, text, style):
    """ Creates a message box """
    result = ctypes.windll.user32.MessageBoxW(0, text, title, style)
    return result

def caprange(df, sig, tx, WL, Hz, mpd=3.8, delmax=[], delmin=[]):
    """Detects maximum and minimum of signal and annotates on plot,
    calculates tHb range and averages exercise perfusion segments"""
    maxs = detect_peaks(df[sig, tx], mpd=mpd*Hz)
    mins = detect_peaks(df[sig, tx], mpd=mpd*Hz, valley=True)
    maxz = np.delete(maxs, delmax)
    minz = np.delete(mins, delmin)
    max = df.iloc[maxz][sig].mean()
    min = df.iloc[minz][sig].mean()
    plt.figure(4, figsize=(14, 10))
    plt.plot(df[sig, tx], 'g')
    plt.plot(df[sig, tx].iloc[maxz], ls='None', marker='D', color='b')
    plt.plot(df[sig, tx].iloc[minz], ls='None', marker='o', color='r')
    antpoint(df.iloc[maxs].index.values, df[sig, tx].iloc[maxs])
    antpoint(df.iloc[mins].index.values, df[sig, tx].iloc[mins])
    plt.show(block=False)
    range = max - min
    result = Mbox('Continue?', 'Are max and min points correct?', 4)
    if result == 7:
        plt.close()
    if result == 6:
        plt.title('Select 8x st/end between-kick perfusion segments')
        points = pickpoint(df, 16, reset=False)
        points = np.array(points, dtype=int)
        perfSegs = []
        for i in np.arange(1, 9):
            y = df[sig].ix[points[(i * 2 - 1) - 1]:points[(i * 2) - 1]]
            perfSegs.append(y)
        segs = pd.concat(perfSegs)
        perf = DataFrame.mean(segs)
    cap = [max, min, range, perf]
    cap = DataFrame(cap, columns=['t1', 't2', 't3'], index=[[WL]*4, ['max', 'min', 'range', 'perf']])
    print(cap)
    return cap

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

def stage_dfs(df, num_stages, points):
    """ Puts stage DataFrame segments into a list """
    stages = []
    for i in np.arange(1, num_stages+1):
        y = df.ix[points[(i*2-1)-1]:points[(i*2)-1]]
        stages.append(y)
    return stages

def isc_norm(df, tx):
    plt.figure(1, figsize=(14, 10))
    plt.plot(df['HbDif'])
    points = pickpoint(df['HbDif'], 2)
    isc = df.ix[points[0]:points[1]]
    min = DataFrame.min(isc['cHbDif'])
    max = DataFrame.max(isc['cHbDif'])
    cnHbDif = (df['cHbDif'] - min) * (100/(max-min))
    cnHbDif.columns = [['cnHbDif'] * len(tx), tx]
    dat3 = [df, cnHbDif]
    df = pd.concat(dat3, axis=1)
    return df

def mito_segs(data, sig, mitoSt):
    """ User selects st/end of mito test a and b """
    plt.figure(1, figsize=(14, 10))
    try:
        mitoSt
    except NameError:
        mitoSt = 0
    plt.plot(data[sig].ix[mitoSt:])  # Plot HbDif signal
    plt.title('Select st/end of MIa and MIb segments')
    plt.legend(loc='best')
    points = pickpoint(data['HbDif'], 4, reset=True)
    return points

def aostart(df, sig, Hz=10):
    """ Defines start of first occlusion"""
    plt.plot(df[sig][20*Hz:40*Hz])
    plt.title('Pick start of first occlusion')
    start = pickpoint(df, 1)
    return start


def occmarks(df, mark, start, sig):
    """ Adds marker template to df index to get occlusion st/end """
    inf = np.zeros(len(df))
    inf[inf == 0] = min(df[sig, 't1']) - (min(df[sig, 't1']) * 0.03)
    mark = mark + start
    inf = Series(inf, index=df.index.values)
    inf.loc[mark] = df[sig, 't3'].loc[mark]
    plt.figure(3, figsize=(14, 10))
    plt.plot(df[sig, 't1'])
    plt.plot(df[sig, 't2'])
    plt.plot(df[sig, 't3'])
    plt.plot(inf, 'C6-', label='Occlusion marker')
    plt.legend(loc='lower right')
    plt.title('corrected HbDif signals')
    plt.xlabel('sample')
    plt.ylabel('O2Hb - HHb[uM]')
    antpoint(inf.loc[mark].index.values, inf.loc[mark])
    plt.show(block=False)
    return mark

def mitoslopes(df, mark, sig):
    """ Calculates slope values for occlusions """
    slps = []
    for i in np.arange(1, len(mark)/2+1, dtype=int):
        y = df.ix[mark[(i*2-1)-1]:mark[(i*2)-1]]
        q, *b = stats.linregress(y['a', 'time'], y[sig, 't1'])
        w, *b = stats.linregress(y['a', 'time'], y[sig, 't2'])
        e, *b = stats.linregress(y['a', 'time'], y[sig, 't3'])
        r = np.mean(y['a', 'time'])
        slps.append([r, q, w, e])
    slps = DataFrame(slps, columns=['time', 't1', 't2', 't3'], dtype=float)
    slps['time'] = slps['time'] - slps['time'][0]
    return slps

def fitFunc(t, A, B, k):
    """ Curve fitting function """
    return A - B*np.exp(-k*t)

def curvefit(slps, fileName, test):
    """ Fits slope points to fitFunc and prints graph """
    fitParams = []
    fitCovs = []
    for i in ['t1', 't2', 't3']:
        x, y = curve_fit(fitFunc, slps['time'], slps[i], p0=np.random.rand(1, 3))
        fitParams.append(x)
        fitCovs.append(y)
    fitParams = DataFrame(fitParams, index=['t1', 't2', 't3'], columns=['A', 'B', 'k'])
    fitParams['tc'] = 1/fitParams['k']
    fitParams = fitParams.transpose()
    t1_A, t1_B, t1_k, t1_tc = fitParams['t1']
    t2_A, t2_B, t2_k, t2_tc = fitParams['t2']
    t3_A, t3_B, t3_k, t3_tc = fitParams['t3']
    t1_y = fitFunc(t1_tc, t1_A, t1_B, t1_k)
    t2_y = fitFunc(t2_tc, t1_A, t1_B, t1_k)
    t3_y = fitFunc(t3_tc, t1_A, t1_B, t1_k)
    x = np.linspace(0, slps.iloc[len(slps) - 1, 0], 200)
    plt.figure(8, figsize=(16,5))
    ax1 = plt.subplot(1,3,1)
    ax1.plot(slps['time'], slps['t1'], marker='o', color='b')
    ax1.plot(x, fitFunc(x, t1_A, t1_B, t1_k), 'r--', label='$f(t) = %.3f$ - %.3f e^(-%.3f t)' % (t1_A,t1_B,t1_k))
    ax1.annotate('tc=' + str(round(t1_tc, 2)) + '(s)', size=15, xy=(t1_tc, t1_y), xytext=(t1_tc+50, t1_y),
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3", facecolor='black'))
    ax1.legend(loc='lower right')
    plt.xlabel('time (s)')
    plt.ylabel('mVO2(slope)')
    ax2 = plt.subplot(1,3,2)
    ax2.plot(slps['time'], slps['t2'], marker='o', color='b')
    ax2.plot(x, fitFunc(x, t2_A, t2_B, t2_k), 'r--', label='$f(t) = %.3f$ - %.3f e^(-%.3f t)' % (t2_A, t2_B, t2_k))
    ax2.annotate('tc=' + str(round(t2_tc, 2)) + '(s)', size=15, xy=(t2_tc, t2_y), xytext=(t2_tc+50, t2_y),
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3", facecolor='black'))
    ax2.legend(loc='lower right')
    plt.xlabel('time (s)')
    plt.ylabel('mVO2(slope)')
    ax3 = plt.subplot(1,3,3)
    ax3.plot(slps['time'], slps['t3'], marker='o', color='b')
    ax3.plot(x, fitFunc(x, t3_A, t3_B, t3_k), 'r--', label='$f(t) = %.3f$ - %.3f e^(-%.3f t)' % (t3_A, t3_B, t3_k))
    ax3.annotate('tc=' + str(round(t3_tc, 2)) + '(s)', size=15, xy=(t3_tc, t3_y), xytext=(t3_tc+50, t3_y),
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3", facecolor='black'))
    ax3.legend(loc='lower right')
    plt.xlabel('time (s)')
    plt.ylabel('mVO2(slope)')
    plt.show(block=False)
    plt.savefig(fileName + test +'.png')
    return fitParams, fitCovs