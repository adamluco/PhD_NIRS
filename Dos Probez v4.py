# ------------------------------------------------------------ #
# This script reads all signals from up to 2 Artinis probes as well as
# LabChart data for the analysis of mBF & mVO2 during resting and exercise stages
#
# Revision History
# Developer:            Version:   Date:        Notes:
# Adam A. Lucero        1.0.0      12/02/2017   User input version, markers serve as guides
# ------------------------------------------------------------ #
# Import required libraries and modules
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
# ------------------------------------------------------------ # Define file path
# Data File Paths, Parameters, and Variables
filepath = 'C:\\Users\\adaml\\Dropbox\\Adam\\Python\\FBF trials\\' #
lc_filename = 'AL_FBF_tr1_LC.txt'
art_filename = 'AL_FBF-tr1_Art.txt'
# ------------------------------------------------------------ # Define variables
# Editable Variables
participant_file = 'AL_FBF_tr2' # Participant file name
p1tx = 't2' # Transmitter to plot for probe 1
p2tx = 't2' # Transmitter to plot for probe 2
Hz = 50 # Sample rate
probe1 = 'FCR' # Muscle under probe1
probe2 = 'FDP' # Muscle under probe2
numstages = 4 # num of stages, currently not editable and must be 4
sig = ['p1O2Hb', 'p1HHb', 'p1tHb', 'p2O2Hb', 'p2HHb', 'p2tHb'] # Signals
tx = ['t1', 't2', 't3'] # Transmitters

# ------------------------------------------------------------ # Class import
# Define class
class Nirs:
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

    def lcdata(lc_path, PS=1):
        lc_names = ['time', 'ECG', 'Period', 'HR', 'HandGrip', 'Occ', 'PS']
        lc_skip = ['NaN', 'Interval=', 'xcelDateTime=', 'TimeFormat=', 'DateFormat=', 'ChannelTitle=', 'Range=']
        lc_table = pd.read_table(lc_path, header=None, names=lc_names, na_values=lc_skip, skip_blank_lines=True,
                                low_memory=False)
        lc_df = DataFrame(lc_table)
        lc_df = lc_df.drop(['Period', 'HR'], axis=1)
        lc_df = lc_df.dropna()
        lc_df = DataFrame(lc_df, dtype=float)
        lc_mark = lc_df.PS.index[lc_df.PS > 1]
        st_time = lc_df.ix[lc_mark[PS-1]].time
        lc_df[lc_df['time'] < st_time] = None
        lc_df = lc_df.dropna()
        return lc_df

    def normalize(y):
        norm = (y - min(y)) / (max(y) - min(y))
        return norm

    def artdata(art_path, PS=1, artSt=67):
        art_names = ['Sample', 'p1t1_O2Hb', 'p1t1_HHb', 'p1t1_tHb',
                     'p1t2_O2Hb', 'p1t2_HHb', 'p1t2_tHb', 'p1t3_O2Hb', 'p1t3_HHb', 'p1t3_tHb',
                     'p2t1_O2Hb', 'p2t1_HHb', 'p2t1_tHb', 'p2t2_O2Hb', 'p2t2_HHb', 'p2t2_tHb',
                     'p2t3_O2Hb', 'p2t3_HHb', 'p2t3_tHb', 'PS_0', 'PS_1', 'event']
        art_table = pd.read_table(art_path, header=None, names=art_names, skiprows=np.arange(artSt),
                                 skip_blank_lines=True, low_memory=False)
        art_df = DataFrame(art_table)
        art_df = art_df.drop('event', axis=1)
        art_df = art_df.dropna()
        art_mark = art_df['PS_0'].index[art_df['PS_0'] > 0.03]
        art_df[art_df['Sample'] < art_mark[PS-1]] = None
        art_df = art_df.dropna()
        art_df = art_df.reindex(columns=['Sample', 'PS_0', 'PS_1', 'p1t1_O2Hb', 'p1t2_O2Hb', 'p1t3_O2Hb', 'p1t1_HHb', 'p1t2_HHb',
                                             'p1t3_HHb', 'p1t1_tHb', 'p1t2_tHb', 'p1t3_tHb', 'p2t1_O2Hb', 'p2t2_O2Hb',
                                             'p2t3_O2Hb', 'p2t1_HHb', 'p2t2_HHb', 'p2t3_HHb', 'p2t1_tHb', 'p2t2_tHb',
                                             'p2t3_tHb'])
        return art_df

    def timealign(lc_df, art_df, Hz):
        lc_df['time'] = lc_df['time'] - lc_df['time'].iloc[0]
        lc_df['time'] = lc_df['time'].round(2)
        art_df['Sample'] = art_df['Sample'] - art_df['Sample'].iloc[0]
        art_df['time'] = art_df['Sample']/Hz
        art_df['time'] = art_df['time'].round(2)
        lc = lc_df.set_index('time', drop=False)
        art = art_df.set_index('time')
        df = lc.join(art)  # LC and Art combined DataFrame
        df = df.fillna(method='ffill')
        df = df.set_index('Sample')
        df.columns = [['lc'] * 5 + ['art'] * 2 + ['p1O2Hb'] * 3 + ['p1HHb'] * 3 + ['p1tHb'] * 3
                      + ['p2O2Hb'] * 3 + ['p2HHb'] * 3 + ['p2tHb'] *3,
                      ['time', 'ECG', 'HandGrip', 'Occ', 'PS', 'PS', 'Occ', 't1', 't2', 't3', 't1', 't2', 't3',
                       't1', 't2', 't3', 't1', 't2', 't3', 't1', 't2', 't3', 't1', 't2', 't3']]
        df = DataFrame(df, dtype='float').round(4)
        return df

    def hbdif(df):
        """Calculates HbDif and adds to DataFrame"""
        p1HbDif = df['p1O2Hb'] - df['p1HHb']
        p2HbDif = df['p2O2Hb'] - df['p2HHb']
        p1HbDif.columns = [['p1HbDif'] * 3, ['t1', 't2', 't3']]
        p2HbDif.columns = [['p2HbDif'] * 3, ['t1', 't2', 't3']]
        dat = [df, p1HbDif, p2HbDif]
        df = pd.concat(dat, axis=1)
        return df

    def chb(df):
        """Calculates corrected Hb signals"""
        p1beta = df['p1O2Hb'] / df['p1tHb']
        p2beta = df['p2O2Hb'] / df['p2tHb']
        p1beta.columns = [['p1beta'] * 3, ['t1', 't2', 't3']]
        p2beta.columns = [['p2beta'] * 3, ['t1', 't2', 't3']]
        dat1 = [df, p1beta, p2beta]
        df = pd.concat(dat1, axis=1)
        p1cO2Hb = df['p1O2Hb'] - df['p1tHb'] * (1 - df['p1beta'])
        p1cHHb = df['p1HHb'] - (df['p1tHb'] * df['p1beta'])
        p2cO2Hb = df['p2O2Hb'] - df['p2tHb'] * (1 - df['p2beta'])
        p2cHHb = df['p2HHb'] - (df['p2tHb'] * df['p2beta'])
        p1cO2Hb.columns = [['p1cO2Hb'] * 3, ['t1', 't2', 't3']]
        p1cHHb.columns = [['p1cHHb'] * 3, ['t1', 't2', 't3']]
        p2cO2Hb.columns = [['p2cO2Hb'] * 3, ['t1', 't2', 't3']]
        p2cHHb.columns = [['p2cHHb'] * 3, ['t1', 't2', 't3']]
        dat2 = [df, p1cO2Hb, p1cHHb, p2cO2Hb, p2cHHb]
        df = pd.concat(dat2, axis=1)
        p1cHbDif = df['p1cO2Hb'] - df['p1cHHb']
        p2cHbDif = df['p2cO2Hb'] - df['p2cHHb']
        p1cHbDif.columns = [['p1cHbDif'] * 3, ['t1', 't2', 't3']]
        p2cHbDif.columns = [['p2cHbDif'] * 3, ['t1', 't2', 't3']]
        dat3 = [df, p1cHbDif, p2cHbDif]
        df = pd.concat(dat3, axis=1)
        return df

    def rnum(x, y):
        i = x.tolist()
        j = y.tolist()
        n = list(range(len(y)))
        return i, j, n

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
        points = plt.ginput(num, timeout=60)
        points = np.array(points)
        points = points[:, 0]
        points = np.trunc(points)
        if reset == True:
            if points[0] < 0:
                points[0] = 0
            else:
                points[0] = points[0]
            if points[-1] > len(df) - 1:
                points[-1] = len(df) - 1
            else:
                points[-1] = points[-1]
        else:
            pass
        plt.close()
        return points

    def markers(df, tempdf, nirs, mark, thresh=0.5, perc=0.01):
        marker = df[nirs, mark].copy()
        tempmin = min(tempdf.min())
        tempmax = max(tempdf.max())
        marker[marker > thresh] = tempmax + (perc * tempmax)
        marker[marker < thresh] = tempmin - (perc * tempmin)
        return marker

    def datSeg(df, p1sig, p2sig, p1tx, p2tx, num, Hz, instr=0, reset=False):
        """Plots all data for stage selection"""
        tempdf = DataFrame(df, columns=[(p1sig, p1tx), (p2sig, p2tx)])
        curr_PS = Nirs.markers(df, tempdf, 'lc', 'PS', thresh=1, perc=0.02)
        curr_Occ = Nirs.markers(df, tempdf, 'lc', 'Occ', thresh=1, perc=0.01)
        plt.figure(1, figsize=(14, 10))
        plt.plot(tempdf.iloc[:, 0], 'lightgreen', label='p1tHb')
        plt.plot(tempdf.iloc[:, 1], 'lawngreen', label='p2tHb')
        plt.plot(curr_PS, 'C1-', label='PS')
        plt.plot(curr_Occ, 'C6-', label='Occ')
        plt.legend(loc='best')
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
        points = Nirs.pickpoint(df, num, reset=reset)
        plt.close()
        return points

    def stage_dfs(df, num_stages, points):
        """Puts stage DataFrame segments into a list"""
        stages = []
        for i in np.arange(1, num_stages + 1):
            y = df.ix[points[(i * 2 - 1) - 1]:points[(i * 2) - 1]]
            stages.append(y)
        return stages

    def rOccSeg(df, points, Hz, sec=17):
        rOccs = []
        for i in range(len(points) - 3):
            x = df.ix[points[i]:points[i] + sec * Hz]
            rOccs.append(x)
        y = df.ix[points[-3]:points[-3] + sec * Hz * 2]
        rOccs.append(y)
        perf = df.ix[points[-2]:points[-1]]
        frames = [DataFrame.mean(perf['p1tHb']), DataFrame.mean(perf['p2tHb'])]
        rPerf = DataFrame(frames).transpose()
        rPerf.columns = ['Probe 1', 'Probe 2']
        rPerf.index = [['rest']*3,['t1', 't2', 't3']]
        print(rPerf)
        return rOccs, rPerf

    def ecgnormnpeak(df):
        ecg = df['lc', 'ECG'].copy()
        ecg = Nirs.normalize(ecg)
        ecgdf = DataFrame(ecg)
        ecg_occ = Nirs.markers(df, ecgdf, 'lc', 'Occ')
        mph = (max(ecg)-min(ecg))*0.6
        cc = Nirs.detect_peaks(ecg, mph=mph)
        return ecgdf, ecg_occ, cc

    def voslps(occs, p1tx, p2tx, numVO=4):
        tempdf = DataFrame(occs[0], columns=[('p1tHb', p1tx), ('p2tHb', p2tx)])
        occList = []
        for i in range(numVO):
            occ = Nirs.markers(occs[i], tempdf, 'lc', 'Occ')
            occList.append(occ)
        ecgvars = []
        for j in range(numVO):
            y = Nirs.ecgnormnpeak(occs[j])
            ecgvars.append(y) # ecgvars: 0=ecg signal, 1=ecg occ, 2=ecg peaks
        ccs = [ecgvars[0][0].iloc[ecgvars[0][2]].index, ecgvars[1][0].iloc[ecgvars[1][2]].index,
                       ecgvars[2][0].iloc[ecgvars[2][2]].index, ecgvars[3][0].iloc[ecgvars[3][2]].index]
        plt.figure(figsize=(14, 10))
        # VO1
        ax1 = plt.subplot2grid((6, 2), (0, 0), rowspan=2)
        ax1.plot(occs[0]['p1tHb', p1tx], 'C2-')
        ax1.plot(occs[0]['p2tHb', p2tx], 'g')
        ax1.plot(occList[0], 'C6-')
        # VO2
        ax2 = plt.subplot2grid((6, 2), (0, 1), rowspan=2)
        ax2.plot(occs[1]['p1tHb', p1tx], 'C2-')
        ax2.plot(occs[1]['p2tHb', p2tx], 'g')
        ax2.plot(occList[1], 'C6-')
        ax2.legend(loc='best')
        # ECG1
        ax3 = plt.subplot2grid((6, 2), (2, 0))
        ax3.plot(ecgvars[0][0], 'r')
        ax3.plot(ecgvars[0][1], 'C6-')
        ax3.plot(ecgvars[0][0].iloc[ecgvars[0][2]], ls='None', marker='+', color='b')
        i, j, n = Nirs.rnum(ecgPeakLocs[0], ecgvars[0][0].iloc[ecgvars[0][2]].iloc[:,0])
        for k, txt in enumerate(n):
            ax3.annotate(txt, (i[k], j[k]+0.05))
        # ECG2
        ax4 = plt.subplot2grid((6, 2), (2, 1))
        ax4.plot(ecgvars[1][0], 'r')
        ax4.plot(ecgvars[1][1], 'C6-')
        ax4.plot(ecgvars[1][0].iloc[ecgvars[1][2]], ls='None', marker='+', color='b')
        i, j, n = Nirs.rnum(ecgPeakLocs[1], ecgvars[1][0].iloc[ecgvars[1][2]].iloc[:,0])
        for k, txt in enumerate(n):
            ax4.annotate(txt, (i[k], j[k]+0.05))
        # vo3
        ax5 = plt.subplot2grid((6, 2), (3, 0), rowspan=2)
        ax5.plot(occs[2]['p1tHb', p1tx], 'C2-')
        ax5.plot(occs[2]['p2tHb', p2tx], 'g')
        ax5.plot(occList[2], 'C6-')
        # vo4
        ax6 = plt.subplot2grid((6, 2), (3, 1), rowspan=2)
        ax6.plot(occs[3]['p1tHb', p1tx], 'C2-')
        ax6.plot(occs[3]['p2tHb', p2tx], 'g')
        ax6.plot(occList[3], 'C6-')
        # ECG3
        ax7 = plt.subplot2grid((6, 2), (5, 0))
        ax7.plot(ecgvars[2][0], 'r')
        ax7.plot(ecgvars[2][1], 'C6-')
        ax7.plot(ecgvars[2][0].iloc[ecgvars[2][2]], ls='None', marker='+', color='b')
        i, j, n = Nirs.rnum(ecgPeakLocs[2], ecgvars[2][0].iloc[ecgvars[2][2]].iloc[:,0])
        for k, txt in enumerate(n):
            ax7.annotate(txt, (i[k], j[k]+0.05))
        # ECG4
        ax8 = plt.subplot2grid((6, 2), (5, 1))
        ax8.plot(ecgvars[3][0], 'r')
        ax8.plot(ecgvars[3][1], 'C6-')
        ax8.plot(ecgvars[3][0].iloc[ecgvars[3][2]], ls='None', marker='+', color='b')
        i, j, n = Nirs.rnum(ecgPeakLocs[3], ecgvars[3][0].iloc[ecgvars[3][2]].iloc[:,0])
        for k, txt in enumerate(n):
            ax8.annotate(txt, (i[k], j[k]+0.05))
        # Plot options
        plt.tight_layout()
        plt.ion()
        plt.show()
        # ECG r peak CC inputs
        ccInp = input('Input first four r peaks for VO1-VO4: (example: 3,4,5,6,3,4,5,6,9,10,11,13,2,3,5,8)')
        plt.close()
        ccInp = [x.strip() for x in ccInp.split(',')]
        ccInp = np.array(list(ccInp), dtype=int)
        ccslist = []
        vo = list(range(0, numVO * 4, 4))
        for q in ['p1tHb', 'p2tHb']:
            for k in range(numVO):
                j = vo[k]
                for i in range(1,5):
                    y = occs[k][q].iloc[ccs[k][ccInp[j+i]:ccs[k][ccInp[j+i+1]]]]
                    q, *b = stats.linregress(y.index.values, y['t1'])
                    w, *b = stats.linregress(y.index.values, y['t2'])
                    e, *b = stats.linregress(y.index.values, y['t3'])
                    ccslist.append([q,w,e])
                z = occs[k][q].iloc[ccs[k][ccInp[j]:ccs[k][ccInp[j + 3]]]]
                q, *b = stats.linregress(z.index.values, z['t1'])
                w, *b = stats.linregress(z.index.values, z['t2'])
                e, *b = stats.linregress(z.index.values, z['t3'])
                ccslist.append([q, w, e])
        columns = ['p1t1cc1', 'p1t1cc2', 'p1t2cc1', 'p1t2cc2', 'p1t3cc1', 'p1t3cc2',
                   'p2t1cc1', 'p2t1cc2', 'p2t2cc1', 'p2t2cc2', 'p2t3cc1', 'p2t3cc2']
        df = DataFrame(table, columns=columns, index=[1, 2, 3, 4])
        print(df)
        return df, ccs





    def aoplot(ao, y1=p1_HHb, y2=p1_HbDif, y3=p2_HHb, y4=p2_HbDif, probe1=probe1, probe2=probe2):
        ao_Occ = Nirs.markers(ao, ao[y1], ao[y2])
        plt.figure(3, figsize=(14, 8))
        plt.plot(ao.Sample, ao[y1], 'b', label=probe1 + 'HHb')
        plt.plot(ao.Sample, ao[y2], 'y', label=probe1 + 'HbDif')
        plt.plot(ao.Sample, ao[y3], 'k', label=probe2 + 'HHb')
        plt.plot(ao.Sample, ao[y4], 'm', label=probe2 + 'HbDif')
        plt.plot(ao.Sample, ao_Occ, 'C6-', label='Occ')
        plt.legend(loc='best')
        plt.title('pick st/end of SO slope')
        print('pick st/end of SO slope')
        aoseg = plt.ginput(2)
        plt.close()
        ao_seg = Nirs.stage_t(aoseg, ao)
        return ao_seg

    def aoslopes(ao1, ao2, ao1_seg, ao2_seg):
        p1t1_HHbao1 = ao1.p1t1_HHb.ix[ao1_seg[0]:ao1_seg[1]]
        p1t2_HHbao1 = ao1.p1t2_HHb.ix[ao1_seg[0]:ao1_seg[1]]
        p1t3_HHbao1 = ao1.p1t3_HHb.ix[ao1_seg[0]:ao1_seg[1]]
        p2t1_HHbao1 = ao1.p2t1_HHb.ix[ao1_seg[0]:ao1_seg[1]]
        p2t2_HHbao1 = ao1.p2t2_HHb.ix[ao1_seg[0]:ao1_seg[1]]
        p2t3_HHbao1 = ao1.p2t3_HHb.ix[ao1_seg[0]:ao1_seg[1]]
        p1t1_HbDifao1 = ao1.p1t1_HbDif.ix[ao1_seg[0]:ao1_seg[1]]
        p1t2_HbDifao1 = ao1.p1t2_HbDif.ix[ao1_seg[0]:ao1_seg[1]]
        p1t3_HbDifao1 = ao1.p1t3_HbDif.ix[ao1_seg[0]:ao1_seg[1]]
        p2t1_HbDifao1 = ao1.p2t1_HbDif.ix[ao1_seg[0]:ao1_seg[1]]
        p2t2_HbDifao1 = ao1.p2t2_HbDif.ix[ao1_seg[0]:ao1_seg[1]]
        p2t3_HbDifao1 = ao1.p2t3_HbDif.ix[ao1_seg[0]:ao1_seg[1]]
        p1t1_HHbao2 = ao2.p1t1_HHb.ix[ao2_seg[0]:ao2_seg[1]]
        p1t2_HHbao2 = ao2.p1t2_HHb.ix[ao2_seg[0]:ao2_seg[1]]
        p1t3_HHbao2 = ao2.p1t3_HHb.ix[ao2_seg[0]:ao2_seg[1]]
        p2t1_HHbao2 = ao2.p2t1_HHb.ix[ao2_seg[0]:ao2_seg[1]]
        p2t2_HHbao2 = ao2.p2t2_HHb.ix[ao2_seg[0]:ao2_seg[1]]
        p2t3_HHbao2 = ao2.p2t3_HHb.ix[ao2_seg[0]:ao2_seg[1]]
        p1t1_HbDifao2 = ao2.p1t1_HbDif.ix[ao2_seg[0]:ao2_seg[1]]
        p1t2_HbDifao2 = ao2.p1t2_HbDif.ix[ao2_seg[0]:ao2_seg[1]]
        p1t3_HbDifao2 = ao2.p1t3_HbDif.ix[ao2_seg[0]:ao2_seg[1]]
        p2t1_HbDifao2 = ao2.p2t1_HbDif.ix[ao2_seg[0]:ao2_seg[1]]
        p2t2_HbDifao2 = ao2.p2t2_HbDif.ix[ao2_seg[0]:ao2_seg[1]]
        p2t3_HbDifao2 = ao2.p2t3_HbDif.ix[ao2_seg[0]:ao2_seg[1]]
        p1t1_HHbao1, q, w, e, r = stats.linregress(p1t1_HHbao1.index, p1t1_HHbao1)
        p1t2_HHbao1, q, w, e, r = stats.linregress(p1t2_HHbao1.index, p1t2_HHbao1)
        p1t3_HHbao1, q, w, e, r = stats.linregress(p1t3_HHbao1.index, p1t3_HHbao1)
        p2t1_HHbao1, q, w, e, r = stats.linregress(p2t1_HHbao1.index, p2t1_HHbao1)
        p2t2_HHbao1, q, w, e, r = stats.linregress(p2t2_HHbao1.index, p2t2_HHbao1)
        p2t3_HHbao1, q, w, e, r = stats.linregress(p2t3_HHbao1.index, p2t3_HHbao1)
        p1t1_HbDifao1, q, w, e, r = stats.linregress(p1t1_HbDifao1.index, p1t1_HbDifao1)
        p1t2_HbDifao1, q, w, e, r = stats.linregress(p1t2_HbDifao1.index, p1t2_HbDifao1)
        p1t3_HbDifao1, q, w, e, r = stats.linregress(p1t3_HbDifao1.index, p1t3_HbDifao1)
        p2t1_HbDifao1, q, w, e, r = stats.linregress(p2t1_HbDifao1.index, p2t1_HbDifao1)
        p2t2_HbDifao1, q, w, e, r = stats.linregress(p2t2_HbDifao1.index, p2t2_HbDifao1)
        p2t3_HbDifao1, q, w, e, r = stats.linregress(p2t3_HbDifao1.index, p2t3_HbDifao1)
        p1t1_HHbao2, q, w, e, r = stats.linregress(p1t1_HHbao2.index, p1t1_HHbao2)
        p1t2_HHbao2, q, w, e, r = stats.linregress(p1t2_HHbao2.index, p1t2_HHbao2)
        p1t3_HHbao2, q, w, e, r = stats.linregress(p1t3_HHbao2.index, p1t3_HHbao2)
        p2t1_HHbao2, q, w, e, r = stats.linregress(p2t1_HHbao2.index, p2t1_HHbao2)
        p2t2_HHbao2, q, w, e, r = stats.linregress(p2t2_HHbao2.index, p2t2_HHbao2)
        p2t3_HHbao2, q, w, e, r = stats.linregress(p2t3_HHbao2.index, p2t3_HHbao2)
        p1t1_HbDifao2, q, w, e, r = stats.linregress(p1t1_HbDifao2.index, p1t1_HbDifao2)
        p1t2_HbDifao2, q, w, e, r = stats.linregress(p1t2_HbDifao2.index, p1t2_HbDifao2)
        p1t3_HbDifao2, q, w, e, r = stats.linregress(p1t3_HbDifao2.index, p1t3_HbDifao2)
        p2t1_HbDifao2, q, w, e, r = stats.linregress(p2t1_HbDifao2.index, p2t1_HbDifao2)
        p2t2_HbDifao2, q, w, e, r = stats.linregress(p2t2_HbDifao2.index, p2t2_HbDifao2)
        p2t3_HbDifao2, q, w, e, r = stats.linregress(p2t3_HbDifao2.index, p2t3_HbDifao2)
        p1t1_HHb = [p1t1_HHbao1, p1t1_HHbao2]
        p1t2_HHb = [p1t2_HHbao1, p1t2_HHbao2]
        p1t3_HHb = [p1t3_HHbao1, p1t3_HHbao2]
        p2t1_HHb = [p2t1_HHbao1, p2t1_HHbao2]
        p2t2_HHb = [p2t2_HHbao1, p2t2_HHbao2]
        p2t3_HHb = [p2t3_HHbao1, p2t3_HHbao2]
        p1t1_HbDif = [p1t1_HbDifao1, p1t1_HbDifao2]
        p1t2_HbDif = [p1t2_HbDifao1, p1t2_HbDifao2]
        p1t3_HbDif = [p1t3_HbDifao1, p1t3_HbDifao2]
        p2t1_HbDif = [p2t1_HbDifao1, p2t1_HbDifao2]
        p2t2_HbDif = [p2t2_HbDifao1, p2t2_HbDifao2]
        p2t3_HbDif = [p2t3_HbDifao1, p2t3_HbDifao2]
        table = np.array([p1t1_HHb, p1t2_HHb, p1t3_HHb, p2t1_HHb, p2t2_HHb, p2t3_HHb, p1t1_HbDif,
                          p1t2_HbDif, p1t3_HbDif, p2t1_HbDif, p2t2_HbDif, p2t3_HbDif]).T
        columns = ['p1t1_HHb', 'p1t2_HHb', 'p1t3_HHb', 'p2t1_HHb', 'p2t2_HHb', 'p2t3_HHb',
                   'p1t1_HbDif', 'p1t2_HbDif', 'p1t3_HbDif', 'p2t1_HbDif', 'p2t2_HbDif', 'p2t3_HbDif']
        df = DataFrame(table, columns=columns, index=[1, 2])
        print(df)
        return df

    def experfplot(df, num, mpd=140, y1=p1_tHb, y2=p2_tHb, instr='', probe1=probe1, probe2=probe2):
        curr_Occ = Nirs.markers(df, df[y1], df[y2], thresh=1, perc=0.01)
        p1maxs = Nirs.detect_peaks(df[y1], mpd=mpd, mph=0.4)
        p1mins = Nirs.detect_peaks(df[y1], mpd=mpd, valley=True)
        p2maxs = Nirs.detect_peaks(df[y2], mpd=mpd, mph=0.4)
        p2mins = Nirs.detect_peaks(df[y2], mpd=mpd, valley=True)
        p1max = np.average(df[y1].iloc[p1maxs])
        p1min = np.average(df[y1].iloc[p1mins])
        p1range = p1max - p1min
        p2max = np.average(df[y2].iloc[p2maxs])
        p2min = np.average(df[y2].iloc[p2mins])
        p2range = p2max - p2min
        table = [[p1max, p2max], [p1min, p2min], [p1range, p2range]]
        cap = DataFrame(table, index=['Max', 'Min', 'Range'], columns=['probe1', 'probe2'])
        # Plot
        fig = plt.figure(2, figsize=(14, 8))
        plt.plot(df.Sample, df[y1], 'C2-', label=probe1 + 'tHb')
        plt.plot(df.Sample, df[y2], 'g-', label=probe2 + 'tHb')
        plt.plot(df.Sample, curr_Occ, 'C6-', label='Occ')
        plt.plot(df.Sample.iloc[p1maxs], df[y1].iloc[p1maxs], ls='None', marker='D', color='b')
        plt.plot(df.Sample.iloc[p1mins], df[y1].iloc[p1mins], ls='None', marker='o', color='r')
        plt.plot(df.Sample.iloc[p2maxs], df[y2].iloc[p2maxs], ls='None', marker='D', color='b')
        plt.plot(df.Sample.iloc[p2mins], df[y2].iloc[p2mins], ls='None', marker='o', color='r')
        plt.title(instr)
        print(instr)
        stages = plt.ginput(num)
        plt.close(fig)
        stage_t = Nirs.stage_t(stages, df)
        return stage_t, cap

    def experfusion(WL, perfsegs, cap):
        q = WL.ix[perfsegs[0]:perfsegs[1]]
        w = WL.ix[perfsegs[2]:perfsegs[3]]
        e = WL.ix[perfsegs[4]:perfsegs[5]]
        r = WL.ix[perfsegs[6]:perfsegs[7]]
        t = WL.ix[perfsegs[8]:perfsegs[9]]
        y = WL.ix[perfsegs[10]:perfsegs[11]]
        u = WL.ix[perfsegs[12]:perfsegs[13]]
        i = WL.ix[perfsegs[14]:perfsegs[15]]
        frames = [q, w, e, r, t, y, u, i]
        result = pd.concat(frames)
        a = np.average(result.p1t1_tHb)
        s = np.average(result.p1t2_tHb)
        d = np.average(result.p1t3_tHb)
        f = np.average(result.p2t1_tHb)
        g = np.average(result.p2t2_tHb)
        h = np.average(result.p2t3_tHb)
        table = np.array([[a, f], [s, g], [d, h]])
        data = DataFrame(table, columns=['probe1', 'probe2'], index=['t1', 't2', 't3'])
        data = [data, cap]
        data = pd.concat(data)
        print(data)
        return data

# ------------------- # Data import
# Data import steps
lc_data = Nirs.lcdata(filepath + lc_filename) # Read in LC data and create DataFrame to first PortaSync mark
art_data = Nirs.artdata(filepath + art_filename) # Read in Artinis data and create DataFrame to first PortaSync mark
data = Nirs.timealign(lc_data, art_data, Hz) # Time align and combine into single DataFrame
data = Nirs.hbdif(data) # Calculate HbDif
data = Nirs.chb(data) # Calculate corrected Hb signals
# ------------------- # Create stages
datPoints = Nirs.datSeg(data, 'p1tHb', 'p2tHb', p1tx, p2tx, numstages*2, Hz, reset=True) # Plot and pick stages
stages = Nirs.stage_dfs(data, numstages, datPoints) # create stage DataFrames
# ------------------- # Stage analysis
rPoints = Nirs.datSeg(stages[0], 'p1tHb', 'p2tHb', p1tx, p2tx, 8, Hz, instr=1) # Plot and pick occlusions
rOccs, rPerf = Nirs.rOccSeg(stages[0], rPoints, Hz) # Create occlusion DataFrames
r_ccs = Nirs.voslrOccs, p1tx, p2tx) # VO plot and slope calculations



# Resting AO plot
rao1_seg = Nirs.aoplot(rao1)
rao2_seg = Nirs.aoplot(rao2)
# Resting AO calculations
r_mVO2 = Nirs.aoslopes(rao1, rao2, rao1_seg, rao2_seg)
# ------------------------------------------------------------ # Stage analysis
# WL1 occlusion allocations
instructions = 'Click start of each occlusion for 4xVO and 2xAO'
WL1occs = Nirs.stageplot(WL1, 6, instr=instructions)
# WL1 DataFrames
WL1perf = WL1.ix[WL1occs[0]-45:WL1occs[0]]
WL1vo1 = WL1.ix[WL1occs[0]:WL1occs[0]+18]
WL1vo2 = WL1.ix[WL1occs[1]:WL1occs[1]+18]
WL1vo3 = WL1.ix[WL1occs[2]:WL1occs[2]+18]
WL1vo4 = WL1.ix[WL1occs[3]:WL1occs[3]+18]
WL1ao1 = WL1.ix[WL1occs[4]:WL1occs[4]+16]
WL1ao2 = WL1.ix[WL1occs[5]:WL1occs[5]+16]
# WL1 perfusion
instructions = 'Select st/end of 8 perfusion segments'
WL1perfsegs, WL1_cap = Nirs.experfplot(WL1perf, 16) # plot
WL1_perf = Nirs.experfusion(WL1perf, WL1perfsegs, WL1_cap) # calculations
# WL1 VO plot
instructions = 'Record first 3 cardiac cycles for each occlusion'
WL1_cc1, WL1_cc2, WL1_cc3, WL1_cc4 = Nirs.voplot(WL1vo1, WL1vo2, WL1vo3, WL1vo4) # VO plot
# WL1 VO calculations
WL1_ccz = input('Input first three r peaks for VO1-VO4: (example: 3,4,5,3,4,5,9,10,11,2,3,5)')
WL1_mBF, WL1_ccs = Nirs.slopesup(WL1vo1, WL1vo2, WL1vo3, WL1vo4, WL1_cc1, WL1_cc2, WL1_cc3, WL1_cc4, WL1_ccz)
# WL1 AO plot
WL1ao1_seg = Nirs.aoplot(WL1ao1)
WL1ao2_seg = Nirs.aoplot(WL1ao2)
# WL1 AO calculations
WL1_mVO2 = Nirs.aoslopes(WL1ao1, WL1ao2, WL1ao1_seg, WL1ao2_seg)
# ------------------------------------------------------------ # Stage analysis
# WL2 occlusion allocations
instructions = 'Click start of each occlusion for 4xVO and 2xAO'
WL2occs = Nirs.stageplot(WL2, 6, instr=instructions)
# WL2 DataFrames
WL2perf = WL2.ix[WL2occs[0]-45:WL2occs[0]]
WL2vo1 = WL2.ix[WL2occs[0]:WL2occs[0]+18]
WL2vo2 = WL2.ix[WL2occs[1]:WL2occs[1]+18]
WL2vo3 = WL2.ix[WL2occs[2]:WL2occs[2]+18]
WL2vo4 = WL2.ix[WL2occs[3]:WL2occs[3]+18]
WL2ao1 = WL2.ix[WL2occs[4]:WL2occs[4]+16]
WL2ao2 = WL2.ix[WL2occs[5]:WL2occs[5]+16]
# WL2 perfusion
instructions = 'Select st/end of 8 perfusion segments'
WL2perfsegs, WL2_cap = Nirs.experfplot(WL2perf, 16) # plot
WL2_perf = Nirs.experfusion(WL2perf, WL2perfsegs, WL2_cap) # calculations
# WL2 VO plot
instructions = 'Record first 3 cardiac cycles for each occlusion'
WL2_cc1, WL2_cc2, WL2_cc3, WL2_cc4 = Nirs.voplot(WL2vo1, WL2vo2, WL2vo3, WL2vo4) # VO plot
# WL2 VO calculations
WL2_ccz = input('Input first three r peaks for VO1-VO4: (example: 3,4,5,3,4,5,9,10,11,2,3,5)')
WL2_mBF, WL2_ccs = Nirs.slopesup(WL2vo1, WL2vo2, WL2vo3, WL2vo4, WL2_cc1, WL2_cc2, WL2_cc3, WL2_cc4, WL2_ccz)
# WL2 AO plot
WL2ao1_seg = Nirs.aoplot(WL2ao1)
WL2ao2_seg = Nirs.aoplot(WL2ao2)
# WL2 AO calculations
WL2_mVO2 = Nirs.aoslopes(WL2ao1, WL2ao2, WL2ao1_seg, WL2ao2_seg)
# ------------------------------------------------------------ # Stage analysis
# WL3 occlusion allocations
instructions = 'Click start of each occlusion for 4xVO and 2xAO'
WL3occs = Nirs.stageplot(WL3, 6, instr=instructions)
# WL3 DataFrames
WL3perf = WL3.ix[WL3occs[0]-45:WL3occs[0]]
WL3vo1 = WL3.ix[WL3occs[0]:WL3occs[0]+18]
WL3vo2 = WL3.ix[WL3occs[1]:WL3occs[1]+18]
WL3vo3 = WL3.ix[WL3occs[2]:WL3occs[2]+18]
WL3vo4 = WL3.ix[WL3occs[3]:WL3occs[3]+18]
WL3ao1 = WL3.ix[WL3occs[4]:WL3occs[4]+16]
WL3ao2 = WL3.ix[WL3occs[5]:WL3occs[5]+16]
# WL3 perfusion
instructions = 'Select st/end of 8 perfusion segments'
WL3perfsegs, WL3_cap = Nirs.experfplot(WL3perf, 16) # plot
WL3_perf = Nirs.experfusion(WL3perf, WL3perfsegs, WL3_cap) # calculations
# WL3 VO plot
instructions = 'Record first 3 cardiac cycles for each occlusion'
WL3_cc1, WL3_cc2, WL3_cc3, WL3_cc4 = Nirs.voplot(WL3vo1, WL3vo2, WL3vo3, WL3vo4) # VO plot
# WL3 VO calculations
WL3_ccz = input('Input first three r peaks for VO1-VO4: (example: 3,4,5,3,4,5,9,10,11,2,3,5)')
WL3_mBF, WL3_ccs = Nirs.slopesup(WL3vo1, WL3vo2, WL3vo3, WL3vo4, WL3_cc1, WL3_cc2, WL3_cc3, WL3_cc4, WL3_ccz)
# WL3 AO plot
WL3ao1_seg = Nirs.aoplot(WL3ao1)
WL3ao2_seg = Nirs.aoplot(WL3ao2)
# WL3 AO calculations
WL3_mVO2 = Nirs.aoslopes(WL3ao1, WL3ao2, WL3ao1_seg, WL3ao2_seg)
# ------------------------------------------------------------ # Write variables to Excel
# Write to excel
writer = pd.ExcelWriter(participant_file+'.xlsx', engine='xlsxwriter')
# Perfusion data
perf = [r_perf, WL1_perf, WL2_perf, WL3_perf]
perf = pd.concat(perf, keys=['rest','WL1','WL2','WL3'])
perf.to_excel(writer, sheet_name='Sheet1')
# mBF data
mBF = [r_mBF, WL1_mBF, WL2_mBF, WL3_mBF]
mBF = pd.concat(mBF, keys=['r_mBF', 'WL1_mBF', 'WL2_mBF', 'WL3_mBF'])
mBF.to_excel(writer, sheet_name='Sheet2')
# mVO2 data
mVO2 = [r_mVO2, WL1_mVO2, WL2_mVO2, WL3_mVO2]
mVO2 = pd.concat(mVO2, keys=['r_mVO2', 'WL1_mVO2', 'WL2_mVO2', 'WL3_mVO2'])
mVO2.to_excel(writer, sheet_name='Sheet3')
# Close and Save
writer.save()
# ----------------------------fin---------------------------- #