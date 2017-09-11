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
from scipy import stats
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
p1_tHb = 'p1t2_tHb' # active probe 1 tHb signal
p2_tHb = 'p2t2_tHb' # active probe 2 tHb signal
p1_HHb = 'p1t2_HHb' # active probe 1 HHb signal
p2_HHb = 'p2t2_HHb' # active probe 2 HHb signal
p1_HbDif = 'p2t2_HbDif' # active probe 1 HbDif signal
p2_HbDif = 'p2t2_HbDif' # active probe 2 HbDif signal
# ------------------------------------------------------------ #
# Define class
class Nirs:
    def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
        """Plot results of the detect_peaks function, see its help."""
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

    def lcdata(lc_path):
        LC_names = ['time', 'ECG', 'Period', 'HR', 'HandGrip', 'Occ', 'PS']
        LC_skip = ['NaN', 'Interval=', 'xcelDateTime=', 'TimeFormat=', 'DateFormat=', 'ChannelTitle=', 'Range=']
        LC_data = pd.read_table(lc_path, header=None, names=LC_names, na_values=LC_skip, skip_blank_lines=True,
                                low_memory=False)
        LC_data = DataFrame(LC_data)
        LC_data = LC_data.drop(['Period', 'HR'], axis=1)
        LC_data = LC_data.dropna()
        LC_data = DataFrame(LC_data, dtype=float)
        LC_mark = LC_data.PS.index[LC_data.PS > 1]
        st_time = LC_data.ix[LC_mark[0]].time
        LC_data[LC_data['time'] < st_time] = None
        LC_data = LC_data.dropna()
        return LC_data

    def normalize(y):
        norm = (y - min(y)) / (max(y) - min(y))
        return norm

    # def markerloc(df, markcol, mph, ):
     #   x = Nirs.detect_peaks(markcol, mph, edge='rising')

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

    def timealign(lc_data, art_data, hz):
        lc_data.time = lc_data.time - lc_data.time[lc_data.index[0]]  # LC
        lc_data.time = lc_data.time.round(2)
        lc = lc_data.set_index('time')
        art_data['time'] = art_data.Sample / hz  # Art
        art_data.Sample = art_data.Sample - art_data.Sample[art_data.index[0]]
        art_data.time = art_data.time - art_data.time[art_data.index[0]]
        art_data.time = art_data.time.round(2)
        art = art_data.set_index('time')
        data = lc.join(art)  # LC and Art combined DataFrame
        data = data.dropna()
        return data

    def hbdif(data):
        data['p1t1_HbDif'] = data.p1t1_O2Hb - data.p1t1_HHb
        data['p1t2_HbDif'] = data.p1t2_O2Hb - data.p1t2_HHb
        data['p1t3_HbDif'] = data.p1t3_O2Hb - data.p1t3_HHb
        data['p2t1_HbDif'] = data.p2t1_O2Hb - data.p2t1_HHb
        data['p2t2_HbDif'] = data.p2t2_O2Hb - data.p2t2_HHb
        data['p2t3_HbDif'] = data.p2t3_O2Hb - data.p2t3_HHb
        return data

    def expplot(data, y1=p1_tHb, y2=p2_tHb, y3=p1_HHb, y4=p2_HHb):
        curr_PS = data.PS.copy()
        curr_Occ = data.Occ.copy()
        tempmin = min([min(data[y3]), min(data[y4])])
        tempmax = max([max(data[y1]), max(data[y2])])
        curr_PS[curr_PS > 1] = tempmax + (0.10 * tempmax)
        curr_PS[curr_PS < 1] = tempmin - (0.10 * tempmin)
        curr_Occ[curr_Occ > 1] = tempmax + (0.05 * tempmax)
        curr_Occ[curr_Occ < 1] = tempmin - (0.05 * tempmin)
        plt.figure(1, figsize=(14, 8))
        plt.plot(data[y1], 'g-', label=probe1 + 'tHb')
        plt.plot(data[y2], 'C2-', label=probe2 + 'tHb')
        plt.plot(data[y3], 'b-', label=probe1 + 'HHb')
        plt.plot(data[y4], 'C0-', label=probe2 + 'HHb')
        plt.plot(curr_PS, 'C1-', label='PS')
        plt.plot(curr_Occ, 'C6-', label='Occ')
        plt.legend(loc='best')
        plt.title('Entire Test')
        plt.show()

    def stage_t(stages, df):
        stages = np.array(stages)
        stages = stages[:, 0]
        stages = np.trunc(stages)
        if stages[0] < 0:
            stages[0] = 0
        else:
            stages[0] = stages[0]
        if stages[-1] > len(df.Sample):
            stages[-1] = len(df.Sample)
        else:
            stages[-1] = stages[-1]
        stage_t = list()
        for x in range(len(stages)):
            y = data[data.Sample == stages[x]].index.tolist()
            stage_t.append(y)
        stage_t = sum(stage_t, [])
        stage_t = np.array(stage_t)
        stage_t = np.trunc(stage_t)
        return stage_t

    def stageplot(df, y1, y2, num, instruct, probe1=probe1, probe2=probe2):
        curr_PS = Nirs.markers(df, y1, y2, mark='PS', thresh=1, perc=0.05)
        curr_Occ = Nirs.markers(df, y1, y2, thresh=1, perc=0.02)
        fig = plt.figure(2, figsize=(14, 8))
        plt.plot(df.Sample, y1, 'C2-', label=probe1 + 'tHb')
        plt.plot(df.Sample, y2, 'g-', label=probe2 + 'tHb')
        plt.plot(df.Sample, curr_PS, 'C1-', label='PS')
        plt.plot(df.Sample, curr_Occ, 'C6-', label='Occ')
        plt.title(instruct)
        print(instruct)
        stages = plt.ginput(num)
        plt.close(fig)
        stage_t = Nirs.stage_t(stages, df)
        return stage_t

    def markers(df, y1, y2, mark='Occ', thresh=0.5, perc=0.01):
        marker = df[marker].copy()
        tempmin = min(min(y1), min(y2))
        tempmax = max(max(y1), max(y2))
        marker[marker > threshold] = tempmax + (percent * tempmax)
        marker[marker < threshold] = tempmin - (percent * tempmin)
        return marker

    def ecgnormnpeak(df):
        ecg = df.ECG.copy()
        ecg = Nirs.normalize(ecg)
        ecg_occ = Nirs.markers(df, ecg, ecg)
        mph = (max(ecg)-min(ecg))*0.5
        cc = Nirs.detect_peaks(ecg, mph=mph)
        return ecg, ecg_occ, cc

    def rnum(x, y):
        i = x.tolist()
        j = y.tolist()
        n = list(range(len(y)))
        return i, j, n

    def voplot(vo1, vo2, vo3, vo4, y1=p1_tHb, y2=P2_tHb, probe1=probe1, probe2=probe2):
        # Resize markers
        vo1_Occ = Nirs.markers(vo1, vo1[y1], vo1[y2])
        vo2_Occ = Nirs.markers(vo2, vo2[y1], vo2[y2])
        vo3_Occ = Nirs.markers(vo3, vo3[y1], vo3[y2])
        vo4_Occ = Nirs.markers(vo4, vo4[y1], vo4[y2])

        # Normalize ECG and get peaks
        ecg1, ecg1_occ, cc1 = Nirs.ecgnormnpeak(vo1)
        ecg2, ecg2_occ, cc2 = Nirs.ecgnormnpeak(vo2)
        ecg3, ecg3_occ, cc3 = Nirs.ecgnormnpeak(vo3)
        ecg4, ecg4_occ, cc4 = Nirs.ecgnormnpeak(vo4)
        plt.figure(figsize=(14, 10))
        print('instructions')
        # VO1
        ax1 = plt.subplot2grid((6, 2), (0, 0), rowspan=2)
        ax1.plot(vo1.Sample, vo1[y1], 'C2-')
        ax1.plot(vo1.Sample, vo1[y2], 'g')
        ax1.plot(vo1.Sample, vo1_Occ, 'C6-')
        # VO2
        ax2 = plt.subplot2grid((6, 2), (0, 1), rowspan=2)
        ax2.plot(vo2.Sample, y2, 'C2-', label = probe1)
        ax1.plot(vo1.Sample, vo1.p2t2_tHb, 'g', label = probe2)
        ax2.plot(vo2.Sample, vo2_Occ, 'C6-')
        # ECG1
        ax3 = plt.subplot2grid((6, 2), (2, 0))
        ax3.plot(vo1.Sample, ecg1, 'r')
        ax3.plot(vo1.Sample, ecg1_occ, 'C6-')
        ax3.plot(vo1.Sample.iloc[cc1], ecg1.iloc[cc1], ls='None', marker='+', color='b')
        i, j, n = Nirs.rnum(vo1.Sample.iloc[cc1], ecg1.iloc[cc1])
        for k, txt in enumerate(n):
            ax3.annotate(txt, (i[k], j[k]+0.05))
        # ECG2
        ax4 = plt.subplot2grid((6, 2), (2, 1))
        ecg2_occ = Nirs.markers(vo2, ecg2)
        ax4.plot(vo2.Sample, ecg2, 'r')
        ax4.plot(vo2.Sample, ecg2_occ, 'C6-')
        ax4.plot(vo2.Sample.iloc[cc2], ecg2.iloc[cc2], ls='None', marker='+', color='b')
        i, j, n = Nirs.rnum(vo2.Sample.iloc[cc2], ecg2.iloc[cc2])
        for k, txt in enumerate(n):
            ax4.annotate(txt, (i[k], j[k]+0.05))
        # vo3
        ax5 = plt.subplot2grid((6, 2), (3, 0), rowspan=2)
        ax5.plot(vo3.Sample, y3, 'C2-')
        ax5.plot(vo3.Sample, vo3_Occ, 'C6-')
        # vo4
        ax6 = plt.subplot2grid((6, 2), (3, 1), rowspan=2)
        ax6.plot(vo4.Sample, y4, 'C2-')
        ax6.plot(vo4.Sample, vo4_Occ, 'C6-')
        # ECG3
        ax7 = plt.subplot2grid((6, 2), (5, 0))
        ecg3_occ = Nirs.markers(vo3, ecg3)
        ax7.plot(vo3.Sample, ecg3, 'r')
        ax7.plot(vo3.Sample, ecg3_occ, 'C6-')
        ax7.plot(vo3.Sample.iloc[cc3], ecg3.iloc[cc3], ls='None', marker='+', color='b')
        i, j, n = Nirs.rnum(vo3.Sample.iloc[cc3], ecg3.iloc[cc3])
        for k, txt in enumerate(n):
            ax7.annotate(txt, (i[k], j[k]+0.05))
        # ECG4
        ax8 = plt.subplot2grid((6, 2), (5, 1))
        ecg4_occ = Nirs.markers(vo4, ecg4)
        ax8.plot(vo4.Sample, ecg4, 'r')
        ax8.plot(vo4.Sample, ecg4_occ, 'C6-')
        ax8.plot(vo4.Sample.iloc[cc4], ecg4.iloc[cc4], ls='None', marker='+', color='b')
        i, j, n = Nirs.rnum(vo4.Sample.iloc[cc4], ecg4.iloc[cc4])
        for k, txt in enumerate(n):
            ax8.annotate(txt, (i[k], j[k]+0.05))
        plt.tight_layout()
        plt.show()
        return cc1, cc2, cc3, cc4

    def slopesup(vo1, vo2, vo3, vo4, cc1, cc2, cc3, cc4, ccs):
        p1t1_vo1cc1 = vo1.p1t1_tHb.iloc[np.arange(cc1[ccs[0]], cc1[ccs[1]])]
        p1t2_vo1cc1 = vo1.p1t2_tHb.iloc[np.arange(cc1[ccs[0]], cc1[ccs[1]])]
        p1t3_vo1cc1 = vo1.p1t3_tHb.iloc[np.arange(cc1[ccs[0]], cc1[ccs[1]])]
        p1t1_vo1cc2 = vo1.p1t1_tHb.iloc[np.arange(cc1[ccs[1]], cc1[ccs[2]])]
        p1t2_vo1cc2 = vo1.p1t2_tHb.iloc[np.arange(cc1[ccs[1]], cc1[ccs[2]])]
        p1t3_vo1cc2 = vo1.p1t3_tHb.iloc[np.arange(cc1[ccs[1]], cc1[ccs[2]])]
        p2t1_vo1cc1 = vo1.p2t1_tHb.iloc[np.arange(cc1[ccs[0]], cc1[ccs[1]])]
        p2t2_vo1cc1 = vo1.p2t2_tHb.iloc[np.arange(cc1[ccs[0]], cc1[ccs[1]])]
        p2t3_vo1cc1 = vo1.p2t3_tHb.iloc[np.arange(cc1[ccs[0]], cc1[ccs[1]])]
        p2t1_vo1cc2 = vo1.p2t1_tHb.iloc[np.arange(cc1[ccs[1]], cc1[ccs[2]])]
        p2t2_vo1cc2 = vo1.p2t2_tHb.iloc[np.arange(cc1[ccs[1]], cc1[ccs[2]])]
        p2t3_vo1cc2 = vo1.p2t3_tHb.iloc[np.arange(cc1[ccs[1]], cc1[ccs[2]])]
        p1t1_vo2cc1 = vo2.p1t1_tHb.iloc[np.arange(cc2[ccs[3]], cc2[ccs[4]])]
        p1t2_vo2cc1 = vo2.p1t2_tHb.iloc[np.arange(cc2[ccs[3]], cc2[ccs[4]])]
        p1t3_vo2cc1 = vo2.p1t3_tHb.iloc[np.arange(cc2[ccs[3]], cc2[ccs[4]])]
        p1t1_vo2cc2 = vo2.p1t1_tHb.iloc[np.arange(cc2[ccs[4]], cc2[ccs[5]])]
        p1t2_vo2cc2 = vo2.p1t2_tHb.iloc[np.arange(cc2[ccs[4]], cc2[ccs[5]])]
        p1t3_vo2cc2 = vo2.p1t3_tHb.iloc[np.arange(cc2[ccs[4]], cc2[ccs[5]])]
        p2t1_vo2cc1 = vo2.p2t1_tHb.iloc[np.arange(cc2[ccs[3]], cc2[ccs[4]])]
        p2t2_vo2cc1 = vo2.p2t2_tHb.iloc[np.arange(cc2[ccs[3]], cc2[ccs[4]])]
        p2t3_vo2cc1 = vo2.p2t3_tHb.iloc[np.arange(cc2[ccs[3]], cc2[ccs[4]])]
        p2t1_vo2cc2 = vo2.p2t1_tHb.iloc[np.arange(cc2[ccs[4]], cc2[ccs[5]])]
        p2t2_vo2cc2 = vo2.p2t2_tHb.iloc[np.arange(cc2[ccs[4]], cc2[ccs[5]])]
        p2t3_vo2cc2 = vo2.p2t3_tHb.iloc[np.arange(cc2[ccs[4]], cc2[ccs[5]])]
        p1t1_vo3cc1 = vo3.p1t1_tHb.iloc[np.arange(cc3[ccs[6]], cc3[ccs[7]])]
        p1t2_vo3cc1 = vo3.p1t2_tHb.iloc[np.arange(cc3[ccs[6]], cc3[ccs[7]])]
        p1t3_vo3cc1 = vo3.p1t3_tHb.iloc[np.arange(cc3[ccs[6]], cc3[ccs[7]])]
        p1t1_vo3cc2 = vo3.p1t1_tHb.iloc[np.arange(cc3[ccs[7]], cc3[ccs[8]])]
        p1t2_vo3cc2 = vo3.p1t2_tHb.iloc[np.arange(cc3[ccs[7]], cc3[ccs[8]])]
        p1t3_vo3cc2 = vo3.p1t3_tHb.iloc[np.arange(cc3[ccs[7]], cc3[ccs[8]])]
        p2t1_vo3cc1 = vo3.p2t1_tHb.iloc[np.arange(cc3[ccs[6]], cc3[ccs[7]])]
        p2t2_vo3cc1 = vo3.p2t2_tHb.iloc[np.arange(cc3[ccs[6]], cc3[ccs[7]])]
        p2t3_vo3cc1 = vo3.p2t3_tHb.iloc[np.arange(cc3[ccs[6]], cc3[ccs[7]])]
        p2t1_vo3cc2 = vo3.p2t1_tHb.iloc[np.arange(cc3[ccs[7]], cc3[ccs[8]])]
        p2t2_vo3cc2 = vo3.p2t2_tHb.iloc[np.arange(cc3[ccs[7]], cc3[ccs[8]])]
        p2t3_vo3cc2 = vo3.p2t3_tHb.iloc[np.arange(cc3[ccs[7]], cc3[ccs[8]])]
        p1t1_vo4cc1 = vo4.p1t1_tHb.iloc[np.arange(cc4[ccs[9]], cc4[ccs[10]])]
        p1t2_vo4cc1 = vo4.p1t2_tHb.iloc[np.arange(cc4[ccs[9]], cc4[ccs[10]])]
        p1t3_vo4cc1 = vo4.p1t3_tHb.iloc[np.arange(cc4[ccs[9]], cc4[ccs[10]])]
        p1t1_vo4cc2 = vo4.p1t1_tHb.iloc[np.arange(cc4[ccs[10]], cc4[ccs[11]])]
        p1t2_vo4cc2 = vo4.p1t2_tHb.iloc[np.arange(cc4[ccs[10]], cc4[ccs[11]])]
        p1t3_vo4cc2 = vo4.p1t3_tHb.iloc[np.arange(cc4[ccs[10]], cc4[ccs[11]])]
        p2t1_vo4cc1 = vo4.p2t1_tHb.iloc[np.arange(cc4[ccs[9]], cc4[ccs[10]])]
        p2t2_vo4cc1 = vo4.p2t2_tHb.iloc[np.arange(cc4[ccs[9]], cc4[ccs[10]])]
        p2t3_vo4cc1 = vo4.p2t3_tHb.iloc[np.arange(cc4[ccs[9]], cc4[ccs[10]])]
        p2t1_vo4cc2 = vo4.p2t1_tHb.iloc[np.arange(cc4[ccs[10]], cc4[ccs[11]])]
        p2t2_vo4cc2 = vo4.p2t2_tHb.iloc[np.arange(cc4[ccs[10]], cc4[ccs[11]])]
        p2t3_vo4cc2 = vo4.p2t3_tHb.iloc[np.arange(cc4[ccs[10]], cc4[ccs[11]])]
        p1t1_vo1cc1, q, w, e, r = stats.linregress(p1t1_vo1cc1.index, p1t1_vo1cc1)
        p1t2_vo1cc1, q, w, e, r = stats.linregress(p1t2_vo1cc1.index, p1t2_vo1cc1)
        p1t3_vo1cc1, q, w, e, r = stats.linregress(p1t3_vo1cc1.index, p1t3_vo1cc1)
        p1t1_vo1cc2, q, w, e, r = stats.linregress(p1t1_vo1cc2.index, p1t1_vo1cc2)
        p1t2_vo1cc2, q, w, e, r = stats.linregress(p1t2_vo1cc2.index, p1t2_vo1cc2)
        p1t3_vo1cc2, q, w, e, r = stats.linregress(p1t3_vo1cc2.index, p1t3_vo1cc2)
        p2t1_vo1cc1, q, w, e, r = stats.linregress(p2t1_vo1cc1.index, p2t1_vo1cc1)
        p2t2_vo1cc1, q, w, e, r = stats.linregress(p2t2_vo1cc1.index, p2t2_vo1cc1)
        p2t3_vo1cc1, q, w, e, r = stats.linregress(p2t3_vo1cc1.index, p2t3_vo1cc1)
        p2t1_vo1cc2, q, w, e, r = stats.linregress(p2t1_vo1cc2.index, p2t1_vo1cc2)
        p2t2_vo1cc2, q, w, e, r = stats.linregress(p2t2_vo1cc2.index, p2t2_vo1cc2)
        p2t3_vo1cc2, q, w, e, r = stats.linregress(p2t3_vo1cc2.index, p2t3_vo1cc2)
        p1t1_vo2cc1, q, w, e, r = stats.linregress(p1t1_vo2cc1.index, p1t1_vo2cc1)
        p1t2_vo2cc1, q, w, e, r = stats.linregress(p1t2_vo2cc1.index, p1t2_vo2cc1)
        p1t3_vo2cc1, q, w, e, r = stats.linregress(p1t3_vo2cc1.index, p1t3_vo2cc1)
        p1t1_vo2cc2, q, w, e, r = stats.linregress(p1t1_vo2cc2.index, p1t1_vo2cc2)
        p1t2_vo2cc2, q, w, e, r = stats.linregress(p1t2_vo2cc2.index, p1t2_vo2cc2)
        p1t3_vo2cc2, q, w, e, r = stats.linregress(p1t3_vo2cc2.index, p1t3_vo2cc2)
        p2t1_vo2cc1, q, w, e, r = stats.linregress(p2t1_vo2cc1.index, p2t1_vo2cc1)
        p2t2_vo2cc1, q, w, e, r = stats.linregress(p2t2_vo2cc1.index, p2t2_vo2cc1)
        p2t3_vo2cc1, q, w, e, r = stats.linregress(p2t3_vo2cc1.index, p2t3_vo2cc1)
        p2t1_vo2cc2, q, w, e, r = stats.linregress(p2t1_vo2cc2.index, p2t1_vo2cc2)
        p2t2_vo2cc2, q, w, e, r = stats.linregress(p2t2_vo2cc2.index, p2t2_vo2cc2)
        p2t3_vo2cc2, q, w, e, r = stats.linregress(p2t3_vo2cc2.index, p2t3_vo2cc2)
        p1t1_vo3cc1, q, w, e, r = stats.linregress(p1t1_vo3cc1.index, p1t1_vo3cc1)
        p1t2_vo3cc1, q, w, e, r = stats.linregress(p1t2_vo3cc1.index, p1t2_vo3cc1)
        p1t3_vo3cc1, q, w, e, r = stats.linregress(p1t3_vo3cc1.index, p1t3_vo3cc1)
        p1t1_vo3cc2, q, w, e, r = stats.linregress(p1t1_vo3cc2.index, p1t1_vo3cc2)
        p1t2_vo3cc2, q, w, e, r = stats.linregress(p1t2_vo3cc2.index, p1t2_vo3cc2)
        p1t3_vo3cc2, q, w, e, r = stats.linregress(p1t3_vo3cc2.index, p1t3_vo3cc2)
        p2t1_vo3cc1, q, w, e, r = stats.linregress(p2t1_vo3cc1.index, p2t1_vo3cc1)
        p2t2_vo3cc1, q, w, e, r = stats.linregress(p2t2_vo3cc1.index, p2t2_vo3cc1)
        p2t3_vo3cc1, q, w, e, r = stats.linregress(p2t3_vo3cc1.index, p2t3_vo3cc1)
        p2t1_vo3cc2, q, w, e, r = stats.linregress(p2t1_vo3cc2.index, p2t1_vo3cc2)
        p2t2_vo3cc2, q, w, e, r = stats.linregress(p2t2_vo3cc2.index, p2t2_vo3cc2)
        p2t3_vo3cc2, q, w, e, r = stats.linregress(p2t3_vo3cc2.index, p2t3_vo3cc2)
        p1t1_vo4cc1, q, w, e, r = stats.linregress(p1t1_vo4cc1.index, p1t1_vo4cc1)
        p1t2_vo4cc1, q, w, e, r = stats.linregress(p1t2_vo4cc1.index, p1t2_vo4cc1)
        p1t3_vo4cc1, q, w, e, r = stats.linregress(p1t3_vo4cc1.index, p1t3_vo4cc1)
        p1t1_vo4cc2, q, w, e, r = stats.linregress(p1t1_vo4cc2.index, p1t1_vo4cc2)
        p1t2_vo4cc2, q, w, e, r = stats.linregress(p1t2_vo4cc2.index, p1t2_vo4cc2)
        p1t3_vo4cc2, q, w, e, r = stats.linregress(p1t3_vo4cc2.index, p1t3_vo4cc2)
        p2t1_vo4cc1, q, w, e, r = stats.linregress(p2t1_vo4cc1.index, p2t1_vo4cc1)
        p2t2_vo4cc1, q, w, e, r = stats.linregress(p2t2_vo4cc1.index, p2t2_vo4cc1)
        p2t3_vo4cc1, q, w, e, r = stats.linregress(p2t3_vo4cc1.index, p2t3_vo4cc1)
        p2t1_vo4cc2, q, w, e, r = stats.linregress(p2t1_vo4cc2.index, p2t1_vo4cc2)
        p2t2_vo4cc2, q, w, e, r = stats.linregress(p2t2_vo4cc2.index, p2t2_vo4cc2)
        p2t3_vo4cc2, q, w, e, r = stats.linregress(p2t3_vo4cc2.index, p2t3_vo4cc2)
        p1t1cc1 = [p1t1_vo1cc1, p1t1_vo2cc1, p1t1_vo3cc1, p1t1_vo4cc1]
        p1t1cc2 = [p1t1_vo1cc2, p1t1_vo2cc2, p1t1_vo3cc2, p1t1_vo4cc2]
        p1t2cc1 = [p1t2_vo1cc1, p1t2_vo2cc1, p1t2_vo3cc1, p1t2_vo4cc1]
        p1t2cc2 = [p1t2_vo1cc2, p1t2_vo2cc2, p1t2_vo3cc2, p1t2_vo4cc2]
        p1t3cc1 = [p1t3_vo1cc1, p1t3_vo2cc1, p1t3_vo3cc1, p1t3_vo4cc1]
        p1t3cc2 = [p1t3_vo1cc2, p1t3_vo2cc2, p1t3_vo3cc2, p1t3_vo4cc2]
        p2t1cc1 = [p2t1_vo1cc1, p2t1_vo2cc1, p2t1_vo3cc1, p2t1_vo4cc1]
        p2t1cc2 = [p2t1_vo1cc2, p2t1_vo2cc2, p2t1_vo3cc2, p2t1_vo4cc2]
        p2t2cc1 = [p2t2_vo1cc1, p2t2_vo2cc1, p2t2_vo3cc1, p2t2_vo4cc1]
        p2t2cc2 = [p2t2_vo1cc2, p2t2_vo2cc2, p2t2_vo3cc2, p2t2_vo4cc2]
        p2t3cc1 = [p2t3_vo1cc1, p2t3_vo2cc1, p2t3_vo3cc1, p2t3_vo4cc1]
        p2t3cc2 = [p2t3_vo1cc2, p2t3_vo2cc2, p2t3_vo3cc2, p2t3_vo4cc2]
        table = np.array([p1t1cc1, p1t1cc2, p1t2cc1, p1t2cc2, p1t3cc1, p1t3cc2, p2t1cc1,
                          p2t1cc2, p2t2cc1, p2t2cc2, p2t3cc1, p2t3cc2]).T
        columns = ['p1t1cc1', 'p1t1cc2', 'p1t2cc1', 'p1t2cc2', 'p1t3cc1', 'p1t3cc2',
                   'p2t1cc1', 'p2t1cc2', 'p2t2cc1', 'p2t2cc2', 'p2t3cc1', 'p2t3cc2']
        df = DataFrame(table, columns=columns, index = [1, 2, 3, 4])
        return df

# ------------------------------------------------------------ #
# Data import steps
LC_data = Nirs.lcdata(LC_path) # Read in LC data and create DataFrame to PortaSync 1
Art_data = Nirs.artdata(Art_path) # Read in Artinis data and create DataFrame to PortaSync 1
data = Nirs.timealign(LC_data, Art_data, Hz) # Time align and combine into single DataFrame
data = Nirs.hbdif(data) # Get marker locations and calculate HbDif
# ------------------------------------------------------------ #
# Plot experimental data
# Nirs.expplot(data)
# ------------------------------------------------------------ #
# Stage allocation steps
instructions = 'Click st/end of rest, WL1, WL2, WL3'
stage_t = Nirs.stageplot(data, data.p1t2_tHb, data.p2t2_tHb, numstages*2, instructions) # Stage plot
# Create stage DataFrames
rest = data.ix[stage_t[0]:stage_t[1]]
WL1 = data.ix[stage_t[2]:stage_t[3]]
WL2 = data.ix[stage_t[4]:stage_t[5]]
WL3 = data.ix[stage_t[6]:stage_t[7]]
# ------------------------------------------------------------ #
# rest occlusion DataFrames
instructions = 'Click just before beginning of each occlusion'
restoccs = Nirs.stageplot(rest, rest.p1t1_tHb, rest.p2t2_tHb, 6, instructions)
vo1 = rest.ix[restoccs[0]:restoccs[0]+18]
vo2 = rest.ix[restoccs[1]:restoccs[1]+18]
vo3 = rest.ix[restoccs[2]:restoccs[2]+18]
vo4 = rest.ix[restoccs[3]:restoccs[3]+18]
ao1 = rest.ix[restoccs[4]:restoccs[4]:18]
ao2 = rest.ix[restoccs[5]:restoccs[5]:33]
# Resting occlusion slopes
instructions = 'Record first 3 cardiac cycles for each occlusion'
r_cc1, r_cc2, r_cc3, r_cc4 = Nirs.voplot(vo1, vo2, vo3, vo4, vo1.p1t2_tHb, vo2.p1t2_tHb,
                                         vo3.p1t2_tHb, vo4.p1t2_tHb)
ccs = input('Input first three r peaks for each occlusion, example: 345345234456')
ccs = np.array(list(ccs), dtype=int)
mBF = Nirs.slopesup(vo1, vo2, vo3, vo4, r_cc1, r_cc2, r_cc3, r_cc4, ccs)

y = 'p1t1_tHb'
vo1.y