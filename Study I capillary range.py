# ------------------------------------------------------------ #
# This script was written for Study I analysis
#
# Revision History
# Developer:            Version:   Date:        Notes:
# Adam A. Lucero        1.0.0      12/02/2017   PhD Script
# ------------------------------------------------------------ # Import Statements
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import pickle


# ------------------------------------------------------------ # Class definition
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

    def rnum(x, y):
        i = x.tolist()
        j = y.tolist()
        n = list(range(len(y)))
        return i, j, n

    def caprange(df, sig, mpd=38, delmax=[], delmin=[]):
        maxs = Nirs.detect_peaks(df[sig], mpd=mpd)
        mins = Nirs.detect_peaks(df[sig], mpd=mpd, valley=True)
        maxz = np.delete(maxs, delmax)
        minz = np.delete(mins, delmin)
        max = df.ix[maxz].mean()
        min = df.ix[minz].mean()
        plt.plot(df[sig], 'g')
        plt.plot(df[sig].ix[maxz], ls='None', marker='D', color='b')
        plt.plot(df[sig].ix[minz], ls='None', marker='o', color='r')
        i, j, n = Nirs.rnum(maxs, df[sig].ix[maxs])
        for k, txt in enumerate(n):
            plt.annotate(txt, (i[k], j[k]+0.05))
        a, s, d = Nirs.rnum(mins, df[sig].ix[mins])
        for k, txt in enumerate(d):
            plt.annotate(txt, (a[k], s[k]+0.05))
        range = max - min
        cap = [max, min, range]
        cap = DataFrame(cap, columns=['t1_tHb', 't2_tHb', 't3_tHb'], index=['max', 'min', 'range'])
        return cap

# ------------------------------------------------------------ # Editable Variables5
filepath = 'C:\\Users\\adaml\\OneDrive\\Massey\\Research\\BF-REP\\Excel\\Perfusion\\'
filename = '12CM-BF-REP-3.xlsx'
# ------------------------------------------------------------ # Read in data and create variables
parse_cols = [1, 2, 3, 8, 9, 10, 15, 16, 17, 22, 23, 24, 29, 30, 31, 36, 37, 38]
data = pd.read_excel(filepath + filename, sheetname='tHb - WLs', skiprows=8, header=None, parse_cols=parse_cols)
data = DataFrame(data)
WL = ['WL1']*3 + ['WL2']*3 + ['WL3']*3 + ['WL4']*3 + ['WL5']*3 + ['WL6']*3
sig = ['t1_tHb', 't2_tHb', 't3_tHb']*6
columns = dict(zip(range(18), sig))
data = data.rename(columns=columns)
data.set_axis(axis=1, labels=[WL, sig])
# ------------------------------------------------------------ # detect peaks
WL1_cap = Nirs.caprange(data.WL1, 't2_tHb', mpd=36, delmax=[8], delmin=[5,8])
WL2_cap = Nirs.caprange(data.WL2, 't2_tHb', mpd=36, delmax=[], delmin=[])
WL3_cap = Nirs.caprange(data.WL3, 't2_tHb', mpd=36, delmax=[0], delmin=[0])
WL4_cap = Nirs.caprange(data.WL4, 't2_tHb', mpd=36, delmax=[], delmin=[])
WL5_cap = Nirs.caprange(data.WL5, 't2_tHb', mpd=36, delmax=[], delmin=[])
WL6_cap = Nirs.caprange(data.WL6, 't2_tHb', mpd=36, delmax=[], delmin=[])
# ------------------------------------------------------------ # Concat and write to excel
keys = ['WL1', 'WL2', 'WL3', 'WL4', 'WL5', 'WL6']
output = pd.concat([WL1_cap, WL2_cap, WL3_cap, WL4_cap, WL5_cap, WL6_cap], keys=keys)
writer = pd.ExcelWriter(filepath + 'Capillary range\\' + filename, engine='xlsxwriter')
output.to_excel(writer, sheet_name='Sheet1')
writer.save()
print(output)
# ------------------------------------------------------------ # Create and plot range df
rangez = output.ix[[2,5,8,11,14,17],:].unstack()
plt.plot(np.arange(6), rangez)
cap_data.append(rangez)
# ------------------------------------------------------------ # Save and open cap_data, list of all data sets
with open(filepath + 'Capillary range\\cap_data', 'wb') as fp:
    pickle.dump(cap_data, fp)
with open(filepath + 'Capillary range\\cap_data', 'rb') as fp:
    cap_data = pickle.load(fp)
# ------------------- # Split list into participants
for i in np.arange(12*3):
    cap_data[i].columns = cap_data[i].columns.get_level_values(0) # remove range column

bf = []
for i in np.arange(0,34,3):
    y = Series(cap_data[i:i+4]) # put data in participant list
    bf.append(y)

bf_r = []
for i in np.arange(12):
    s = (bf[i][0]+bf[i][1]+bf[i][2])/3 # average reos 1-3
    bf_r.append(s)

bf_r = Series(bf_r)


with open(filepath + 'Capillary range\\bf_r', 'wb') as fp: # save participant averages
    pickle.dump(bf_r, fp)

plt.plot(np.arange(1,7), bf_r[11])
plt.legend(bf_r[11].columns.values, loc='upper left')

cap_r = bf_r.sum()/12 # average all data

with open(filepath + 'Capillary range\\cap_r', 'wb') as fp: # save all data average
    pickle.dump(cap_r, fp)

plt.plot(np.arange(1,7), cap_r)
plt.legend(cap_r.columns.values, loc='upper left')

# ------------------- # write to excel
writer = pd.ExcelWriter('BF-REP cap range for stats'+'.xlsx', engine='xlsxwriter') # Write to excel
cap = pd.concat(f, keys=[str(x) for x in np.arange(1,13)])
f.to_excel(writer, sheet_name='caprange')
writer.save()

'WL1','WL2','WL3','WL4','WL5','WL6']

# ------------------- # Put in WL tables for reliability stats
with open(filepath + 'Capillary range\\cap_data', 'rb') as fp:
    cap_data = pickle.load(fp)
cap = pd.concat(cap_data)
cap = cap.sort_index(ascending=True) # sort by workload
cap_t2 = cap['t2_tHb']
df = cap.values.reshape(72, 3)
f = DataFrame(df, index=[['WL1']*12 + ['WL2']*12 + ['WL3']*12 + ['WL4']*12 + ['WL5']*12+['WL6']*12,
                         [str(x) for x in np.arange(1,13)]*6], columns=['day1', 'day2','day3'])


