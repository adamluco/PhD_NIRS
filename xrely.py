# ------------------------------------------------------------ #
# This script contains reliability functions from excel xrely template
# Based on reliability spreadsheet available at: http://sportsci.org/resource/stats/
#
# Revision History
# Developer:            Version:   Date:        Notes:
# Adam A. Lucero        1.0.0      08/22/2017   PhD Script
# ------------------------------------------------------------ # Import statements
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import ctypes
import pickle
import operator
import functools

# ------------------------------------------------------------ # Common functions
def sumproduct(*lists):
    return sum(functools.reduce(operator.mul, data) for data in zip(*lists))

def fisher(x):
    """ Returns the Fisher transformation at x. This transformation produces a function that is
    normally distributed rather than skewed. Use this function to perform hypothesis testing on
    the correlation coefficient. """
    if x <= - 1:
        print('input out of range')
    elif x >= 1:
        print('input out of range')
    else:
        y = (1 / 2) * np.log((1 + x) / (1 - x))
        return y

def fisher_inv(y):
    """Returns the inverse of the Fisher transformation. Use this transformation when analyzing correlations
    between ranges or arrays of data. If y = fisher(x), then fisher_inv(y) = x."""
    x = (np.exp(2 * y)-1) / (np.exp(2 * y) + 1)
    return x
# ------------------------------------------------------------ # REP stat functions
# Mean and SD of change
def rep_means(rep_df):
    """
    Calculates the trial and group mean and standard deviation for reproducibility measures
    from a given participant(r), trial(c) DataFrame
    Parameters
    -----------
    rep_df: reproducibility DataFrame - must be in participant(r), trial(c)
    """
    trial_mean, trial_sd, n, deg_f = [], [], [], []
    for i in np.arange(len(rep_df.columns)):
        x = np.average(rep_df.iloc[:, i])
        y = np.std(rep_df.iloc[:, i], ddof=1)
        z = len(rep_df.iloc[:, i])
        trial_mean.append(x)
        trial_sd.append(y)
        n.append(z)
        deg_f.append(z-1)
    grpMean = sumproduct(trial_mean, n)/np.sum(n)
    grpSD = np.sqrt(sumproduct(trial_sd, trial_sd, deg_f)/np.sum(deg_f))
    grp_n = np.min(n)
    means = DataFrame([trial_mean, trial_sd, n], index=['Mean', 'SD', 'Num Subj'])
    grp_means = DataFrame([grpMean, grpSD, grp_n], index=['Mean', 'SD', 'Num Subj'], columns=['Mean'])
    table = pd.concat([means, grp_means], axis=1)
    return table

def change_scores(rep_df):
    delta, ind = [], []
    for i in np.arange(len(rep_df.columns)-1):
        x = rep_df.iloc[:, i+1] - rep_df.iloc[:, i]
        y = str(i+1) + '-' + str(i)
        delta.append(x)
        ind.append(y)
    change_tbl = DataFrame(delta, index=ind).T
    mean_change_tbl = rep_means(change_tbl)  # Calculate mean of change scores table
    cols = change_tbl.columns.tolist()  # Create list of change table column names
    cols.extend(['Mean'])  # Extend column names to add 'Mean'
    mean_change_tbl.columns = cols  # Rename mean of change table columns
    mean_change_tbl.loc['Mean', 'Mean'] = np.nan  # Remove grp Mean
    mean_change_tbl.rename(index={'Mean':'Mean of change', 'SD':'SD of change'})
    return change_tbl, mean_change_tbl

def change_in_mean(mean_change_tbl, conf_lim, log=False):
    mean_change = mean_change_tbl.loc['Mean', :]
    l_cl = mean_change - stats.t.ppf(1-(1-conf_lim)/2, mean_change_tbl.loc['Num Subj', :].round(0) - 1) \
                                            * mean_change_tbl.loc['SD', :]/np.sqrt(mean_change_tbl.loc['Num Subj', :])
    u_cl = mean_change + stats.t.ppf(1-(1-conf_lim)/2, mean_change_tbl.loc['Num Subj', :].round(0) - 1) \
                                            * mean_change_tbl.loc['SD', :]/np.sqrt(mean_change_tbl.loc['Num Subj', :])
    plus_min = (u_cl-l_cl)/2
    if log == False:
        dat = [mean_change, l_cl, u_cl, plus_min]
        ind = ['Change in mean', 'Lower CL', 'Upper CL', 'CL as +/-value']
        mean_cl_tbl = DataFrame(dat, index=ind)
    else:
        dat = [mean_change, l_cl, u_cl]
        ind = ['Change in mean', 'Lower CL', 'Upper CL']
        mean_cl_tbl = DataFrame(dat, index=ind)
    return mean_cl_tbl

def typical_error(mean_tbl, mean_change_tbl, conf_lim, log=False):
    n = mean_tbl.ix['Num Subj']
    typ_err = mean_change_tbl.loc['SD', :]/np.sqrt(2)  # Calculate typical error
    deg_f = mean_change_tbl.loc['Num Subj', :].copy() - 1  # Calculate degrees of freedom
    deg_f.columns = ['Degrees of Freedom'] # Change column title
    if len(deg_f) == 1:
        deg_f['Mean'] == deg_f['Mean']
        typ_err['Mean'] == typ_err['Mean']
    else:
        deg_f['Mean'] = ((1 - 0.22 * np.sum(n[:-1])/(len(n[:-1]) * n[-1])) * np.sum(deg_f[:-1]))
        typ_err['Mean'] = np.sqrt(sumproduct(typ_err[:-1], typ_err[:-1], deg_f[:-1]) / np.sum(deg_f[:-1]))
    l_cl = np.sqrt(deg_f * typ_err ** 2 / stats.chi2.ppf(1 - (1 - conf_lim) / 2, deg_f.round(0)))
    u_cl = np.sqrt(deg_f * typ_err ** 2 / stats.chi2.ppf((1 - conf_lim) / 2, deg_f.round(0)))
    factor = np.sqrt(u_cl/l_cl)
    bias = 1 + 1 / (4 * deg_f)
    if log == False:
        dat = [typ_err, l_cl, u_cl, factor, bias]
        ind = ['Typical Error', 'Lower CL', 'Upper CL', 'CL as x/\u00F7factor', 'Bias corr factor']
        typ_err_tbl = DataFrame(dat, index=ind)
    else:
        dat = [typ_err, l_cl, u_cl, bias]
        ind = ['Typical Error', 'Lower CL', 'Upper CL', 'Bias corr factor']
        typ_err_tbl = DataFrame(dat, index=ind)
    return typ_err_tbl, deg_f

def pearson_correlation(rep_df, mean_change_tbl, conf_lim):
    pearson, fishr, fishr_se2, bias_corr = [], [], [], []
    for i in np.arange(len(rep_df.columns)-1):
        pcc, p = stats.pearsonr(rep_df.iloc[:, i], rep_df.iloc[:, i+1])
        n = mean_change_tbl.ix['Num Subj'][i]
        z = pcc * (1 + (1 - pcc ** 2) / 2 / (n-3)) # Pearson Correlation
        b = 1 + (1 - pcc ** 2) / 2 / (n - 3) # Bias correction factor
        pearson.append(z)
        bias_corr.append(b)
        if n > 3:
            f_trans = fisher(z)
            fishr.append(f_trans)
            fishr_se2.append(n-3)
        else:
            print('n < 3')
    bias_corr.append(np.nan)
    fishr_mean = sumproduct(fishr, fishr_se2)/np.sum(fishr_se2)
    pcc_mean = fisher_inv(fishr_mean)
    pearson.append(pcc_mean)
    fishr.append(fishr_mean) # currently not exported
    pearson = np.asarray(pearson) #, columns=['Pearson correlation'], index=[mean_change_tbl.columns]).T
    # Lower confidence limit
    l_cl = (np.exp(2 * (
    0.5 * np.log((1 + pearson) / (1 - pearson)) - (stats.norm.ppf(1 - (1 - conf_lim) / 2)) / np.sqrt(n - 3))) - 1) / (
        np.exp(2 * (0.5 * np.log((1 + pearson) / (1 - pearson)) - (stats.norm.ppf(1 - (1 - conf_lim) / 2)) / np.sqrt(
            n - 3))) + 1)
    # Upper confidence limit
    u_cl = (np.exp(2 * (
    0.5 * np.log((1 + pearson) / (1 - pearson)) + (stats.norm.ppf(1 - (1 - conf_lim) / 2)) / np.sqrt(n - 3))) - 1) / (
        np.exp(2 * (0.5 * np.log((1 + pearson) / (1 - pearson)) + (stats.norm.ppf(1 - (1 - conf_lim) / 2)) / np.sqrt(
            n - 3))) + 1)
    l_cl[-1], u_cl[-1] = np.nan, np.nan # Remove mean value
    # Create DataFrame
    dat = [pearson, l_cl, u_cl, bias_corr]
    ind = ['Pearson Correlation', 'Lower CL', 'Upper CL', 'Bias corr factor']
    pearson_tbl = DataFrame(dat, index=ind, columns=mean_change_tbl.columns)
    fdat = [fishr, fishr_se2]
    find = ['Fisher', 'FisherSE^-2']
    fisher_tbl = DataFrame(fdat, index=find, columns=mean_change_tbl.columns)
    return pearson_tbl, fisher_tbl

def icc(mean_tbl, typ_err_tbl, mean_change_tbl, deg_f, conf_lim, log=False):
    iccs, eff_num_trials, F, F_lower, F_upper, l_cl, u_cl = [], [], [], [], [], [], []
    num_degf, den_degf, pure_var, se_pure_var, var_lcl, var_ucl, tr_sd = [], [], [], [], [], [], []
    for i in np.arange(len(typ_err_tbl.columns) - 1):
        te = typ_err_tbl.ix['Typical Error'][i]
        sd = mean_tbl.ix['SD'][i:i+2]
        degfreedom = mean_tbl.ix['Num Subj'][i:i+2] - 1
        n1 = mean_tbl.ix['Num Subj'][i]
        n2 = mean_tbl.ix['Num Subj'][i+1]
        n_mean = mean_change_tbl.ix['Num Subj'][i]
        trial_sd = sumproduct(sd, sd, degfreedom) / np.sum(degfreedom)
        icc = (1 - te ** 2 / trial_sd) * (1 + (1 - (1 - te ** 2 / trial_sd) ** 2) / (n1 + n2 - n_mean - 3))
        numerator_degf = n1 + n2 - n_mean - 1
        denominator_degf = deg_f[i]
        eff_num_tr = 1 + denominator_degf / numerator_degf
        F_val = 1 + icc * eff_num_tr / (1 - icc)
        F_val_l = F_val / stats.f.isf((1 - conf_lim) / 2, numerator_degf, denominator_degf)
        F_val_u = F_val * stats.f.isf((1 - conf_lim) / 2, numerator_degf, denominator_degf)
        lcl = (F_val_l - 1) / (F_val_l + eff_num_tr - 1)
        ucl = (F_val_u - 1) / (F_val_u + eff_num_tr - 1)
        if log == False:
            df_te = te
        else:
            df_te = degfreedom[i]
        if icc > 0:
            pv = trial_sd - te ** 2
            se_pv = np.sqrt(2 * (trial_sd ** 2 / numerator_degf - df_te ** 4 / denominator_degf))
            pv_lcl = pv + stats.norm.ppf((1 - conf_lim) / 2) * se_pv
            pv_ucl = pv - stats.norm.ppf((1 - conf_lim) / 2) * se_pv
        else:
            pv = np.nan
            se_pv = np.nan
            pv_lcl = np.nan
            pv_ucl = np.nan
        iccs.append(icc)
        eff_num_trials.append(eff_num_tr)
        F.append(F_val)
        F_lower.append(F_val_l)
        F_upper.append(F_val_u)
        l_cl.append(lcl)
        u_cl.append(ucl)
        num_degf.append(numerator_degf)
        den_degf.append(denominator_degf)
        pure_var.append(pv)
        se_pure_var.append(se_pv)
        var_lcl.append(pv_lcl)
        var_ucl.append(pv_ucl)
        tr_sd.append(trial_sd)
    # Calculate and append mean values
    num_degf_mean = mean_change_tbl.loc['Num Subj', 'Mean'] - 1
    den_degf_mean = deg_f['Mean']
    trial_sd_mean = mean_tbl.loc['SD', 'Mean']
    icc_mean = (1 - typ_err_tbl.ix['Typical Error'][-1] ** 2 / trial_sd_mean ** 2) * (
    1 + (1 - (1 - typ_err_tbl.ix['Typical Error'][-1] ** 2 / trial_sd_mean ** 2) ** 2) / (
    mean_change_tbl.ix['Num Subj'][-1] - 3))
    ent_mean = 1 + den_degf_mean / num_degf_mean
    F_mean = 1 + icc_mean * ent_mean / (1 - icc_mean)
    F_lower_mean = F_mean / stats.f.isf((1 - conf_lim) / 2, num_degf_mean, den_degf_mean)
    F_upper_mean = F_mean * stats.f.isf((1 - conf_lim) / 2, den_degf_mean, num_degf_mean)
    lcl_mean = (F_lower_mean - 1) / (F_lower_mean + ent_mean - 1)
    ucl_mean = (F_upper_mean - 1) / (F_upper_mean + ent_mean - 1)
    if icc_mean > 0:
        pv_mean = trial_sd_mean ** 2 - typ_err_tbl.ix['Typical Error'][-1] ** 2
        se_pv_mean = np.sqrt(2 * trial_sd_mean ** 4 / num_degf_mean -
                            typ_err_tbl.ix['Typical Error'][-1] ** 4 / den_degf_mean)
        pv_lcl_mean = pv_mean + stats.norm.ppf((1-conf_lim)/2) * se_pv_mean
        pv_ucl_mean = pv_mean - stats.norm.ppf((1 - conf_lim) / 2) * se_pv_mean
    else:
        pv_mean = np.nan
        se_pv_mean = np.nan
        pv_lcl_mean = np.nan
        pv_ucl_mean = np.nan
    iccs.append(icc_mean)
    eff_num_trials.append(ent_mean)
    F.append(F_mean)
    F_lower.append(F_lower_mean)
    F_upper.append(F_upper_mean)
    l_cl.append(lcl_mean)
    u_cl.append(ucl_mean)
    num_degf.append(num_degf_mean)
    den_degf.append(den_degf_mean)
    pure_var.append(pv_mean)
    se_pure_var.append(se_pv_mean)
    var_lcl.append(pv_lcl_mean)
    var_ucl.append(pv_ucl_mean)
    tr_sd.append(trial_sd_mean)
    # Create Data tables
    dat = [iccs, l_cl, u_cl]
    ind = ['ICC', 'Lower CL', 'Upper CL']
    icc_tbl = DataFrame(dat, index=ind, columns=mean_change_tbl.columns)
    pdat = [eff_num_trials, F, F_lower, F_upper, num_degf, den_degf, pure_var, se_pure_var, var_lcl, var_ucl, tr_sd]
    pind = ['Effective num trials', 'F', 'F lower', 'F upper', 'Numerator DF', 'Denominator DF',
           'Pure variance', 'SE of pure variance', 'Lower CL', 'Upper CL', 'Trial SD']
    pvar_tbl = DataFrame(pdat, index=pind, columns=mean_change_tbl.columns)
    return icc_tbl, pvar_tbl

def smallest_eff(pvar_tbl, small_eff, conf_lim):
    p_sd, p_sd_lcl, p_sd_ucl, o_sd, o_sd_lcl, o_sd_ucl = [], [], [], [], [], []
    for i in np.arange(len(pvar_tbl.columns)):
        pure_sd = small_eff * np.sqrt(pvar_tbl.ix['Pure variance'][i])
        if pvar_tbl.ix['Lower CL'][i] > 0:
            pure_sd_lcl = small_eff * np.sqrt(pvar_tbl.ix['Lower CL'][i])
        else:
            pure_sd_lcl = small_eff * -np.sqrt(-pvar_tbl.ix['Lower CL'][i])
        pure_sd_ucl = small_eff * np.sqrt(pvar_tbl.ix['Upper CL'][i])
        if i < len(pvar_tbl.columns) - 1:
            obs_sd = small_eff * np.sqrt(pvar_tbl.ix['Trial SD'][i])
        else:
            obs_sd = small_eff * pvar_tbl.ix['Trial SD'][i]
        obs_sd_lcl = np.sqrt(pvar_tbl.ix['Numerator DF'][i] * obs_sd ** 2 / stats.chi2.ppf(1 - (1 - conf_lim) / 2, int(
            pvar_tbl.ix['Numerator DF'][i])))
        obs_sd_ucl = np.sqrt(pvar_tbl.ix['Numerator DF'][i] * obs_sd ** 2 / stats.chi2.ppf((1 - conf_lim) / 2, int(
            pvar_tbl.ix['Numerator DF'][i])))
        p_sd.append(pure_sd)
        p_sd_lcl.append(pure_sd_lcl)
        p_sd_ucl.append(pure_sd_ucl)
        o_sd.append(obs_sd)
        o_sd_lcl.append(obs_sd_lcl)
        o_sd_ucl.append(obs_sd_ucl)
    smallest_eff_table = [p_sd, p_sd_lcl, p_sd_ucl, o_sd, o_sd_lcl, o_sd_ucl]
    ind = ['Smallest effect from pure SD', 'Lower CL pSD', 'Upper CL pSD', 'Smallest effect from observed SD',
           'Lower CL oSD', 'Upper CL oSD']
    smallest_eff_tbl = DataFrame(smallest_eff_table, index=ind, columns=pvar_tbl.columns)
    return smallest_eff_tbl

def standardized_icc(mean_cl_tbl, typ_err_tbl, pvar_tbl, icc_tbl):
    if icc_tbl.ix['ICC'].all() > 0:  # a.any() or a.all()?
        mean_change = mean_cl_tbl.ix['Change in mean'] / np.sqrt(pvar_tbl.ix['Trial SD'] - typ_err_tbl.ix['Typical Error'] ** 2)
        mean_change_lcl = mean_cl_tbl.ix['Lower CL'] / np.sqrt(pvar_tbl.ix['Trial SD'][:-1] - typ_err_tbl.ix[
            'Typical Error'] ** 2)
        mean_change_ucl = mean_cl_tbl.ix['Upper CL'] / np.sqrt(pvar_tbl.ix['Trial SD'][:-1] - typ_err_tbl.ix[
            'Typical Error'] ** 2)
        value = (mean_change_ucl - mean_change_lcl) / 2
        te = typ_err_tbl.ix['Typical Error'] / np.sqrt(pvar_tbl.ix['Trial SD'] - typ_err_tbl.ix['Typical Error'] ** 2)
        te_lcl = typ_err_tbl.ix['Lower CL'] / np.sqrt(pvar_tbl.ix['Trial SD'] - typ_err_tbl.ix['Typical Error'] ** 2)
        te_ucl = typ_err_tbl.ix['Upper CL'] / np.sqrt(pvar_tbl.ix['Trial SD'] - typ_err_tbl.ix['Typical Error'] ** 2)
        te[2] = typ_err_tbl.ix['Typical Error'][2] / np.sqrt(pvar_tbl.ix['Trial SD'][2] ** 2)
        te_lcl[2] = typ_err_tbl.ix['Lower CL'][2] / np.sqrt(pvar_tbl.ix['Trial SD'][2] ** 2)
        te_ucl[2] = typ_err_tbl.ix['Upper CL'][2] / np.sqrt(pvar_tbl.ix['Trial SD'][2] ** 2)
        factor = np.sqrt(te_ucl / te_lcl)
        dat = [mean_change, mean_change_lcl, mean_change_ucl, value, te, te_lcl, te_ucl, factor]
        ind = ['change in mean', 'Lower CL', 'Upper CL', 'CL as +/-value', 'Typical Error', 'Lower CL',
                   'Upper CL', 'CL as x/\u00F7factor']
    else:
        dat = np.nan
        ind = ['change in mean', 'Lower CL', 'Upper CL', 'CL as +/-value', 'Typical Error', 'Lower CL', 'Upper CL',
               'CL as x/\u00F7factor']
    standardized_tbl = DataFrame(dat, index=ind, columns=typ_err_tbl.columns)
    return standardized_tbl

# ------------------------------------------------------------ # log transformed functions
def back_tf_mean(mean_tbl):
    btf_mean = np.exp(mean_tbl.ix['Mean'] / 100)
    sd_factor = np.exp(mean_tbl.ix['SD'] / 100)
    sd_cv = 100 * sd_factor - 100
    n = mean_tbl.ix['Num Subj']
    table = [btf_mean, sd_factor, sd_cv, n]
    ind = ['Back-transformed mean', 'SD as x/\u00F7factor', 'SD as a %CV', 'Num Subj']
    btf_mean_tbl = DataFrame(table, index=ind)
    return btf_mean_tbl

def log_change(mean_change_tbl):
    mean_factor = np.exp(mean_change_tbl.ix['Mean'] / 100)
    sd_factor = np.exp(mean_change_tbl.ix['SD'] / 100)
    mean_perc = 100 * np.exp(mean_change_tbl.ix['Mean'] / 100) - 100
    sd_cv = 100 * sd_factor - 100
    n = mean_change_tbl.ix['Num Subj']
    table = [mean_factor, sd_factor, mean_perc, sd_cv, n]
    ind = ['Factor change in mean', 'SD as x/\u00F7factor', '% Change in mean', 'SD as a %CV', 'Num Subj']
    log_change_tbl = DataFrame(table, index=ind)
    return log_change_tbl

def log_change_mean(mean_cl_tbl, conf_lim):
    mean_change = mean_change_tbl.loc['Mean', :]
    l_cl = mean_change - stats.t.ppf(1-(1-conf_lim)/2, mean_change_tbl.loc['Num Subj', :].round(0) - 1) \
                                            * mean_change_tbl.loc['SD', :]/np.sqrt(mean_change_tbl.loc['Num Subj', :])
    u_cl = mean_change + stats.t.ppf(1-(1-conf_lim)/2, mean_change_tbl.loc['Num Subj', :].round(0) - 1) \
                                            * mean_change_tbl.loc['SD', :]/np.sqrt(mean_change_tbl.loc['Num Subj', :])
    plus_min = (u_cl-l_cl)/2
    mean_cl_tbl = DataFrame([mean_change, l_cl, u_cl, plus_min], index=['Change in mean', 'Lower CL', 'Upper CL', 'CL as +/-value'])
    return mean_cl_tbl

# ------------------------------------------------------------ # Reliability table fxs
def raw_rely(rep_df, small_eff=0.2, conf_lim=0.9, log=False):
    mean_tbl = rep_means(rep_df)  # Calculate mean
    change_tbl, mean_change_tbl = change_scores(rep_df)  # Calculate change score table and mean change score table
    mean_cl_tbl = change_in_mean(mean_change_tbl, conf_lim, log=log)
    typ_err_tbl, deg_f = typical_error(mean_tbl, mean_change_tbl, conf_lim, log=log)
    pearson_tbl, fisher_tbl = pearson_correlation(rep_df, mean_change_tbl, conf_lim)
    icc_tbl, pvar_tbl = icc(mean_tbl, typ_err_tbl, mean_change_tbl, deg_f, conf_lim, log=log)
    smallest_eff_tbl = smallest_eff(pvar_tbl, small_eff, conf_lim)
    standardized_tbl = standardized_icc(mean_cl_tbl, typ_err_tbl, pvar_tbl, icc_tbl)
    dict = {'Avg':mean_cl_tbl, 'TE':typ_err_tbl, 'SmSD':smallest_eff_tbl, 'STDZ':standardized_tbl, 'Psn':pearson_tbl,
            'ICC':icc_tbl, 'Fshr':fisher_tbl , 'Var':pvar_tbl}
    rely_tbl = pd.concat(dict.values(), keys=dict.keys())
    return rely_tbl, mean_tbl, change_tbl, mean_change_tbl

def log_rely(rep_df, small_eff=0.2, conf_lim=0.9):
    # log transform
    log_df = 100 * np.log(rep_df)
    # run reliability table function
    log_rely_tbl, mean_tbl, change_tbl, mean_change_tbl = raw_rely(log_df, small_eff=small_eff, conf_lim=conf_lim,
                                                                   log=True)
    # create log data tables
    btf_mean_tbl = back_tf_mean(mean_tbl)
    log_change_tbl = log_change(mean_change_tbl)
    # create back transform reliability table and calculate variables
    mean_cl_tbl = np.exp(log_rely_tbl.ix['Avg'] / 100)
    factor = np.sqrt(mean_cl_tbl.ix['Upper CL'] / mean_cl_tbl.ix['Lower CL'])
    mean_cl_tbl = mean_cl_tbl.append(factor, ignore_index=True)
    mean_cl_tbl= mean_cl_tbl.rename(index={0: 'Change in mean', 1: 'Lower CL', 2: 'Upper CL', 3: 'CL as +/-value'})
    typ_err_tbl = np.exp(log_rely_tbl.ix['TE'] / 100)
    te_factor = np.sqrt(typ_err_tbl.ix['Upper CL'] / typ_err_tbl.ix['Lower CL'])
    typ_err_tbl = typ_err_tbl.append(te_factor, ignore_index=True)
    te_bcf = typ_err_tbl.ix[3].copy()
    typ_err_tbl.ix[3] = typ_err_tbl.ix[4].copy()
    typ_err_tbl.ix[4] = te_bcf
    typ_err_tbl= typ_err_tbl.rename(index={0:'Typical Error', 1:'Lower CL', 2:'Upper CL', 3:'CL as x/\u00F7factor',
                                           4:'Bias corr factor'})
    smallest_eff_tbl = np.exp(log_rely_tbl.ix['SmSD'] / 100)
    perc_mean_change = 100 * np.exp(log_rely_tbl.ix['Avg'] / 100) - 100
    cl_value = (perc_mean_change.ix['Upper CL'] - perc_mean_change.ix['Lower CL']) / 2
    standardized_tbl = log_rely_tbl.ix['STDZ']
    pearson_tbl, fisher_tbl = log_rely_tbl.ix['Psn'], log_rely_tbl.ix['Fshr']
    icc_tbl, pvar_tbl = log_rely_tbl.ix['ICC'], log_rely_tbl.ix['Var']
    # create percentage datatable
    perc_avg = perc_mean_change.append(cl_value, ignore_index=True)
    perc_avg = perc_avg.rename(index={0:'Change in mean', 1:'Lower CL', 2:'Upper CL', 3:'CL as +/-value'})
    perc_typ_err = 100 * np.exp(log_rely_tbl.ix['TE'] / 100) - 100
    perc_te_factor = np.sqrt(perc_typ_err.ix['Upper CL'] / perc_typ_err.ix['Lower CL'])
    perc_te = perc_typ_err.append(perc_te_factor, ignore_index=True)
    bcf = perc_te.ix[3].copy()
    perc_te.ix[3] = perc_te.ix[4].copy()
    perc_te.ix[4] = bcf
    perc_te = perc_te.rename(index={0:'Typical Error', 1:'Lower CL', 2:'Upper CL', 3:'CL as x/\u00F7factor',
                                    4:'Bias corr factor'})
    perc_smsd = 100 * np.exp(log_rely_tbl.ix['SmSD'] / 100) - 100
    # df1
    dict = {'Avg':mean_cl_tbl, 'TE':typ_err_tbl, 'SmSD':smallest_eff_tbl, '% Avg':perc_avg, '% TE':perc_te,
            '% SmSD':perc_smsd, 'STDZ':standardized_tbl, 'Psn':pearson_tbl, 'ICC':icc_tbl, 'Fshr':fisher_tbl,
            'Var':pvar_tbl}
    rely_tbl = pd.concat(dict.values(), keys=dict.keys())
    tables = [mean_tbl, change_tbl, mean_change_tbl]
    return rely_tbl, btf_mean_tbl, log_change_tbl, tables

# raw_rely_tbl, raw_mean, raw_change = raw_rely(rep_df)

