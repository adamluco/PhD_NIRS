# ISS analysis rational using AL-REP-1
constScat:
AC scat  wl1    4.959283
         wl2    3.908631
DC scat  wl1    4.936472
         wl2    3.878083

# get scat for file
ACscat = scatMan_AC(data, 'radSlopePh', 'SlopeAC')
DCscat = scatMan_DC(data, 'radSlopePh', 'SlopeDC')
# Apply rolling mean
ACscat = ACscat.rolling(rolWin).mean().bfill()
DCscat = DCscat.rolling(rolWin).mean().bfill()
# plot scattering
plt.plot(ACscat)
plt.plot(DCscat)
plt.legend(['692 nm','834 nm'], loc='best')
plt.title('DC scattering')
plt.xlabel('Sample')
plt.ylabel('Scattering coefficient ua')
# plot scattering with constant scatter
plt.plot(DCscat)
a = np.empty(len(ACscat.index.values))
a.fill(3.878083)
plt.plot(ACscat.index.values, a)
plt.legend(['692 nm','834 nm', '692 nm const', '834 nm const'], loc='best')
plt.title('DC scattering')
plt.xlabel('Sample')
plt.ylabel('Scattering coefficient ua')
# VO and AO with constant and specific scattering for each stage and mito
VOpts = datSeg(data, 11*2, instr=1)
occs = stage_dfs(data, 11, VOpts, seg=2)
# get const scattering for occs
occScat = []
def ScatVO(df):
    rVOscat = scatMan_DC(occs[11], 'radSlopePh', 'SlopeDC')
    instr = 'Occlusion Scattering'
    hPlotter(rVOscat, 'DC scat', ['wl1', 'wl2'], instr, 'Scattering (ua 1/cm)', 'Sample')
    occ = pickpoint(rVOscat, 2)
    scat = stage_dfs(rVOscat, 1, occ, seg=2)
    scat = scat[0].mean()
    occScat.append(scat)
    return scat
# calculate Hb data from occ scat
HbOccScat = []
for i in range(11):
    corrScatDC = ua_DCscat(occScat[i]['DC scat'], occs[i]['SlopeDC'])
    Hb_occScatDC = Hb_sigs(corrScatDC['wl1'], corrScatDC['wl2'], 'Hb_OccScatDC')
    Hb_occScatDC = Hb_occScatDC.rolling(rolWin).mean().bfill()
    HbOccScat.append(Hb_occScatDC)
# plot VOs
i = 1
plt.plot(occs[i]['Hb_constScatDC', 'tHb'], 'g', label='Const Scat')
plt.plot(HbOccScat[i]['Hb_OccScatDC', 'tHb'], 'lawngreen', label='Occlusion Scat')
g = HbOccScat[i]['Hb_OccScatDC', 'tHb']+(occs[i]['Hb_constScatDC', 'tHb'].iloc[70]-HbOccScat[i]['Hb_OccScatDC', 'tHb'].iloc[70])
plt.plot(g, 'lawngreen', ls='--', label='OccScat overlaid on ConstScat')
plt.legend(loc='best')
plt.title('Constant Scatter vs Occlusion Scatter')
plt.xlabel('Sample')
plt.ylabel('tHb (uM)')
plt.show()
# plot AOs
i = 10
constHbDif = occs[i]['Hb_constScatDC', 'O2Hb']-occs[i]['Hb_constScatDC', 'HHb']
plt.plot(occs[i]['Hb_constScatDC', 'HHb'], 'b', label='Const Scat')
plt.plot(HbOccScat[i]['Hb_OccScatDC', 'HHb'], 'dodgerblue', label='Occlusion Scat')
g = HbOccScat[i]['Hb_OccScatDC', 'HHb']-(HbOccScat[i]['Hb_OccScatDC', 'HHb'].iloc[70]-occs[i]['Hb_constScatDC', 'HHb'].iloc[70])
plt.plot(g, 'dodgerblue', ls='--', label='OccScat overlaid on ConstScat')
plt.legend(loc='best')
plt.title('Constant Scatter vs Occlusion Scatter')
plt.xlabel('Sample')
plt.ylabel('HHb (uM)')
plt.show()

# ex perf plotting
exPerfScat = proc_scat(ex1PerfSeg, rolWin, proc=2)
plt.plot(exPerfScat)
exHb = hbsigcalc(ex1PerfSeg, exPerfScat, rolWin)
plt.plot()
plt.plot(ex1PerfSeg['Hb_csAC', 'tHb'], label='Hb_csAC')
plt.plot(ex1PerfSeg['Hb_csDC', 'tHb'], label='Hb_csDC')
plt.plot(exHb['Hb_AC', 'tHb'], label='Hb_AC')
plt.plot(exHb['Hb_DC', 'tHb'], label='Hb_DC')
plt.legend(loc='best')
plt.xlabel('Sample')
plt.ylabel('[tHb] (uM)')
plt.title('Exercise Perfusion segment')


# Scrap fxs
plt.plot(aoslps.ix['MIa', 'HHb'].index.values, aoslps.ix['MIa', 'HHb'])









def rolScat(df, rolWin):
    us_AC = scatMan_AC(df, 'radSlopePh', 'SlopeAC')
    us_DC = scatMan_DC(df, 'radSlopePh', 'SlopeDC')
    us_AC = us_AC.rolling(rolWin).mean().bfill()
    us_DC = us_DC.rolling(rolWin).mean().bfill()
    scat = pd.concat([us_AC, us_DC], axis=1)
    corrAC_ua = ua_ACscat(scat['AC scat'], df['SlopeAC'], df['radSlopePh'])
    corrDC_ua = ua_DCscat(scat['DC scat'], df['SlopeDC'])
    Hb_AC = Hb_sigs(corrAC_ua['wl1'], corrAC_ua['wl2'], 'Hb_AC')
    Hb_AC = Hb_AC.rolling(rolWin).mean().bfill()
    Hb_DC = Hb_sigs(corrDC_ua['wl1'], corrDC_ua['wl2'], 'Hb_DC')
    Hb_DC = Hb_DC.rolling(rolWin).mean().bfill()
    HbDat = pd.concat([Hb_AC, Hb_DC], axis=1)
    return Hb_AC, Hb_DC

us_AC = np.mean(us_AC)
us_DC = np.mean(us_DC)

plt.plot(scat['AC scat'])

plt.PLot
plt.plot(df['Hb_csAC', 'tHb'], 'g')
plt.plot(df['Hb_csAC', 'tHb'], 'g')
plt.plot(df['Hb_csDC', 'tHb'], 'r')
plt.plot(HbDat['Hb_AC', 'tHb'], 'lawngreen')
plt.plot(HbDat['Hb_DC', 'tHb'], 'slategray')
['Hb_csAC', 'Hb_csDC', 'SlopeAC', 'SlopeDC', 'SlopePh', 'T', 'radSlopePh']
plt.plot(r_occs[0]['Hb_csAC', 'tHb'])
plt.plot(r_occs[0]['Hb_csDC', 'tHb'])

plt.plot(r_occHb[4]['Hb_DC','O2Hb'])
plt.plot(r_occHb[4]['Hb_DC','HHb'])
plt.plot(r_occHb[4]['Hb_DC','HbDif'])
plt.plot(r_occHb[4]['Hb_DC','cHHb'])
plt.plot(r_occHb[4]['Hb_DC','cHbDif'])
