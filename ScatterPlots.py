# This file requires the data extracted from PopulationCharacteristics.py to run
# Generation of violin and scatter plots for hypertension, 1st degree AV block, dilated & hypertrophic cardiomyopathy

import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dfWT = pd.read_csv('WallThickness.csv')
dfAV = pd.read_csv('AVBlock.csv')
dfHyp = pd.read_csv('Hypertension.csv')
dfDCM = pd.read_csv('DCM.csv')


def clean(x):
    if x != '0':
        x = x.replace("'", " ").strip('[').strip(']').strip().replace(' ', '').split(',')
    return x


def ICD10(df):
    df.fillna('0', inplace=True)
    df['ICD10'] = df['ICD10'].apply(clean)  # clean list of icd10 codes from csv format
    df['ICD10'] = [','.join(map(str, l)) for l in df['ICD10']]  # convert list of idc10 codes to a string
    return df


ICD10(dfWT)
ICD10(dfAV)
ICD10(dfHyp)
ICD10(dfDCM)

dfHyp['HypDiag'] = np.where(dfHyp['ICD10'].str.contains('I10'), 1, 0)  # 1 if diseased, 0 if healthy
dfAV['AVDiag'] = np.where(dfAV['ICD10'].str.contains('I440'), 1, 0)
HCM = ['I421', 'I422']  # hypertrophic cardiomyopathy
pattern = '|'.join(HCM)
dfWT['HCMDiag'] = np.where(dfWT['ICD10'].str.contains(pattern), 1, 0)

dfAV['PQ_pos'] = np.where(dfAV['PQ interval'] > 200, 1, 0)  # 1 is PQ > 200ms (diseased), 0 if healthy
dfHyp['Hyp_pos'] = np.where((dfHyp['SBP'] >= 140) | (dfHyp['DBP'] >= 90), 1, 0)

## DCM data processing, convert Vol to Diam: -2.4*LVEDV*LVEDD**0 - LVEDV*LVEDD**1 + 0*LVEDD**2 + 7*LVEDD**3
roots = []


def pos(lst):
    return [x for x in lst if x > 0] or None


for i in range(len(dfDCM)):
    # Teichholz Formula: Vol [mL], Diam [cm]
    roots.append(poly.polyroots([-2.4*dfDCM['LVEDV'][i], -dfDCM['LVEDV'][i], 0, 7]))
    roots[i] = np.real(roots[i][np.isreal(roots[i])])  # get real roots and remove 0j
    roots[i] = float(pos(roots[i])[0])*10  # get positive roots and convert to float, 10 is unit conversion to mm


dfDCM['PredictDiam'] = 45.3*dfDCM['BSA']**0.3-0.03*dfDCM['Age']-7.2
dfDCM['LVEDD'] = roots
# LVEDD > 112% of predicted value & EF <= 45% is DCM diagnosis
dfDCM['DCM_pos'] = np.where((dfDCM['LVEDD']/dfDCM['PredictDiam'] > 1.12) & (dfDCM['LVEF'] < 45), 1, 0)
dfDCM['DCMDiag'] = np.where(dfDCM['ICD10'].str.contains('I420'), 1, 0)  # 1 if diseased, 0 if healthy
dfDCM.to_csv('DCM_analyzed.csv', index=False)

# print(dfAV.groupby(['Sex']).sum(numeric_only=True))  # number of female (0) and male (1) ppl diagnosed with heart disease
# print('# total', dfAV['Sex'].count())
# print('# male', dfAV['Sex'].sum())
# print(dfWT.groupby(['Sex']).sum(numeric_only=True))
# print('# total', dfWT['Sex'].count())
# print('# male', dfWT['Sex'].sum())
# print(dfHyp.groupby(['Sex']).sum(numeric_only=True))
# print('# total', dfHyp['Sex'].count())
# print('# male', dfHyp['Sex'].sum())
# print(dfDCM.groupby(['Sex']).sum(numeric_only=True))
# print('# total', dfDCM['Sex'].count())
# print('# male', dfDCM['Sex'].sum())
# print('# Diagnosed', dfDCM['DCMDiag'].sum())

colors = ['#F9A03F', '#540D6E']  # orange, purple
hue_order = ['Not Diagnosed', 'Diagnosed']

### AV Block: Scatter Plot
dfAV['Category'] = np.where((dfAV['AVDiag'] == 1), 'Diagnosed', 'Not Diagnosed')
dfAVsort = dfAV.sort_values('Category', key=np.vectorize(hue_order.index))
xnoise = np.random.random(len(dfAV))/1.1
ax = sns.relplot(data=dfAVsort, x=dfAVsort["Sex"]+xnoise, y='PQ interval', hue='Category', hue_order=hue_order,
                 size='Category', sizes=[30, 30], height=12.5, aspect=1, palette=colors)
ax.set_ylabels(fontsize=24)
sns.move_legend(ax, "upper center", title=None, fontsize=0)
plt.legend([], [], frameon=False)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks([0.45, 1.45], ['Female', 'Male'])
plt.xlabel('')
plt.xlim([-0.05, 1.95])
plt.ylim([0, 750])
plt.plot(np.linspace(-0.05, 1.95, 100), np.linspace(200, 200, 100), 'r', lw=3, zorder=1000000, alpha=0.6)
plt.ylabel('PQ interval [ms]')
plt.tight_layout()
plt.savefig('AVBlock_dots')
## violin
dfAV['Category'] = np.where((dfAV['AVDiag'] == 1), 'Diagnosed', 'Not Diagnosed')
dfAVsort = dfAV.sort_values('Category', key=np.vectorize(hue_order.index))
xnoise = np.random.random(len(dfAV))/1.1
ax = sns.catplot(kind='violin', data=dfAVsort, x='Sex', y='PQ interval', split=False, hue='Category', hue_order=hue_order,
                 height=12.5, aspect=1, palette=colors, cut=0)
ax.set_ylabels(fontsize=24)
sns.move_legend(ax, "upper center", title=None, fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks([0, 1], ['Female', 'Male'])
plt.xlabel('')
plt.xlim([-0.5, 1.5])
plt.ylim([0, 750])
plt.plot(np.linspace(-0.5, 1.5, 100), np.linspace(200, 200, 100), 'r', lw=3, zorder=1000000)
plt.ylabel('PQ interval [ms]')
plt.tight_layout()
plt.savefig('AVBlock_violin')


### Hypertension: Scatter Plots
dfHyp['Category'] = np.where((dfHyp['HypDiag'] == 1), 'Diagnosed', 'Not Diagnosed')
dfHypsort = dfHyp.sort_values('Category', key=np.vectorize(hue_order.index))
# SBP
ax = sns.catplot(kind='violin', data=dfHypsort, x='Sex', y='SBP', split=False, hue='Category', hue_order=hue_order,
                 height=12.5, aspect=1, palette=colors, cut=0)
ax.set_ylabels(fontsize=24)
sns.move_legend(ax, "upper center", title=None, fontsize=0)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks([0, 1], ['Female', 'Male'])
plt.xlabel('')
plt.xlim([-0.5, 1.5])
plt.ylim([50, 300])
plt.plot(np.linspace(-0.5, 1.5, 100), np.linspace(140, 140, 100), 'r', lw=3, zorder=1000000)
plt.ylabel('Systolic Blood Pressure [mmHg]')
plt.tight_layout()
plt.savefig('HypSBP_violin')
# DBP
ax = sns.catplot(kind='violin', data=dfHypsort, x='Sex', y='DBP', split=False, hue='Category', hue_order=hue_order,
                 height=12.5, aspect=1, palette=colors, cut=0)
ax.set_ylabels(fontsize=24)
sns.move_legend(ax, "upper center", title=None, fontsize=0)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks([0, 1], ['Female', 'Male'])
plt.xlabel('')
plt.xlim([-0.5, 1.5])
plt.ylim([20, 150])
plt.plot(np.linspace(-0.5, 1.5, 100), np.linspace(90, 90, 100), 'r', lw=3, zorder=1000000)
plt.ylabel('Diastolic Blood Pressure [mmHg]')
plt.tight_layout()
plt.savefig('HypDBP_violin')
## Dots
dfHyp['Category'] = np.where((dfHyp['HypDiag'] == 1), 'Diagnosed', 'Not Diagnosed')
dfHypsort = dfHyp.sort_values('Category', key=np.vectorize(hue_order.index))
xnoise = np.random.random(len(dfHyp))/1.1
# SBP
ax = sns.relplot(data=dfHypsort, x=dfHypsort["Sex"]+xnoise, y='SBP', hue='Category', hue_order=hue_order,
                 size='Category', sizes=[10, 10], height=12.5, aspect=1, palette=colors)
ax.set_ylabels(fontsize=24)
sns.move_legend(ax, "upper center", title=None, fontsize=0)
plt.legend([], [], frameon=False)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks([0.45, 1.45], ['Female', 'Male'])
plt.xlabel('')
plt.xlim([-0.05, 1.95])
plt.ylim([50, 300])
plt.plot(np.linspace(-0.05, 1.95, 100), np.linspace(140, 140, 100), 'r', lw=3, zorder=1000000, alpha=0.6)
plt.ylabel('Systolic Blood Pressure [mmHg]')
plt.tight_layout()
plt.savefig('HypSBP_dots')
# DBP
ax = sns.relplot(data=dfHypsort, x=dfHypsort["Sex"]+xnoise, y='DBP', hue='Category', hue_order=hue_order,
                 size='Category', sizes=[10, 10], height=12.5, aspect=1, palette=colors)
ax.set_ylabels(fontsize=24)
sns.move_legend(ax, "upper center", title=None, fontsize=0)
plt.legend([], [], frameon=False)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks([0.45, 1.45], ['Female', 'Male'])
plt.xlabel('')
plt.xlim([-0.05, 1.95])
plt.ylim([20, 150])
plt.plot(np.linspace(-0.05, 1.95, 100), np.linspace(90, 90, 100), 'r', lw=3, zorder=1000000, alpha=0.6)
plt.ylabel('Diastolic Blood Pressure [mmHg]')
plt.tight_layout()
plt.savefig('HypDBP_dots')

### HCM: Scatter Plot
dfWT['Category'] = np.where((dfWT['HCMDiag'] == 1), 'Diagnosed', 'Not Diagnosed')
dfWTsort = dfWT.sort_values('Category', key=np.vectorize(hue_order.index))
xnoise = np.random.random(len(dfWT))/1.1
ax = sns.relplot(data=dfWTsort, x=dfWTsort["Sex"]+xnoise, y='WTmax', hue='Category', hue_order=hue_order,
                 size='Category', sizes=[30], height=12.5, aspect=1, palette=colors)
ax.set_ylabels(fontsize=24)
sns.move_legend(ax, "upper center", title=None, fontsize=0)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks([0.45, 1.45], ['Female', 'Male'])
plt.xlabel('')
plt.ylabel('Wall Thickness [mm]')
plt.xlim([-0.05, 1.95])
plt.ylim([3.5, 15.03])
plt.plot(np.linspace(-0.05, 1.95, 100), np.linspace(15, 15, 100), 'r', lw=3, zorder=1000000)
plt.tight_layout()
plt.savefig('HCM_dots')
## violin
dfWT['Category'] = np.where((dfWT['HCMDiag'] == 1), 'Diagnosed', 'Not Diagnosed')
dfWTsort = dfWT.sort_values('Category', key=np.vectorize(hue_order.index))
ax = sns.catplot(kind='violin', data=dfWTsort, x='Sex', y='WTmax', split=False, hue='Category', hue_order=hue_order,
                 height=12.5, aspect=1, palette=colors, cut=0)
ax.set_ylabels(fontsize=24)
sns.move_legend(ax, "upper center", title=None, fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks([0, 1], ['Female', 'Male'])
plt.xlabel('')
plt.xlim([-0.5, 1.5])
plt.plot(np.linspace(-0.5, 1.5, 100), np.linspace(15, 15, 100), 'r', lw=3, zorder=1000000)
plt.ylabel('Wall Thickness [mm]')
plt.ylim([3.5, 15.03])
plt.tight_layout()
plt.savefig('HCM_violin')

### DCM: Scatter Plots
dfDCM['Category'] = np.where((dfDCM['DCM_pos'] == 1) & (dfDCM['DCMDiag'] == 1), 'Diagnosed, Meets Cut-off Approximation', "Not Diagnosed, Doesn't Meet Cut-off Approximation")
dfDCM['Category'] = np.where((dfDCM['DCM_pos'] == 1) & (dfDCM['DCMDiag'] == 0), 'Not Diagnosed, Meets Cut-off Approximation', dfDCM['Category'])
dfDCM['Category'] = np.where((dfDCM['DCM_pos'] == 0) & (dfDCM['DCMDiag'] == 1), "Diagnosed, Doesn't Meet Cut-off Approximation", dfDCM['Category'])
hue_order = ["Not Diagnosed, Doesn't Meet Cut-off Approximation", 'Not Diagnosed, Meets Cut-off Approximation',
             "Diagnosed, Doesn't Meet Cut-off Approximation", 'Diagnosed, Meets Cut-off Approximation']
colors4 = ['#F9A03F', '#540D6E', '#0077B6', '#D90429']
dfDCMsort = dfDCM.sort_values('Category', key=np.vectorize(hue_order.index))
xnoise = np.random.random(len(dfDCM))/1.1
# LV EF
ax = sns.relplot(data=dfDCMsort, x=dfDCMsort["Sex"]+xnoise, y='LVEF',
                 col_order=hue_order, hue='Category', hue_order=hue_order, size='Category', sizes=[30, 30, 30, 30], height=12.5,
                 aspect=1, palette=colors4)
ax.set_ylabels(fontsize=24)
sns.move_legend(ax, "upper center", title=None, fontsize=0)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks([0.45, 1.45], ['Female', 'Male'])
plt.xlabel('')
plt.ylabel('LV EF [%]')
plt.xlim([-0.05, 1.95])
plt.ylim([0, 110])
plt.plot(np.linspace(-0.05, 1.95, 100), np.linspace(45, 45, 100), 'r', lw=3, zorder=1000000, alpha=0.6)
plt.tight_layout()
plt.savefig('DCM_EF_dots')
## violin
ax = sns.catplot(kind='violin', data=dfDCMsort, x='Sex', y='LVEF', split=False, hue='Category', hue_order=hue_order,
                 height=12.5, aspect=1, palette=colors4, cut=0)
ax.set_ylabels(fontsize=24)
sns.move_legend(ax, "upper center", title=None, fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.ylabel('LV EF [%]')
plt.xticks([0, 1], ['Female', 'Male'])
plt.xlabel('')
plt.xlim([-0.5, 1.5])
plt.ylim([0, 110])
plt.plot(np.linspace(-0.5, 1.5, 100), np.linspace(45, 45, 100), 'r', lw=3, zorder=1000000)
# plt.ylim([0, 110])
plt.tight_layout()
plt.savefig('DCM_EF_violin')

# LV End Diastolic Diameter
ax = sns.relplot(data=dfDCMsort, x=dfDCMsort["Sex"]+xnoise, y='LVEDD',
                 hue='Category', hue_order=hue_order, size='Category',
                 sizes=[30, 30, 30, 30], height=12.5, aspect=1, palette=colors4)
ax.set_ylabels(fontsize=24)
sns.move_legend(ax, "upper center", title=None, fontsize=0)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks([0.45, 1.45], ['Female', 'Male'])
plt.xlabel('')
plt.ylabel('LVEDD (calculated) [mm]')
plt.xlim([-0.05, 1.95])
plt.ylim([0, 500])
plt.tight_layout()
plt.savefig('DCM_EDD_dots')
## violin
ax = sns.catplot(kind='violin', data=dfDCMsort, x='Sex', y='LVEDD', split=False, hue='Category', hue_order=hue_order,
                 height=12.5, aspect=1, palette=colors4, cut=0)
ax.set_ylabels(fontsize=24)
sns.move_legend(ax, "upper center", title=None, fontsize=0)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks([0, 1], ['Female', 'Male'])
plt.xlabel('')
plt.xlim([-0.5, 1.5])
plt.ylim([0, 500])
plt.ylabel('LVEDD (calculated) [mm]')
plt.tight_layout()
plt.savefig('DCM_EDD_violin')

