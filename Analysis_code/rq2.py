import numpy as np
from numpy import mean, std,sqrt
from scipy.stats import spearmanr
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

model_list = ['lr', 'rf','svm', 'dl']
dataset_list = ['adult-sex','adult-race','compas-sex','compas-race','german-sex', 'german-age','bank-age','mep-race']

data = {}
for i in model_list:
    data[i]={}
    for j in dataset_list:
        data[i][j]={}
        for k in ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'srp', 'sru', 'fprp', 'fpru', 'tprp', 'tpru', 'spd', 'aod', 'eod']:
            data[i][j][k]={}

for j in model_list:
    for name in ['origin', 'rew','eop','fairsmote','ltdd','maat','fairmask', 'mirrorfair']:
        for dataset in dataset_list:
            (dataset_pre,dataset_aft) = dataset.split('-')
            fin = open('../Results_Single/'+name+'_'+j+'_'+dataset_pre+'_'+dataset_aft+'.txt','r')
            for line in fin:
                k = line.strip().split('\t')[0]
                data[j][dataset][k][name]=list(map(float,line.strip().split('\t')[1:21]))
            fin.close()
for name in ['adv']:
    for dataset in dataset_list:
        (dataset_pre,dataset_aft) = dataset.split('-')
        fin = open('../Results_Single/'+name+'_lr_'+dataset_pre+'_'+dataset_aft+'.txt','r')
        for line in fin:
            k = line.strip().split('\t')[0]
            for j in model_list:
                data[j][dataset][k][name]=list(map(float,line.strip().split('\t')[1:21]))
        fin.close()

corre_max = {}
corre_max['correlation'] = {}
corre_max['pvalue'] = {}
for metric1 in ['srp', 'tprp', 'fprp', 'sru', 'tpru', 'fpru', 'spd','eod', 'aod']:
    corre_max['correlation'] [metric1] = {}
    corre_max['pvalue'][metric1] = {}
    for metric2 in ['srp', 'tprp', 'fprp', 'sru', 'tpru', 'fpru', 'spd','eod', 'aod']:
        list_metric1 = []
        list_metric2 = []
        for i in model_list:
            for name in ['adv', 'rew', 'eop', 'fairsmote', 'ltdd', 'maat', 'fairmask', 'mirrorfair']:
                for k in dataset_list:
                    list_metric1.append(mean(data[i][k][metric1][name])-mean(data[i][k][metric1]['origin']))
                    list_metric2.append(mean(data[i][k][metric2][name])-mean(data[i][k][metric2]['origin']))
        corre_max['correlation'][metric1][metric2] = spearmanr(list_metric1, list_metric2)[0]
        corre_max['pvalue'][metric1][metric2] = spearmanr(list_metric1, list_metric2)[1]

fout = open('rq2_result', 'w')

fout.write('correlation coefficients\n')
for metric1 in ['srp', 'tprp', 'fprp', 'sru', 'tpru', 'fpru', 'spd','eod', 'aod']:
    fout.write('\t'+metric1)
fout.write('\n')
for metric1 in ['srp', 'tprp', 'fprp', 'sru', 'tpru', 'fpru', 'spd','eod', 'aod']:
    fout.write(metric1)
    for metric2 in ['srp', 'tprp', 'fprp', 'sru', 'tpru', 'fpru', 'spd','eod', 'aod']:
        fout.write('\t%f' % corre_max['correlation'][metric1][metric2])
    fout.write('\n')
fout.write('\n')

fout.write('p-value\n')
for metric1 in ['srp', 'tprp', 'fprp', 'sru', 'tpru', 'fpru', 'spd','eod', 'aod']:
    fout.write('\t'+metric1)
fout.write('\n')
for metric1 in ['srp', 'tprp', 'fprp', 'sru', 'tpru', 'fpru', 'spd','eod', 'aod']:
    fout.write(metric1)
    for metric2 in ['srp', 'tprp', 'fprp', 'sru', 'tpru', 'fpru', 'spd','eod', 'aod']:
        fout.write('\t%f' % corre_max['pvalue'][metric1][metric2])
    fout.write('\n')
fout.close()

corr = pd.DataFrame(corre_max['correlation'])

mask = np.tril(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(10, 8))

cax = ax.imshow(np.where(mask, corr, np.nan), cmap='coolwarm', vmin=-1, vmax=1)

cbar = fig.colorbar(cax)
cbar.ax.tick_params(labelsize=14)

ax.set_xticks(np.arange(len(corr.columns)))
ax.set_yticks(np.arange(len(corr.columns)))

ax.set_xticklabels(['SR_p', 'TPR_p', 'FPR_p', 'SR_u', 'TPR_u', 'FPR_u', 'SPD', 'EOD', 'AOD'], fontsize=14)
ax.set_yticklabels(['SR_p', 'TPR_p', 'FPR_p', 'SR_u', 'TPR_u', 'FPR_u', 'SPD', 'EOD', 'AOD'], fontsize=14)

plt.xticks(rotation=45)

for i in range(len(corr)):
    for j in range(i + 1):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', color='black', fontsize=14)


plt.savefig("correlation.png", format='png')