from numpy import mean, std,sqrt
import scipy.stats as stats
from cliffs_delta import cliffs_delta

def mann(x,y):
	return stats.mannwhitneyu(x,y)[1]

model_list = ['lr', 'rf','svm', 'dl']
dataset_list = ['adult', 'compas', 'german']

data = {}
for i in model_list:
    data[i] = {}
    for j in dataset_list:
        data[i][j]={}
        for k in ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'sr1', 'sr2', 'sr3', 'sr4', 'fpr1', 'fpr2', 'fpr3', 'fpr4', 'tpr1', 'tpr2', 'tpr3', 'tpr4', 'wcspd', 'wcaod', 'wceod']:
            data[i][j][k]={}

keymap = {}
keymap['adult'] = {'sr00': 'sr4', 'sr01': 'sr3', 'sr10': 'sr2', 'sr11': 'sr1', 'fpr00': 'fpr4', 'fpr01': 'fpr3', 'fpr10': 'fpr2', 'fpr11': 'fpr1','tpr00': 'tpr4', 'tpr01': 'tpr3', 'tpr10': 'tpr2', 'tpr11': 'tpr1'}
keymap['compas'] = {'sr00': 'sr4', 'sr01': 'sr3', 'sr10': 'sr1', 'sr11': 'sr2', 'fpr00': 'fpr4', 'fpr01': 'fpr3', 'fpr10': 'fpr1', 'fpr11': 'fpr2','tpr00': 'tpr4', 'tpr01': 'tpr3', 'tpr10': 'tpr1', 'tpr11': 'tpr2'}
keymap['german'] = {'sr00': 'sr4', 'sr01': 'sr2', 'sr10': 'sr3', 'sr11': 'sr1', 'fpr00': 'fpr4', 'fpr01': 'fpr2', 'fpr10': 'fpr3', 'fpr11': 'fpr1','tpr00': 'tpr4', 'tpr01': 'tpr2', 'tpr10': 'tpr3', 'tpr11': 'tpr1'}

for j in model_list:
    for name in ['mirrorfair', 'mirrorfairu']:
        for dataset in dataset_list:
            fin = open('../Results_Multiple/'+name+'_'+j+'_'+dataset+'_multi.txt','r')
            for line in fin:
                k = line.strip().split('\t')[0]
                if k in keymap[dataset]:
                    k = keymap[dataset][k]
                data[j][dataset][k][name]=list(map(float,line.strip().split('\t')[1:21]))
            fin.close()

fout = open('rq4_2_result', 'w')
fout.write('Table6 Results\n')
for j in ['sr1', 'tpr1', 'fpr1', 'sr2', 'tpr2', 'fpr2', 'sr3', 'tpr3', 'fpr3', 'sr4', 'tpr4', 'fpr4', 'accuracy', 'recall', 'precision', 'f1score', 'mcc', 'wcspd', 'wceod', 'wcaod']:
    fout.write(j+'\n')
    for name in ['mirrorfairu']:
        count_list = {}
        for ind in ['win', 'tie', 'loss']:
            count_list[ind] = 0
        for i in model_list:
            for k in dataset_list:
                num_origin = data[i][k][j]['mirrorfair']
                num_method = data[i][k][j][name]
                if mann(num_origin, num_method) < 0.05:
                    if j in ['fpr1', 'fpr2', 'fpr3', 'fpr4','wcspd', 'wcaod', 'wceod']:
                        if mean(num_origin) < mean(num_method):
                            count_list['loss'] += 1
                        else:
                            count_list['win'] += 1
                    else:
                        if mean(num_origin) > mean(num_method):
                            count_list['loss'] += 1
                        else:
                            count_list['win'] += 1
                else:
                    count_list['tie'] += 1
        fout.write(name)
        for ind in ['win', 'tie', 'loss']:
            fout.write('\t%d' % count_list[ind])
        fout.write('\n')
fout.close()

