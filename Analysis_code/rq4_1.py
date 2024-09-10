from numpy import mean, std,sqrt
import scipy.stats as stats
from cliffs_delta import cliffs_delta

def mann(x,y):
	return stats.mannwhitneyu(x,y)[1]

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
    for name in ['mirrorfair', 'mirrorfairu', 'naivebase']:
        for dataset in dataset_list:
            (dataset_pre,dataset_aft) = dataset.split('-')
            fin = open('../Results_Single/'+name+'_'+j+'_'+dataset_pre+'_'+dataset_aft+'.txt','r')
            for line in fin:
                k = line.strip().split('\t')[0]
                data[j][dataset][k][name]=list(map(float,line.strip().split('\t')[1:21]))
            fin.close()


fout = open('rq4_1_result', 'w')
fout.write('Table5 Results\n')
for j in ['srp', 'tprp', 'fprp', 'sru', 'tpru', 'fpru', 'accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd', 'eod', 'aod']:
    fout.write(j+'\n')
    for name in ['mirrorfairu','naivebase']:
        count_list = {}
        for ind in ['win', 'tie', 'loss']:
            count_list[ind] = 0
        for i in model_list:
            for k in dataset_list:
                num_origin = data[i][k][j]['mirrorfair']
                num_method = data[i][k][j][name]
                if mann(num_origin, num_method) < 0.05:
                    if j in ['fprp', 'fpru','spd', 'aod', 'eod']:
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
