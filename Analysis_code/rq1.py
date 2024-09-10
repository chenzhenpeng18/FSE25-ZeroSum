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

count_list = {}
for i in ['increase', 'tie', 'decrease', 'largedecrease', 'largeincrease']:
    count_list[i]={}
    for j in ['sr', 'fpr', 'tpr']:
        count_list[i][j] ={}
        for name in ['adv', 'rew', 'eop','fairsmote', 'ltdd', 'maat', 'fairmask', 'mirrorfair']:
            count_list[i][j][name] = {}
            for group in ['p', 'u']:
                count_list[i][j][name][group] = 0

value_list = {}
for i in ['rela', 'abso']:
    value_list[i]={}
    for j in ['sr', 'fpr', 'tpr']:
        value_list[i][j]={}
        for name in ['adv', 'rew', 'eop','fairsmote', 'ltdd', 'maat', 'fairmask', 'mirrorfair']:
            value_list[i][j][name]={}
            for group in ['p', 'u']:
                value_list[i][j][name][group] = []

for i in model_list:
    for j in ['sr', 'fpr', 'tpr']:
        for name in ['adv', 'rew', 'eop', 'fairsmote','ltdd', 'maat', 'fairmask', 'mirrorfair']:
            for k in dataset_list:
                for group in ['p', 'u']:
                    num_origin = data[i][k][j+group]['origin']
                    num_method = data[i][k][j+group][name]
                    if mann(num_origin, num_method) < 0.05:
                        if mean(num_origin) < mean(num_method):
                            count_list['increase'][j][name][group]+=1
                            if abs(cliffs_delta(num_origin,num_method)[0]) >=0.428:
                                count_list['largeincrease'][j][name][group] += 1
                        else:
                            count_list['decrease'][j][name][group] += 1
                            if abs(cliffs_delta(num_origin,num_method)[0]) >=0.428:
                                count_list['largedecrease'][j][name][group] += 1
                    else:
                        count_list['tie'][j][name][group] += 1
                    value_list['abso'][j][name][group].append(mean(num_method)-mean(num_origin))
                    value_list['rela'][j][name][group].append((mean(num_method)-mean(num_origin))/mean(num_origin))

fout = open('rq1_result', 'w')
fout.write('Table2 Results\n')
fout.write('method\tsrp_in\tsrp_tie\tsrp_de\tsru_in\tsru_tie\tsru_de\ttpp_in\ttpp_tie\ttpp_de\ttpu_in\ttpu_tie\ttpu_de\tfpp_in\tfpp_tie\tfpp_de\tfpu_in\tfpu_tie\tfpu_de\n')
for name in ['adv', 'rew', 'eop','fairsmote', 'ltdd', 'maat', 'fairmask', 'mirrorfair']:
    fout.write(name)
    for j in ['sr', 'tpr', 'fpr']:
        for group in ['p', 'u']:
            fout.write('\t%d\t%d\t%d' % (count_list['increase'][j][name][group], count_list['tie'][j][name][group], count_list['decrease'][j][name][group]))
    fout.write('\n')

fout.write('\n')
fout.write('\n')

fout.write('Table3 Results\n')

fout.write('method\tsrp_mean\tsrp_max\tspr_large\ttpp_mean\ttpp_max\ttpp_large\tfpp_mean\tfpp_max\tfpp_large\n')
for name in ['adv', 'rew', 'eop','fairsmote', 'ltdd', 'maat', 'fairmask', 'mirrorfair']:
    fout.write(name)
    for j in ['sr', 'tpr', 'fpr']:
        for group in ['p']:
            fout.write('\t%f\t%f\t%f' % (mean(value_list['abso'][j][name][group]), min(value_list['abso'][j][name][group]), count_list['largedecrease'][j][name][group]/32))
    fout.write('\n')

fout.write('method\tsru_mean\tsru_max\tspu_large\ttpu_mean\ttpu_max\ttpu_large\tfpu_mean\tfpu_max\tfpu_large\n')
for name in ['adv', 'rew', 'eop','fairsmote', 'ltdd', 'maat', 'fairmask', 'mirrorfair']:
    fout.write(name)
    for j in ['sr', 'tpr', 'fpr']:
        for group in ['u']:
            fout.write('\t%f\t%f\t%f' % (mean(value_list['abso'][j][name][group]), max(value_list['abso'][j][name][group]), count_list['largeincrease'][j][name][group]/32))
    fout.write('\n')

fout.close()
