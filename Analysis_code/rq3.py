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
    for name in ['origin', 'rew','eop','fairsmote','maat','fairmask', 'mirrorfair']:
        for dataset in dataset_list:
            fin = open('../Results_Multiple/'+name+'_'+j+'_'+dataset+'_multi.txt','r')
            for line in fin:
                k = line.strip().split('\t')[0]
                if k in keymap[dataset]:
                    k = keymap[dataset][k]
                data[j][dataset][k][name]=list(map(float,line.strip().split('\t')[1:21]))
            fin.close()
for name in ['adv']:
    for dataset in dataset_list:
        fin = open('../Results_Multiple/'+name+'_lr_'+dataset+'_multi.txt','r')
        for line in fin:
            k = line.strip().split('\t')[0]
            if k in keymap[dataset]:
                k = keymap[dataset][k]
            for j in model_list:
                data[j][dataset][k][name]=list(map(float,line.strip().split('\t')[1:21]))
        fin.close()

count_list = {}
for i in ['increase', 'tie', 'decrease']:
    count_list[i]={}
    for j in ['sr1', 'tpr1', 'fpr1', 'sr2', 'tpr2', 'fpr2','sr3', 'tpr3', 'fpr3', 'sr4', 'tpr4', 'fpr4']:
        count_list[i][j] ={}
        for name in ['adv', 'rew', 'eop','fairsmote', 'maat', 'fairmask', 'mirrorfair']:
            count_list[i][j][name] = 0

for i in model_list:
    for j in ['sr1', 'tpr1', 'fpr1', 'sr2', 'tpr2', 'fpr2','sr3', 'tpr3', 'fpr3', 'sr4', 'tpr4', 'fpr4']:
        for name in ['adv', 'rew', 'eop', 'fairsmote', 'maat', 'fairmask', 'mirrorfair']:
            for k in dataset_list:
                num_origin = data[i][k][j]['origin']
                num_method = data[i][k][j][name]
                if mann(num_origin, num_method) < 0.05:
                    if mean(num_origin) < mean(num_method):
                        count_list['increase'][j][name]+=1
                    else:
                        count_list['decrease'][j][name] += 1
                else:
                    count_list['tie'][j][name] += 1

fout = open('rq3_result', 'w')
fout.write('Table4 Results\n')
fout.write('method\tsr1in\tsr1tie\tsr1de\ttpr1in\ttpr1tie\ttpr1de\tfpr1in\tfpr1tie\tfpr1de\tsr2in\tsr2tie\tsr2de\ttpr2in\ttpr2tie\ttpr2de\tfpr2in\tfpr2tie\tfpr2de\tsr3in\tsr3tie\tsr3de\ttpr3in\ttpr3tie\ttpr3de\tfpr3in\tfpr3tie\tfpr3de\tsr4in\tsr4tie\tsr4de\ttpr4in\ttpr4tie\ttpr4de\tfpr4in\tfpr4tie\tfpr4de\n')
for name in ['adv', 'rew', 'eop','fairsmote', 'maat', 'fairmask', 'mirrorfair']:
    fout.write(name)
    for j in ['sr1', 'sr2','sr3', 'sr4', 'tpr1', 'tpr2', 'tpr3', 'tpr4', 'fpr1','fpr2','fpr3','fpr4']:
        fout.write('\t%d\t%d' % (count_list['increase'][j][name], count_list['decrease'][j][name]))
    fout.write('\n')

fout.close()
