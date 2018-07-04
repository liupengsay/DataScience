import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import numpy as np
import time
import random
import warnings
warnings.filterwarnings("ignore")
##读取数据
print("Reading...")
data = pd.read_csv('data_preprocessing/train_test_merge_p.csv')
##划分训练与测试集
print('Index...')
train_part_index = list(data[(data['label']!=-1)&(data['n_parts']!=1)].index)
evals_index = list(data[(data['label']!=-1)&(data['n_parts']==1)].index)
test2_index = list(data[data['n_parts']==7].index)

vec_feature = ['creativeSize','ct','marriageStatus','interest1', 'interest2', 'interest3', 
               'interest4','interest5', 'kw1', 'kw2','kw3', 'topic1', 'topic2', 
               'topic3','appIdAction', 'appIdInstall']
data = data[vec_feature]
df_feature = data['creativeSize']
del data['creativeSize']

df_feature = df_feature.astype(int)
value = df_feature.values

df_feature = pd.DataFrame()
df_feature['creativeSize'] = value

for co in vec_feature[1:]:
    print(co)
    s = time.time()
    value = []
    lis = list(data[co].values)
    for i in range(len(lis)):
        value.append(len(lis[i].split(' ')))
    col_name = co+'_length'
    df_feature[col_name]=value
    print(co,int(time.time()-s),'s')
    del data[co]
print('interests...')
df_feature['interests_length'] = df_feature['interest1_length']+df_feature['interest2_length']+df_feature['interest3_length']+df_feature['interest4_length']+df_feature['interest5_length']
interests = ['interest1', 'interest2', 'interest3', 'interest4','interest5']
for co in interests:
    col_name = 'ratio_of_'+co
    df_feature[col_name] = (df_feature[co+'_length']/df_feature['interests_length']*100).astype(int)
print('kws...')
df_feature['kws_length'] = df_feature['kw1_length']+df_feature['kw2_length']+df_feature['kw3_length']
kws = ['kw1', 'kw2', 'kw3']
for co in kws:
    col_name = 'ratio_of_'+co
    df_feature[col_name] = (df_feature[co+'_length']/df_feature['kws_length']*100).astype(int)
print('topics...')
df_feature['topics_length'] = df_feature['topic1_length']+df_feature['topic2_length']+df_feature['topic3_length']
topics = ['topic1', 'topic2', 'topic3']
for co in topics:
    col_name = 'ratio_of_'+co
    df_feature[col_name] = (df_feature[co+'_length']/df_feature['topics_length']*100).astype(int)

print('apps...')
df_feature['apps_length'] = df_feature['appIdAction_length']+df_feature['appIdInstall_length']
apps = ['appIdAction', 'appIdInstall']
for co in apps:
    col_name = 'ratio_of_'+co
    df_feature[col_name] = (df_feature[co+'_length']/df_feature['apps_length']*100).astype(int)
data = []
col_new = ['creativeSize', 'interest2_length', 'ratio_of_interest2',
       'interest1_length', 'ratio_of_interest1', 'ct_length',
       'marriageStatus_length', 'interests_length', 'ratio_of_interest5',
       'interest5_length', 'kw2_length', 'ratio_of_kw1', 'ratio_of_interest4',
       'topics_length']
print('train_part...')
df_feature[col_new].loc[train_part_index].to_csv('data_preprocessing/train_part_x_length_p.csv',index=False)
print('evals...')
df_feature[col_new].loc[evals_index].to_csv('data_preprocessing/evals_x_length_p.csv',index=False)
print('test2...')
df_feature[col_new].loc[test2_index].to_csv('data_preprocessing/test2_x_length_p.csv',index=False)
print('Over')