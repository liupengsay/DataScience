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
ad_feature=pd.read_csv('preliminary_competition_data/adFeature.csv')
if os.path.exists('preliminary_competition_data/userFeature.csv'):
    user_feature=pd.read_csv('preliminary_competition_data/userFeature.csv')
    print('User feature prepared')
else:
    userFeature_data = []
    with open('preliminary_competition_data/userFeature.data', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            userFeature_data.append(userFeature_dict)
            if i % 1000000 == 0:
                print(i)
        user_feature = pd.DataFrame(userFeature_data)
        print('User feature...')
        user_feature.to_csv('preliminary_competition_data/userFeature.csv', index=False)
        print('User feature prepared')
data=pd.read_csv('preliminary_competition_data/train.csv')
##关联数据
print("Merge...")
data=pd.merge(data,ad_feature,on='aid',how='left')
data=pd.merge(data,user_feature,on='uid',how='left')
user_feature = []
ad_feature = []
data=data.fillna('-1')
data = pd.DataFrame(data.values,columns=data.columns)
data['label'] = data['label'].astype(float)
data['n_parts']=-1
##concat初复赛数据
print('Concating...')
data = pd.concat([data,pd.read_csv('data_preprocessing/train_test_merge.csv')],axis=0,ignore_index=True)
print('Index...')
train_part_index = list(data[(data['label']!=-1)&(data['n_parts']!=1)].index)
evals_index = list(data[(data['label']!=-1)&(data['n_parts']==1)].index)
test2_index  = list(data[data['n_parts']==7].index)
print('label...')
data.loc[train_part_index]['label'].to_csv('data_preprocessing/train_part_y_p.csv',index=False)
data.loc[evals_index]['label'].to_csv('data_preprocessing/evals_y_p.csv',index=False)
data.loc[test2_index][['aid','uid']].to_csv('data_preprocessing/aid_uid_test2_p.csv',index=False)
##保存文件
print('Saving...')
data.to_csv('data_preprocessing/train_test_merge_p.csv',index=False)
print('Over')