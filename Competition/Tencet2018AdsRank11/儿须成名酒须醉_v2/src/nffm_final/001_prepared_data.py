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
import gc
import warnings
warnings.filterwarnings("ignore")

# 复赛数据
# 获取预处理的复赛数据
data = pd.read_csv('../data_preprocessing/train_test_merge.csv')

##根据生成的分块字段n_parts划分训练与验证集(n_parts=1)
print('Index...')
train_part_index = list(data[(data['label']!=-1)&(data['n_parts']!=1)].index)
evals_index = list(data[(data['label']!=-1)&(data['n_parts']==1)].index)
test2_index  = list(data[data['n_parts']==7].index)

del data['n_parts']

print('添加新的统计特征')
# 频数很少的种类，划为其他
def del_little_feature(data,feature):
    data1 = data[feature].value_counts().reset_index().rename(columns = {'index':feature,feature:'count'})
    data2 = data1[data1['count']<5]
    del_kind = data2[feature].values.tolist()
    for i in range(len(del_kind)):
        data.loc[data[feature]==del_kind[i],feature]=-2
    return data
data = del_little_feature(data, 'LBS')
print('LBS is prepared!')

print('添加新的交叉特征')
data['aid_age']=((data['aid']*100)+(data['age']))
data['aid_gender']=((data['aid']*100)+(data['gender']))
data['aid_LBS']=((data['aid']*1000)+(data['LBS']).astype(int)) 

train = data.loc[train_part_index]
evals = data.loc[evals_index]
test2 = data.loc[test2_index]

train.to_csv('./data/train.csv', index=False)
evals.to_csv('./data/evals.csv', index=False)
test2.to_csv('./data/test2.csv', index=False)
