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
data = pd.read_csv('data_preprocessing/train_test_merge.csv')
##划分训练与测试集
print('Index...')
train_part_index = list(data[(data['label']!=-1)&(data['n_parts']!=1)].index)
evals_index = list(data[(data['label']!=-1)&(data['n_parts']==1)].index)
test1_index = list(data[data['n_parts']==6].index)
test2_index = list(data[data['n_parts']==7].index)

print('LabelEncoder...')
label_feature=['aid','uid','advertiserId', 'campaignId', 'creativeId',
       'creativeSize', 'adCategoryId', 'productId', 'productType', 'age',
       'gender','education', 'consumptionAbility', 'LBS',
       'os', 'carrier', 'house']

for feature in label_feature:
    s = time.time()
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])
    print(feature,int(time.time()-s),'s')

data = data[label_feature]
df_feature = pd.DataFrame()
data['cnt']=1
print('Begin stat...')
col_type = label_feature.copy()
n = len(col_type)
num = 0
for i in range(n):
    for j in range(n):
        if i!=j:
            s = time.time()
            col_name = "count_type_"+col_type[j]+"_in_"+col_type[i]
            se = data.groupby([col_type[i]])[col_type[j]].value_counts()
            se = pd.Series(1,index=se.index).sum(level=col_type[i])
            df_feature[col_name] = (data[col_type[i]].map(se)).fillna(value=0).astype(int).values
            num+=1
            print(num,col_name,int(time.time()-s),'s')  
            if num%31==0:
                k = int(num//31)
                print('Saving...')
                print('train_part...')
                df_feature.loc[train_part_index].to_csv('data_preprocessing/train_part_x_unique_'+str(k)+'.csv',index=False)
                print('evals...')
                df_feature.loc[evals_index].to_csv('data_preprocessing/evals_x_unique_'+str(k)+'.csv',index=False)
                print('test1...')
                df_feature.loc[test1_index].to_csv('data_preprocessing/test1_x_unique_'+str(k)+'.csv',index=False)
                print('test2...')
                df_feature.loc[test2_index].to_csv('data_preprocessing/test2_x_unique_'+str(k)+'.csv',index=False)
                df_feature = pd.DataFrame()
                print('Over')
k = k+1
print('Saving...')
print('train_part...')
df_feature.loc[train_part_index].to_csv('data_preprocessing/train_part_x_unique_'+str(k)+'.csv',index=False)
print('evals...')
df_feature.loc[evals_index].to_csv('data_preprocessing/evals_x_unique_'+str(k)+'.csv',index=False)
print('test1...')
df_feature.loc[test1_index].to_csv('data_preprocessing/test1_x_unique_'+str(k)+'.csv',index=False)
print('test2...')
df_feature.loc[test2_index].to_csv('data_preprocessing/test2_x_unique_'+str(k)+'.csv',index=False)
df_feature = pd.DataFrame()
print('Over')