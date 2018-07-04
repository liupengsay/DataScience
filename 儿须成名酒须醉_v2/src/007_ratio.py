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
print('Index...')
train_part_index = list(data[(data['label']!=-1)&(data['n_parts']!=1)].index)
evals_index = list(data[(data['label']!=-1)&(data['n_parts']==1)].index)
test1_index = list(data[data['n_parts']==6].index)
test2_index = list(data[data['n_parts']==7].index)
print('LabelEncoder...')
label_feature=['aid','uid', 'advertiserId', 'campaignId', 'creativeId',
       'creativeSize', 'adCategoryId', 'productId', 'productType', 'age',
       'gender','education', 'consumptionAbility', 'LBS',
       'os', 'carrier', 'house']
data = data[label_feature]
for feature in label_feature:
    s = time.time()
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])
    print(feature,int(time.time()-s),'s')
print('Done')

df_feature = pd.DataFrame()
data['cnt']=1
num = 0
print('Ratio clcik...')
col_type = label_feature.copy()
n = len(col_type)
for i in range(n):
    col_name = "ratio_click_of_"+col_type[i]
    s = time.time()
    df_feature[col_name] =(data[col_type[i]].map(data[col_type[i]].value_counts())/len(data)*100).astype(int)
    num+=1
    print(num,col_name,int(time.time()-s),'s')
n = len(col_type)
for i in range(n):
    for j in range(n):
        if i!=j:
            col_name = "ratio_click_of_"+col_type[j]+"_in_"+col_type[i]
            s = time.time()
            se = data.groupby([col_type[i],col_type[j]])['cnt'].sum()
            dt = data[[col_type[i],col_type[j]]]
            cnt = data[col_type[i]].map(data[col_type[i]].value_counts())
            df_feature[col_name] = ((pd.merge(dt,se.reset_index(),how='left',on=[col_type[i],col_type[j]]).sort_index()['cnt'].fillna(value=0)/cnt)*100).astype(int).values
            num+=1
            print(num,col_name,int(time.time()-s),'s')
            if num%30==0:
                k = int(num//30)
                print('Saving...')
                print('train_part...')
                df_feature.loc[train_part_index].to_csv('data_preprocessing/train_part_x_ratio_'+str(k)+'.csv',index=False)
                print('evals...')
                df_feature.loc[evals_index].to_csv('data_preprocessing/evals_x_ratio_'+str(k)+'.csv',index=False)
                print('test1...')
                df_feature.loc[test1_index].to_csv('data_preprocessing/test1_x_ratio_'+str(k)+'.csv',index=False)
                print('test2...')
                df_feature.loc[test2_index].to_csv('data_preprocessing/test2_x_ratio_'+str(k)+'.csv',index=False)
                df_feature = pd.DataFrame()
                print('Over')
print('Saving...')
k = k+1
print('train_part...')
df_feature.loc[train_part_index].to_csv('data_preprocessing/train_part_x_ratio_'+str(k)+'.csv',index=False)
print('evals...')
df_feature.loc[evals_index].to_csv('data_preprocessing/evals_x_ratio_'+str(k)+'.csv',index=False)
print('test1...')
df_feature.loc[test1_index].to_csv('data_preprocessing/test1_x_ratio_'+str(k)+'.csv',index=False)
print('test2...')
df_feature.loc[test2_index].to_csv('data_preprocessing/test2_x_ratio_'+str(k)+'.csv',index=False)
df_feature = pd.DataFrame()
print('Over')