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
for feature in label_feature:
    s = time.time()
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])
    print(feature,int(time.time()-s),'s')
print('Done')
col_type = label_feature.copy()
label_feature.append('label')
label_feature.append('n_parts')
data = data[label_feature]
df_feature = pd.DataFrame()
data['cnt']=1
print('Begin stat...')
n_parts = 7
num = 0
for co in col_type:
    s = time.time()
    col_name = 'cvr_of_'+co
    se = pd.Series()
    for i in range(n_parts):
        if i==0:
            df = data[data['n_parts']==i+1][[co]]
            stat = data[(data['n_parts']!=i+1)&(data['n_parts']<=5)][[co,'label']].groupby(co)['label'].mean()
            se = se.append(pd.Series(df[co].map(stat).values,index=df.index))
        elif i<=4 and 1<=i:
            df = data[data['n_parts']==i+1][[co]]
            stat = data[(data['n_parts']!=i+1)&(data['n_parts']<=5)&(data['n_parts']>=2)][[co,'label']].groupby(co)['label'].mean()
            se = se.append(pd.Series(df[co].map(stat).values,index=df.index))
        elif i>=5:
            df = data[data['n_parts']==i+1][[co]]
            stat = data[data['n_parts']<=5][[co,'label']].groupby(co)['label'].mean()
            se = se.append(pd.Series(df[co].map(stat).values,index=df.index))
    df_feature[col_name] = ((pd.Series(data.index).map(se)*10000)-400).fillna(value=-1).astype(int)
    num+=1
    print(num,col_name,int(time.time()-s),'s')
n = len(col_type)
for i in range(n):
    for j in range(n-i-1):
        s = time.time()
        col_name = 'cvr_of_'+col_type[i]+"_and_"+col_type[i+j+1]
        se = pd.Series()
        for k in range(n_parts):
            if k==0:
                stat = data[(data['n_parts']!=k+1)&(data['n_parts']<=5)].groupby([col_type[i],col_type[i+j+1]])['label'].mean()
                dt = data[data['n_parts']==k+1][[col_type[i],col_type[i+j+1]]]
                dt.insert(0,'index',list(dt.index))
                dt = pd.merge(dt,stat.reset_index(),how='left',on=[col_type[i],col_type[i+j+1]])
                se = se.append(pd.Series(dt['label'].values,index=list(dt['index'].values)))

            elif 1<=k and k<=4:
                stat = data[(data['n_parts']!=k+1)&(data['n_parts']<=5)&(data['n_parts']>=2)].groupby([col_type[i],col_type[i+j+1]])['label'].mean()
                dt = data[data['n_parts']==k+1][[col_type[i],col_type[i+j+1]]]
                dt.insert(0,'index',list(dt.index))
                dt = pd.merge(dt,stat.reset_index(),how='left',on=[col_type[i],col_type[i+j+1]])
                se = se.append(pd.Series(dt['label'].values,index=list(dt['index'].values)))
            elif k>=5:
                stat = data[data['n_parts']<=5].groupby([col_type[i],col_type[i+j+1]])['label'].mean()
                dt = data[data['n_parts']==k+1][[col_type[i],col_type[i+j+1]]]
                dt.insert(0,'index',list(dt.index))
                dt = pd.merge(dt,stat.reset_index(),how='left',on=[col_type[i],col_type[i+j+1]])
                se = se.append(pd.Series(dt['label'].values,index=list(dt['index'].values)))
        df_feature[col_name] = (pd.Series(data.index).map(se)*10000-400).fillna(value=-1).astype(int)
        num+=1
        print(num,col_name,int(time.time()-s),'s')
        if num%40==0:
            k = int(num/40)
            print(k,'Saving...')
            print('train_part...')
            df_feature.loc[train_part_index].to_csv('data_preprocessing/train_part_x_cvr_'+str(k)+'.csv',index=False)
            print('evals...')
            df_feature.loc[evals_index].to_csv('data_preprocessing/evals_x_cvr_'+str(k)+'.csv',index=False)
            print('test1...')
            df_feature.loc[test1_index].to_csv('data_preprocessing/test1_x_cvr_'+str(k)+'.csv',index=False)
            print('test2...')
            df_feature.loc[test2_index].to_csv('data_preprocessing/test2_x_cvr_'+str(k)+'.csv',index=False)
            df_feature = pd.DataFrame()
            print('Over')