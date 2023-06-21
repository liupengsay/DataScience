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
print('Index...')
train_part_index = list(data[(data['label']!=-1)&(data['n_parts']!=1)].index)
evals_index = list(data[(data['label']!=-1)&(data['n_parts']==1)].index)
test2_index = list(data[data['n_parts']==7].index)

label_feature = ['aid','uid', 'advertiserId', 'campaignId', 'creativeId',
       'creativeSize', 'adCategoryId', 'productId', 'productType', 'age',
       'gender','education', 'consumptionAbility', 'LBS',
       'os', 'carrier', 'house']

col_type = label_feature.copy()
label_feature.append('label')
label_feature.append('n_parts')
for co in data.columns:
    if co not in label_feature:
        del data[co]
        print('del',co)

col_select = ['ratio_click_of_aid_in_uid',
 'ratio_click_of_creativeSize_in_uid',
 'ratio_click_of_age_in_aid',
 'ratio_click_of_age_in_creativeSize',
 'ratio_click_of_gender_in_advertiserId',
 'ratio_click_of_gender_in_creativeSize',
 'ratio_click_of_consumptionAbility_in_aid',
 'ratio_click_of_age_in_advertiserId',
 'ratio_click_of_productType_in_uid',
 'ratio_click_of_productType_in_consumptionAbility',
 'ratio_click_of_productType_in_age',
 'ratio_click_of_gender_in_consumptionAbility',
 'ratio_click_of_creativeSize_in_age',
 'ratio_click_of_gender_in_aid',
 'ratio_click_of_creativeSize_in_productType',
 'ratio_click_of_house_in_campaignId',
 'ratio_click_of_house_in_creativeSize',
 'ratio_click_of_aid_in_creativeSize',
 'ratio_click_of_productId_in_uid',
 'ratio_click_of_os_in_advertiserId',
 'ratio_click_of_adCategoryId_in_uid',
 'ratio_click_of_productType_in_creativeSize',
 'ratio_click_of_productType_in_os',
 'ratio_click_of_productType_in_education',
 'ratio_click_of_advertiserId_in_uid',
 'ratio_click_of_gender_in_productId',
 'ratio_click_of_consumptionAbility_in_age',
 'ratio_click_of_adCategoryId_in_creativeSize',
 'ratio_click_of_creativeSize_in_education',
 'ratio_click_of_campaignId_in_uid',
 'ratio_click_of_consumptionAbility_in_advertiserId']
df_feature = pd.DataFrame()
data['cnt']=1
num = 0
print('Ratio clcik...')
n = len(col_type)
for i in range(n):
    col_name = "ratio_click_of_"+col_type[i]
    if col_name in col_select:
        s = time.time()
        df_feature[col_name] =(data[col_type[i]].map(data[col_type[i]].value_counts())/len(data)*100).astype(int)
        num+=1
        print(num,col_name,int(time.time()-s),'s')
for i in range(n):
    for j in range(n):
        if i!=j:
            col_name = "ratio_click_of_"+col_type[j]+"_in_"+col_type[i]
            if col_name in col_select:
                s = time.time()
                se = data.groupby([col_type[i],col_type[j]])['cnt'].sum()
                dt = data[[col_type[i],col_type[j]]]
                cnt = data[col_type[i]].map(data[col_type[i]].value_counts())
                df_feature[col_name] = ((pd.merge(dt,se.reset_index(),how='left',on=[col_type[i],col_type[j]]).sort_index()['cnt'].fillna(value=0)/cnt)*100).astype(int).values
                num+=1
                print(num,col_name,int(time.time()-s),'s')
print('train_part...')
df_feature.loc[train_part_index].to_csv('data_preprocessing/train_part_x_ratio_p.csv',index=False)
print('evals...')
df_feature.loc[evals_index].to_csv('data_preprocessing/evals_x_ratio_p.csv',index=False)
print('test2...')
df_feature.loc[test2_index].to_csv('data_preprocessing/test2_x_ratio_p.csv',index=False)
print('Over')

col_select = ['cnt_click_of_uid',
 'cnt_click_of_creativeSize_and_uid',
 'cnt_click_of_age_and_creativeSize',
 'cnt_click_of_gender_and_creativeSize',
 'cnt_click_of_productType_and_uid',
 'cnt_click_of_gender_and_aid',
 'cnt_click_of_productId_and_creativeSize',
 'cnt_click_of_gender_and_advertiserId',
 'cnt_click_of_adCategoryId_and_uid',
 'cnt_click_of_age_and_aid',
 'cnt_click_of_age_and_productType',
 'cnt_click_of_consumptionAbility',
 'cnt_click_of_productType_and_creativeSize',
 'cnt_click_of_age_and_advertiserId',
 'cnt_click_of_productType',
 'cnt_click_of_advertiserId',
 'cnt_click_of_aid',
 'cnt_click_of_adCategoryId_and_creativeSize',
 'cnt_click_of_productId_and_advertiserId',
 'cnt_click_of_gender_and_campaignId',
 'cnt_click_of_education_and_creativeSize',
 'cnt_click_of_age_and_adCategoryId',
 'cnt_click_of_productId_and_uid',
 'cnt_click_of_gender',
 'cnt_click_of_consumptionAbility_and_advertiserId',
 'cnt_click_of_os_and_age',
 'cnt_click_of_consumptionAbility_and_productId',
 'cnt_click_of_carrier_and_os',
 'cnt_click_of_consumptionAbility_and_gender',
 'cnt_click_of_age_and_productId']
num = 0
df_feature = pd.DataFrame()
for i in range(n):
    col_name = "cnt_click_of_"+col_type[i]
    if col_name in col_select:
        s = time.time()
        se = (data[col_type[i]].map(data[col_type[i]].value_counts())).astype(int)
        semax = se.max()
        semin = se.min()
        df_feature[col_name] = ((se-se.min())/(se.max()-se.min())*100).astype(int).values
        num+=1
        print(num,col_name,int(time.time()-s),'s')

print('Begin stat...')##再加入离散特征
n = len(col_type)
for i in range(n):
    for j in range(n-i-1):
        col_name = "cnt_click_of_"+col_type[i+j+1]+"_and_"+col_type[i]
        if col_name in col_select:
            s = time.time()
            se = data.groupby([col_type[i],col_type[i+j+1]])['cnt'].sum()
            dt = data[[col_type[i],col_type[i+j+1]]]
            se = (pd.merge(dt,se.reset_index(),how='left',
                            on=[col_type[i],col_type[j+i+1]]).sort_index()['cnt'].fillna(value=0)).astype(int)
            semax = se.max()
            semin = se.min()
            df_feature[col_name] = ((se-se.min())/(se.max()-se.min())*100).fillna(value=0).astype(int).values
            num+=1
            print(num,col_name,int(time.time()-s),'s')
print('train_part...')
df_feature.loc[train_part_index].to_csv('data_preprocessing/train_part_x_click_p.csv',index=False)
print('evals...')
df_feature.loc[evals_index].to_csv('data_preprocessing/evals_x_click_p.csv',index=False)
print('test2...')
df_feature.loc[test2_index].to_csv('data_preprocessing/test2_x_click_p.csv',index=False)
print('Over')

col_select = ['count_type_aid_in_uid',
 'count_type_uid_in_age',
 'count_type_productId_in_uid',
 'count_type_uid_in_aid',
 'count_type_advertiserId_in_creativeSize',
 'count_type_productType_in_creativeSize',
 'count_type_uid_in_consumptionAbility',
 'count_type_LBS_in_aid',
 'count_type_aid_in_age',
 'count_type_uid_in_gender',
 'count_type_LBS_in_advertiserId',
 'count_type_aid_in_productType',
 'count_type_LBS_in_campaignId',
 'count_type_uid_in_advertiserId',
 'count_type_productType_in_uid',
 'count_type_uid_in_os',
 'count_type_advertiserId_in_uid',
 'count_type_uid_in_education',
 'count_type_LBS_in_education',
 'count_type_creativeSize_in_adCategoryId',
 'count_type_LBS_in_carrier',
 'count_type_creativeSize_in_advertiserId',
 'count_type_uid_in_creativeSize',
 'count_type_aid_in_LBS',
 'count_type_uid_in_adCategoryId',
 'count_type_advertiserId_in_adCategoryId']
df_feature = pd.DataFrame()
print('Begin stat...')
n = len(col_type)
num = 0
for i in range(n):
    for j in range(n):
        if i!=j:
            s = time.time()
            col_name = "count_type_"+col_type[j]+"_in_"+col_type[i]
            if col_name in col_select:
                se = data.groupby([col_type[i]])[col_type[j]].value_counts()
                se = pd.Series(1,index=se.index).sum(level=col_type[i])
                se = (data[col_type[i]].map(se))
                semax = se.max()
                semin = se.min()
                df_feature[col_name] = ((se-se.min())/(se.max()-se.min())*100).fillna(value=0).astype(int).values
                num+=1
                print(num,col_name,int(time.time()-s),'s')  
print('train_part...')
df_feature.loc[train_part_index].to_csv('data_preprocessing/train_part_x_unique_p.csv',index=False)
print('evals...')
df_feature.loc[evals_index].to_csv('data_preprocessing/evals_x_unique_p.csv',index=False)
print('test2...')
df_feature.loc[test2_index].to_csv('data_preprocessing/test2_x_unique_p.csv',index=False)
print('Over')

col_select = ['cvr_of_aid_and_age',
 'cvr_of_aid_and_gender',
 'cvr_of_uid',
 'cvr_of_aid_and_consumptionAbility',
 'cvr_of_aid_and_os',
 'cvr_of_creativeSize_and_LBS',
 'cvr_of_aid_and_education',
 'cvr_of_uid_and_creativeSize',
 'cvr_of_creativeSize',
 'cvr_of_uid_and_adCategoryId',
 'cvr_of_uid_and_productType',
 'cvr_of_advertiserId_and_consumptionAbility',
 'cvr_of_uid_and_productId',
 'cvr_of_creativeSize_and_education',
 'cvr_of_aid_and_LBS',
 'cvr_of_aid_and_carrier',
 'cvr_of_creativeSize_and_gender',
 'cvr_of_creativeSize_and_productType',
 'cvr_of_campaignId_and_education',
 'cvr_of_aid',
 'cvr_of_uid_and_advertiserId',
 'cvr_of_aid_and_house',
 'cvr_of_advertiserId_and_LBS',
 'cvr_of_adCategoryId_and_consumptionAbility',
 'cvr_of_campaignId_and_os',
 'cvr_of_campaignId_and_consumptionAbility',
 'cvr_of_consumptionAbility_and_os',
 'cvr_of_advertiserId_and_creativeSize',
 'cvr_of_adCategoryId_and_gender',
 'cvr_of_productType',
 'cvr_of_advertiserId',
 'cvr_of_productType_and_gender',
 'cvr_of_age_and_consumptionAbility',
 'cvr_of_creativeSize_and_consumptionAbility',
 'cvr_of_campaignId_and_gender']
print('转化率特征开始...')
df_feature = pd.DataFrame()
data['cnt']=1
n_parts = [-1]
n_parts.extend(list(range(1,8)))
num = 0
for co in col_type:
    s = time.time()
    col_name = 'cvr_of_'+co
    if col_name in col_select:
        se = pd.Series()
        for i in n_parts:
            if i==1:
                df = data[data['n_parts']==i][[co]]
                stat = data[(data['n_parts']!=i)&(data['n_parts']<=5)][[co,'label']].groupby(co)['label'].mean()
                se = se.append(pd.Series(df[co].map(stat).values,index=df.index))
            elif i<=5 and i!=1:
                df = data[data['n_parts']==i][[co]]
                stat = data[(data['n_parts']!=i)&(data['n_parts']<=5)&(data['n_parts']!=1)][[co,'label']].groupby(co)['label'].mean()
                se = se.append(pd.Series(df[co].map(stat).values,index=df.index))
            elif i>=5:
                df = data[data['n_parts']==i][[co]]
                stat = data[data['n_parts']<=5][[co,'label']].groupby(co)['label'].mean()
                se = se.append(pd.Series(df[co].map(stat).values,index=df.index))
        se = pd.Series(data.index).map(se)
        semax = se.max()
        semin = se.min()
        se = 100*(se-semin)/(semax-semin)
        df_feature[col_name] = (se.fillna(value=-1)).astype(int)
        num+=1
        print(num,col_name,int(time.time()-s),'s')
n = len(col_type)
for i in range(n):
    for j in range(n-i-1):
        s = time.time()
        col_name = 'cvr_of_'+col_type[i]+"_and_"+col_type[i+j+1]
        if col_name in col_select:
            se = pd.Series()
            for k in n_parts:
                if k==1:
                    stat = data[(data['n_parts']!=k)&(data['n_parts']<=5)].groupby([col_type[i],col_type[i+j+1]])['label'].mean()
                    dt = data[data['n_parts']==k][[col_type[i],col_type[i+j+1]]]
                    dt.insert(0,'index',list(dt.index))
                    dt = pd.merge(dt,stat.reset_index(),how='left',on=[col_type[i],col_type[i+j+1]])
                    se = se.append(pd.Series(dt['label'].values,index=list(dt['index'].values)))

                elif k<=5 and k!=1:
                    stat = data[(data['n_parts']!=k)&(data['n_parts']<=5)&(data['n_parts']!=1)].groupby([col_type[i],col_type[i+j+1]])['label'].mean()
                    dt = data[data['n_parts']==k][[col_type[i],col_type[i+j+1]]]
                    dt.insert(0,'index',list(dt.index))
                    dt = pd.merge(dt,stat.reset_index(),how='left',on=[col_type[i],col_type[i+j+1]])
                    se = se.append(pd.Series(dt['label'].values,index=list(dt['index'].values)))
                elif k>=5:
                    stat = data[data['n_parts']<=5].groupby([col_type[i],col_type[i+j+1]])['label'].mean()
                    dt = data[data['n_parts']==k][[col_type[i],col_type[i+j+1]]]
                    dt.insert(0,'index',list(dt.index))
                    dt = pd.merge(dt,stat.reset_index(),how='left',on=[col_type[i],col_type[i+j+1]])
                    se = se.append(pd.Series(dt['label'].values,index=list(dt['index'].values)))
            se = pd.Series(data.index).map(se)
            semax = se.max()
            semin = se.min()
            se = 100*(se-semin)/(semax-semin)
            df_feature[col_name] = (se.fillna(value=-1)).astype(int)
            num+=1
            print(num,col_name,int(time.time()-s),'s')
print('train_part...')
df_feature.loc[train_part_index].to_csv('data_preprocessing/train_part_x_cvr_p.csv',index=False)
print('evals...')
df_feature.loc[evals_index].to_csv('data_preprocessing/evals_x_cvr_p.csv',index=False)
print('test2...')
df_feature.loc[test2_index].to_csv('data_preprocessing/test2_x_cvr_p.csv',index=False)
print('Over')