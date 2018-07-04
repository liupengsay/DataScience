col_new = ['count_type_aid_in_uid',
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
##筛选特征
from lightgbm import LGBMClassifier
import time
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

print('Reading train...')
train_part_x = pd.DataFrame()
evals_x = pd.DataFrame()

for i in range(1,10):
    train_part_x = pd.concat([train_part_x,pd.read_csv('data_preprocessing/train_part_x_unique_'+str(i)+'.csv')],axis=1)
    evals_x = pd.concat([evals_x,pd.read_csv('data_preprocessing/evals_x_unique_'+str(i)+'.csv')],axis=1)
    for co in evals_x.columns:
        if co not in col_new:
            del evals_x[co]
            del evals_x[co]
    print(i)
print('train_part...')
train_part_x[col_new].to_csv('data_preprocessing/train_part_x_unique_select.csv',index=False)
print('evals...')
evals_x[col_new].to_csv('data_preprocessing/evals_x_unique_select.csv',index=False)
train_part_x = pd.DataFrame()
evals_x = pd.DataFrame()
test1_x = pd.DataFrame()
test2_x = pd.DataFrame()
print('Reading test...')
for i in range(1,10):
    test1_x = pd.concat([test1_x,pd.read_csv('data_preprocessing/test1_x_unique_'+str(i)+'.csv')],axis=1)
    test2_x = pd.concat([test2_x,pd.read_csv('data_preprocessing/test2_x_unique_'+str(i)+'.csv')],axis=1)
    for co in test1_x.columns:
        if co not in col_new:
            del test1_x[co]
            del test2_x[co]
    print(i)
print('test1...')
test1_x[col_new].to_csv('data_preprocessing/test1_x_unique_select.csv',index=False)
print('test2...')
test2_x[col_new].to_csv('data_preprocessing/test2_x_unique_select.csv',index=False)
print('Over')