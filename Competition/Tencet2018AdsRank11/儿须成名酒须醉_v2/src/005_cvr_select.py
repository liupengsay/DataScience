##筛选特征
from lightgbm import LGBMClassifier
import time
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

print('Feature selecting...')
col_new =['cvr_of_aid_and_age',
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



train_part_x = pd.DataFrame()
evals_x = pd.DataFrame()
print('Reading train...')
for i in range(1,5):
    train_part_x = pd.concat([train_part_x,pd.read_csv('data_preprocessing/train_part_x_cvr_'+str(i)+'.csv')],axis=1)
    evals_x = pd.concat([evals_x,pd.read_csv('data_preprocessing/evals_x_cvr_'+str(i)+'.csv')],axis=1)
    for co in evals_x.columns:
        if co not in col_new:
            del evals_x[co]
            del train_part_x[co]
    print(i)

print('train_part...')
train_part_x[col_new].to_csv('data_preprocessing/train_part_x_cvr_select.csv',index=False)
print('evals...')
evals_x[col_new].to_csv('data_preprocessing/evals_x_cvr_select.csv',index=False)
train_part_x = pd.DataFrame()
evals_x = pd.DataFrame()
test1_x = pd.DataFrame()
test2_x = pd.DataFrame()

print('Reading test...')
for i in range(1,5):
    test1_x = pd.concat([test1_x,pd.read_csv('data_preprocessing/test1_x_cvr_'+str(i)+'.csv')],axis=1)
    test2_x = pd.concat([test2_x,pd.read_csv('data_preprocessing/test2_x_cvr_'+str(i)+'.csv')],axis=1)
    for co in test1_x.columns:
        if co not in col_new:
            del test1_x[co]
            del test2_x[co]
    print(i)
print('test1...')
test1_x[col_new].to_csv('data_preprocessing/test1_x_cvr_select.csv',index=False)
print('test2...')
test2_x[col_new].to_csv('data_preprocessing/test2_x_cvr_select.csv',index=False)
print('Over')