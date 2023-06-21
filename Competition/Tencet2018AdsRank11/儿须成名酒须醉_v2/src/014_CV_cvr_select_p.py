##筛选特征
col_new = ['cvr_of_aid_and_onehot7',
         'cvr_of_age_and_onehot7',
         'cvr_of_consumptionAbility_and_onehot1',
         'cvr_of_aid_and_onehot3', 'cvr_of_advertiserId_and_onehot3',
         'cvr_of_advertiserId_and_onehot1', 'cvr_of_gender_and_onehot2', 
         'cvr_of_campaignId_and_onehot7', 'cvr_of_campaignId_and_onehot3',
         'cvr_of_adCategoryId_and_onehot3', 'cvr_of_creativeSize_and_onehot6', 
         'cvr_of_advertiserId_and_onehot10', 'cvr_of_campaignId_and_onehot1',
         'cvr_of_age_and_onehot1', 'cvr_of_creativeSize_and_onehot5',
         'cvr_of_aid_and_onehot5', 'cvr_of_creativeSize_and_onehot2', 
         'cvr_of_advertiserId_and_onehot6', 'cvr_of_age_and_onehot10', 
         'cvr_of_consumptionAbility_and_onehot7', 'cvr_of_age_and_onehot2',
         'cvr_of_os_and_onehot4', 'cvr_of_age_and_onehot6', 
         'cvr_of_creativeSize_and_onehot3', 'cvr_of_advertiserId_and_onehot8', 
         'cvr_of_carrier_and_onehot4', 'cvr_of_adCategoryId_and_onehot2', 
         'cvr_of_creativeSize_and_onehot10', 'cvr_of_aid_and_onehot1', 
         'cvr_of_creativeSize_and_onehot7', 'cvr_of_campaignId_and_onehot5', 
         'cvr_of_advertiserId_and_onehot4', 'cvr_of_aid_and_onehot10',
         'cvr_of_productId_and_onehot7', 'cvr_of_creativeSize_and_onehot8',
         'cvr_of_aid_and_onehot6', 'cvr_of_productType_and_onehot9', 
         'cvr_of_advertiserId_and_onehot7',
         'cvr_of_consumptionAbility_and_onehot4', 'cvr_of_advertiserId_and_onehot2']
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
train_part_y = pd.read_csv('data_preprocessing/train_part_y_p.csv',header=None)
evals_y = pd.read_csv('data_preprocessing/evals_y_p.csv',header=None)

for i in range(1,16):
    train_part_x = pd.concat([train_part_x,pd.read_csv('data_preprocessing/train_part_x_CV_cvr_'+str(i)+'.csv')],axis=1)
    evals_x = pd.concat([evals_x,pd.read_csv('data_preprocessing/evals_x_CV_cvr_'+str(i)+'.csv')],axis=1)
    for co in evals_x.columns:
        if co not in col_new:
            del evals_x[co] 
            del train_part_x[co]
    print(i)
print('train_part...')
train_part_x[col_new].to_csv('data_preprocessing/train_part_x_CV_cvr_select_p.csv',index=False)
print('evals...')
evals_x[col_new].to_csv('data_preprocessing/evals_x_CV_cvr_select_p.csv',index=False)
train_part_x = pd.DataFrame()
evals_x = pd.DataFrame()
test2_x = pd.DataFrame()
print('Reading test...')
for i in range(1,16):
    test2_x = pd.concat([test2_x,pd.read_csv('data_preprocessing/test2_x_CV_cvr_'+str(i)+'.csv')],axis=1)
    for co in test2_x.columns:
        if co not in col_new:
            del test2_x[co]
    print(i)
print('test2...')
test2_x[col_new].to_csv('data_preprocessing/test2_x_CV_cvr_select_p.csv',index=False)
print('Over')