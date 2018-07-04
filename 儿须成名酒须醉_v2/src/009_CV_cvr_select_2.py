col_new =['cvr_of_campaignId_and_onehot2', 'cvr_of_campaignId_and_onehot15',
           'cvr_of_campaignId_and_onehot9', 'cvr_of_campaignId_and_onehot16',
           'cvr_of_creativeSize_and_onehot15', 'cvr_of_consumptionAbility_and_onehot3', 
          'cvr_of_campaignId_and_onehot18', 'cvr_of_campaignId_and_onehot10', 
          'cvr_of_LBS_and_onehot9', 'cvr_of_creativeId_and_onehot5', 'cvr_of_education_and_onehot15',
          'cvr_of_gender_and_onehot7', 'cvr_of_creativeId_and_onehot12', 
          'cvr_of_campaignId_and_onehot4', 'cvr_of_campaignId_and_onehot20',
          'cvr_of_creativeSize_and_onehot2', 'cvr_of_creativeId_and_onehot3', 
          'cvr_of_age_and_onehot18', 'cvr_of_age_and_onehot2', 'cvr_of_campaignId_and_onehot11', 
          'cvr_of_campaignId_and_onehot14', 'cvr_of_consumptionAbility_and_onehot20',
          'cvr_of_campaignId_and_onehot1', 'cvr_of_age_and_onehot1',
          'cvr_of_creativeSize_and_onehot8', 'cvr_of_campaignId_and_onehot17',
          'cvr_of_os_and_onehot12', 'cvr_of_campaignId_and_onehot13', 
          'cvr_of_creativeId_and_onehot8', 'cvr_of_creativeSize_and_onehot13', 
          'cvr_of_consumptionAbility_and_onehot9', 'cvr_of_campaignId_and_onehot6',
          'cvr_of_age_and_onehot6', 'cvr_of_productType_and_onehot5', 
          'cvr_of_productType_and_onehot1', 'cvr_of_gender_and_onehot4',
          'cvr_of_gender_and_onehot19',
           'cvr_of_age_and_onehot4', 'cvr_of_advertiserId_and_onehot11', 'cvr_of_age_and_onehot20']
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

for i in range(1,33):
    train_part_x = pd.concat([train_part_x,pd.read_csv('data_preprocessing/train_part_x_CV_cvr_'+str(i)+'.csv')],axis=1)
    evals_x = pd.concat([evals_x,pd.read_csv('data_preprocessing/evals_x_CV_cvr_'+str(i)+'.csv')],axis=1)
    for co in evals_x.columns:
        if co not in col_new:
            del evals_x[co]
            del evals_x[co]
    print(i)
print('train_part...')
train_part_x[col_new].to_csv('data_preprocessing/train_part_x_CV_cvr_select_2.csv',index=False)
print('evals...')
evals_x[col_new].to_csv('data_preprocessing/evals_x_CV_cvr_select_2.csv',index=False)
train_part_x = pd.DataFrame()
evals_x = pd.DataFrame()
test1_x = pd.DataFrame()
test2_x = pd.DataFrame()
print('Reading test...')
for i in range(1,33):
    test1_x = pd.concat([test1_x,pd.read_csv('data_preprocessing/test1_x_CV_cvr_'+str(i)+'.csv')],axis=1)
    test2_x = pd.concat([test2_x,pd.read_csv('data_preprocessing/test2_x_CV_cvr_'+str(i)+'.csv')],axis=1)
    for co in test1_x.columns:
        if co not in col_new:
            del test1_x[co]
            del test2_x[co]
    print(i)
print('test1...')
test1_x[col_new].to_csv('data_preprocessing/test1_x_CV_cvr_select_2.csv',index=False)
print('test2...')
test2_x[col_new].to_csv('data_preprocessing/test2_x_CV_cvr_select_2.csv',index=False)
print('Over')