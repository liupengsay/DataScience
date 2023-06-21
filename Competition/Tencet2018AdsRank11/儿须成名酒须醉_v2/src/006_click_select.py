col_new = ['cnt_click_of_uid',
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

for i in range(1,6):
    train_part_x = pd.concat([train_part_x,pd.read_csv('data_preprocessing/train_part_x_click_'+str(i)+'.csv')],axis=1)
    evals_x = pd.concat([evals_x,pd.read_csv('data_preprocessing/evals_x_click_'+str(i)+'.csv')],axis=1)
    for co in evals_x.columns:
    if co not in col_new:
        del evals_x[co]
        del train_part_x[co]
    print(i)

print('train_part...')
train_part_x[col_new].to_csv('data_preprocessing/train_part_x_click_select.csv',index=False)
print('evals...')
evals_x[col_new].to_csv('data_preprocessing/evals_x_click_select.csv',index=False)
train_part_x = pd.DataFrame()
evals_x = pd.DataFrame()
test1_x = pd.DataFrame()
test2_x = pd.DataFrame()
print('Reading test...')
for i in range(1,6):
    test1_x = pd.concat([test1_x,pd.read_csv('data_preprocessing/test1_x_click_'+str(i)+'.csv')],axis=1)
    test2_x = pd.concat([test2_x,pd.read_csv('data_preprocessing/test2_x_click_'+str(i)+'.csv')],axis=1)
    for co in test1_x.columns:
        if co not in col_new:
            del test1_x[co]
            del test2_x[co]
    print(i)
print('test1...')
test1_x[col_new].to_csv('data_preprocessing/test1_x_click_select.csv',index=False)
print('test2...')
test2_x[col_new].to_csv('data_preprocessing/test2_x_click_select.csv',index=False)
print('Over')