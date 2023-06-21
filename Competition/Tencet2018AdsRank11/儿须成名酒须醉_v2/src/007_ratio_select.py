col_new = ['ratio_click_of_aid_in_uid',
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

for i in range(1,11):
    train_part_x = pd.concat([train_part_x,pd.read_csv('data_preprocessing/train_part_x_ratio_'+str(i)+'.csv')],axis=1)
    evals_x = pd.concat([evals_x,pd.read_csv('data_preprocessing/evals_x_ratio_'+str(i)+'.csv')],axis=1)
    for co in evals_x.columns:
        if co not in col_new:
            del evals_x[co]
            del evals_x[co]
    print(i)
print('train_part...')
train_part_x[col_new].to_csv('data_preprocessing/train_part_x_ratio_select.csv',index=False)
print('evals...')
evals_x[col_new].to_csv('data_preprocessing/evals_x_ratio_select.csv',index=False)
train_part_x = pd.DataFrame()
evals_x = pd.DataFrame()
test1_x = pd.DataFrame()
test2_x = pd.DataFrame()
print('Reading test...')
for i in range(1,11):
    test1_x = pd.concat([test1_x,pd.read_csv('data_preprocessing/test1_x_ratio_'+str(i)+'.csv')],axis=1)
    test2_x = pd.concat([test2_x,pd.read_csv('data_preprocessing/test2_x_ratio_'+str(i)+'.csv')],axis=1)
    for co in test1_x.columns:
        if co not in col_new:
            del test1_x[co]
            del test2_x[co]
    print(i)
print('test1...')
test1_x[col_new].to_csv('data_preprocessing/test1_x_ratio_select.csv',index=False)
print('test2...')
test2_x[col_new].to_csv('data_preprocessing/test2_x_ratio_select.csv',index=False)
print('Over')