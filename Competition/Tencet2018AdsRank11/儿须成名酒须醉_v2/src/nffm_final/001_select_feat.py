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
import gc
import warnings
warnings.filterwarnings("ignore")

##################################################################### evals
ev_cv_cvr = pd.read_csv('../data_preprocessing/evals_x_CV_cvr_select.csv')
li =  ['cvr_of_creativeId_and_onehot2', 'cvr_of_creativeId_and_onehot9',
       'cvr_of_creativeId_and_onehot16','cvr_of_consumptionAbility_and_onehot1',
       'cvr_of_creativeId_and_onehot10', 'cvr_of_creativeId_and_onehot15',
       'cvr_of_creativeId_and_onehot13', 'cvr_of_creativeId_and_onehot18',
       'cvr_of_age_and_onehot10', 'cvr_of_creativeId_and_onehot14']
ev_cv_cvr = ev_cv_cvr[li]

ev_cvr = pd.read_csv('../data_preprocessing/evals_x_cvr_select.csv')
li = ['cvr_of_aid_and_age', 'cvr_of_aid_and_gender', 
       'cvr_of_uid', 'cvr_of_aid_and_consumptionAbility',
       'cvr_of_aid_and_os', 'cvr_of_creativeSize_and_LBS', 
       'cvr_of_aid_and_education', 'cvr_of_uid_and_creativeSize', 
       'cvr_of_creativeSize', 'cvr_of_uid_and_adCategoryId']
ev_cvr = ev_cvr[li]

ev_ratio = pd.read_csv('../data_preprocessing/evals_x_ratio.csv')
li = ['ratio_click_of_aid_in_uid', 'ratio_click_of_creativeSize_in_uid',
       'ratio_click_of_age_in_aid', 'ratio_click_of_age_in_creativeSize',
       'ratio_click_of_gender_in_advertiserId',
       'ratio_click_of_gender_in_creativeSize',
       'ratio_click_of_consumptionAbility_in_aid',
       'ratio_click_of_age_in_advertiserId',
       'ratio_click_of_productType_in_uid',
       'ratio_click_of_productType_in_consumptionAbility']
ev_ratio = ev_ratio[li]

ev_cv_cvr.to_csv('./data/evals_x_CV_cvr.csv', index=False)
ev_cvr.to_csv('./data/evals_x_cvr.csv', index=False)
ev_ratio.to_csv('./data/evals_x_ratio.csv', index=False)

####################################################################### test
ev_cv_cvr = pd.read_csv('../data_preprocessing/test2_x_CV_cvr_select.csv')
li =  ['cvr_of_creativeId_and_onehot2', 'cvr_of_creativeId_and_onehot9',
       'cvr_of_creativeId_and_onehot16','cvr_of_consumptionAbility_and_onehot1',
       'cvr_of_creativeId_and_onehot10', 'cvr_of_creativeId_and_onehot15',
       'cvr_of_creativeId_and_onehot13', 'cvr_of_creativeId_and_onehot18',
       'cvr_of_age_and_onehot10', 'cvr_of_creativeId_and_onehot14',]
ev_cv_cvr = ev_cv_cvr[li]

ev_cvr = pd.read_csv('../data_preprocessing/test2_x_cvr_select.csv')
li = ['cvr_of_aid_and_age', 'cvr_of_aid_and_gender', 
       'cvr_of_uid', 'cvr_of_aid_and_consumptionAbility',
       'cvr_of_aid_and_os', 'cvr_of_creativeSize_and_LBS', 
       'cvr_of_aid_and_education', 'cvr_of_uid_and_creativeSize', 
       'cvr_of_creativeSize', 'cvr_of_uid_and_adCategoryId',]
ev_cvr = ev_cvr[li]

ev_ratio = pd.read_csv('../data_preprocessing/test2_x_ratio.csv')
li = ['ratio_click_of_aid_in_uid', 'ratio_click_of_creativeSize_in_uid',
       'ratio_click_of_age_in_aid', 'ratio_click_of_age_in_creativeSize',
       'ratio_click_of_gender_in_advertiserId',
       'ratio_click_of_gender_in_creativeSize',
       'ratio_click_of_consumptionAbility_in_aid',
       'ratio_click_of_age_in_advertiserId',
       'ratio_click_of_productType_in_uid',
       'ratio_click_of_productType_in_consumptionAbility']
ev_ratio = ev_ratio[li]

ev_cv_cvr.to_csv('./data/test2_x_CV_cvr.csv', index=False)
ev_cvr.to_csv('./data/test2_x_cvr.csv', index=False)
ev_ratio.to_csv('./data/test2_x_ratio.csv', index=False)

####################################################################### train
ev_cv_cvr = pd.read_csv('../data_preprocessing/train_part_x_CV_cvr_select.csv')
li =  ['cvr_of_creativeId_and_onehot2', 'cvr_of_creativeId_and_onehot9',
       'cvr_of_creativeId_and_onehot16','cvr_of_consumptionAbility_and_onehot1',
       'cvr_of_creativeId_and_onehot10', 'cvr_of_creativeId_and_onehot15',
       'cvr_of_creativeId_and_onehot13', 'cvr_of_creativeId_and_onehot18',
       'cvr_of_age_and_onehot10', 'cvr_of_creativeId_and_onehot14',]
ev_cv_cvr = ev_cv_cvr[li]

ev_cvr = pd.read_csv('../data_preprocessing/train_part_x_cvr_select.csv')
li = ['cvr_of_aid_and_age', 'cvr_of_aid_and_gender', 
       'cvr_of_uid', 'cvr_of_aid_and_consumptionAbility',
       'cvr_of_aid_and_os', 'cvr_of_creativeSize_and_LBS', 
       'cvr_of_aid_and_education', 'cvr_of_uid_and_creativeSize', 
       'cvr_of_creativeSize', 'cvr_of_uid_and_adCategoryId',]
ev_cvr = ev_cvr[li]

ev_ratio = pd.read_csv('../data_preprocessing/train_part_x_ratio.csv')
li = ['ratio_click_of_aid_in_uid', 'ratio_click_of_creativeSize_in_uid',
       'ratio_click_of_age_in_aid', 'ratio_click_of_age_in_creativeSize',
       'ratio_click_of_gender_in_advertiserId',
       'ratio_click_of_gender_in_creativeSize',
       'ratio_click_of_consumptionAbility_in_aid',
       'ratio_click_of_age_in_advertiserId',
       'ratio_click_of_productType_in_uid',
       'ratio_click_of_productType_in_consumptionAbility']
ev_ratio = ev_ratio[li]

ev_cv_cvr.to_csv('./data/train_x_CV_cvr.csv', index=False)
ev_cvr.to_csv('./data/train_x_cvr.csv', index=False)
ev_ratio.to_csv('./data/train_x_ratio.csv', index=False)