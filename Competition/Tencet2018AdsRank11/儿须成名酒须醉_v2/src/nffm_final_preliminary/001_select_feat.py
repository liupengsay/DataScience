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
ev_cv_cvr = pd.read_csv('../data_preprocessing/evals_x_CV_cvr_select_p.csv')
li =  ['cvr_of_aid_and_onehot7', 'cvr_of_age_and_onehot7',
       'cvr_of_consumptionAbility_and_onehot1', 'cvr_of_aid_and_onehot3',
       'cvr_of_advertiserId_and_onehot3', 'cvr_of_advertiserId_and_onehot1',
       'cvr_of_gender_and_onehot2', 'cvr_of_campaignId_and_onehot7',
       'cvr_of_campaignId_and_onehot3', 'cvr_of_adCategoryId_and_onehot3']
ev_cv_cvr = ev_cv_cvr[li]

ev_cvr = pd.read_csv('../data_preprocessing/evals_x_cvr_p.csv')
li = ['cvr_of_aid', 'cvr_of_uid', 'cvr_of_advertiserId',
       'cvr_of_creativeSize', 'cvr_of_productType', 'cvr_of_aid_and_age',
       'cvr_of_aid_and_gender', 'cvr_of_aid_and_education',
       'cvr_of_aid_and_consumptionAbility', 'cvr_of_aid_and_LBS']
ev_cvr = ev_cvr[li]

ev_ratio = pd.read_csv('../data_preprocessing/evals_x_ratio_p.csv')
li = ['ratio_click_of_age_in_aid', 'ratio_click_of_gender_in_aid',
       'ratio_click_of_consumptionAbility_in_aid',
       'ratio_click_of_aid_in_uid', 'ratio_click_of_advertiserId_in_uid',
       'ratio_click_of_campaignId_in_uid',
       'ratio_click_of_creativeSize_in_uid',
       'ratio_click_of_adCategoryId_in_uid',
       'ratio_click_of_productId_in_uid',
       'ratio_click_of_productType_in_uid']
ev_ratio = ev_ratio[li]

ev_length = pd.read_csv('../data_preprocessing/evals_x_length_p.csv')
li = ['interest2_length', 'ratio_of_interest2', 'interest1_length',
       'ratio_of_interest1', 'ct_length', 'marriageStatus_length',
       'interests_length', 'ratio_of_interest5', 'interest5_length',
       'kw2_length', 'ratio_of_kw1', 'ratio_of_interest4',
       'topics_length']
ev_length = ev_length[li]

ev_click = pd.read_csv('../data_preprocessing/evals_x_click_p.csv')
li = ['cnt_click_of_aid', 'cnt_click_of_uid', 'cnt_click_of_advertiserId',
       'cnt_click_of_productType', 'cnt_click_of_gender',
       'cnt_click_of_consumptionAbility', 'cnt_click_of_age_and_aid',
       'cnt_click_of_gender_and_aid', 'cnt_click_of_creativeSize_and_uid',
       'cnt_click_of_adCategoryId_and_uid']
ev_click = ev_click[li]

ev_cv_cvr.to_csv('./data/evals_x_CV_cvr_p.csv', index=False)
ev_cvr.to_csv('./data/evals_x_cvr_p.csv', index=False)
ev_ratio.to_csv('./data/evals_x_ratio_p.csv', index=False)
ev_length.to_csv('./data/evals_x_length_p.csv', index=False)
ev_click.to_csv('./data/evals_x_click_p.csv', index=False)

####################################################################### test
ev_cv_cvr = pd.read_csv('../data_preprocessing/test2_x_CV_cvr_select_p.csv')
li =  ['cvr_of_aid_and_onehot7', 'cvr_of_age_and_onehot7',
       'cvr_of_consumptionAbility_and_onehot1', 'cvr_of_aid_and_onehot3',
       'cvr_of_advertiserId_and_onehot3', 'cvr_of_advertiserId_and_onehot1',
       'cvr_of_gender_and_onehot2', 'cvr_of_campaignId_and_onehot7',
       'cvr_of_campaignId_and_onehot3', 'cvr_of_adCategoryId_and_onehot3']
ev_cv_cvr = ev_cv_cvr[li]

ev_cvr = pd.read_csv('../data_preprocessing/test2_x_cvr_p.csv')
li = ['cvr_of_aid', 'cvr_of_uid', 'cvr_of_advertiserId',
       'cvr_of_creativeSize', 'cvr_of_productType', 'cvr_of_aid_and_age',
       'cvr_of_aid_and_gender', 'cvr_of_aid_and_education',
       'cvr_of_aid_and_consumptionAbility', 'cvr_of_aid_and_LBS']
ev_cvr = ev_cvr[li]

ev_ratio = pd.read_csv('../data_preprocessing/test2_x_ratio_p.csv')
li = ['ratio_click_of_age_in_aid', 'ratio_click_of_gender_in_aid',
       'ratio_click_of_consumptionAbility_in_aid',
       'ratio_click_of_aid_in_uid', 'ratio_click_of_advertiserId_in_uid',
       'ratio_click_of_campaignId_in_uid',
       'ratio_click_of_creativeSize_in_uid',
       'ratio_click_of_adCategoryId_in_uid',
       'ratio_click_of_productId_in_uid',
       'ratio_click_of_productType_in_uid']
ev_ratio = ev_ratio[li]

ev_length = pd.read_csv('../data_preprocessing/test2_x_length_p.csv')
li = ['interest2_length', 'ratio_of_interest2', 'interest1_length',
       'ratio_of_interest1', 'ct_length', 'marriageStatus_length',
       'interests_length', 'ratio_of_interest5', 'interest5_length',
       'kw2_length', 'ratio_of_kw1', 'ratio_of_interest4',
       'topics_length']
ev_length = ev_length[li]

ev_click = pd.read_csv('../data_preprocessing/test2_x_click_p.csv')
li = ['cnt_click_of_aid', 'cnt_click_of_uid', 'cnt_click_of_advertiserId',
       'cnt_click_of_productType', 'cnt_click_of_gender',
       'cnt_click_of_consumptionAbility', 'cnt_click_of_age_and_aid',
       'cnt_click_of_gender_and_aid', 'cnt_click_of_creativeSize_and_uid',
       'cnt_click_of_adCategoryId_and_uid']
ev_click = ev_click[li]

ev_cv_cvr.to_csv('./data/test2_x_CV_cvr_p.csv', index=False)
ev_cvr.to_csv('./data/test2_x_cvr_p.csv', index=False)
ev_ratio.to_csv('./data/test2_x_ratio_p.csv', index=False)
ev_length.to_csv('./data/test2_x_length_p.csv', index=False)
ev_click.to_csv('./data/test2_x_click_p.csv', index=False)

####################################################################### train
ev_cv_cvr = pd.read_csv('../data_preprocessing/train_part_x_CV_cvr_select_p.csv')
li =  ['cvr_of_aid_and_onehot7', 'cvr_of_age_and_onehot7',
       'cvr_of_consumptionAbility_and_onehot1', 'cvr_of_aid_and_onehot3',
       'cvr_of_advertiserId_and_onehot3', 'cvr_of_advertiserId_and_onehot1',
       'cvr_of_gender_and_onehot2', 'cvr_of_campaignId_and_onehot7',
       'cvr_of_campaignId_and_onehot3', 'cvr_of_adCategoryId_and_onehot3']
ev_cv_cvr = ev_cv_cvr[li]

ev_cvr = pd.read_csv('../data_preprocessing/train_part_x_cvr_p.csv')
li = ['cvr_of_aid', 'cvr_of_uid', 'cvr_of_advertiserId',
       'cvr_of_creativeSize', 'cvr_of_productType', 'cvr_of_aid_and_age',
       'cvr_of_aid_and_gender', 'cvr_of_aid_and_education',
       'cvr_of_aid_and_consumptionAbility', 'cvr_of_aid_and_LBS']
ev_cvr = ev_cvr[li]

ev_ratio = pd.read_csv('../data_preprocessing/train_part_x_ratio_p.csv')
li = ['ratio_click_of_age_in_aid', 'ratio_click_of_gender_in_aid',
       'ratio_click_of_consumptionAbility_in_aid',
       'ratio_click_of_aid_in_uid', 'ratio_click_of_advertiserId_in_uid',
       'ratio_click_of_campaignId_in_uid',
       'ratio_click_of_creativeSize_in_uid',
       'ratio_click_of_adCategoryId_in_uid',
       'ratio_click_of_productId_in_uid',
       'ratio_click_of_productType_in_uid']
ev_ratio = ev_ratio[li]

ev_length = pd.read_csv('../data_preprocessing/train_part_x_length_p.csv')
li = ['interest2_length', 'ratio_of_interest2', 'interest1_length',
       'ratio_of_interest1', 'ct_length', 'marriageStatus_length',
       'interests_length', 'ratio_of_interest5', 'interest5_length',
       'kw2_length', 'ratio_of_kw1', 'ratio_of_interest4',
       'topics_length']
ev_length = ev_length[li]

ev_click = pd.read_csv('../data_preprocessing/train_part_x_click_p.csv')
li = ['cnt_click_of_aid', 'cnt_click_of_uid', 'cnt_click_of_advertiserId',
       'cnt_click_of_productType', 'cnt_click_of_gender',
       'cnt_click_of_consumptionAbility', 'cnt_click_of_age_and_aid',
       'cnt_click_of_gender_and_aid', 'cnt_click_of_creativeSize_and_uid',
       'cnt_click_of_adCategoryId_and_uid']
ev_click = ev_click[li]

ev_cv_cvr.to_csv('./data/train_x_CV_cvr_p.csv', index=False)
ev_cvr.to_csv('./data/train_x_cvr_p.csv', index=False)
ev_ratio.to_csv('./data/train_x_ratio_p.csv', index=False)
ev_length.to_csv('./data/train_x_length_p.csv', index=False)
ev_click.to_csv('./data/train_x_click_p.csv', index=False)
