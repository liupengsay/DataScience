import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import numpy as np
import random
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from collections import OrderedDict
import pickle as pkl
from collections import Counter
import scipy.special as special
import gc
import os
threshold=1000
random.seed(2018)

def load_data(update=False):
    train_df = pd.read_csv('./data/train_cv_cvr_ratio_cli_len.csv')
    test_df = pd.read_csv('./data/test2_cv_cvr_ratio_cli_len.csv')

    train_df, dev_df,_,_ = train_test_split(train_df,train_df,test_size=0.01, random_state=2018)

    print(len(train_df),len(dev_df),len(test_df))

    return train_df,dev_df,test_df

def output_label(train_df,dev_df,test_df):
    with open('ffm_data/dev/label','w') as f:
        for i in list(dev_df['label']):
            f.write(str(i)+'\n')
    with open('ffm_data/test/label','w') as f:
        for i in list(test_df['label']):
            f.write(str(i)+'\n')
    with open('ffm_data/train/label','w') as f:
        for i in list(train_df['label']):
            f.write(str(i)+'\n')
                       
def single_features(train_df,dev_df,test_df,word2index):   
    single_ids_features = [
       # 原始特征
       'aid','advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId',
       'productType', 'age', 'gender','education', 'consumptionAbility', 'LBS', 'carrier', 'house',
       # 交叉特征
       'aid_age', 'aid_LBS', 'aid_gender',
       # CV_cvr特征
       'cvr_of_aid_and_onehot7', 'cvr_of_age_and_onehot7',
       'cvr_of_consumptionAbility_and_onehot1', 'cvr_of_aid_and_onehot3',
       'cvr_of_advertiserId_and_onehot3', 'cvr_of_advertiserId_and_onehot1',
       'cvr_of_gender_and_onehot2', 'cvr_of_campaignId_and_onehot7',
       'cvr_of_campaignId_and_onehot3', 'cvr_of_adCategoryId_and_onehot3',
       # cvr特征
       'cvr_of_aid', 'cvr_of_uid', 'cvr_of_advertiserId',
       'cvr_of_creativeSize', 'cvr_of_productType', 'cvr_of_aid_and_age',
       'cvr_of_aid_and_gender', 'cvr_of_aid_and_education',
       'cvr_of_aid_and_consumptionAbility', 'cvr_of_aid_and_LBS',
       # ratio特征
       'ratio_click_of_age_in_aid', 'ratio_click_of_gender_in_aid',
       'ratio_click_of_consumptionAbility_in_aid',
       'ratio_click_of_aid_in_uid', 'ratio_click_of_advertiserId_in_uid',
       'ratio_click_of_campaignId_in_uid',
       'ratio_click_of_creativeSize_in_uid',
       'ratio_click_of_adCategoryId_in_uid',
       'ratio_click_of_productId_in_uid',
       'ratio_click_of_productType_in_uid',
       # click特征
       'cnt_click_of_aid', 'cnt_click_of_uid', 'cnt_click_of_advertiserId',
       'cnt_click_of_productType', 'cnt_click_of_gender',
       'cnt_click_of_consumptionAbility', 'cnt_click_of_age_and_aid',
       'cnt_click_of_gender_and_aid', 'cnt_click_of_creativeSize_and_uid',
       'cnt_click_of_adCategoryId_and_uid',
       # length特征
       'interest2_length', 'ratio_of_interest2', 'interest1_length',
       'ratio_of_interest1', 'ct_length', 'marriageStatus_length',
       'interests_length', 'ratio_of_interest5', 'interest5_length',
       'kw2_length', 'ratio_of_kw1', 'ratio_of_interest4',
       'topics_length'
       ]      
    
    for s in single_ids_features:   
        cont={}
        
        with open('ffm_data/train/'+str(s),'w') as f:
            for line in list(train_df[s].values):
                f.write(str(line)+'\n')
                if str(line) not in cont:
                    cont[str(line)]=0
                cont[str(line)]+=1                
        
        with open('ffm_data/dev/'+str(s),'w') as f:
            for line in list(dev_df[s].values):
                f.write(str(line)+'\n')
                
        with open('ffm_data/test/'+str(s),'w') as f:
            for line in list(test_df[s].values):
                f.write(str(line)+'\n')

        index=[]
        for k in cont:
            if cont[k]>=threshold:
                index.append(k)
        word2index[s]={}
        for idx,val in enumerate(index):
            word2index[s][val]=idx+2
        print(s+' done!')


def mutil_ids(train_df,dev_df,test_df,word2index):  
    features_mutil = [
                     'interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3',
                     'topic1','topic2','topic3','appIdAction','appIdInstall','marriageStatus','ct','os'
                     ]
    for s in features_mutil:
        cont={}        
        with open('ffm_data/train/'+str(s),'w') as f:
            for lines in list(train_df[s].values):
                f.write(str(lines)+'\n')
                for line in lines.split():
                    if str(line)not in cont:
                        cont[str(line)]=0
                    cont[str(line)]+=1
                                 
        with open('ffm_data/dev/'+str(s),'w') as f:
            for line in list(dev_df[s].values):
                f.write(str(line)+'\n')
                
        with open('ffm_data/test/'+str(s),'w') as f:
            for line in list(test_df[s].values):
                f.write(str(line)+'\n')

        index=[]
        for k in cont:
            if cont[k]>=threshold:
                index.append(k)
        word2index[s]={}
        for idx,val in enumerate(index):
            word2index[s][val]=idx+2
        print(s+' done!')  
                
if os.path.exists('ffm_data/dic.pkl'):  
    word2index=pkl.load(open('ffm_data/dic.pkl','rb'))
else:
    word2index={}
    
print('Loading data...')
train_df,dev_df,test_df=load_data(update=False)

print('Output label files...')
output_label(train_df,dev_df,test_df)
print('Single ids features...')
single_features(train_df,dev_df,test_df,word2index)
pkl.dump(word2index,open('ffm_data/dic.pkl','wb'))
print('Mutil features...') 
mutil_ids(train_df,dev_df,test_df,word2index)  
pkl.dump(word2index,open('ffm_data/dic.pkl','wb'))

print('Vocabulary bulding...')   
pkl.dump(word2index,open('ffm_data/dic.pkl','wb'))
