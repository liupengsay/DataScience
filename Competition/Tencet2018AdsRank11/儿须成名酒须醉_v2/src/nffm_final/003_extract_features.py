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
    train_df = pd.read_csv('./data/train_cv_cvr_ratio.csv')
    test_df = pd.read_csv('./data/test2_cv_cvr_ratio.csv')

    train_df, dev_df,_,_ = train_test_split(train_df,train_df,test_size=0.02, random_state=2018)

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
       'aid_age', 'aid_gender', 'aid_LBS',
       # CV_cvr特征
       'cvr_of_creativeId_and_onehot2', 'cvr_of_creativeId_and_onehot9',
       'cvr_of_creativeId_and_onehot16','cvr_of_consumptionAbility_and_onehot1',
       'cvr_of_creativeId_and_onehot10', 'cvr_of_creativeId_and_onehot15',
       'cvr_of_creativeId_and_onehot13', 'cvr_of_creativeId_and_onehot18',
       'cvr_of_age_and_onehot10', 'cvr_of_creativeId_and_onehot14',
       # cvr特征
       'cvr_of_aid_and_age', 'cvr_of_aid_and_gender', 
       'cvr_of_uid', 'cvr_of_aid_and_consumptionAbility',
       'cvr_of_aid_and_os', 'cvr_of_creativeSize_and_LBS', 
       'cvr_of_aid_and_education', 'cvr_of_uid_and_creativeSize', 
       'cvr_of_creativeSize', 'cvr_of_uid_and_adCategoryId',
       # ratio特征
       'ratio_click_of_aid_in_uid', 'ratio_click_of_creativeSize_in_uid',
       'ratio_click_of_age_in_aid', 'ratio_click_of_age_in_creativeSize',
       'ratio_click_of_gender_in_advertiserId',
       'ratio_click_of_gender_in_creativeSize',
       'ratio_click_of_consumptionAbility_in_aid',
       'ratio_click_of_age_in_advertiserId',
       'ratio_click_of_productType_in_uid',
       'ratio_click_of_productType_in_consumptionAbility'
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
