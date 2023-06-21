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
import warnings
warnings.filterwarnings("ignore")
##读取数据
print("Reading...")
data = pd.read_csv('data_preprocessing/train_test_merge.csv')
print('Index...')
train_part_index = list(data[(data['label']!=-1)&(data['n_parts']!=1)].index)
evals_index = list(data[(data['label']!=-1)&(data['n_parts']==1)].index)
test1_index = list(data[data['n_parts']==6].index)
test2_index = list(data[data['n_parts']==7].index)
del data['uid']
##labelencoder
print('LabelEncoder...')
label_feature=['aid', 'advertiserId', 'campaignId', 'creativeId',
       'creativeSize', 'adCategoryId', 'productId', 'productType', 'age',
       'gender','education', 'consumptionAbility', 'LBS',
       'os', 'carrier', 'house']

for feature in label_feature:
    s = time.time()
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])
    print(feature,int(time.time()-s),'s')
print('CountVector...')
vector_feature = ['ct','marriageStatus','interest1', 'interest2', 'interest3', 'interest4','interest5', 
                  'kw1', 'kw2','kw3', 'topic1', 'topic2',  'topic3','appIdAction', 'appIdInstall']
##ct特殊处理       
print('Ct...')
value = []
ct_ = ['0','1','2','3','4']
ct_all = list(data['ct'].values)
for i in range(len(data)):
    ct = ct_all[i]
    va = []
    for j in range(5):
        if ct_[j] in ct:
            va.append(1)
        else:va.append(0)
    value.append(va)
df = pd.DataFrame(value,columns=['ct0','ct1','ct2','ct3','ct4'])
del data['ct']

cntv=CountVectorizer()
for feature in vector_feature[1:7]:
    s = time.time()
    arr = cntv.fit_transform(data[feature])
    df = sparse.hstack((df,arr)).tocsr()
    arr = []
    print(feature,int(time.time()-s),'s')
    del data[feature]
for feature in vector_feature[7:]:
    s = time.time()
    del data[feature]
    print(feature,int(time.time()-s),'s')

train_part_y = pd.read_csv('data_preprocessing/train_part_y.csv',header=None)
evals_y = pd.read_csv('data_preprocessing/evals_y.csv',header=None)

df.astype(float)

from lightgbm import LGBMClassifier
import time
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")
clf = LGBMClassifier(boosting_type='gbdt',
                     num_leaves=31, max_depth=-1, 
                     learning_rate=0.1, n_estimators=100, 
                     subsample_for_bin=200000, objective=None,
                     class_weight=None, min_split_gain=0.0, 
                     min_child_weight=0.001,
                     min_child_samples=20, subsample=1.0, subsample_freq=1,
                     colsample_bytree=1.0,
                     reg_alpha=0.0, reg_lambda=0.0, random_state=None,
                     n_jobs=-1, silent=True)
print('Fiting...')
clf.fit(df[train_part_index,:].astype(float),train_part_y)
train_part_y = []
evals_y = []

se = pd.Series(clf.feature_importances_).sort_values(ascending=False)
col_sort = list(se.index[:20])
print('Feature choosing...')
cols = []
for i in range(1,21):
    cols.append('onehot'+str(i))
df = pd.DataFrame(df.tocsc()[:,col_sort].toarray(),columns=cols)
data = pd.concat([data,df],axis=1)
df.to_csv('data_preprocessing/cv_onehot.csv',index=False)
df = []
print('Over...')

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
import warnings
warnings.filterwarnings("ignore")
##读取数据
print("Reading...")
data = pd.read_csv('data_preprocessing/train_test_merge.csv')
print('Index...')
train_part_index = list(data[(data['label']!=-1)&(data['n_parts']!=1)].index)
evals_index = list(data[(data['label']!=-1)&(data['n_parts']==1)].index)
test1_index = list(data[data['n_parts']==6].index)
test2_index = list(data[data['n_parts']==7].index)
del data['uid']
##labelencoder
print('LabelEncoder...')
label_feature=['aid', 'advertiserId', 'campaignId', 'creativeId',
       'creativeSize', 'adCategoryId', 'productId', 'productType', 'age',
       'gender','education', 'consumptionAbility', 'LBS',
       'os', 'carrier', 'house']

for feature in label_feature[:1]:
    s = time.time()
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])
    print(feature,int(time.time()-s),'s')
print('CountVector...')
vector_feature = ['ct','marriageStatus','interest1', 'interest2', 'interest3', 'interest4','interest5', 
                  'kw1', 'kw2','kw3', 'topic1', 'topic2',  'topic3','appIdAction', 'appIdInstall']
##ct特殊处理       
for feature in vector_feature:
    s = time.time()
    del data[feature]
    print(feature,int(time.time()-s),'s')
df = pd.read_csv('data_preprocessing/cv_onehot.csv')
data = pd.concat([data,df],axis=1)
df = []

df_feature = pd.DataFrame()
num = 0
n_parts= 7
col_type =['aid', 'advertiserId', 'campaignId', 'creativeId',
       'creativeSize', 'adCategoryId', 'productId', 'productType', 'age',
       'gender','education', 'consumptionAbility', 'LBS',
       'os', 'carrier', 'house']
cols = []
for i in range(1,21):
    cols.append('onehot'+str(i))
print('cvr stat...')
col_type.extend(cols)
n = len(col_type)
for i in range(n):
    for j in range(n-i-1):
        if i<=15 and i+j+1 >15 and i>=1:
                num+=1
                s = time.time()
                col_name = 'cvr_of_'+col_type[i]+"_and_"+col_type[i+j+1]
                print(num,col_name)
                se = pd.Series()
                for k in range(n_parts):
                    if k==0:
                        stat = data[(data['n_parts']!=k+1)&(data['n_parts']<=5)].groupby([col_type[i],col_type[i+j+1]])['label'].mean()
                        dt = data[data['n_parts']==k+1][[col_type[i],col_type[i+j+1]]]
                        dt.insert(0,'index',list(dt.index))
                        dt = pd.merge(dt,stat.reset_index(),how='left',on=[col_type[i],col_type[i+j+1]])
                        se = se.append(pd.Series(dt['label'].values,index=list(dt['index'].values)))

                    elif 1<=k and k<=4:
                        stat = data[(data['n_parts']!=k+1)&(data['n_parts']<=5)&(data['n_parts']>=2)].groupby([col_type[i],col_type[i+j+1]])['label'].mean()
                        dt = data[data['n_parts']==k+1][[col_type[i],col_type[i+j+1]]]
                        dt.insert(0,'index',list(dt.index))
                        dt = pd.merge(dt,stat.reset_index(),how='left',on=[col_type[i],col_type[i+j+1]])
                        se = se.append(pd.Series(dt['label'].values,index=list(dt['index'].values)))
                    elif k>=5:
                        stat = data[data['n_parts']<=5].groupby([col_type[i],col_type[i+j+1]])['label'].mean()
                        dt = data[data['n_parts']==k+1][[col_type[i],col_type[i+j+1]]]
                        dt.insert(0,'index',list(dt.index))
                        dt = pd.merge(dt,stat.reset_index(),how='left',on=[col_type[i],col_type[i+j+1]])
                        se = se.append(pd.Series(dt['label'].values,index=list(dt['index'].values)))
                stat = []
                dt = []
                se = pd.Series(data.index).map(se)
                se = (se-se.min())/(se.max()-se.min())
                df_feature[col_name] = (se*100).fillna(value=-1).astype(int)
                
                print(num,col_name,int(time.time()-s),'s')
                if num%10==0:
                    k = int(num/10)
                    print(k,'Saving...')
                    print('train_part...')
                    df_feature.loc[train_part_index].to_csv('data_preprocessing/train_part_x_CV_cvr_'+str(k)+'.csv',index=False)
                    print('evals...')
                    df_feature.loc[evals_index].to_csv('data_preprocessing/evals_x_CV_cvr_'+str(k)+'.csv',index=False)
                    print('test1...')
                    df_feature.loc[test1_index].to_csv('data_preprocessing/test1_x_CV_cvr_'+str(k)+'.csv',index=False)
                    print('test2...')
                    df_feature.loc[test2_index].to_csv('data_preprocessing/test2_x_CV_cvr_'+str(k)+'.csv',index=False)
                    df_feature = pd.DataFrame()
                    print('CV_cvr is Over')
    del data[col_type[i]]
    print('del',col_type[i])