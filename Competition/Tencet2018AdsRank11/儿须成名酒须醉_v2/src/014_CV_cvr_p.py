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
data = pd.read_csv('data_preprocessing/train_test_merge_p.csv')
print('Index...')
train_part_index = list(data[(data['label']!=-1)&(data['n_parts']!=1)].index)
evals_index = list(data[(data['label']!=-1)&(data['n_parts']==1)].index)
test2_index = list(data[data['n_parts']==7].index)

print('CountVector...')
vector_feature = ['ct','marriageStatus','interest1', 'interest2', 'interest3', 'interest4','interest5']
col_select = vector_feature.copy()
col_select.extend(['aid', 'advertiserId', 'campaignId', 'creativeId',
       'creativeSize', 'adCategoryId', 'productId', 'productType', 'age',
       'gender','education', 'consumptionAbility', 'LBS',
       'os', 'carrier', 'house','n_parts','label'])
data = data[col_select]
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
for feature in vector_feature[1:]:
    s = time.time()
    arr = cntv.fit_transform(data[feature])
    df = sparse.hstack((df,arr)).tocsr()
    arr = []
    del data[feature]
    print(feature,int(time.time()-s),'s')

sparse.save_npz("data_preprocessing/cv_onehot_1.npz",df)
sparse.save_npz("data_preprocessing/cv_onehot_2.npz",arr)


from lightgbm import LGBMClassifier
import time
from scipy import sparse
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import random
import warnings
df = sparse.hstack((sparse.load_npz("data_preprocessing/cv_onehot_1.npz"),sparse.load_npz("data_preprocessing/cv_onehot_2.npz")))
warnings.filterwarnings("ignore")

train_part_y = pd.read_csv('data_preprocessing/train_part_y_p.csv',header=None)
evals_y = pd.read_csv('data_preprocessing/evals_y_p.csv',header=None)
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
##读取数据
print("Reading...")
data = pd.read_csv('data_preprocessing/train_test_merge_p.csv')
print('Index...')
train_part_index = list(data[(data['label']!=-1)&(data['n_parts']!=1)].index)
evals_index = list(data[(data['label']!=-1)&(data['n_parts']==1)].index)
test2_index = list(data[data['n_parts']==7].index)

print('Selecting...')
col_select = []
col_select.extend(['aid', 'advertiserId', 'campaignId', 'creativeId',
       'creativeSize', 'adCategoryId', 'productId', 'productType', 'age',
       'gender','education', 'consumptionAbility', 'LBS',
       'os', 'carrier', 'house','n_parts','label'])
data = data[col_select]

df = df.tocsr()

print('Fiting...')
clf.fit(df[evals_index,:].astype(float),evals_y)
train_part_y = []
evals_y = []

se = pd.Series(clf.feature_importances_).sort_values(ascending=False)
col_sort = list(se.index[:10])
pd.Series(col_sort).to_csv('data_preprocessing/col_sort_cv.csv',index=False)
print('Feature choosing...')
cols = []
for i in range(1,11):
    cols.append('onehot'+str(i))
df = pd.DataFrame(df.tocsc()[:,col_sort].toarray(),columns=cols)
print('Concating...')
data = pd.concat([data,df],axis=1)
print('Saving...')
df.to_csv('data_preprocessing/cv_onehot_p.csv',index=False)
df = []

col_type =['aid', 'advertiserId', 'campaignId', 'creativeId',
       'creativeSize', 'adCategoryId', 'productId', 'productType', 'age',
       'gender','education', 'consumptionAbility', 'LBS',
       'os', 'carrier', 'house']

df_feature = pd.DataFrame()
num = 0
n_parts = [-1]
n_parts.extend(list(range(1,8)))

print('cvr stat...')
col_type.extend(cols)
n = len(col_type)
for i in range(n):
    for j in range(n-i-1):
        if i<=15 and i+j+1 >15:
                num+=1
                s = time.time()
                col_name = 'cvr_of_'+col_type[i]+"_and_"+col_type[i+j+1]
                print(num,col_name)
                se = pd.Series()
                for k in n_parts:
                    if k==1:
                        stat = data[(data['n_parts']!=k)&(data['n_parts']<=5)].groupby([col_type[i],col_type[i+j+1]])['label'].mean()
                        dt = data[data['n_parts']==k][[col_type[i],col_type[i+j+1]]]
                        dt.insert(0,'index',list(dt.index))
                        dt = pd.merge(dt,stat.reset_index(),how='left',on=[col_type[i],col_type[i+j+1]])
                        se = se.append(pd.Series(dt['label'].values,index=list(dt['index'].values)))

                    elif k<=5 and k!=1:
                        stat = data[(data['n_parts']!=k)&(data['n_parts']<=5)&(data['n_parts']!=1)].groupby([col_type[i],col_type[i+j+1]])['label'].mean()
                        dt = data[data['n_parts']==k][[col_type[i],col_type[i+j+1]]]
                        dt.insert(0,'index',list(dt.index))
                        dt = pd.merge(dt,stat.reset_index(),how='left',on=[col_type[i],col_type[i+j+1]])
                        se = se.append(pd.Series(dt['label'].values,index=list(dt['index'].values)))
                    elif k>=5:
                        stat = data[data['n_parts']<=5].groupby([col_type[i],col_type[i+j+1]])['label'].mean()
                        dt = data[data['n_parts']==k][[col_type[i],col_type[i+j+1]]]
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
                    s = time.time()
                    print('train_part...')
                    df_feature.loc[train_part_index].to_csv('data_preprocessing/train_part_x_CV_cvr_'+str(k)+'.csv',index=False)
                    print('evals...')
                    df_feature.loc[evals_index].to_csv('data_preprocessing/evals_x_CV_cvr_'+str(k)+'.csv',index=False)
                    print('test2...')
                    df_feature.loc[test2_index].to_csv('data_preprocessing/test2_x_CV_cvr_'+str(k)+'.csv',index=False)
                    df_feature = pd.DataFrame()
                    print('CV_cvr is Over')
                    print(int(time.time()-s),'s')
    print(col_type[i])