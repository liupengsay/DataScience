##由于机器内存问题，编码时分批进行处理
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
##划分训练与测试集
print('Index...')
train_part_index = list(data[(data['label']!=-1)&(data['n_parts']!=1)].index)
evals_index = list(data[(data['label']!=-1)&(data['n_parts']==1)].index)
test1_index = list(data[data['n_parts']==6].index)
test2_index = list(data[data['n_parts']==7].index)
train_part_y = data['label'].loc[train_part_index]
evals_y = data['label'].loc[evals_index]
print('Done')

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
print('Done')



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
print('Done')

print('Sparse...')
col  = ['ct0','ct1','ct2','ct3','ct4']
train_part_x=df.loc[train_part_index][col]
evals_x=df.loc[evals_index][col]
test1_x=df.loc[test1_index][col]
test2_x=df.loc[test2_index][col]
df = []
print('OneHoting1...')
enc = OneHotEncoder()
one_hot_feature1 = ['aid', 'advertiserId', 'campaignId', 'creativeId',
       'creativeSize']
one_hot_feature2 = ['adCategoryId', 'productId', 'productType', 'age',
       'gender','education']
one_hot_feature3 = ['consumptionAbility', 'LBS',
       'os', 'carrier', 'house']
for feature in one_hot_feature1:
    s = time.time()
    enc.fit(data[feature].values.reshape(-1, 1))
    arr = enc.transform(data.loc[train_part_index][feature].values.reshape(-1, 1))
    train_part_x = sparse.hstack((train_part_x,arr))
    
    arr = enc.transform(data.loc[evals_index][feature].values.reshape(-1, 1))
    evals_x = sparse.hstack((evals_x,arr))
    
    arr = enc.transform(data.loc[test1_index][feature].values.reshape(-1, 1))
    test1_x = sparse.hstack((test1_x,arr))
    
    arr = enc.transform(data.loc[test2_index][feature].values.reshape(-1, 1))
    test2_x = sparse.hstack((test2_x,arr))
    
    arr= []
    del data[feature]
    print(feature,int(time.time()-s),"s")
print("Saving...")
print('train_part_x...')
sparse.save_npz("data_preprocessing/train_part_x_sparse_one_1.npz",train_part_x)
print('evals_x...')
sparse.save_npz("data_preprocessing/evals_x_sparse_one_1.npz",evals_x)
print('test1_x...')
sparse.save_npz("data_preprocessing/test1_x_sparse_one_1.npz",test1_x)
print('test2_x...')
sparse.save_npz("data_preprocessing/test2_x_sparse_one_1.npz",test2_x)
print('Done')

print('OneHoting2...')
train_part_x =  pd.DataFrame()
evals_x = pd.DataFrame()
test1_x = pd.DataFrame()
test2_x = pd.DataFrame()
for feature in one_hot_feature2:
    s = time.time()
    enc.fit(data[feature].values.reshape(-1, 1))
    arr = enc.transform(data.loc[train_part_index][feature].values.reshape(-1, 1))
    train_part_x = sparse.hstack((train_part_x,arr))
    
    arr = enc.transform(data.loc[evals_index][feature].values.reshape(-1, 1))
    evals_x = sparse.hstack((evals_x,arr))
    
    arr = enc.transform(data.loc[test1_index][feature].values.reshape(-1, 1))
    test1_x = sparse.hstack((test1_x,arr))
    
    arr = enc.transform(data.loc[test2_index][feature].values.reshape(-1, 1))
    test2_x = sparse.hstack((test2_x,arr))
    
    arr= []
    del data[feature]
    print(feature,int(time.time()-s),"s")
print("Saving...")
print('train_part_x...')
sparse.save_npz("data_preprocessing/train_part_x_sparse_one_2.npz",train_part_x)
print('evals_x...')
sparse.save_npz("data_preprocessing/evals_x_sparse_one_2.npz",evals_x)
print('test1_x...')
sparse.save_npz("data_preprocessing/test1_x_sparse_one_2.npz",test1_x)
print('test2_x...')
sparse.save_npz("data_preprocessing/test2_x_sparse_one_2.npz",test2_x)
print('Done')

print('Sparse...')
train_part_x =  pd.DataFrame()
evals_x = pd.DataFrame()
test1_x = pd.DataFrame()
test2_x = pd.DataFrame()
for feature in one_hot_feature3:
    s = time.time()
    enc.fit(data[feature].values.reshape(-1, 1))
    arr = enc.transform(data.loc[train_part_index][feature].values.reshape(-1, 1))
    train_part_x = sparse.hstack((train_part_x,arr))
    
    arr = enc.transform(data.loc[evals_index][feature].values.reshape(-1, 1))
    evals_x = sparse.hstack((evals_x,arr))
    
    arr = enc.transform(data.loc[test1_index][feature].values.reshape(-1, 1))
    test1_x = sparse.hstack((test1_x,arr))
    
    arr = enc.transform(data.loc[test2_index][feature].values.reshape(-1, 1))
    test2_x = sparse.hstack((test2_x,arr))
    
    arr= []
    del data[feature]
    print(feature,int(time.time()-s),"s")
print("Saving...")
print('train_part_x...')
sparse.save_npz("data_preprocessing/train_part_x_sparse_one_3.npz",train_part_x)
print('evals_x...')
sparse.save_npz("data_preprocessing/evals_x_sparse_one_3.npz",evals_x)
print('test1_x...')
sparse.save_npz("data_preprocessing/test1_x_sparse_one_3.npz",test1_x)
print('test2_x...')
sparse.save_npz("data_preprocessing/test2_x_sparse_one_3.npz",test2_x)
print('Done')

print('CountVector1...')
train_part_x =  pd.DataFrame()
evals_x = pd.DataFrame()
test1_x = pd.DataFrame()
test2_x = pd.DataFrame()
vector_feature1 = ['marriageStatus','interest1', 'interest2', 'interest3', 'interest4','interest5']

vector_feature2 =  ['kw1', 'kw2','kw3', 'topic1', 'topic2',  'topic3','appIdAction', 'appIdInstall']
cntv=CountVectorizer()
for feature in vector_feature1[:-1]:
    s = time.time()
    cntv.fit(data[feature])
    
    arr = cntv.transform(data.loc[train_part_index][feature])
    train_part_x = sparse.hstack((train_part_x,arr))
    
    arr = cntv.transform(data.loc[evals_index][feature])
    evals_x = sparse.hstack((evals_x,arr))
    
    arr = cntv.transform(data.loc[test1_index][feature])
    test1_x = sparse.hstack((test1_x,arr))
    
    arr = cntv.transform(data.loc[test2_index][feature])
    test2_x = sparse.hstack((test2_x,arr))
    arr = []
    del data[feature]
    print(feature,int(time.time()-s),'s')



print("Saving...")
print('train_part_x...')
sparse.save_npz("data_preprocessing/train_part_x_sparse_one_4.npz",train_part_x)
print('evals_x...')
sparse.save_npz("data_preprocessing/evals_x_sparse_one_4.npz",evals_x)
print('test1_x...')
sparse.save_npz("data_preprocessing/test1_x_sparse_one_4.npz",test1_x)
print('test2_x...')
sparse.save_npz("data_preprocessing/test2_x_sparse_one_4.npz",test2_x)
print('Done')

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
data = pd.read_csv('train_test_merge.csv')
##划分训练与测试集
data.columns

print('Dropping...')
label_feature= ['label','n_parts','interest5','kw1','kw2','kw3', 'topic1', 'topic2',  'topic3','appIdAction', 'appIdInstall']
data = data[label_feature]
print('Index...')
train_part_index = list(data[(data['label']!=-1)&(data['n_parts']!=1)].index)
evals_index = list(data[(data['label']!=-1)&(data['n_parts']==1)].index)
test1_index = list(data[data['n_parts']==6].index)
test2_index = list(data[data['n_parts']==7].index)
del data['label']
del data['n_parts']



data.loc[test1_index][label_feature[-2:]].isnull().sum()

data.loc[test2_index][label_feature[-2:]].isnull().sum()



print('Cntv...')
s = time.time()
train_part_x =  pd.DataFrame()
evals_x = pd.DataFrame()
test1_x = pd.DataFrame()
test2_x = pd.DataFrame()

feature = 'interest5'

cntv=CountVectorizer()
cntv.fit(data[feature])

arr = cntv.transform(data.loc[train_part_index][feature])
train_part_x = sparse.hstack((train_part_x,arr))

arr = cntv.transform(data.loc[evals_index][feature])
evals_x = sparse.hstack((evals_x,arr))

arr = cntv.transform(data.loc[test1_index][feature])
test1_x = sparse.hstack((test1_x,arr))

arr = cntv.transform(data.loc[test2_index][feature])
test2_x = sparse.hstack((test2_x,arr))
arr = []
del data[feature]
print(feature,int(time.time()-s),'s')
print("Saving...")
print('train_part_x...')
sparse.save_npz("data_preprocessing/train_part_x_sparse_one_5.npz",train_part_x)
print('evals_x...')
sparse.save_npz("data_preprocessing/evals_x_sparse_one_5.npz",evals_x)
print('test1_x...')
sparse.save_npz("data_preprocessing/test1_x_sparse_one_5.npz",test1_x)
print('test2_x...')
sparse.save_npz("data_preprocessing/test2_x_sparse_one_5.npz",test2_x)
print('Done')


print('CountVector1...')
train_part_x =  pd.DataFrame()
evals_x = pd.DataFrame()
test1_x = pd.DataFrame()
test2_x = pd.DataFrame()
num = 0
vector_feature2 =  ['kw1', 'kw2','kw3', 'topic1', 'topic2',  'topic3','appIdAction', 'appIdInstall']
cntv=CountVectorizer()
for feature in vector_feature2:
    print(feature)
    s = time.time()
    cntv.fit(data[feature])

    arr = cntv.transform(data.loc[train_part_index][feature])
    train_part_x = sparse.hstack((train_part_x,arr))

    arr = cntv.transform(data.loc[evals_index][feature])
    evals_x = sparse.hstack((evals_x,arr))

    arr = cntv.transform(data.loc[test1_index][feature])
    test1_x = sparse.hstack((test1_x,arr))

    arr = cntv.transform(data.loc[test2_index][feature])
    test2_x = sparse.hstack((test2_x,arr))
    
    arr = []
    del data[feature]
    print(feature,int(time.time()-s),'s')
    num+=1
    if num%3==0:
        k = int(num/3+5)
        print("Saving...")
        print(k)
        print('train_part_x...',train_part_x.shape)
        sparse.save_npz('data_preprocessing/train_part_x_sparse_one_'+str(k)+'.npz',train_part_x)
        
        print('evals_x...',evals_x.shape)
        sparse.save_npz('data_preprocessing/evals_x_sparse_one_'+str(k)+'.npz',evals_x)
        
        print('test1_x...',test1_x.shape)
        sparse.save_npz('data_preprocessing/test1_x_sparse_one_'+str(k)+'.npz',test1_x)
        
        print('test2_x...',test2_x.shape)
        sparse.save_npz('data_preprocessing/test2_x_sparse_one_'+str(k)+'.npz',test2_x)
        
        print('Over')
        train_part_x=pd.DataFrame()
        evals_x=pd.DataFrame()
        test1_x=pd.DataFrame()
        test2_x=pd.DataFrame()
print("Saving...")
print(8)
print('train_part_x...',train_part_x.shape)
sparse.save_npz('data_preprocessing/train_part_x_sparse_one_'+str(8)+'.npz',train_part_x)

print('evals_x...',evals_x.shape)
sparse.save_npz('data_preprocessing/evals_x_sparse_one_'+str(8)+'.npz',evals_x)

print('test1_x...',test1_x.shape)
sparse.save_npz('data_preprocessing/test1_x_sparse_one_'+str(8)+'.npz',test1_x)

print('test2_x...',test2_x.shape)
sparse.save_npz('data_preprocessing/test2_x_sparse_one_'+str(8)+'.npz',test2_x)
print('Over')
