
# coding: utf-8

# In[ ]:


import pandas as pd
import datetime
import gc
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import lightgbm as lgb
label = 'click'
train = pd.read_csv('data/round2_iflyad_train.txt',sep='\t',encoding='utf-8')
train = pd.concat([train,pd.read_csv('data/round1_iflyad_train.txt',sep='\t',encoding='utf-8')],axis=0,ignore_index=True)
train = train.drop_duplicates().reset_index()
del train['index']
test = pd.read_csv('data/round2_iflyad_test_feature.txt',sep='\t',encoding='utf-8')
res = test[['instance_id']]
test[label] = -1
data = pd.concat([train,test],axis=0,ignore_index=True)
del data['instance_id']
del train
del test
def getTimeList(arr):
    day = []
    hour = []
    for a in arr:
        t = datetime.datetime.fromtimestamp(a+3600*24*5)
        day.append(t.day)
        hour.append(t.hour)
    return day,hour
#######################
day,hour = getTimeList(data['time'].values)
data['day'] = day
data['hour'] = hour
data['hour_part'] =  (data['hour']//6).astype(int)
del data['time']
#######################
lst1 = []
lst2 = []
for va in data['advert_industry_inner'].values:
    lst1.append(va.split('_')[0])
    lst2.append(va.split('_')[1])
data['advert_industry_inner1'] = lst1
data['advert_industry_inner2'] = lst2
#########################
lst  = []
for va in data['inner_slot_id'].values:
    lst.append(va.split('_')[0])
data['inner_slot_id1'] = lst
#######################
del data['creative_is_js']
del data['creative_is_voicead']
del data['app_paid']
#######################
import numpy as np
lst = []
for va in data['make'].values:
    va = str(va)
    if ',' in va:
        lst.append(va.split(',')[0].lower())
    elif ' ' in va:
        lst.append(va.split(' ')[0].lower())
    elif '-' in va:
        lst.append(va.split('-')[0].lower())
    else:
        lst.append(va.lower())
for i in range(len(lst)):
    if 'iphone' in lst[i]:
        lst[i] = 'apple'
    elif 'redmi' in lst[i]:
        lst[i] = 'xiaomi'
    elif lst[i]=='mi':
        lst[i] = 'xiaomi'
    elif lst[i]=='meitu':
        lst[i] = 'meizu'
    elif lst[i]=='nan':
        lst[i] = np.nan
    elif lst[i]=='honor':
        lst[i] = 'huawei'
    elif lst[i]=='le' or lst[i]=='letv' or lst[i]=='lemobile' or lst[i]=='lephone' or lst[i]=='blephone':
        lst[i] = 'leshi'
data['make_new'] = lst
#######################
lst = []
for va in data['model'].values:
    va = str(va)
    if ',' in va:
        lst.append(va.replace(',',' '))
    elif '+' in va:
        lst.append(va.replace('+',' '))
    elif '-' in va:
        lst.append(va.replace('-',' '))
    elif 'nan'==va:
        lst.append(np.nan)
    else:
        lst.append(va)
data['model_new'] = lst
######################
data['os'].replace(0,2,inplace=True)
lst = []
for va in data['osv'].values:
    va = str(va)
    va = va.replace('iOS','')
    va = va.replace('android','')
    va = va.replace(' ','')
    va = va.replace('iPhoneOS','')
    va = va.replace('_','.')
    va = va.replace('Android5.1','.')   
    try:
        int(va)
        lst.append(np.nan)
    except:
        sp = ['nan','11.39999961853027','10.30000019073486','unknown','11.30000019073486']
        if va in sp:
            lst.append(np.nan)
        elif va=='3.0.4-RS-20160720.1914':
            lst.append('3.0.4')
        else:
            lst.append(va)
temp = pd.Series(lst).value_counts()
temp = temp[temp<=2].index.tolist()
for i in range(len(lst)):
    if lst[i] in temp:
        lst[i] = np.nan
data['osv'] = lst
lst1 = []
lst2 = []
lst3 = []
for va in data['osv'].values:
    va = str(va).split('.')
    if len(va)<3:
        va.extend(['0','0','0'])
    lst1.append(va[0])
    lst2.append(va[1])
    lst3.append(va[2])
data['osv1'] = lst1
data['osv2'] = lst2
data['osv3'] = lst3
def getLst(co1,co2):
    arr2 = data[co2].astype(str).fillna(value='-1').values
    arr1 = data[co1].astype(str).values
    lst = []
    for i in range(len(arr1)):
        lst.append(arr1[i]+'_'+arr2[i])
    return lst
data['os_name'] = getLst('os','osv')
data['os_name1'] = getLst('os','osv1')
#######################
bool_feature = ['creative_is_jump', 'creative_is_download', 'creative_has_deeplink']
for feature in bool_feature:
    data[feature] = data[feature].astype(int)

#######################
se = pd.Series(data['user_tags'].drop_duplicates().values)
data['user_id'] = data['user_tags'].map(pd.Series(se.index,index=se.values))
data['user_tags'].fillna(value='-1',inplace=True)

lst = []
for va in data['user_tags'].values:
    va = va.replace(',',' ')
    lst.append(va)
data['user_tags'] = lst


#######################
#######################
type_feature =  ['city', 'province','carrier', 'devtype', 'make', 'nnt','osv',
                 'model','make_new','model_new',
       'os', 'os_name', 'os_name1', 'user_id','inner_slot_id1',
                 'adid', 'advert_id', 'orderid', 'campaign_id','advert_industry_inner',
       'creative_id', 'creative_tp_dnf', 'app_cate_id', 'f_channel', 'app_id',
       'inner_slot_id', 'creative_type', 
       'advert_name', 'hour', 'advert_industry_inner1',
       'advert_industry_inner2']

for feature in type_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].fillna(-1).apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature].fillna('-1'))
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5,random_state=2020,shuffle=True)
cols =  ['city', 'province', 'carrier', 'devtype', 'make', 'model', 'nnt', 'os',
       'osv', 'os_name', 'adid', 'advert_id', 'orderid',
       'advert_industry_inner', 'campaign_id', 'creative_id',
       'creative_tp_dnf', 'app_cate_id', 'app_id', 'inner_slot_id',
       'creative_type', 'creative_width', 'creative_height',
       'creative_is_jump', 'creative_is_download', 'creative_has_deeplink',
       'advert_name', 'day', 'hour']
train_index = data[data['click']!=-1].index.tolist()
test_index = data[data['click']==-1].index.tolist()
train_y = pd.Series(data['click'].loc[train_index].values)
NBR = 2500
VBE = 50
ESR = 200

import lightgbm as lgb
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
arr = sparse.hstack((pd.DataFrame(),cv.fit_transform(data['user_tags']))).tocsr()

params_initial = {
'objective':'binary'
}
train_x = arr[train_index,:]
test_x = arr[test_index,:]
num = 1
train_user = pd.Series()
test_user = pd.Series(0,index=list(range(test_x.shape[0])))
fscore_se = pd.Series(0,index=list(range(arr.shape[1])))
for train_part_index,evals_index in skf.split(train_x,train_y):
    EVAL_RESULT = {}
    train_part = lgb.Dataset(train_x[train_part_index,:],label=train_y.loc[train_part_index])
    evals = lgb.Dataset(train_x[evals_index,:],label=train_y.loc[evals_index])
    bst = lgb.train(params_initial,train_part, 
          num_boost_round=NBR, valid_sets=[train_part,evals], 
          valid_names=['train','evals'],early_stopping_rounds=ESR,
          evals_result=EVAL_RESULT, verbose_eval=VBE)
    fscore_se = fscore_se+pd.Series(bst.feature_importance())
    train_user = train_user.append(pd.Series(bst.predict(train_x[evals_index,:]),index=evals_index))
    test_user = test_user+pd.Series(bst.predict(test_x))
    print(num)
    num+=1

fscore_se = fscore_se.sort_values(ascending=False)
###########################################
col_cvr = cols[:]
train_x = data[col_cvr].loc[train_index].reset_index()
test_x = data[col_cvr].loc[test_index].reset_index()
train_x['user_stack'] = train_user.sort_index().values
test_x['user_stack'] = (test_user.values)/5
del train_x['index']
del test_x['index']
##########################################

def getStackFeature(df_,seed_):
    skf = StratifiedKFold(n_splits=5,random_state=seed_,shuffle=True)
    train = df_.loc[train_index]
    test = df_.loc[test_index]
    train_user = pd.Series()
    test_user = pd.Series(0,index=list(range(test_x.shape[0])))
    for train_part_index,evals_index in skf.split(train,train_y):
        EVAL_RESULT = {}
        train_part = lgb.Dataset(train.loc[train_part_index],label=train_y.loc[train_part_index])
        evals = lgb.Dataset(train.loc[evals_index],label=train_y.loc[evals_index])
        bst = lgb.train(params_initial,train_part, 
              num_boost_round=NBR, valid_sets=[train_part,evals], 
              valid_names=['train','evals'],early_stopping_rounds=ESR,
              evals_result=EVAL_RESULT, verbose_eval=VBE)
        train_user = train_user.append(pd.Series(bst.predict(train.loc[evals_index]),index=evals_index))
        test_user = test_user+pd.Series(bst.predict(test))
    return train_user,test_user

###########################################
data['cnt']=1
col_ratio = []
num = 0
s = time.time()
df = pd.DataFrame()
print('Ratio clcik...')
col_type = type_feature.copy()
n = len(col_type)
for i in range(n):
    col_name = "ratio_click_of_"+col_type[i]
    s = time.time()
    df[col_name] =(data[col_type[i]].map(data[col_type[i]].value_counts())/len(data)*100).astype(int)
    num+=1
    print(num,col_name,int(time.time()-s),'s')
    col_ratio.append(col_name)
stack = 1
n = len(col_type)
for i in range(n):
    for j in range(n):
        if i!=j:
            col_name = "ratio_click_of_"+col_type[j]+"_in_"+col_type[i]       
            se = data.groupby([col_type[i],col_type[j]])['cnt'].sum()
            dt = data[[col_type[i],col_type[j]]]
            cnt = data[col_type[i]].map(data[col_type[i]].value_counts())
            df[col_name] = ((pd.merge(dt,se.reset_index(),how='left',on=[col_type[i],col_type[j]]).sort_index()['cnt'].fillna(value=0)/cnt)*100).astype(int).values
            num+=1
            col_ratio.append(col_name)
            if len(df.columns)==200:
                print(num,col_name,int(time.time()-s),'s')
                train_user,test_user = getStackFeature(df,stack)
                train_x['ratio_'+str(stack)] = train_user.sort_index().values
                test_x['ratio_'+str(stack)] = (test_user.values)/5
                stack+=1
                df = pd.DataFrame()
                s = time.time()
                gc.collect()
###################################
k = 100
num = 1
col_select = fscore_se.index.tolist()
df = pd.DataFrame()
for ind in col_select[:100]: 
    data['temp'] = arr[:,ind].toarray()[:,0]
    for co in type_feature:
        col_name = co+'_user_tags_'+str(ind)+'_ratio'
        se = data.groupby([co])['temp'].mean()
        df[col_name] = ((data[co].map(se))*10000).astype(int)
        if len(df.columns)==200:
            print(num,col_name,int(time.time()-s),'s')
            print('\n')
            print('\n')
            train_user,test_user = getStackFeature(df,stack)
            train_x['cvr_'+str(stack)] = train_user.sort_index().values
            test_x['cvr_'+str(stack)] = (test_user.values)/5
            stack+=1
            df = pd.DataFrame()
            s = time.time()
            gc.collect()
    num+=1
###################################

for i in range(n):
    col_name = "cnt_click_of_"+col_type[i]
    s = time.time()
    se = (data[col_type[i]].map(data[col_type[i]].value_counts())).astype(int)
    semax = se.max()
    semin = se.min()
    df[col_name] = ((se-se.min())/(se.max()-se.min())*100).astype(int).values
    num+=1
    
print('Begin stat...')##再加入离散特征
n = len(col_type)
for i in range(n):
    for j in range(n-i-1):
        col_name = "cnt_click_of_"+col_type[i+j+1]+"_and_"+col_type[i]
        s = time.time()
        se = data.groupby([col_type[i],col_type[i+j+1]])['cnt'].sum()
        dt = data[[col_type[i],col_type[i+j+1]]]
        se = (pd.merge(dt,se.reset_index(),how='left',
                        on=[col_type[i],col_type[j+i+1]]).sort_index()['cnt'].fillna(value=0)).astype(int)
        semax = se.max()
        semin = se.min()
        df[col_name] = ((se-se.min())/(se.max()-se.min())*100).fillna(value=0).astype(int).values
        num+=1
        if len(df.columns)>=200:
            print(num,col_name,int(time.time()-s),'s')
            print('\n')
            print('\n')
            train_user,test_user = getStackFeature(df,stack)
            train_x['click_'+str(stack)] = train_user.sort_index().values
            test_x['click_'+str(stack)] = (test_user.values)/5
            stack+=1
            df = pd.DataFrame()
            s = time.time()
            gc.collect()
            print(num,col_name,int(time.time()-s),'s')
##################################
n = len(col_type)
num = 0
for i in range(n):
    for j in range(n):
        if i!=j:
            s = time.time()
            col_name = "count_type_"+col_type[j]+"_in_"+col_type[i]
            se = data.groupby([col_type[i]])[col_type[j]].value_counts()
            se = pd.Series(1,index=se.index).sum(level=col_type[i])
            df[col_name] = (data[col_type[i]].map(se)).fillna(value=0).astype(int).values
            num+=1
            if len(df.columns)>=200:
                print(num,col_name,int(time.time()-s),'s')
                print('\n')
                print('\n')
                train_user,test_user = getStackFeature(df,stack)
                train_x['nunique_'+str(stack)] = train_user.sort_index().values
                test_x['nunique_'+str(stack)] = (test_user.values)/5
                stack+=1
                df = pd.DataFrame()
                s = time.time()
                gc.collect()
                print(num,col_name,int(time.time()-s),'s')
                
k = 100
col_select = fscore_se[:k].index.tolist()
train_x = sparse.hstack((train_x,arr[train_index,:].tocsc()[:,col_select])).tocsr()
test_x = sparse.hstack((test_x,arr[test_index,:].tocsc()[:,col_select])).tocsr()
print(train_x.shape)
print(test_x.shape)





params_initial = {
   'num_leaves':48,
    'objective':'binary',
    'learning_rate':0.035,
   'max_bin':425,
   'min_data_in_leaf':10,

    'feature_fraction':0.5,

   'bagging_freq':1,
   'bagging_seed':0,
    'bagging_fraction':0.9,

   'reg_alpha':3,
   'reg_lambda':5
}




score = []
res['predicted_score'] = 0
for train_part_index,evals_index in skf.split(train_x,train_y):
    EVAL_RESULT = {}
    train_part = lgb.Dataset(train_x[train_part_index,:],label=train_y.loc[train_part_index])
    evals = lgb.Dataset(train_x[evals_index,:],label=train_y.loc[evals_index])
    bst = lgb.train(params_initial,train_part, 
          num_boost_round=NBR, valid_sets=[train_part,evals], 
          valid_names=['train','evals'],early_stopping_rounds=ESR,
          evals_result=EVAL_RESULT, verbose_eval=VBE)
    lst = EVAL_RESULT['evals']['binary_logloss']
    best_score = min(lst)
    print(best_score)
    score.append(best_score)
    best_iter = lst.index(best_score)+1
    print(best_iter)
    res['predicted_score'] =  res['predicted_score'].values+bst.predict(test_x,num_iteration = best_iter)
res['predicted_score'] = res['predicted_score']/5
filename = 'liupeng_result.csv'
res.to_csv('result/'+filename,index=False)
print(filename)
print(score)
print(sum(score)/5)
print(res.head(10))
print(res['predicted_score'].describe())
timeArray = time.localtime()
time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
print('\n')

