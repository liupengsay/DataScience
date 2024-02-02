# coding: utf-8
import pandas as pd

import time
import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import gc
from scipy.stats import scoreatpercentile
from xgboost import XGBClassifier
from operator import itemgetter

from tqdm import tqdm
from sklearn import metrics


from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm
PROCESSED_FILE = False # 是否使用预处理过用户特征的文件，True为使用预处理过的数据,速度相对较快 
PATH = "./data/"
def getTag(curlist,totalList):
    userfea = np.zeros(len(totalList))
    for item in curlist:
        if item not in totalList:
            continue
        ind = totalList.index(item)
        userfea[ind] = userfea[ind]+1
    return userfea
def dup_remove_set(list_raw):
    result = set()
    for sublist in list_raw:
        item = set(sublist)
        result = result.union(item)
    return list(result)
def order_dict(dicts, n):
    result = []
    result1 = []
    p = sorted([(k, v) for k, v in dicts.items()], reverse=True)
    s = set()
    for i in p:
        s.add(i[1])
    for i in sorted(s, reverse=True)[:n]:
        for j in p:
            if j[1] == i:
                result.append(j)
    for r in result:
        result1.append(r[0])
    return result1
if PROCESSED_FILE == True:
    train_df = pd.read_csv(PATH+'processed_train_fusai2.csv')
    test_df = pd.read_csv(PATH+'processed_test_fusai2.csv')
    
    

if PROCESSED_FILE == False: #对用户特征进行预处理 
    
    train_df1 = pd.read_table(PATH+'round1_iflyad_train.txt')
    train_df2 = pd.read_table(PATH+'round2_iflyad_train.txt')
    test_df = pd.read_table(PATH+'round2_iflyad_test_feature.txt')
    test_df2 = pd.read_table(PATH+'round2_iflyad_test_feature.txt')
    train_df = pd.concat([train_df1,train_df2],ignore_index=True).drop_duplicates()
    print ("train_df_len",len(train_df))
    train_df['f_channel'].head(5)
    train_df.columns
    train_df['user_tags'] = train_df['user_tags'].apply(lambda x: str(x).split(","))
    test_df['user_tags'] = test_df['user_tags'].apply(lambda x: str(x).split(","))
    taglist = list(train_df['user_tags'])
    taglist2 = []
    for f in taglist:
        taglist2.extend(f)
    dict1 = {}
    for key in taglist2:
        dict1[key] = dict1.get(key, 0) + 1
    print (len(dict1))
    #d_tmp = dict((key, value) for key, value in dict1.items() if float(value) >= 10000)
    #print(d_tmp)
    fillter = order_dict(dict1,399)
    train_df['userfea'] = train_df['user_tags'].apply(lambda x: getTag(x,fillter))
    test_df['userfea'] = test_df['user_tags'].apply(lambda x: getTag(x,fillter))
    train_df['userfea2'] = train_df['userfea'].apply(lambda x: " ".join([str(i) for i in x]))
    test_df['userfea2'] = test_df['userfea'].apply(lambda x: " ".join([str(i) for i in x]))
    train_df.to_csv(PATH+"processed_train_fusai2.csv")
    test_df.to_csv(PATH+"processed_test_fusai2.csv")

test_df2 = pd.read_table(PATH+'round2_iflyad_test_feature.txt')
# In[3]:


# In[3]:

train_df['f_channel'].head(5)


# In[4]:

train_df.columns


# In[4]:


# In[5]:

train_df['userfea'] = train_df['userfea2'].apply(lambda x: str(x).split())
test_df['userfea'] = test_df['userfea2'].apply(lambda x: str(x).split())


test_nunique = test_df2.nunique()
test_nunique.sort_values(inplace=True)
one_hot_col = test_nunique[test_nunique<50].index.values.tolist()
one_hot_col = list(set(one_hot_col))
print (train_df['userfea'].head(5))
train_df['userfea'] = train_df['userfea'].apply(lambda x: [np.int(np.float(i)) for i in x])
test_df['userfea'] = test_df['userfea'].apply(lambda x: [np.int (np.float(i)) for i in x])
print (train_df['userfea'].head(5))



train_df['f_channel'].head(5)

train_df.columns




print (train_df['campaign_id'].head(5))




train_df['where'] = train_df['inner_slot_id'].apply(lambda x: x.split('_')[0])
test_df['where'] = test_df['inner_slot_id'].apply(lambda x: x.split('_')[0])

train_df['where2'] = train_df['inner_slot_id'].apply(lambda x: x.split('_')[1])
test_df['where2'] = test_df['inner_slot_id'].apply(lambda x: x.split('_')[1])


train_df['channel'] = train_df['f_channel'].apply(lambda x: str(x).split('_')[0])
test_df['channel'] = test_df['f_channel'].apply(lambda x: str(x).split('_')[0])

train_df['f_channel'] = train_df['f_channel'].fillna("NaN_NaN")
test_df['f_channel'] = test_df['f_channel'].fillna("NaN_NaN")
train_df['channel1'] = train_df['f_channel'].apply(lambda x: str(x).split('_')[1])
test_df['channel1'] = test_df['f_channel'].apply(lambda x: str(x).split('_')[1])



train_df['advert_industry_inner0'] = train_df['advert_industry_inner'].apply(lambda x: str(x).split('_')[0])
test_df['advert_industry_inner0'] = test_df['advert_industry_inner'].apply(lambda x: str(x).split('_')[0])

train_df['advert_industry_inner1'] = train_df['advert_industry_inner'].apply(lambda x: str(x).split('_')[1])
test_df['advert_industry_inner1'] = test_df['advert_industry_inner'].apply(lambda x: str(x).split('_')[1])
# In[14]:
has_string = ['make','model']
for f in has_string:
    train_df[f] = train_df[f].apply(lambda x: str(x).lower())
    test_df[f] = test_df[f].apply(lambda x: str(x).lower())
    

# In[19]:

def unxi_time(t):
    t = time.localtime(t)
    dt = time.strftime("%b-%d-%Y-%H-%M-%S", t)
    return dt

train_df['time2']  = train_df['time'].apply(lambda x: unxi_time(x))
test_df['time2']  = test_df['time'].apply(lambda x: unxi_time(x))


# In[20]:

train_df['time'].head(5)


# In[21]:

train_df["hour"] = train_df["time2"].apply(lambda x: int(str(x).split('-')[3]))
test_df["hour"] = test_df["time2"].apply(lambda x: int(str(x).split('-')[3]))


# In[22]:

train_df["hour"].head(5)



features_to_use=[]
def do_count( df, group_cols, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Aggregating by ", group_cols , '...' )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name]
    features_to_use.append(agg_name)
    gc.collect()
    return( df )
MISSING8 = 255
test_df['click'] = MISSING8

train_df =  pd.concat([train_df,test_df], axis=0)
del test_df

lst = []
for va in train_df['make'].values:
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
train_df['make_new'] = lst

train_df['make'] = train_df['make_new'] 
#######################
lst = []
for va in train_df['model'].values:
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
train_df['model_new'] = lst
train_df['model'] = train_df['model_new']
######################
train_df['os'].replace(0,2,inplace=True)
lst = []
for va in train_df['osv'].values:
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
train_df['osv'] = lst
lst1 = []
lst2 = []
lst3 = []
for va in train_df['osv'].values:
    va = str(va).split('.')
    if len(va)<3:
        va.extend(['0','0','0'])
    lst1.append(va[0])
    lst2.append(va[1])
    lst3.append(va[2])
train_df['osv1'] = lst1
train_df['osv2'] = lst2
train_df['osv3'] = lst3
def getLst(co1,co2):
    arr2 = train_df[co2].astype(str).fillna(value='-1').values
    arr1 = train_df[co1].astype(str).values
    lst = []
    for i in range(len(arr1)):
        lst.append(arr1[i]+'_'+arr2[i])
    return lst
train_df['os_name'] = getLst('os','osv')
train_df['os_name1'] = getLst('os','osv1')



train_df['day'] = train_df['time'].apply(lambda x : int(time.strftime("%d", time.localtime(x))))

train_df['period'] = train_df['day']


train_df['period'] = train_df['day']
train_df['period'][train_df['period']<27] = train_df['period'][train_df['period']<27] + 31

for feat_1 in ['advert_id','advert_industry_inner','advert_name','campaign_id', 'creative_height',
               'creative_tp_dnf', 'creative_width', 'province', 'f_channel']:
    gc.collect()
    res=pd.DataFrame()
    temp=train_df[[feat_1,'period','click']]
    for period in range(train_df['period'].min(),train_df['period'].max()+1):
        if period == train_df['period'].min():
            count=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<=period).values].count()).reset_index(name=feat_1+'_all')
            count1=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<=period).values].sum()).reset_index(name=feat_1+'_1')
        else: 
            count=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<=(period-1)).values].count()).reset_index(name=feat_1+'_all')
            count1=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<=(period-1))].sum()).reset_index(name=feat_1+'_1')
        count[feat_1+'_1']=count1[feat_1+'_1']
        count.fillna(value=0, inplace=True)
        count[feat_1+'_rate'] = count[feat_1+'_1'] / count[feat_1+'_all']*1.0
        count['period']=period
        count.drop([feat_1+'_all', feat_1+'_1'],axis=1,inplace=True)
        count.fillna(value=0, inplace=True)
        res=res.append(count,ignore_index=True)
    features_to_use.append(feat_1+'_rate')
    print(feat_1,' over')
    train_df = pd.merge(train_df,res, how='left', on=[feat_1,'period'])



for_count = ["creative_id","channel1",'inner_slot_id','advert_id','orderid','adid','campaign_id',"hour","nnt","city","province","carrier","devtype","os","app_cate_id","app_id","creative_type","creative_tp_dnf","os_name","where2","osv","where","make","advert_name","model","advert_industry_inner0",'advert_industry_inner1','advert_industry_inner']

first_f = ['app_cate_id','channel','app_id']
for_count2 = ["make","model","osv1","osv2","osv3","adid","advert_name","campaign_id","creative_id","carrier","nnt","devtype","os"]
categorical = ["userfea2","channel1","os_name","where2","osv","where","make","advert_name","model","advert_industry_inner0",'advert_industry_inner1','advert_industry_inner']


categorical.append('osv1')
categorical.append('osv2')
categorical.append('osv3')
categorical.append('os_name1')




train_df['a'] = 1
for f1 in first_f:
    for f2 in for_count2:
        if (f1!=f2 and 'count_'+f1+"_"+f2 not in list(train_df.columns) and 'count_'+f2+"_"+f1 not in list(train_df.columns)):
            train_df = do_count( train_df, [f1,f2], 'count_'+f1+"_"+f2, 'uint32', show_max=True ); gc.collect()
            train_df[f1+"_"+f2] = train_df[f1].apply(lambda x: str(x))+"_"+train_df[f2].apply(lambda x: str(x))
            categorical.append(f1+"_"+f2)
for f in for_count:
    train_df = do_count( train_df, [f], 'count_'+f, 'uint32', show_max=True ); gc.collect()
for feat_1 in for_count:
    gc.collect()
    res=pd.DataFrame()
    temp=train_df[[feat_1,'period','a']]
    for period in range(train_df['period'].min(),train_df['period'].max()+1):
        if period == train_df['period'].min():
            count=temp.groupby([feat_1]).apply(lambda x: x['a'][(x['period']<=period).values].count()).reset_index(name=feat_1+'_all2')
            count1=temp.loc[temp['period']<=period]

        else: 
            count=temp.groupby([feat_1]).apply(lambda x: x['a'][(x['period']==(period-1)).values].count()).reset_index(name=feat_1+'_all2')
            count1=temp.loc[temp['period']==(period-1)]
        count.fillna(value=0, inplace=True)
        count['period']=period
        count[feat_1+"rate2"] = count[feat_1+'_all2']/len(count1)
        count.fillna(value=0, inplace=True)
        res=res.append(count,ignore_index=True)
    features_to_use.append(feat_1+'_all2')
    features_to_use.append(feat_1+"rate2")
    print(feat_1,' over')
    train_df = pd.merge(train_df,res, how='left', on=[feat_1,'period'])
    
train_df = do_count( train_df, ['campaign_id', 'channel1', 'where2'], 'cam_chan__wh2', 'uint16', show_max=True ); gc.collect()
#test_df = do_count( test_df, ['campaign_id', 'channel', 'where2'], 'cam_chan__wh2', 'uint16', show_max=True ); gc.collect()
train_df = do_count( train_df, ['f_channel','app_cate_id'], 'x5', 'uint32', show_max=True ); gc.collect()
train_df = do_count( train_df, ['make', 'carrier'], 'x4', 'uint32', show_max=True ); gc.collect()

#train_df = do_count( train_df, ['app_cate_id'], 'x6', 'uint32', show_max=True ); gc.collect()

train_df = do_count( train_df, ['app_cate_id','nnt'], 'x8', 'uint32', show_max=True ); gc.collect()
train_df = do_count( train_df, ['app_cate_id','creative_id'], 'x9', 'uint32', show_max=True ); gc.collect()
train_df = do_count( train_df, ['os','advert_industry_inner0'], 'x10', 'uint32', show_max=True ); gc.collect()
train_df = do_count( train_df, ['os','advert_name'], 'x11', 'uint32', show_max=True ); gc.collect()
train_df['area'] = train_df['creative_width'] * train_df['creative_height']
#train_df = do_count( train_df, ['advert_id'], 'x33', 'uint32', show_max=True ); gc.collect()

train_df = do_count( train_df, ['hour','app_cate_id'], 'x13', 'uint32', show_max=True ); gc.collect()
train_df = do_count( train_df, ['os','app_cate_id'], 'x34', 'uint32', show_max=True ); gc.collect()

#train_df = do_count( train_df, ['adid'], 'x36', 'uint32', show_max=True ); gc.collect()

train_df = do_count( train_df, ['hour','nnt','os'], 'x37', 'uint32', show_max=True ); gc.collect()
train_df = do_count( train_df, ['make','app_cate_id'], 'x38', 'uint32', show_max=True ); gc.collect()
train_df = do_count( train_df, ['app_cate_id','advert_name','hour'], 'x39', 'uint32', show_max=True ); gc.collect()
train_df = do_count( train_df, ['app_id','hour'], 'x40', 'uint32', show_max=True ); gc.collect()





test_df = train_df.loc[train_df['click']==MISSING8]
train_df = train_df.loc[train_df['click']!=MISSING8]
test_df['advert_industry_inner1'].describe()
train_df['advert_id'].head(5)
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
def cattrans(df,name):
    enc = LabelEncoder()
    feature_to_use = [name]
    X_train_fea = np.array((df[feature_to_use]))

    enc.fit(X_train_fea)
    df[name] = enc.transform(X_train_fea)
    
    return df
def one_hot(df,name):
    feature_to_use = [name]
    X_train_fea = np.array((df[feature_to_use]))
    enc = OneHotEncoder()
    enc.fit(X_train_fea)
    df[name] = enc.transform(X_train_fea)
    return df




from sklearn import model_selection, preprocessing, ensemble

for f in categorical:
        if train_df[f].dtype=='object':
            print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            features_to_use.append(f)
boolval = ['creative_is_jump', 'creative_is_download', 'creative_is_js',
       'creative_is_voicead', 'creative_has_deeplink']
for f in boolval:
    train_df[f] = train_df[f].apply(lambda x: 1 if x == True else 0)
    features_to_use.append(f)

train_df["os_name"].head(5)


buer = ['creative_is_jump', 'creative_is_download', 'creative_is_js',
       'creative_is_voicead', 'creative_has_deeplink']
 
# In[18]:

train_df.describe()


for_one_hot=["creative_id","orderid",'adid',"area",'campaign_id',"hour","nnt","city","province","carrier","devtype","os","creative_height","creative_width","app_cate_id","app_id","creative_type","creative_tp_dnf"]


for f in for_one_hot:
    features_to_use.append(f)


X_train1  = np.array((train_df[features_to_use]))
X_train  = np.array((train_df[features_to_use]))
y_train = np.array(list(train_df['click']))




X_test  = np.array((test_df[features_to_use]))
train_y = np.array(list(train_df['click']))
train_X1 = np.array(list(train_df['userfea'])) # user_feature
test_X1 = np.array(list(test_df['userfea'])) #user_feature



train_X =  np.hstack([X_train,train_X1]) 
test_X =  np.hstack([X_test,test_X1]) 
del train_X1
del test_X1


print (one_hot_col)
def getTag2(x,totalList):
    userfea = np.zeros(len(totalList))
    if x in totalList:
        ind = totalList.index(x)
        userfea[ind] = 1
    return userfea
def dup_remove_set2(list_raw):
    return list(set(list_raw))



for f in one_hot_col:
  
    print (f)
    taglist_2 = list(test_df[f].fillna("NaN"))  
    taglist_2 = dup_remove_set2(taglist_2)
    print (len(taglist_2))
    print (taglist_2)
    train_df[f] = train_df[f].apply(lambda x: getTag2(x,taglist_2))
    test_df[f] = test_df[f].apply(lambda x: getTag2(x,taglist_2))
    train_X =  np.hstack([train_X,np.array(list(train_df[f]))]) 
    test_X =  np.hstack([test_X,np.array(list(test_df[f]))]) 


print (train_X.shape)
del train_df 
# In[ ]:

import xgboost as xgb
import lightgbm as lgb

res = test_df.loc[:, ['instance_id']]
# model
model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=127, max_depth=-1, learning_rate=0.02, n_estimators=5000,
                           max_bin=425, subsample_for_bin=150000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=30, subsample=0.8, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=10, silent=True)
from sklearn.cross_validation import StratifiedKFold
skf=list(StratifiedKFold(y_train, n_folds=5, shuffle=True, random_state=1024))
baseloss = []
loss = 0
for i, (train_index, test_index) in enumerate(skf):
    print("Fold", i)
    lgb_model = model.fit(train_X[train_index], train_y[train_index],
                          eval_names =['train','valid'],
                          eval_metric='logloss',
                          eval_set=[(train_X[train_index], train_y[train_index]), 
                                    (train_X[test_index], train_y[test_index])],early_stopping_rounds=100)
    baseloss.append(lgb_model.best_score_['valid']['binary_logloss'])
    loss += lgb_model.best_score_['valid']['binary_logloss']
    test_pred= lgb_model.predict_proba(test_X, num_iteration=lgb_model.best_iteration_)[:, 1]
    print('test mean:', test_pred.mean())
    res['prob_%s' % str(i)] = test_pred
print('logloss:', baseloss, loss/5)

res['predicted_score'] = 0
for i in range(5):
    res['predicted_score'] += res['prob_%s' % str(i)]
res['predicted_score'] = res['predicted_score']/5


mean = res['predicted_score'].mean()
print('mean:',mean)
res[['instance_id', 'predicted_score']].to_csv(".result/water_result.csv", index=False)
