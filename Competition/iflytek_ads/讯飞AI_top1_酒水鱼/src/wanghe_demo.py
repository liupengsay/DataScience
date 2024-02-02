# coding: utf-8
import numpy as np
import pandas as pd
import time
import datetime
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
from scipy import sparse
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb
import gc
import os
import warnings
warnings.filterwarnings("ignore")

# 加载数据
train = pd.read_table('./data/round2_iflyad_train.txt')
train1 = pd.read_table('./data/round1_iflyad_train.txt')
train = pd.concat([train, train1], axis=0, ignore_index=True)
train.drop_duplicates(subset=None, keep='first', inplace=False)
test = pd.read_table('./data/round2_iflyad_test_feature.txt')
data = pd.concat([train, test], axis=0, ignore_index=True)
data['day'] = data['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
data['hour'] = data['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))

# 数据预处理
data['os'].replace(0, 2, inplace=True)
lst = []
for va in data['osv'].values:
    va = str(va)
    va = va.replace('iOS', '')
    va = va.replace('android', '')
    va = va.replace(' ', '')
    va = va.replace('iPhoneOS', '')
    va = va.replace('_', '.')
    va = va.replace('Android5.1', '.')
    try:
        int(va)
        lst.append(np.nan)
    except:
        sp = ['nan', '11.39999961853027', '10.30000019073486', 'unknown', '11.30000019073486']
        if va in sp:
            lst.append(np.nan)
        elif va == '3.0.4-RS-20160720.1914':
            lst.append('3.0.4')
        else:
            lst.append(va)
temp = pd.Series(lst).value_counts()
temp = temp[temp <= 2].index.tolist()
for i in range(len(lst)):
    if lst[i] in temp:
        lst[i] = np.nan
data['osv'] = lst
lst1 = []
lst2 = []
lst3 = []
for va in data['osv'].values:
    va = str(va).split('.')
    if len(va) < 3:
        va.extend(['0', '0', '0'])
    lst1.append(va[0])
    lst2.append(va[1])
    lst3.append(va[2])
data['osv1'] = lst1
data['osv2'] = lst2
data['osv3'] = lst3

# add cross feature
first_feature = ['app_cate_id', 'f_channel', 'app_id']
second_feature = ["make", "model", "osv1", "osv2", "osv3", "adid", "advert_name", "campaign_id", "creative_id",
                  "carrier", "nnt", "devtype", "os"]
cross_feature = []
for feat_1 in first_feature:
    for feat_2 in second_feature:
        col_name = "cross_" + feat_1 + "_and_" + feat_2
        cross_feature.append(col_name)
        data[col_name] = data[feat_1].astype(str).values + '_' + data[feat_2].astype(str).values

# fillna
data['make'] = data['make'].fillna(str(-1))
data['model'] = data['model'].fillna(str(-1))
data['osv'] = data['osv'].fillna(str(-1))
data['app_cate_id'] = data['app_cate_id'].fillna(-1)
data['app_id'] = data['app_id'].fillna(-1)
data['click'] = data['click'].fillna(-1)
data['user_tags'] = data['user_tags'].fillna(str(-1))
data['f_channel'] = data['f_channel'].fillna(str(-1))
# replace
replace = ['creative_is_jump', 'creative_is_download', 'creative_is_js', 'creative_is_voicead', 'creative_has_deeplink',
           'app_paid']
for feat in replace:
    data[feat] = data[feat].replace([False, True], [0, 1])
# labelencoder
encoder = ['city', 'province', 'make', 'model', 'osv', 'os_name', 'adid', 'advert_id', 'orderid',
           'advert_industry_inner', 'campaign_id', 'creative_id', 'app_cate_id',
           'app_id', 'inner_slot_id', 'advert_name', 'f_channel', 'osv1', 'osv2', 'osv3'
           ]
encoder = encoder + cross_feature
col_encoder = LabelEncoder()
for feat in encoder:
    col_encoder.fit(data[feat])
    data[feat] = col_encoder.transform(data[feat])

def feature_count(data, features=[]):
    if len(set(features)) != len(features):
        print('equal feature !!!!')
        return data
    new_feature = 'count'
    for i in features:
        new_feature += '_' + i.replace('add_', '')
    try:
        del data[new_feature]
    except:
        pass
    temp = data.groupby(features).size().reset_index().rename(columns={0: new_feature})
    data = data.merge(temp, 'left', on=features)
    return data
for i in cross_feature:
    n = data[i].nunique()
    if n > 5:
        print(i)
        data = feature_count(data, [i])
    else:
        print(i, ':', n)

# user_tags CountVectorizer
train_new = pd.DataFrame()
test_new = pd.DataFrame()
train = data[:train.shape[0]]
test = data[train.shape[0]:]
train_y = train['click']

cntv = CountVectorizer()
cntv.fit(data['user_tags'])
train_a = cntv.transform(train['user_tags'])
test_a = cntv.transform(test['user_tags'])
train_new = sparse.hstack((train_new, train_a), 'csr', 'bool')
test_new = sparse.hstack((test_new, test_a), 'csr', 'bool')
SKB = SelectPercentile(chi2, percentile=95).fit(train_new, train_y)
train_new = SKB.transform(train_new)
test_new = SKB.transform(test_new)

# add nunique feature
## 广告
adid_nuq = ['model', 'make', 'os', 'city', 'province', 'user_tags', 'f_channel', 'app_id', 'carrier', 'nnt', 'devtype',
            'app_cate_id', 'inner_slot_id']
for feat in adid_nuq:
    gp1 = data.groupby('adid')[feat].nunique().reset_index().rename(columns={feat: "adid_%s_nuq_num" % feat})
    gp2 = data.groupby(feat)['adid'].nunique().reset_index().rename(columns={'adid': "%s_adid_nuq_num" % feat})
    data = pd.merge(data, gp1, how='left', on=['adid'])
    data = pd.merge(data, gp2, how='left', on=[feat])
## 广告主
advert_id_nuq = ['model', 'make', 'os', 'city', 'province', 'user_tags', 'f_channel', 'app_id', 'carrier', 'nnt',
                 'devtype',
                 'app_cate_id', 'inner_slot_id']
for fea in advert_id_nuq:
    gp1 = data.groupby('advert_id')[fea].nunique().reset_index().rename(columns={fea: "advert_id_%s_nuq_num" % fea})
    gp2 = data.groupby(fea)['advert_id'].nunique().reset_index().rename(
        columns={'advert_id': "%s_advert_id_nuq_num" % fea})
    data = pd.merge(data, gp1, how='left', on=['advert_id'])
    data = pd.merge(data, gp2, how='left', on=[fea])
## app_id
app_id_nuq = ['model', 'make', 'os', 'city', 'province', 'user_tags', 'f_channel', 'carrier', 'nnt', 'devtype',
              'app_cate_id', 'inner_slot_id']
for fea in app_id_nuq:
    gp1 = data.groupby('app_id')[fea].nunique().reset_index().rename(columns={fea: "app_id_%s_nuq_num" % fea})
    gp2 = data.groupby(fea)['app_id'].nunique().reset_index().rename(columns={'app_id': "%s_app_id_nuq_num" % fea})
    data = pd.merge(data, gp1, how='left', on=['app_id'])
    data = pd.merge(data, gp2, how='left', on=[fea])

## user_id
user_id = ['model', 'make', 'os', 'city', 'province', 'user_tags', 'campaign_id']
data['user_id'] = data['model'].astype(str) + data['make'].astype(str) + data['city'].astype(str) + data[
    'province'].astype(str) + data['user_tags'].astype(str)
gp1 = data.groupby('adid')['user_id'].nunique().reset_index().rename(columns={'user_id': "adid_user_id_nuq_num"})
gp2 = data.groupby('user_id')['adid'].nunique().reset_index().rename(columns={'adid': "user_id_adid_nuq_num"})
data = pd.merge(data, gp1, how='left', on=['adid'])
data = pd.merge(data, gp2, how='left', on=['user_id'])
del data['user_id']

# add ratio feature
ratio_feat = pd.read_csv('./data/ratio_feat.csv')
ratio_list = ratio_feat['feat'][ratio_feat['imp'] > 10].values
label_feature = ['advert_id', 'advert_industry_inner', 'advert_name', 'campaign_id', 'creative_height',
                 'creative_tp_dnf', 'creative_width', 'province', 'f_channel',
                 'carrier', 'creative_type', 'devtype', 'nnt',
                 'adid', 'app_id', 'app_cate_id', 'city', 'os', 'orderid', 'inner_slot_id', 'make', 'osv',
                 'os_name', 'creative_has_deeplink', 'creative_is_download', 'hour', 'creative_id', 'model']
data_temp = data[label_feature]
df_feature = pd.DataFrame()
data_temp['cnt'] = 1
print('Begin ratio clcik...')
col_type = label_feature.copy()
n = len(col_type)
for i in range(n):
    col_name = "ratio_click_of_" + col_type[i]
    if (col_name in ratio_list):
        df_feature[col_name] = (
                    data_temp[col_type[i]].map(data_temp[col_type[i]].value_counts()) / len(data) * 100).astype(int)
n = len(col_type)
for i in range(n):
    for j in range(n):
        if i != j:
            col_name = "ratio_click_of_" + col_type[j] + "_in_" + col_type[i]
            if (col_name in ratio_list):
                se = data_temp.groupby([col_type[i], col_type[j]])['cnt'].sum()
                dt = data_temp[[col_type[i], col_type[j]]]
                cnt = data_temp[col_type[i]].map(data[col_type[i]].value_counts())
                df_feature[col_name] = ((pd.merge(dt, se.reset_index(), how='left',
                                                  on=[col_type[i], col_type[j]]).sort_index()['cnt'].fillna(
                    value=0) / cnt) * 100).astype(int).values
data = pd.concat([data, df_feature], axis=1)
print('The end')

# add ctr feature
data['period'] = data['day']
data['period'][data['period'] < 27] = data['period'][data['period'] < 27] + 31
for feat_1 in ['advert_id', 'advert_industry_inner', 'advert_name', 'campaign_id', 'creative_height',
               'creative_tp_dnf', 'creative_width', 'province', 'f_channel']:
    res = pd.DataFrame()
    temp = data[[feat_1, 'period', 'click']]
    for period in range(27, 35):
        if period == 27:
            count = temp.groupby([feat_1]).apply(
                lambda x: x['click'][(x['period'] <= period).values].count()).reset_index(name=feat_1 + '_all')
            count1 = temp.groupby([feat_1]).apply(
                lambda x: x['click'][(x['period'] <= period).values].sum()).reset_index(name=feat_1 + '_1')
        else:
            count = temp.groupby([feat_1]).apply(
                lambda x: x['click'][(x['period'] < period).values].count()).reset_index(name=feat_1 + '_all')
            count1 = temp.groupby([feat_1]).apply(
                lambda x: x['click'][(x['period'] < period).values].sum()).reset_index(name=feat_1 + '_1')
        count[feat_1 + '_1'] = count1[feat_1 + '_1']
        count.fillna(value=0, inplace=True)
        count[feat_1 + '_rate'] = round(count[feat_1 + '_1'] / count[feat_1 + '_all'], 5)
        count['period'] = period
        count.drop([feat_1 + '_all', feat_1 + '_1'], axis=1, inplace=True)
        count.fillna(value=0, inplace=True)
        res = res.append(count, ignore_index=True)
    print(feat_1, ' over')
    data = pd.merge(data, res, how='left', on=[feat_1, 'period'])

# 减小内存
for i in data.columns:
    if (i != 'instance_id'):
        if (data[i].dtypes == 'int64'):
            data[i] = data[i].astype('int16')
        if (data[i].dtypes == 'int32'):
            data[i] = data[i].astype('int16')

drop = ['click', 'time', 'instance_id', 'user_tags',
        'app_paid', 'creative_is_js', 'creative_is_voicead']

train = data[:train.shape[0]]
test = data[train.shape[0]:]
del data
gc.collect()
y_train = train.loc[:, 'click']
res = test.loc[:, ['instance_id']]

train.drop(drop, axis=1, inplace=True)
print('train:', train.shape)
test.drop(drop, axis=1, inplace=True)
print('test:', test.shape)

X_loc_train = train.values
y_loc_train = y_train.values
X_loc_test = test.values
del train
del test
gc.collect()

# hstack CountVectorizer
X_loc_train = sparse.hstack((X_loc_train, train_new), 'csr')
X_loc_test = sparse.hstack((X_loc_test, test_new), 'csr')
del train_new
del test_new
gc.collect()

# 利用不同折数加参数，特征，样本（随机数种子）扰动，再加权平均得到最终成绩
lgb_clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.02, n_estimators=6000,
                             max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                             min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_alpha=3, reg_lambda=0.1, seed=1000, n_jobs=-1, silent=True)
skf = list(StratifiedKFold(y_loc_train, n_folds=5, shuffle=True, random_state=1024))
baseloss = []
loss = 0
for i, (train_index, test_index) in enumerate(skf):
    print("Fold", i)
    lgb_model = lgb_clf.fit(X_loc_train[train_index], y_loc_train[train_index],
                            eval_names=['train', 'valid'],
                            eval_metric='logloss',
                            eval_set=[(X_loc_train[train_index], y_loc_train[train_index]),
                                      (X_loc_train[test_index], y_loc_train[test_index])], early_stopping_rounds=100)
    baseloss.append(lgb_model.best_score_['valid']['binary_logloss'])
    loss += lgb_model.best_score_['valid']['binary_logloss']
    test_pred = lgb_model.predict_proba(X_loc_test, num_iteration=lgb_model.best_iteration_)[:, 1]
    print('test mean:', test_pred.mean())
    res['prob_%s' % str(i)] = test_pred
print('logloss:', baseloss, loss / 5)

res['predicted_score'] = 0
for i in range(5):
    res['predicted_score'] += res['prob_%s' % str(i)]
res['predicted_score'] = res['predicted_score'] / 5

mean = res['predicted_score'].mean()
print('mean:', mean)
res[['instance_id', 'predicted_score']].to_csv("result/wanghe_result/.csv", index=False)