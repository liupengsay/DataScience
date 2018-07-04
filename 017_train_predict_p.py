##使用的特征描述，及是否为部分数据
import pandas as pd
import lightgbm as lgb
from scipy import sparse
import time
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')
print('下面请开始你的标签读取...')
evals_y=pd.read_csv('data_preprocessing/evals_y_p.csv',header=None)
from scipy import sparse
print('生成分块索引...')
se=pd.read_csv('data_preprocessing/train_part_y_p.csv',header=None)
ind = se.sample(frac=1,random_state=2018).index.tolist()
cut = [0]
for i in range(1,6):
    cut.append(int(len(se)*i/5)+1)
index = [] 
for i in range(5):
    index.append(ind[cut[i]:cut[i+1]])
se = []

clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,
        learning_rate=0.1, max_depth=-1, min_child_samples=20,
        min_child_weight=0.001, min_split_gain=0.0, n_estimators=10000,
        n_jobs=-1, num_leaves=110, objective=None, random_state=2018,
        reg_alpha=2, reg_lambda=3, silent=True, subsample=0.7,
        subsample_for_bin=200000, subsample_freq=1)

for i in range(0,3):
    s = time.time()
    print('内存太小了只好读取第',i+1,'部分数据...')
    print('蜗牛般读取训练集...')
    train_index = index[i]
    train_part_y=pd.read_csv('data_preprocessing/train_part_y_p.csv',header=None).loc[train_index]
    spar = sparse.load_npz('data_preprocessing/train_part_x_sparse_one_p.npz').tocsr()[train_index,:]
    df = pd.read_csv('data_preprocessing/train_part_x_cvr_p.csv').loc[train_index]
    df = pd.concat([df,pd.read_csv('data_preprocessing/train_part_x_CV_cvr_select_p.csv').loc[train_index]],axis=1)
    print('Sparse...')
    train_part_x = sparse.hstack((spar,df))
    df = []
    spar = []
    print('闪电般读取验证集...')
    spar = sparse.load_npz('data_preprocessing/evals_x_sparse_one_p.npz')
    df = pd.read_csv('data_preprocessing/evals_x_cvr_p.csv')
    df = pd.concat([df,pd.read_csv('data_preprocessing/evals_x_CV_cvr_select_p.csv')],axis=1)
    evals_x = sparse.hstack((spar,df))
    df = []
    spar = []
    print('天灵灵地灵灵开始拟合模型...')
    clf.fit(train_part_x,train_part_y, eval_set=[(train_part_x, train_part_y),(evals_x, evals_y)], 
            eval_names =['train','valid'],
            eval_metric='auc',early_stopping_rounds=100)
    auc = int(clf.best_score_['valid']['auc']*1000000)
    train_part_x = []
    train_part_y = []
    from sklearn.metrics import roc_auc_score
    print('=================================evals=================================')
    print('开始预测验证集...')
    evals_ypre = clf.predict_proba(evals_x,num_iteration = clf.best_iteration_)[:,1]
    evals_x = []
    print(pd.Series(evals_ypre).describe())
    round(pd.Series(evals_ypre),6).to_csv('data_preprocessing/evals_ypre_'+str(i+1)+'_p.csv',index=False)
    print('很高兴本次模型验证集得分为',roc_auc_score(evals_y[0].values,evals_ypre))
    evals_ypre = []
    print('=================================test2=================================')
    print('花火样读取测试集2...')
    spar = sparse.load_npz('data_preprocessing/test2_x_sparse_one_p.npz')
    df = pd.read_csv('data_preprocessing/test2_x_cvr_p.csv')
    df = pd.concat([df,pd.read_csv('data_preprocessing/test2_x_CV_cvr_select_p.csv')],axis=1)
    test_x = sparse.hstack((spar,df))
    df = []
    spar = []
    print('开始预测测试集2...')
    res = pd.read_csv('data_preprocessing/aid_uid_test2.csv')
    res['score'] = clf.predict_proba(test_x,num_iteration = clf.best_iteration_)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('data_preprocessing/submission2_'+str(i+1)+'_p.csv',index=False)
    print(res['score'].describe())
    test_x = []
    res = []
    print('测试集2的预测结果已存好...')
    print(int((time.time()-s)/60),"minutes")
    print('\n')
########################################################################################################################
print('下面请开始你的标签读取...')
evals_y=pd.read_csv('data_preprocessing/evals_y_p.csv',header=None)

clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.8,
        learning_rate=0.1, max_depth=-1, min_child_samples=20,
        min_child_weight=0.001, min_split_gain=0.0, n_estimators=10000,
        n_jobs=-1, num_leaves=110, objective=None, random_state=None,
        reg_alpha=10, reg_lambda=0.0, silent=True, subsample=1.0,
        subsample_for_bin=200000, subsample_freq=1)
from scipy import sparse
print('生成分块索引...')
se=pd.read_csv('data_preprocessing/train_part_y_p.csv',header=None)
ind = se.sample(frac=1,random_state=2018).index.tolist()
cut = [0]
for i in range(1,6):
    cut.append(int(len(se)*i/5)+1)
index = [] 
for i in range(5):
    index.append(ind[cut[i]:cut[i+1]])
se = []

clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,
        learning_rate=0.1, max_depth=-1, min_child_samples=20,
        min_child_weight=0.001, min_split_gain=0.0, n_estimators=10000,
        n_jobs=-1, num_leaves=110, objective=None, random_state=2018,
        reg_alpha=2, reg_lambda=3, silent=True, subsample=0.7,
        subsample_for_bin=200000, subsample_freq=1)
for i in range(3,4):
    s = time.time()
    print('内存太小了只好读取第',i+1,'部分数据...')
    print('蜗牛般读取训练集...')
    train_index = index[i]
    train_part_y=pd.read_csv('data_preprocessing/train_part_y_p.csv',header=None).loc[train_index]
    spar = sparse.load_npz('data_preprocessing/train_part_x_sparse_one_p.npz').tocsr()[train_index,:]
    df = pd.read_csv('data_preprocessing/train_part_x_ratio_p.csv').loc[train_index]
    df = pd.concat([df,pd.read_csv('data_preprocessing/train_part_x_length_p.csv').loc[train_index]],axis=1)
    print('Sparse...')
    train_part_x = sparse.hstack((spar,df))
    df = []
    spar = []
    print('闪电般读取验证集...')
    spar = sparse.load_npz('data_preprocessing/evals_x_sparse_one_p.npz')
    df = pd.read_csv('data_preprocessing/evals_x_ratio_p.csv')
    df = pd.concat([df,pd.read_csv('data_preprocessing/evals_x_length_p.csv')],axis=1)
    evals_x = sparse.hstack((spar,df))
    df = []
    spar = []
    print('天灵灵地灵灵开始拟合模型...')
    clf.fit(train_part_x,train_part_y, eval_set=[(train_part_x, train_part_y),(evals_x, evals_y)], 
            eval_names =['train','valid'],
            eval_metric='auc',early_stopping_rounds=100)
    auc = int(clf.best_score_['valid']['auc']*1000000)
    train_part_x = []
    train_part_y = []
    from sklearn.metrics import roc_auc_score
    print('=================================evals=================================')
    print('开始预测验证集...')
    evals_ypre = clf.predict_proba(evals_x,num_iteration = clf.best_iteration_)[:,1]
    evals_x = []
    print(pd.Series(evals_ypre).describe())
    round(pd.Series(evals_ypre),6).to_csv('data_preprocessing/evals_ypre_'+str(i+1)+'_p.csv',index=False)
    print('很高兴本次模型验证集得分为',roc_auc_score(evals_y[0].values,evals_ypre))
    evals_ypre = []
    print('=================================test2=================================')
    print('花火样读取测试集2...')
    spar = sparse.load_npz('data_preprocessing/test2_x_sparse_one_p.npz')
    df = pd.read_csv('data_preprocessing/test2_x_ratio_p.csv')
    df = pd.concat([df,pd.read_csv('data_preprocessing/test2_x_length_p.csv')],axis=1)
    test_x = sparse.hstack((spar,df))
    df = []
    spar = []
    print('开始预测测试集2...')
    res = pd.read_csv('data_preprocessing/aid_uid_test2.csv')
    res['score'] = clf.predict_proba(test_x,num_iteration = clf.best_iteration_)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('data_preprocessing/submission2_'+str(i+1)+'_p.csv',index=False)
    print(res['score'].describe())
    test_x = []
    res = []
    print('测试集2的预测结果已存好...')
    print(int((time.time()-s)/60),"minutes")
    print('\n')
###########################################################################################################################
print('下面请开始你的标签读取...')
evals_y=pd.read_csv('data_preprocessing/evals_y_p.csv',header=None)

from scipy import sparse
print('生成分块索引...')
se=pd.read_csv('data_preprocessing/train_part_y_p.csv',header=None)
ind = se.sample(frac=1,random_state=2018).index.tolist()
cut = [0]
for i in range(1,6):
    cut.append(int(len(se)*i/5)+1)
index = [] 
for i in range(5):
    index.append(ind[cut[i]:cut[i+1]])
se = []

clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.8,
        learning_rate=0.1, max_depth=-1, min_child_samples=20,
        min_child_weight=0.001, min_split_gain=0.0, n_estimators=10000,
        n_jobs=-1, num_leaves=110, objective=None, random_state=None,
        reg_alpha=10, reg_lambda=0.0, silent=True, subsample=1.0,
        subsample_for_bin=200000, subsample_freq=1)
feature = 'part_data_click_unique_p'

for i in range(4,5):
    s = time.time()
    print('内存太小了只好读取第',i+1,'部分数据...')
    print('蜗牛般读取训练集...')
    train_index = index[i]
    train_part_y=pd.read_csv('data_preprocessing/train_part_y_p.csv',header=None).loc[train_index]
    spar = sparse.load_npz('data_preprocessing/train_part_x_sparse_one_p.npz').tocsr()[train_index,:]
    df = pd.read_csv('data_preprocessing/train_part_x_click_p.csv').loc[train_index]
    df = pd.concat([df,pd.read_csv('data_preprocessing/train_part_x_unique_p.csv').loc[train_index]],axis=1)
    print('Sparse...')
    train_part_x = sparse.hstack((spar,df))
    df = []
    spar = []
    print('闪电般读取验证集...')
    spar = sparse.load_npz('data_preprocessing/evals_x_sparse_one_p.npz')
    df = pd.read_csv('data_preprocessing/evals_x_click_p.csv')
    df = pd.concat([df,pd.read_csv('data_preprocessing/evals_x_unique_p.csv')],axis=1)
    evals_x = sparse.hstack((spar,df))
    df = []
    spar = []
    print('天灵灵地灵灵开始拟合模型...')
    clf.fit(train_part_x,train_part_y, eval_set=[(train_part_x, train_part_y),(evals_x, evals_y)], 
            eval_names =['train','valid'],
            eval_metric='auc',early_stopping_rounds=100)
    auc = int(clf.best_score_['valid']['auc']*1000000)
    train_part_x = []
    train_part_y = []
    from sklearn.metrics import roc_auc_score
    print('=================================evals=================================')
    print('开始预测验证集...')
    evals_ypre = clf.predict_proba(evals_x,num_iteration = clf.best_iteration_)[:,1]
    evals_x = []
    print(pd.Series(evals_ypre).describe())
    round(pd.Series(evals_ypre),6).to_csv('data_preprocessing/evals_ypre_'+str(i+1)+'_p.csv',index=False)
    print('很高兴本次模型验证集得分为',roc_auc_score(evals_y[0].values,evals_ypre))
    evals_ypre = []
    print('=================================test2=================================')
    print('花火样读取测试集2...')
    spar = sparse.load_npz('data_preprocessing/test2_x_sparse_one_p.npz')
    df = pd.read_csv('data_preprocessing/test2_x_click_p.csv')
    df = pd.concat([df,pd.read_csv('data_preprocessing/test2_x_unique_p.csv')],axis=1)
    test_x = sparse.hstack((spar,df))
    df = []
    spar = []
    print('开始预测测试集2...')
    res = pd.read_csv('data_preprocessing/aid_uid_test2.csv')
    res['score'] = clf.predict_proba(test_x,num_iteration = clf.best_iteration_)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('data_preprocessing/submission2_'+str(i+1)+'_p.csv',index=False)
    print(res['score'].describe())
    test_x = []
    res = []
    print('测试集2的预测结果已存好...')
    print(int((time.time()-s)/60),"minutes")
    print('\n')
###########################################################################################################
###### 使用的特征描述，及是否为部分数据
print('下面请开始你的标签读取...')
evals_y=pd.read_csv('data_preprocessing/evals_y_p.csv',header=None)
from scipy import sparse
print('生成分块索引...')
se=pd.read_csv('data_preprocessing/train_part_y_p.csv',header=None)
ind = se.sample(frac=1,random_state=2020).index.tolist()
cut = [0]
for i in range(1,11):
    cut.append(int(len(se)*i/10)+1)
index = [] 
for i in range(10):
    index.append(ind[cut[i]:cut[i+1]])
se = []

clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.8,
        learning_rate=0.1, max_depth=-1, min_child_samples=20,
        min_child_weight=0.001, min_split_gain=0.0, n_estimators=10000,
        n_jobs=-1, num_leaves=110, objective=None, random_state=None,
        reg_alpha=10, reg_lambda=0.0, silent=True, subsample=1.0,
        subsample_for_bin=200000, subsample_freq=1)
feature = 'part_data_all_p'
for i in range(0,2):
    s = time.time()
    print('内存太小了只好读取第',i+1,'部分数据...')
    print('蜗牛般读取训练集...')
    train_index = index[i]
    train_part_y=pd.read_csv('data_preprocessing/train_part_y_p.csv',header=None).loc[train_index]
    spar = sparse.load_npz('data_preprocessing/train_part_x_sparse_one_p.npz').tocsr()[train_index,:]
    df = pd.read_csv('data_preprocessing/train_part_x_ratio_p.csv').loc[train_index]
    df = pd.concat([df,pd.read_csv('data_preprocessing/train_part_x_cvr_p.csv').loc[train_index]],axis=1)
    df = pd.concat([df,pd.read_csv('data_preprocessing/train_part_x_CV_cvr_select_p.csv').loc[train_index]],axis=1)
    df = pd.concat([df,pd.read_csv('data_preprocessing/train_part_x_length_p.csv').loc[train_index]],axis=1)
    print('Sparse...')
    train_part_x = sparse.hstack((spar,df))
    df = []
    spar = []
    print('闪电般读取验证集...')
    spar = sparse.load_npz('data_preprocessing/evals_x_sparse_one_p.npz')
    df = pd.read_csv('data_preprocessing/evals_x_ratio_p.csv')
    df = pd.concat([df,pd.read_csv('data_preprocessing/evals_x_cvr_p.csv')],axis=1)
    df = pd.concat([df,pd.read_csv('data_preprocessing/evals_x_CV_cvr_select_p.csv')],axis=1)
    df = pd.concat([df,pd.read_csv('data_preprocessing/evals_x_length_p.csv')],axis=1)
    evals_x = sparse.hstack((spar,df))
    df = []
    spar = []
    print('天灵灵地灵灵开始拟合模型...')
    clf.fit(train_part_x,train_part_y, eval_set=[(train_part_x, train_part_y),(evals_x, evals_y)], 
            eval_names =['train','valid'],
            eval_metric='auc',early_stopping_rounds=100)
    auc = int(clf.best_score_['valid']['auc']*1000000)
    train_part_x = []
    train_part_y = []
    from sklearn.metrics import roc_auc_score
    print('=================================evals=================================')
    print('开始预测验证集...')
    evals_ypre = clf.predict_proba(evals_x,num_iteration = clf.best_iteration_)[:,1]
    evals_x = []
    print(pd.Series(evals_ypre).describe())
    round(pd.Series(evals_ypre),6).to_csv('data_preprocessing/evals_ypre_'+str(i+6)+'_p.csv',index=False)
    print('很高兴本次模型验证集得分为',roc_auc_score(evals_y[0].values,evals_ypre))
    evals_ypre = []
    print('=================================test2=================================')
    print('花火样读取测试集2...')
    spar = sparse.load_npz('data_preprocessing/test2_x_sparse_one_p.npz')
    df = pd.read_csv('data_preprocessing/test2_x_ratio_p.csv')
    df = pd.concat([df,pd.read_csv('data_preprocessing/test2_x_cvr_p.csv')],axis=1)
    df = pd.concat([df,pd.read_csv('data_preprocessing/test2_x_CV_cvr_select_p.csv')],axis=1)
    df = pd.concat([df,pd.read_csv('data_preprocessing/test2_x_length_p.csv')],axis=1)
    test_x = sparse.hstack((spar,df))
    df = []
    spar = []
    print('开始预测测试集2...')
    res = pd.read_csv('data_preprocessing/aid_uid_test2.csv')
    res['score'] = clf.predict_proba(test_x,num_iteration = clf.best_iteration_)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('data_preprocessing/submission2_'+str(i+6)+'_p.csv',index=False)
    print(res['score'].describe())
    test_x = []
    res = []
    print('测试集2的预测结果已存好...')
    print(int((time.time()-s)/60),"minutes")
    print('\n')

clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,
        learning_rate=0.1, max_depth=-1, min_child_samples=120,
        min_child_weight=0.001, min_split_gain=0.0, n_estimators=10000,
        n_jobs=-1, num_leaves=110, objective=None, random_state=2018,
        reg_alpha=2, reg_lambda=3, silent=True, subsample=0.7,
        subsample_for_bin=200000, subsample_freq=1)
for i in range(2,4):
    s = time.time()
    print('内存太小了只好读取第',i+1,'部分数据...')
    print('蜗牛般读取训练集...')
    train_index = index[i]
    train_part_y=pd.read_csv('data_preprocessing/train_part_y_p.csv',header=None).loc[train_index]
    spar = sparse.load_npz('data_preprocessing/train_part_x_sparse_one_p.npz').tocsr()[train_index,:]
    df = pd.read_csv('data_preprocessing/train_part_x_click_p.csv').loc[train_index]
    df = pd.concat([df,pd.read_csv('data_preprocessing/train_part_x_cvr_p.csv').loc[train_index]],axis=1)
    df = pd.concat([df,pd.read_csv('data_preprocessing/train_part_x_CV_cvr_select_p.csv').loc[train_index]],axis=1)
    df = pd.concat([df,pd.read_csv('data_preprocessing/train_part_x_unique_p.csv').loc[train_index]],axis=1)
    print('Sparse...')
    train_part_x = sparse.hstack((spar,df))
    df = []
    spar = []
    print('闪电般读取验证集...')
    spar = sparse.load_npz('data_preprocessing/evals_x_sparse_one_p.npz')
    df = pd.read_csv('data_preprocessing/evals_x_click_p.csv')
    df = pd.concat([df,pd.read_csv('data_preprocessing/evals_x_cvr_p.csv')],axis=1)
    df = pd.concat([df,pd.read_csv('data_preprocessing/evals_x_CV_cvr_select_p.csv')],axis=1)
    df = pd.concat([df,pd.read_csv('data_preprocessing/evals_x_unique_p.csv')],axis=1)
    evals_x = sparse.hstack((spar,df))
    df = []
    spar = []
    print('天灵灵地灵灵开始拟合模型...')
    clf.fit(train_part_x,train_part_y, eval_set=[(train_part_x, train_part_y),(evals_x, evals_y)], 
            eval_names =['train','valid'],
            eval_metric='auc',early_stopping_rounds=100)
    auc = int(clf.best_score_['valid']['auc']*1000000)
    train_part_x = []
    train_part_y = []
    from sklearn.metrics import roc_auc_score
    print('=================================evals=================================')
    print('开始预测验证集...')
    evals_ypre = clf.predict_proba(evals_x,num_iteration = clf.best_iteration_)[:,1]
    evals_x = []
    print(pd.Series(evals_ypre).describe())
    round(pd.Series(evals_ypre),6).to_csv('data_preprocessing/evals_ypre_'+str(i+6)+'_p.csv',index=False)
    print('很高兴本次模型验证集得分为',roc_auc_score(evals_y[0].values,evals_ypre))
    evals_ypre = []
    print('=================================test2=================================')
    print('花火样读取测试集2...')
    spar = sparse.load_npz('data_preprocessing/test2_x_sparse_one_p.npz')
    df = pd.read_csv('data_preprocessing/test2_x_click_p.csv')
    df = pd.concat([df,pd.read_csv('data_preprocessing/test2_x_cvr_p.csv')],axis=1)
    df = pd.concat([df,pd.read_csv('data_preprocessing/test2_x_CV_cvr_select_p.csv')],axis=1)
    df = pd.concat([df,pd.read_csv('data_preprocessing/test2_x_unique_p.csv')],axis=1)
    test_x = sparse.hstack((spar,df))
    df = []
    spar = []
    print('开始预测测试集2...')
    res = pd.read_csv('data_preprocessing/aid_uid_test2.csv')
    res['score'] = clf.predict_proba(test_x,num_iteration = clf.best_iteration_)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('data_preprocessing/evals_ypre_'+str(i+6)+'_p.csv',index=False)
    print(res['score'].describe())
    test_x = []
    res = []
    print('测试集2的预测结果已存好...')
    print(int((time.time()-s)/60),"minutes")
    print('\n')