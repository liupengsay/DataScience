import pandas as pd
import lightgbm as lgb
from scipy import sparse
import time
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')
print('下面请开始你的标签读取...')
evals_y=pd.read_csv('data_preprocessing/evals_y.csv',header=None)
clf = LGBMClassifier(boosting_type='gbdt', num_leaves=40, max_depth=-1, learning_rate=0.1, 
                n_estimators=10000, subsample_for_bin=200000, objective=None, 
                class_weight=None, min_split_gain=0.0, min_child_weight=0.001, 
                min_child_samples=20, subsample=0.7, subsample_freq=1, 
                colsample_bytree=0.7, 
                reg_alpha=6, reg_lambda=3,
                random_state=2018, n_jobs=-1, silent=True)
from scipy import sparse
print('生成分块索引...')
train_part_x=sparse.load_npz('data_preprocessing/train_part_x_sparse_one_select.npz')
se = pd.Series(range(0,train_part_x.shape[0]))
ind = se.sample(frac=1,random_state=2018).index.tolist()
cut = [0]
for i in range(1,5):
    cut.append(int(len(se)*i/4)+1)
index = [] 
for i in range(4):
    index.append(ind[cut[i]:cut[i+1]])
train_part_x = []

for i in range(0,4):
    s = time.time()
    print('内存太小了只好读取第',i+1,'部分数据...')
    print('蜗牛般读取训练集...')
    train_index = index[i]
    train_part_y=pd.read_csv('data_preprocessing/train_part_y.csv',header=None).loc[train_index]
    df = pd.read_csv('data_preprocessing/train_part_x_cvr_select.csv').loc[train_index]
    df = pd.concat([df,pd.read_csv('data_preprocessing/train_part_x_ratio_select.csv').loc[train_index]],axis=1)
    train_part_x = pd.concat([df,pd.read_csv('data_preprocessing/train_part_x_length.csv').loc[train_index]],axis=1)
    df = []
    train_part_x = sparse.hstack((train_part_x,sparse.load_npz('data_preprocessing/train_part_x_sparse_one_select.npz').tocsr()[train_index,:]))
    print('闪电般读取验证集...')
    df = pd.read_csv('data_preprocessing/evals_x_cvr_select.csv')
    df = pd.concat([df,pd.read_csv('data_preprocessing/evals_x_ratio_select.csv')],axis=1)
    evals_x = pd.concat([df,pd.read_csv('data_preprocessing/evals_x_length.csv')],axis=1)
    df = []
    evals_x = sparse.hstack((evals_x,sparse.load_npz('data_preprocessing/evals_x_sparse_one_select.npz')))
    print('天灵灵地灵灵开始拟合模型...')
    clf.fit(train_part_x,train_part_y, eval_set=[(train_part_x, train_part_y),(evals_x, evals_y)], 
            eval_names =['train','valid'],
            eval_metric='auc',early_stopping_rounds=50)
    auc = int(clf.best_score_['valid']['auc']*1000000)
    train_part_x = []
    train_part_y = []
    from sklearn.metrics import roc_auc_score
    print('=================================evals=================================')
    print('开始预测验证集...')
    evals_ypre = clf.predict_proba(evals_x,num_iteration = clf.best_iteration_)[:,1]
    evals_x = []
    print(pd.Series(evals_ypre).describe())
    round(pd.Series(evals_ypre),6).to_csv('data_preprocessing/evals_ypre_'+str(i+1)+'.csv',index=False)
    print('很高兴本次模型验证集得分为',roc_auc_score(evals_y[0].values,evals_ypre))
    evals_ypre = []
    print('=================================test2=================================')
    print('花火样读取测试集2...')
    df = pd.read_csv('data_preprocessing/test2_x_cvr_select.csv')
    df = pd.concat([df,pd.read_csv('data_preprocessing/test2_x_ratio_select.csv')],axis=1)
    test_x = pd.concat([df,pd.read_csv('data_preprocessing/test2_x_length.csv')],axis=1)
    df = []
    test_x = sparse.hstack((test_x,sparse.load_npz('data_preprocessing/test2_x_sparse_one_select.npz')))
    print('开始预测测试集2...')
    res = pd.read_csv('aid_uid_test2.csv')
    res['score'] = clf.predict_proba(test_x,num_iteration = clf.best_iteration_)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('data_preprocessing/submission_'+str(i+1)+'.csv',index=False)
    print(res['score'].describe())
    test_x = []
    res = []
    print('测试集2的预测结果已存好...')
    print(int((time.time()-s)/60),"minutes")
    print('\n')