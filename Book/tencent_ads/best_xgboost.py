"""
feature + svd + sparse
"""
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from scipy import sparse

def read_data(debug=True):
    """
    读取数据
    :param debug:代码调试
    :return:
    """
    print("read_data...")
    train_df = pd.read_csv("preprocess/train_sample_feature.csv", nrows=NROWS)
    test_df = pd.read_csv("preprocess/test_sample_feature.csv", nrows=NROWS)

    # 去除掉cat_features
    features = train_df.columns.tolist()
    cat_features= [feature for feature in features if 'labelencoder' in feature]
    train_df = train_df.drop(cat_features, axis=1)
    test_df = test_df.drop(cat_features, axis=1)
    train_sp = sparse.load_npz("preprocess/train_sample_sparse.npz").tocsr()[:10000 if debug else train_df.shape[0], :]
    test_sp = sparse.load_npz("preprocess/test_sample_sparse.npz").tocsr()[:10000 if debug else test_df.shape[0], :]
    train_x = sparse.hstack((train_df, train_sp)).tocsr()
    test_x = sparse.hstack((test_df, test_sp)).tocsr()
    print("done")
    return train_x, test_x


def params_append(params):
    """
    对参数进行额外赋值
    :param params:参数字典
    :return:更新后的参数字典
    """
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'auc'
    params["min_child_weight"] = int(params["min_child_weight"])
    params['max_depth'] = int(params['max_depth'])
    return params


def param_beyesian(train):
    """
    beyesian超参寻优
    :param train:训练集
    :return:最优结果对应的参数字典
    """
    def xgb_cv(colsample_bytree, subsample, min_child_weight, max_depth,
               reg_alpha, eta,
               reg_lambda):
        """

        :param colsample_bytree:
        :param subsample:
        :param min_child_weight:
        :param max_depth:
        :param reg_alpha:
        :param eta:
        :param reg_lambda:
        :return:
        """
        params = {'objective': 'binary:logistic',
                  'early_stopping_round': 50,
                  'eval_metric': 'auc'}
        params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
        params['subsample'] = max(min(subsample, 1), 0)
        params["min_child_weight"] = int(min_child_weight)
        params['max_depth'] = int(max_depth)
        params['eta'] = float(eta)
        params['reg_alpha'] = max(reg_alpha, 0)
        params['reg_lambda'] = max(reg_lambda, 0)
        print(params)
        cv_result = xgb.cv(params, train_data,
                           num_boost_round=1000,
                           nfold=2, seed=2,
                           stratified=False,
                           shuffle=True,
                           early_stopping_rounds=30,
                           verbose_eval=False)
        return max(cv_result['test-auc-mean'])
    train_y = pd.read_csv("preprocess/train_sample.csv", nrows=NROWS)['label']
    sample_index = train_y.sample(frac=0.1, random_state=2020).index.tolist()
    train_data = xgb.DMatrix(train.tocsr()[sample_index, :
                             ], train_y.loc[sample_index].values, silent=True)
    xgb_bo = BayesianOptimization(
        xgb_cv,
        {'colsample_bytree': (0.5, 1),
         'subsample': (0.5, 1),
         'min_child_weight': (1, 30),
         'max_depth': (5, 12),
         'reg_alpha': (0, 5),
         'eta':(0.02, 0.2),
         'reg_lambda': (0, 5)}
    )
    xgb_bo.maximize(init_points=21, n_iter=5)  # init_points表示初始点，n_iter代表迭代次数（即采样数）
    print(xgb_bo.max['target'], xgb_bo.max['params'])
    return xgb_bo.max['params']


def train_predict(train, test, params):
    """

    :param train:
    :param test:
    :param params:
    :return:
    """
    train_y = pd.read_csv("preprocess/train_sample.csv")['label']
    test_data = xgb.DMatrix(test)
    truth_test = pd.read_csv('preprocess/test_truth_sample.csv')['label'].values

    params = params_append(params)
    kf = KFold(n_splits=5, random_state=2020, shuffle=True)
    prediction_test = 0
    cv_score = []
    prediction_train = pd.Series(dtype=float)
    ESR = 30
    NBR = 10000
    VBE = 50
    EVAL_RESULT = {}
    for train_part_index, eval_index in kf.split(train, train_y):
        # 模型训练
        train_part = xgb.DMatrix(train.tocsr()[train_part_index, :],
                                 train_y.loc[train_part_index])
        eval = xgb.DMatrix(train.tocsr()[eval_index, :],
                           train_y.loc[eval_index])
        bst = xgb.train(params, train_part, NBR, [(train_part, 'train'),
                                                          (eval, 'eval')], verbose_eval=VBE,
                        maximize=False, early_stopping_rounds=ESR, evals_result=EVAL_RESULT)
        prediction_test += bst.predict(test_data)
        eval_pre = bst.predict(eval)
        prediction_train = prediction_train.append(pd.Series(eval_pre, index=eval_index))
        cv_score.append(max(EVAL_RESULT['eval']['auc']))
        print(cv_score)
        print(roc_auc_score(truth_test, prediction_test / 5))
    print(cv_score, sum(cv_score) / 5)
    print(roc_auc_score(truth_test, prediction_test / 5))
    print(roc_auc_score(train_y.values, prediction_train.sort_index().values))
    pd.Series(prediction_train.sort_index().values).to_csv("result/train_xgboost.csv", index=False)
    pd.Series(prediction_test / 5).to_csv("result/test_xgboost.csv", index=False)
    return


if __name__ == "__main__":

    # 读取数据
    train, test = read_data(False)

    # 贝叶斯参数寻优
    # best_clf = param_beyesian(train)
    # print(best_clf)

    # 根据寻优结果设置参数
    best_clf = {'colsample_bytree': 0.8648389082861029, 'eta': 0.10507276399720099, 'max_depth': 5.587245378907312, 'min_child_weight': 1.8588942349272135, 'reg_alpha': 4.106739442342471, 'reg_lambda': 1.8352371284719622, 'subsample': 0.5630865519911432}

    # 模型训练与预测
    train_predict(train, test, best_clf)

    # 模型效果
    # [0.680968, 0.683661, 0.68366, 0.669765, 0.677175] 0.6790458
    # 0.676715279079092