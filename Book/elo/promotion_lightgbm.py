"""
特征组合：Dict+GroupBy+nlp
特征选择方式：Wrapper
参数寻优办法：hyperopt
模型：lightgbm
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from hyperopt import hp, fmin, tpe
from numpy.random import RandomState
from sklearn.metrics import mean_squared_error

def read_data(debug=True):
    """

    :param debug:
    :return:
    """
    print("read_data...")
    NROWS = 10000 if debug else None
    train_dict = pd.read_csv("preprocess/train_dict.csv", nrows=NROWS)
    test_dict = pd.read_csv("preprocess/test_dict.csv", nrows=NROWS)
    train_groupby = pd.read_csv("preprocess/train_groupby.csv", nrows=NROWS)
    test_groupby = pd.read_csv("preprocess/test_groupby.csv", nrows=NROWS)

    # 去除重复列
    for co in train_dict.columns:
        if co in train_groupby.columns and co!='card_id':
            del train_groupby[co]
    for co in test_dict.columns:
        if co in test_groupby.columns and co!='card_id':
            del test_groupby[co]

    train = pd.merge(train_dict, train_groupby, how='left', on='card_id')
    test = pd.merge(test_dict, test_groupby, how='left', on='card_id')
    print("done")
    return train, test


def feature_select_wrapper(train, test):
    """

    :param train:
    :param test:
    :return:
    """
    print('feature_select_wrapper...')
    label = 'target'
    features = train.columns.tolist()
    features.remove('card_id')
    features.remove('target')

    # 配置模型的训练参数
    params_initial = {
        'num_leaves': 31,
        'learning_rate': 0.1,
        'boosting': 'gbdt',
        'min_child_samples': 20,
        'bagging_seed': 2020,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'feature_fraction': 0.7,
        'max_depth': -1,
        'metric': 'rmse',
        'reg_alpha': 0,
        'reg_lambda': 1,
        'objective': 'regression'
    }
    ESR = 30
    NBR = 10000
    VBE = 50
    kf = KFold(n_splits=5, random_state=2020, shuffle=True)
    fse = pd.Series(0, index=features)
    for train_part_index, eval_index in kf.split(train[features], train[label]):
        # 模型训练
        train_part = lgb.Dataset(train[features].loc[train_part_index],
                                 train[label].loc[train_part_index])
        eval = lgb.Dataset(train[features].loc[eval_index],
                           train[label].loc[eval_index])
        bst = lgb.train(params_initial, train_part, num_boost_round=NBR,
                        valid_sets=[train_part, eval],
                        valid_names=['train', 'valid'],
                        early_stopping_rounds=ESR, verbose_eval=VBE)
        fse += pd.Series(bst.feature_importance(), features)

    feature_select = ['card_id'] + fse.sort_values(ascending=False).index.tolist()[:300]
    print('done')
    return train[feature_select + ['target']], test[feature_select]


def params_append(params):
    """

    :param params:
    :return:
    """
    params['objective'] = 'regression'
    params['metric'] = 'rmse'
    params['bagging_seed'] = 2020
    return params


def param_hyperopt(train):
    """

    :param train:
    :return:
    """
    label = 'target'
    features = train.columns.tolist()
    features.remove('card_id')
    features.remove('target')
    train_data = lgb.Dataset(train[features], train[label], silent=True)
    def hyperopt_objective(params):
        """

        :param params:
        :return:
        """
        params = params_append(params)
        print(params)
        res = lgb.cv(params, train_data, 1000,
                     nfold=2,
                     stratified=False,
                     shuffle=True,
                     metrics='rmse',
                     early_stopping_rounds=20,
                     verbose_eval=False,
                     show_stdv=False,
                     seed=2020)
        return min(res['rmse-mean'])

    params_space = {
        'learning_rate': hp.uniform('learning_rate', 1e-2, 5e-1),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
        'num_leaves': hp.choice('num_leaves', list(range(10, 300, 10))),
        'reg_alpha': hp.randint('reg_alpha', 0, 10),
        'reg_lambda': hp.uniform('reg_lambda', 0, 10),
        'bagging_freq': hp.randint('bagging_freq', 1, 10),
        'min_child_samples': hp.choice('min_child_samples', list(range(1, 30, 5)))
    }
    params_best = fmin(
        hyperopt_objective,
        space=params_space,
        algo=tpe.suggest,
        max_evals=30,
        rstate=RandomState(2020))
    return params_best


def train_predict(train, test, params):
    """

    :param train:
    :param test:
    :param params:
    :return:
    """
    label = 'target'
    features = train.columns.tolist()
    features.remove('card_id')
    features.remove('target')
    params = params_append(params)
    kf = KFold(n_splits=5, random_state=2020, shuffle=True)
    prediction_test = 0
    cv_score = []
    prediction_train = pd.Series()
    ESR = 30
    NBR = 10000
    VBE = 50
    for train_part_index, eval_index in kf.split(train[features], train[label]):
        # 模型训练
        train_part = lgb.Dataset(train[features].loc[train_part_index],
                                 train[label].loc[train_part_index])
        eval = lgb.Dataset(train[features].loc[eval_index],
                           train[label].loc[eval_index])
        bst = lgb.train(params, train_part, num_boost_round=NBR,
                        valid_sets=[train_part, eval],
                        valid_names=['train', 'valid'],
                        early_stopping_rounds=ESR, verbose_eval=VBE)
        prediction_test += bst.predict(test[features])
        prediction_train = prediction_train.append(pd.Series(bst.predict(train[features].loc[eval_index]),
                                                             index=eval_index))
        eval_pre = bst.predict(train[features].loc[eval_index])
        score = np.sqrt(mean_squared_error(train[label].loc[eval_index].values, eval_pre))
        cv_score.append(score)
    print(cv_score, sum(cv_score) / 5)
    pd.Series(prediction_train.sort_index().values).to_csv("preprocess/train_lightgbm.csv", index=False)
    pd.Series(prediction_test / 5).to_csv("preprocess/test_lightgbm.csv", index=False)
    test['target'] = prediction_test / 5
    test[['card_id', 'target']].to_csv("result/submission_lightgbm.csv", index=False)
    return

if __name__ == "__main__":
    train, test = read_data(debug=False)
    train, test = feature_select_wrapper(train, test)
    best_clf = param_hyperopt(train)
    train_predict(train, test, best_clf)
# [3.686192535745703, 3.647032390847285, 3.706089838227353, 3.773664215095074, 3.5735473296458626] 3.677305261912256
