"""
feature + svd
"""
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from hyperopt import hp, fmin, tpe
from numpy.random import RandomState
from sklearn.metrics import roc_auc_score


def read_data(debug=True):
    """
    读取数据
    :param debug:代码调试
    :return:
    """
    print("read_data...")
    NROWS = 10000 if debug else None
    train_df = pd.read_csv("preprocess/train_sample_feature.csv", nrows=NROWS)
    test_df = pd.read_csv("preprocess/test_sample_feature.csv", nrows=NROWS)
    train_svd = pd.read_csv("preprocess/train_sample_svd.csv", nrows=NROWS)
    test_svd = pd.read_csv("preprocess/test_sample_svd.csv", nrows=NROWS)
    train = pd.concat([train_df, train_svd], axis=1)
    test = pd.concat([test_df, test_svd], axis=1)
    train['label'] = pd.read_csv('preprocess/train_sample.csv', nrows=NROWS)['label'].values
    print("done")
    return train, test


def params_append(params):
    """
    对参数进行额外赋值
    :param params:参数字典
    :return:更新后的参数字典
    """
    params['objective'] = 'binary'
    params['metric'] = 'auc'
    params['bagging_seed'] = 2020
    return params


def param_hyperopt(train):
    """
    采用hyperopt框架进行参数寻优
    :param train:训练集
    :return:
    """
    def hyperopt_objective(params):
        """
        超参调优的目标函数
        :param params:参数字典
        :return:越小越好，所以取auc对应的负值
        """
        params = params_append(params)
        print(params)
        res = lgb.cv(params, train_data, 1000,
                     nfold=2,
                     stratified=False,
                     shuffle=True,
                     metrics='auc',
                     early_stopping_rounds=20,
                     verbose_eval=False,
                     show_stdv=False,
                     categorical_feature=cat_features,
                     seed=2020)
        return -max(res['auc-mean'])
    label = 'label'
    features = train.columns.tolist()
    features.remove('label')
    cat_features= [feature for feature in features if 'labelencoder' in feature]
    train_data = lgb.Dataset(train[features], train[label], silent=True)
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


def train_predict(train, test, params, debug):
    """

    :param train:
    :param test:
    :param params:
    :param debug:
    :return:
    """

    label = 'label'
    features = train.columns.tolist()
    features.remove(label)
    print(train.shape, test.shape)
    cat_features= [feature for feature in features if 'labelencoder' in feature]
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
                        valid_names=['train', 'valid'],categorical_feature=cat_features,
                        early_stopping_rounds=ESR, verbose_eval=VBE)
        prediction_test += bst.predict(test[features])
        prediction_train = prediction_train.append(pd.Series(bst.predict(train[features].loc[eval_index]),
                                                             index=eval_index))
        cv_score.append(bst.best_score['valid']['auc'])
    print(cv_score, sum(cv_score) / 5)
    NROWS = 10000 if debug else None
    truth_test = pd.read_csv('preprocess/test_truth_sample.csv', nrows=NROWS)['label'].values
    print(roc_auc_score(truth_test, prediction_test / 5))
    pd.Series(prediction_train.sort_index().values).to_csv("result/train_lightgbm.csv", index=False)
    pd.Series(prediction_test / 5).to_csv("result/test_lightgbm.csv", index=False)
    return


if __name__ == "__main__":

    # 读取数据
    debug = False
    train, test = read_data(debug)

    # 进行参数寻优
    # best_clf = param_hyperopt(train)
    # print(best_clf)

    # 寻优后设置参数
    best_clf = {'bagging_fraction': 0.920269687818116, 'bagging_freq': 1,
                'feature_fraction': 0.6748121703594134, 'learning_rate': 0.2534183350764367,
                'min_child_samples': 3, 'num_leaves': 10, 'reg_alpha': 0,
                'reg_lambda': 9.986223861515665}

    # 训练预测和结果输出
    train_predict(train, test, best_clf, debug)

    # 模型效果
    # [0.6757995747895691, 0.6774535654447692, 0.6977578977587808, 0.6788931645556028, 0.6661883292824802]
    # 0.6186427592654342