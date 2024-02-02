
"""
特征组合：Dict+GroupBy
特征选择方式：Pearson
参数寻优办法：GridSearch
模型：randomforest

"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def read_data(debug=True):
    """
    读取数据
    :param debug:是否调试版，可以极大节省debug时间
    :return:训练集，测试集
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

    # 拼接特征
    train = pd.merge(train_dict, train_groupby, how='left', on='card_id').fillna(0)
    test = pd.merge(test_dict, test_groupby, how='left', on='card_id').fillna(0)
    print("done")
    return train, test


def feature_select_pearson(train, test):
    """
    利用pearson系数进行相关性特征选择
    :param train:训练集
    :param test:测试集
    :return:经过特征选择后的训练集与测试集
    """
    print('feature_select...')
    features = train.columns.tolist()
    features.remove("card_id")
    features.remove("target")
    featureSelect = features[:]

    # 去掉缺失值比例超过0.99的
    for fea in features:
        if train[fea].isnull().sum() / train.shape[0] >= 0.99:
            featureSelect.remove(fea)

    # 进行pearson相关性计算
    corr = []
    for fea in featureSelect:
        corr.append(abs(train[[fea, 'target']].fillna(0).corr().values[0][1]))

    # 取top300的特征进行建模，具体数量可选
    se = pd.Series(corr, index=featureSelect).sort_values(ascending=False)
    feature_select = ['card_id'] + se[:300].index.tolist()
    print('done')
    return train[feature_select + ['target']], test[feature_select]


def param_grid_search(train):
    """
    网格搜索参数寻优
    :param train:训练集
    :return:最优的分类器模型
    """
    print('param_grid_search')
    features = train.columns.tolist()
    features.remove("card_id")
    features.remove("target")
    parameter_space = {
        "n_estimators": [80],
        "min_samples_leaf": [30],
        "min_samples_split": [2],
        "max_depth": [9],
        "max_features": ["auto", 80]
    }

    print("Tuning hyper-parameters for mse")
    clf = RandomForestRegressor(
        criterion="mse",
        min_weight_fraction_leaf=0.,
        max_leaf_nodes=None,
        min_impurity_decrease=0.,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=4,
        random_state=2020,
        verbose=0,
        warm_start=False)
    grid = GridSearchCV(clf, parameter_space, cv=2, scoring="neg_mean_squared_error")
    grid.fit(train[features].values, train['target'].values)

    print("best_params_:")
    print(grid.best_params_)
    means = grid.cv_results_["mean_test_score"]
    stds = grid.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, grid.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    return grid.best_estimator_


def train_predict(train, test, best_clf):
    """
    进行训练和预测输出结果
    :param train:训练集
    :param test:测试集
    :param best_clf:最优的分类器模型
    :return:
    """
    print('train_predict...')
    features = train.columns.tolist()
    features.remove("card_id")
    features.remove("target")

    prediction_test = 0
    cv_score = []
    prediction_train = pd.Series()
    kf = KFold(n_splits=5, random_state=2020, shuffle=True)
    for train_part_index, eval_index in kf.split(train[features], train['target']):
        best_clf.fit(train[features].loc[train_part_index].values, train['target'].loc[train_part_index].values)
        prediction_test += best_clf.predict(test[features].values)
        eval_pre = best_clf.predict(train[features].loc[eval_index].values)
        score = np.sqrt(mean_squared_error(train['target'].loc[eval_index].values, eval_pre))
        cv_score.append(score)
        print(score)
        prediction_train = prediction_train.append(pd.Series(best_clf.predict(train[features].loc[eval_index]),
                                                             index=eval_index))
    print(cv_score, sum(cv_score) / 5)
    pd.Series(prediction_train.sort_index().values).to_csv("preprocess/train_randomforest.csv", index=False)
    pd.Series(prediction_test / 5).to_csv("preprocess/test_randomforest.csv", index=False)
    test['target'] = prediction_test / 5
    test[['card_id', 'target']].to_csv("result/submission_randomforest.csv", index=False)
    return


if __name__ == "__main__":

    # 获取训练集与测试集
    train, test = read_data(debug=False)

    # 获取特征选择结果
    train, test = feature_select_pearson(train, test)

    # 获取最优分类器模型
    best_clf = param_grid_search(train)

    # 获取结果
    train_predict(train, test, best_clf)
# [3.6952175995861753, 3.653405245049519, 3.711542672510601, 3.78859477721067, 3.586786511640954] 3.687109361199584


