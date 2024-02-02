

catboost_info                       catboost模型相关过程记录文件夹
data                                      原始数据
preprocess                           采样的中间数据与特征文件
result                                    存放模型各种结果


EDA.ipynb                             数据探索性分析并对数据做一定的预处理

FeatureEngineering.ipynb    特征工程

baseline_lightgbm.py           采用lightgbm模型跑出baseline
promotion_catboost.py       采用catboost模型进阶提升
best_xgboost.py                  用xgboost抛出的最好单模型

model_stacking.py               将上述三个单模型的结果进行stacking融合建模

model_fusion.ipynb              进行三个但模型的加权结果融合
