环境配置: Anaconda 4.4.10, python 3.6.4


依赖库： pandas 0.22.0, numpy 1.14.2, scikit-learn 0.19.1, lightgbm 2.2.1, tensorflow 1.8.0


步骤说明： 先后完成预处理、特征工程、lgb模型训练、lgb模型融合、nffm模型训练、lgb和nffm预测结果进行融合


特征的使用与生成情况： 除原始特征稀疏矩阵与组合ID稀疏矩阵外，还采用了分块转化率、点击量、点击比例偏好、兴趣类特征长度、兴趣类特征组合转化率共五组统计特征，构造过程分别用cvr、click、ratio、length、CV_cvr表示。其中cvr统计方法为将训练集分成五份，其中一份作为验证集不加入统计计算，其余四份交叉统计。而CV_cvr则是利用ct\marriageStatus\interest字段编码生成稀疏矩阵，通过lightgbm模型的特征重要性排序提取前二十维转化为onehot编码，与其余字段进行组合转化率统计，统计方法同cvr。


模型结构：
使用了lightgbm和nffm
lgb方面，由于数据量的原因，复赛均提取20%的训练集数据来训练lgb模型，通过不同的特征组合构造多个lgb模型进行融合。
nffm方面，模型结构主要是复赛数全量据构造的nffm，初赛和复赛全量数据构造的nffm两种，两种nffm所使用的特征也是不用的。同时为了模型的鲁棒性，每种nffm通过不同的数据分布各训练出三个模型。
融合结构，一个融合后的lgb模型结合六个nffm模型进行加权融合。
(
  ( 
    (
      result['nffm_75866_1']*0.5 + result['nffm_75866_2']*0.5
    ) 
    *0.5 + result['nffm_75866_0']*0.5
  ) 
  *0.2 + result['nffm_7688']*0.4 + (result['nffm_763']*0.4 + result['nffm_765']*0.6)*0.4 
) 
*0.7 + result['lgb']*0.3

模型参数： 
-------lightgbm:
----lgb参数一:
LGBMClassifier(boosting_type='gbdt', num_leaves=40, max_depth=-1, learning_rate=0.1, n_estimators=10000, subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_alpha=6, reg_lambda=3, random_state=2018, n_jobs=-1, silent=True) LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7, learning_rate=0.1, max_depth=-1, min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0, n_estimators=10000, n_jobs=-1, num_leaves=110, objective=None, random_state=2018, reg_alpha=2, reg_lambda=3, silent=True, subsample=0.7, subsample_for_bin=200000, subsample_freq=1)
----lgb参数二:
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.8,
    learning_rate=0.1, max_depth=-1, min_child_samples=20,
    min_child_weight=0.001, min_split_gain=0.0, n_estimators=10000,
    n_jobs=-1, num_leaves=110, objective=None, random_state=None,
    reg_alpha=10, reg_lambda=0.0, silent=True, subsample=1.0,
    subsample_for_bin=200000, subsample_freq=1)

-------nffm:
----nffm参数一:
仅使用复赛数据，跑完所有epoch，对idx分别设置0，1，2，跑三个复赛数据的nffm
k=8,
batch_size=4096,
optimizer="adam",
learning_rate=0.0002,
num_display_steps=100,
num_eval_steps=1000,
l2=0.000002,
hidden_size=[128,128],
evl_batch_size=5000,
all_process=3,
idx=0,
epoch=int(44628906//4096),
----nffm参数二：
使用初赛加复赛数据，由于时间原因，未能跑完全部epoch，故对epoch分别设置8000，8000，10000，idx分别设置0，1，2，跑三个初赛加复赛数据的nffm
k=8,
batch_size=4096,
optimizer="adam",
learning_rate=0.0002,
num_display_steps=100,
num_eval_steps=2000,
l2=0.000002,
hidden_size=[128,128],
evl_batch_size=5000,
all_process=3,
idx=0, # 0，1，2
epoch=8000, # 8000，8000，10000 
