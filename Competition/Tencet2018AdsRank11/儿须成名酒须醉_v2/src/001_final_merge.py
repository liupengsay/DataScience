import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import numpy as np
import time
import random
import warnings
warnings.filterwarnings("ignore")

##读取数据
print("Reading...")
ad_feature=pd.read_csv('final_competition_data/adFeature.csv')
if os.path.exists('final_competition_data/userFeature.csv'):
    user_feature=pd.read_csv('final_competition_data/userFeature.csv')
    print('User feature prepared')
else:
    userFeature_data = []
    with open('final_competition_data/userFeature.data', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            userFeature_data.append(userFeature_dict)
            if i % 1000000 == 0:
                print(i)
        user_feature = pd.DataFrame(userFeature_data)
        print('User feature...')
        user_feature.to_csv('final_competition_data/userFeature.csv', index=False)
        print('User feature prepared')
        
##插入test字段，0代表训练集，1代表测试集1，2代表测试集2
train_pre=pd.read_csv('final_competition_data/train.csv')

predict1=pd.read_csv('final_competition_data/test1.csv')
predict1['test'] = 1

predict2=pd.read_csv('final_competition_data/test2.csv')
predict2['test'] = 2
predict = pd.concat([predict1,predict2],axis=0,ignore_index=True)
train_pre.loc[train_pre['label']==-1,'label']=0
predict['label']=-1
data=pd.concat([train_pre,predict])

data['test'].fillna(value=0,inplace=True)

##关联数据
print("Merge...")
data=pd.merge(data,ad_feature,on='aid',how='left')
data=pd.merge(data,user_feature,on='uid',how='left')
user_feature = []
ad_feature = []
train_pre = []
predict = []
data=data.fillna('-1')
data = pd.DataFrame(data.values,columns=data.columns)
data['label'] = data['label'].astype(float)

##插入字段n_parts数据集进行分块，训练集分成五块1、2、3、4、5，测试集1为6、测试集2为7
##也就是test字段与n_parts字段都是为了区分数据块，n_parts对训练集进行了分块
print('N parts...')
train = data[data['test']==0][['aid','label']]
test1_index  = data[data['test']==1].index
test2_index  = data[data['test']==2].index
n_parts = 5
index = []
for i in range(n_parts):
    index.append([])
aid = list(train['aid'].drop_duplicates().values)
for adid in aid:
    dt = train[train['aid']==adid]
    for k in range(2):
        lis = list(dt[dt['label']==k].sample(frac=1,random_state=2018).index)
        cut = [0]
        for i in range(n_parts):
            cut.append(int((i+1)*len(lis)/n_parts)+1)
        for j in range(n_parts):
            index[j].extend(lis[cut[j]:cut[j+1]])
se = pd.Series()
for r in range(n_parts):
    se = se.append(pd.Series(r+1,index=index[r]))
se = se.append(pd.Series(6,index=test1_index)) 
se = se.append(pd.Series(7,index=test2_index)) 
data.insert(0,'n_parts',list(pd.Series(data.index).map(se).values))
train = []
del data['test']
##根据生成的分块字段n_parts划分训练与验证集(n_parts=1)
print('Index...')
train_part_index = list(data[(data['label']!=-1)&(data['n_parts']!=1)].index)
evals_index = list(data[(data['label']!=-1)&(data['n_parts']==1)].index)
test1_index  = list(data[data['n_parts']==6].index)
test2_index  = list(data[data['n_parts']==7].index)

data.loc[train_part_index]['label'].to_csv('data_preprocessing/train_part_y.csv',index=False)
data.loc[evals_index]['label'].to_csv('data_preprocessing/evals_y.csv',index=False)
data.loc[test1_index][['aid','uid']].to_csv('data_preprocessing/aid_uid_test1.csv',index=False)
data.loc[test2_index][['aid','uid']].to_csv('data_preprocessing/aid_uid_test2.csv',index=False)


part = data['n_parts']
se = pd.Series(part[(part<=5)&(part>=2)].values)
print('训练集总长度',len(se))
length = 0
for i in range(2,6,1):
    dt = pd.Series(se[se==i].index)
    dt.to_csv('train_index_'+str(i)+'.csv',index=False)
    length +=len(dt)
    print(i,len(dt))
print('所有分块总长度',length)


##保存文件
print('Saving...')
data.to_csv('data_preprocessing/train_test_merge.csv',index=False)
print('Over')
