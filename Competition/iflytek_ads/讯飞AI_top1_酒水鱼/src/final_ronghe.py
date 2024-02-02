
# coding: utf-8

# In[ ]:


import pandas as pd
sub1 = pd.read_csv('result/liupeng_result.csv')  
sub2 = pd.read_csv('result/wengyp_result.csv') 
sub3 = pd.read_csv('result/wanghe_result.csv') 
df = pd.DataFrame()
co = 'predicted_score'
df['酒'] = sub1['predicted_score'].values
df['水'] = sub2['predicted_score'].values
df['鱼'] = sub3['predicted_score'].values
print(df.corr())
sub = sub1.copy()
sub[co] = (sub1[co]*5+sub2[co]*3+sub3[co]*2)/10
sub.to_csv('result/final_sub.csv',index=False)
print(sub[co].describe())

