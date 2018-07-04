import pandas as pd
from sklearn.metrics import roc_auc_score
evals_y = pd.read_csv('data_preprocessing/evals_y.csv',header=None)[0].values
print('evals_ypre...')
evals_ypre = pd.DataFrame()
for i in range(1,17):
    col_name = 'model'+str(i)
    evals_ypre[col_name] = pd.read_csv('data_preprocessing/evals_ypre_'+str(i)+'.csv',header=None)[0].values
print('ytest...')
ytest = pd.DataFrame()
for i in range(1,17):
    col_name = 'model'+str(i)
    ytest[col_name] = pd.read_csv('data_preprocessing/submission2_'+str(i)+'.csv')['score'].values

def searchBest(y1,y2):
    se = pd.Series()
    for i in range(0,102,2):
        se = se.append(pd.Series(roc_auc_score(evals_y,(i*y1+(100-i)*y2)/100),index=[i]))
    return se
ind = []
for co in evals_ypre.columns:
    ind.append(int(roc_auc_score(evals_y,evals_ypre[co].values)*1000000))
col_sort_descend = list(pd.Series(evals_ypre.columns,index=ind).sort_index(ascending=False).values)

auc = [(pd.Series(evals_ypre.columns,index=ind).sort_index(ascending=False).index[0]).astype(int)/1000000]
import time
num = 0
for co in col_sort_descend:
    if num==0:
        evals_ypre_ronghe = evals_ypre[co].values
        ytest_ronghe = ytest[co].values
        print(num+1,auc[0])
        print('\n')
        del ytest[co]
        del evals_ypre[co]
    else:
        s = time.time()
        print(num+1)
        se = searchBest(evals_ypre_ronghe,evals_ypre[co].values)
        print(se.sort_values(ascending=False).head(1))
        auc.append(se.sort_values(ascending=False).values[0])
        k = se.sort_values(ascending=False).index[0]
        evals_ypre_ronghe = (evals_ypre_ronghe*k+evals_ypre[co].values*(100-k))/100
        ytest_ronghe = (ytest_ronghe*k+ytest[co].values*(100-k))/100
        print(evals_ypre_ronghe.mean())
        print(ytest_ronghe.mean())
        print(roc_auc_score(evals_y,evals_ypre_ronghe))
        print(int(time.time()-s),"s")
        print('\n')
        del ytest[co]
        del evals_ypre[co]
    num+=1
pd.Series(evals_ypre_ronghe).to_csv('data_preprocessing/evals_ypre.csv',index=False)
sub = pd.read_csv('data_preprocessing/aid_uid_test2.csv')
print(sub.head())
sub['score'] = round(pd.Series(ytest_ronghe),6).values
print(sub['score'].describe())
sub.to_csv('data_preprocessing/submission2.csv',index=False)
print(sub.head())
auc = pd.Series(auc)
auc.index = auc.index+1
print(auc)