import pandas as pd
result = pd.DataFrame()
result['nffm_75866_0'] = pd.read_csv('./nffm_final/submission_nffm_75866_0.csv')['score']
result['nffm_75866_1'] = pd.read_csv('./nffm_final/submission_nffm_75866_1.csv')['score']
result['nffm_75866_2'] = pd.read_csv('./nffm_final/submission_nffm_75866_2.csv')['score']
result['nffm_763'] = pd.read_csv('./nffm_final_preliminary/submission_nffm_763.csv')['score']
result['nffm_765'] = pd.read_csv('./nffm_final_preliminary/submission_nffm_765.csv')['score']
result['nffm_7688'] = pd.read_csv('./nffm_final_preliminary/submission_nffm_7688.csv')['score']
result['lgb'] = pd.read_csv('data_preprocessing/submission2_p.csv')['score']
a = 0.7
sub = pd.read_csv('./nffm_final/submission_nffm_75866_0.csv')
sub['score'] = round((((result['nffm_75866_1']*0.5+result['nffm_75866_2']*0.5)*0.5+
                     result['nffm_75866_0']*0.5)*0.2+result['nffm_7688']*0.4
        +(result['nffm_763']*0.4+result['nffm_765']*0.6)*0.4)*a+result['lgb']*(1-a),6)
print(sub['score'].describe())
sub.to_csv('../submission.csv',index=False)
print('Over')