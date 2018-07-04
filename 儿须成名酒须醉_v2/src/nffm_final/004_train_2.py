import numpy as np
import pandas as pd
import tensorflow as tf
import utils
import nffm
import os
os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]='0'


def create_hparams():
    return tf.contrib.training.HParams(
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
        idx=2,
        epoch=int(44628906//4096),
        mode='train',
        data_path='./ffm_data/',
        sub_name='sub',
        single_features=['aid','advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId',
                         'productType', 'age', 'gender','education', 'consumptionAbility', 'LBS', 'carrier', 'house',
                         
                         'aid_age', 'aid_gender', 'aid_LBS',
                        
                         'cvr_of_creativeId_and_onehot2', 'cvr_of_creativeId_and_onehot9',
                         'cvr_of_creativeId_and_onehot16','cvr_of_consumptionAbility_and_onehot1',
                         'cvr_of_creativeId_and_onehot10', 'cvr_of_creativeId_and_onehot15',
                         'cvr_of_creativeId_and_onehot13', 'cvr_of_creativeId_and_onehot18',
                         'cvr_of_age_and_onehot10', 'cvr_of_creativeId_and_onehot14',
                        
                         'cvr_of_aid_and_age', 'cvr_of_aid_and_gender', 
                         'cvr_of_uid', 'cvr_of_aid_and_consumptionAbility',
                         'cvr_of_aid_and_os', 'cvr_of_creativeSize_and_LBS', 
                         'cvr_of_aid_and_education', 'cvr_of_uid_and_creativeSize', 
                         'cvr_of_creativeSize', 'cvr_of_uid_and_adCategoryId',
                         
                         'ratio_click_of_aid_in_uid', 'ratio_click_of_creativeSize_in_uid',
                         'ratio_click_of_age_in_aid', 'ratio_click_of_age_in_creativeSize',
                         'ratio_click_of_gender_in_advertiserId',
                         'ratio_click_of_gender_in_creativeSize',
                         'ratio_click_of_consumptionAbility_in_aid',
                         'ratio_click_of_age_in_advertiserId',
                         'ratio_click_of_productType_in_uid',
                         'ratio_click_of_productType_in_consumptionAbility'],
        mutil_features=['interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2',
                        'topic3','appIdAction','appIdInstall','marriageStatus','ct','os'],
        )


hparams=create_hparams()
hparams.path='./model/'
utils.print_hparams(hparams)
       
hparams.aid=['aid','advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId', 'productType',
            
             'cvr_of_creativeId_and_onehot2', 'cvr_of_creativeId_and_onehot9',
             'cvr_of_creativeId_and_onehot16', 'cvr_of_creativeId_and_onehot10',
             'cvr_of_creativeId_and_onehot15', 'cvr_of_creativeId_and_onehot14',
             'cvr_of_creativeId_and_onehot13', 'cvr_of_creativeId_and_onehot18',
             
             'cvr_of_aid_and_age',
             'cvr_of_creativeSize', 'cvr_of_uid_and_adCategoryId',
             'cvr_of_uid_and_creativeSize',
            
             'ratio_click_of_productType_in_uid']        
hparams.user=['age', 'gender','education', 'consumptionAbility', 'LBS', 'carrier', 'house','interest1','interest2','interest3','interest4',
              'interest5','kw1','kw2','kw3','topic1','topic2','topic3','appIdAction','appIdInstall','marriageStatus','ct','os',
             
              'aid_age', 'aid_gender', 'aid_LBS',
              
              'cvr_of_consumptionAbility_and_onehot1',
              'cvr_of_age_and_onehot10',
             
              'cvr_of_uid'
              ] 
hparams.num_features=[]

preds=nffm.train(hparams)

test_df=pd.read_csv('../final_competition_data/test2.csv')
test_df['score']=preds
test_df['score']=test_df['score'].apply(lambda x:round(x,9))
test_df[['aid','uid','score']].to_csv('submission_nffm_75866_2.csv',index=False) 
