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
        num_eval_steps=2000,
        l2=0.000002,
        hidden_size=[128,128],
        evl_batch_size=5000,
        all_process=3,
        idx=0,
        epoch=8000,
        mode='train',
        data_path='./ffm_data/',
        sub_name='sub',
        single_features=['aid','advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId',
                         'productType', 'age', 'gender','education', 'consumptionAbility', 'LBS', 'carrier', 'house',
                         #across
                         'aid_age', 'aid_gender', 'aid_LBS',
                         #####################加载cvr数据##################
                         'cvr_of_aid', 'cvr_of_uid', 'cvr_of_advertiserId',
                         'cvr_of_creativeSize', 'cvr_of_productType', 'cvr_of_aid_and_age',
                         'cvr_of_aid_and_gender', 'cvr_of_aid_and_education',
                         'cvr_of_aid_and_consumptionAbility', 'cvr_of_aid_and_LBS',
                         #####################加载ratio数据##################
                         'ratio_click_of_aid_in_uid', 'ratio_click_of_advertiserId_in_uid',
                         'ratio_click_of_campaignId_in_uid',
                         'ratio_click_of_creativeSize_in_uid',
                         'ratio_click_of_adCategoryId_in_uid',
                         'ratio_click_of_productId_in_uid',
                         'ratio_click_of_productType_in_uid',
                         #####################加载click数据##################
                         'cnt_click_of_uid', 
                         'cnt_click_of_gender_and_aid', 'cnt_click_of_creativeSize_and_uid',
                         'cnt_click_of_adCategoryId_and_uid',
                         #####################加载length数据##################
                         'interest2_length', 'ratio_of_interest2', 'interest1_length',
                         'ratio_of_interest1', 'ct_length', 'marriageStatus_length',
                         'interests_length', 'ratio_of_interest5', 'interest5_length',
                         'kw2_length', 'ratio_of_kw1', 'ratio_of_interest4',
                         'topics_length'],
        mutil_features=['interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2',
                        'topic3','appIdAction','appIdInstall','marriageStatus','ct','os'],
        )


hparams=create_hparams()
hparams.path='./model/'
utils.print_hparams(hparams)
       
hparams.aid=['aid','advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId', 'productType',
             # cvr
             'cvr_of_aid','cvr_of_advertiserId',
             'cvr_of_creativeSize', 'cvr_of_productType', 
             # ratio
             'ratio_click_of_aid_in_uid', 'ratio_click_of_advertiserId_in_uid',
             'ratio_click_of_campaignId_in_uid',
             'ratio_click_of_creativeSize_in_uid',
             'ratio_click_of_adCategoryId_in_uid',
             'ratio_click_of_productId_in_uid',
             'ratio_click_of_productType_in_uid',
             # else
             'cnt_click_of_creativeSize_and_uid',
             'cnt_click_of_adCategoryId_and_uid',]        
hparams.user=['age', 'gender','education', 'consumptionAbility', 'LBS', 'carrier', 'house','interest1','interest2','interest3','interest4',
              'interest5','kw1','kw2','kw3','topic1','topic2','topic3','appIdAction','appIdInstall','marriageStatus','ct','os',
             'cvr_of_uid',
             # length
             'interest2_length', 'ratio_of_interest2', 'interest1_length',
             'ratio_of_interest1', 'ct_length', 'marriageStatus_length',
             'interests_length', 'ratio_of_interest5', 'interest5_length',
             'kw2_length', 'ratio_of_kw1', 'ratio_of_interest4',
             'topics_length',
             # click
             'cnt_click_of_uid'
              ] 
hparams.num_features=[]

preds= nffm.train(hparams)

test_df=pd.read_csv('../final_competition_data/test2.csv')
test_df['score']=preds
test_df['score']=test_df['score'].apply(lambda x:round(x,9))
test_df[['aid','uid','score']].to_csv('submission_nffm_763.csv',index=False) 