#!/bin/sh
# 复赛数据预处理
echo --------------------final_competition_data--------------------------
python ./src/001_final_merge.py

# sparse
echo ---------------------------------sparse------------------------------------
python ./src/002_sparse_one.py
python ./src/002_sparse_one_select.py

python ./src/003_sparse_two.py
python ./src/003_sparse_two_select.py

# length
echo -------------------------------length------------------------------------
python ./src/004_length_ratio.py

# cvr
echo ---------------------------------cvr------------------------------------
python ./src/005_cvr.py
python ./src/005_cvr_select.py
python ./src/005_cvr_select2.py

# click
echo --------------------------------click------------------------------------
python ./src/006_click.py
python ./src/006_click_select.py

# ratio
echo --------------------------------ratio------------------------------------
python ./src/007_ratio.py
python ./src/007_ratio_select.py

# unique
echo -------------------------------unique------------------------------------
python ./src/008_unique.py
python ./src/008_unique_select.py

# CV_cvr
echo ------------------------------- CV_cvr------------------------------------
python ./src/009_CV_cvr.py
python ./src/009_CV_cvr_select.py
python ./src/009_CV_cvr_select2.py

# lightgbm 部分
echo --------------------------------lightgbm------------------------------------
python ./src/010_train_predict.py
python ./src/011_ronghe.py

# 加入初赛数据进行构造特征
echo ----------------------preliminary_competition_data--------------------------
python ./src/012_merge_part_p.py
python ./src/013_stat_p.py
python ./src/014_CV_cvr_p.py
python ./src/014_CV_cvr_select_p.py
python ./src/015_length_p.py
python ./src/016_sparse_p.py

# 融合得到lgb模型的最终结果
echo --------------------------------lightgbm------------------------------------
python ./src/017_train_predict_p.py
python ./src/018_rong_p.py

# nffm 部分
# nffm 复赛数据训练
echo --------------------------------nffm final------------------------------------
python ./src/nffm_final/001_prepared_data.py
python ./src/nffm_final/001_select_feat.py
python ./src/nffm_final/002_doFeature.py
python ./src/nffm_final/003_extract_features.py
python ./src/nffm_final/004_train_0.py
python ./src/nffm_final/004_train_1.py
python ./src/nffm_final/004_train_2.py

# nffm 初赛加复赛数据训练
echo --------------------------------nffm final preliminary------------------------------------
python ./src/nffm_final_preliminary/001_prepared_data.py
python ./src/nffm_final_preliminary/001_select_feat.py
python ./src/nffm_final_preliminary/002_doFeature.py
python ./src/nffm_final_preliminary/003_extract_features.py
python ./src/nffm_final_preliminary/nffm_train_763.py
python ./src/nffm_final_preliminary/nffm_train_765.py
python ./src/nffm_final_preliminary/nffm_train_7688.py

# 最终融合
echo ---------------------------------------blending--------------------------------------------
python ./src/019_ronghe_lgb_nffm.py