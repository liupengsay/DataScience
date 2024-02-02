#!/bin/sh
# 三套代码分别构造不同特征，不同参数
echo --------------------liupeng_demo--------------------------
python ./src/liupeng_demo.py
echo --------------------wengyp_demo--------------------------
python ./src/wengyp_demo.py
echo --------------------wanghe_demo--------------------------
python ./src/wanghe_demo.py
# 三个结果进行加权平均
echo --------------------final_ronghe--------------------------
python ./src/final_ronghe.py