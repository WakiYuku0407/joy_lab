import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import *

num_dim = 6 #&軸データ
columns_list = ["accel x", "accel y", "accel z", "pitch", "roll", "yaw"]
subject_names = ['waki2', 'ishii', 'takami']
punch_types = ['s', 'u', 'f']
punch_type2name = {'s' : 'straight', 'u' : 'uppercut', 'f': 'hook'}
#dir_names = ['waki2s', 'waki2u', 'waki2f']

#フォルダ名の作成
dir_names = []
for punch_type in punch_types:
    for subject_name in subject_names:   
        dir_names.append(subject_name+punch_type)

#実験データのcsvを読み込み
datas = {}
for dir_name in dir_names:
    directory_path = "data/" + dir_name
    datas[dir_name] = get_trial_csv_files(directory_path)

#データを取得し加工、分類器インスタンスへの追加
classifier = Classifier(cv=5, columns_list = columns_list, save_path="results") #分類器の定義
for name, data_frames in datas.items():
    #dataframesをndarrayに変換
    data_array = dataframes2ndarray(data_frames, columns_list, num_dim)
    #データの加工
    imu = IMU_data(name, data_array,cutoff_freq=25, thred_rasio=0.3, num_roll_back= 5)
    #データのプロット
    plotter(np.mean(imu.cliped_datas, axis=0), name, columns_list)
    #分類器にデータの追加
    punch_type = name[-1]
    label = punch_type2name[punch_type]
    classifier.append_data_array(imu.normalized_datas, label)

#各分類手法でのクロスバリデーションの実施
classifier.validate_decision_tree()
classifier.validate_svm()
classifier.validate_random_forest()
#classifier.validate_xgboost()
classifier.validate_mlp()
#結果を表示
print(classifier.clossval_scores)


