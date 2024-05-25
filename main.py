import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import *

num_dim = 6 #６軸データ
columns_list = ["accel x", "accel y", "accel z", "pitch", "roll", "yaw"]
dir_names = ['waki2s', 'waki2u', 'waki2f']

#実験データのcsvを読み込み
datas = {}
for dir_name in dir_names:
    directory_path = "data/" + dir_name
    datas[dir_name] = get_trial_csv_files(directory_path)

classifier = Classifier(cv=10, columns_list = columns_list, save_path="results") #分類器の定義
#データを取得し加工、分類器への追加
for name, data_frames in datas.items():
    #取り合えず100フレームで統一残りはゼロパディング
    data_array = np.zeros((len(data_frames), 120, num_dim))
    for i, df in enumerate(data_frames):
        data = df[columns_list].to_numpy()
        data_len = data.shape[0]
        data_array[i, :data_len, :] = data
    #データの加工
    imu = IMU_data(name, data_array,cutoff_freq=25, thred_rasio=0.3, num_roll_back= 5)
    #加工後のデータの表示
    fig, axes = plt.subplots(6, 1, figsize=(8, 16)) 
    for i, ax in enumerate(axes):
        #ax.plot(np.mean(imu.normalized_datas, axis = 0)[:, i], label = columns_list[i])
        ax.plot(imu.cliped_datas[:, :,  i].T)
        ax.set_title(columns_list[i])
    plt.suptitle(name)
    plt.tight_layout()
    plt.savefig("results/figure/mean_{}.png".format(name))
    plt.close()
    #分類器にデータの追加
    classifier.append_data_array(imu.normalized_datas, name)

#決定木での分類
classifier.fit_decision_tree()


print(classifier.clossval_scores)


