import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import *

num_dim = 6 #６軸データ
columns_list = ["accel x", "accel y", "accel z", "pitch", "roll", "yaw"]
dir_names = ['waki2s', 'waki2u', 'waki2f']

#実験データの読み込み
datas = {}
for dir_name in dir_names:
    directory_path = "data/" + dir_name
    datas[dir_name] = get_trial_csv_files(directory_path)


for name, data_frames in datas.items():
    #取り合えず100フレームで統一残りはゼロパディング
    data_array = np.zeros((len(data_frames), 120, num_dim))
    for i, df in enumerate(data_frames):
        data = df[columns_list].to_numpy()
        data_len = data.shape[0]
        data_array[i, :data_len, :] = data
        
    #データの加工
    imu = IMU_data(name, data_array, thred_rasio=0.2, num_roll_back= 10)
    plt.plot(np.mean(imu.cliped_datas, axis = 0))
    #plt.show()





