import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os
import scipy.signal as signal

def get_trial_csv_files(directory):
    # ディレクトリ内のすべてのファイル名を取得
    files = os.listdir(directory)
    # "trial"で始まるCSVファイルをフィルタリング
    trial_csv_files = fnmatch.filter(files, 'trial*.csv')
    # フルパスに変換
    trial_csv_files = [os.path.join(directory, file) for file in trial_csv_files]
    # 各CSVファイルをpandasで読み込む
    data_frames = [pd.read_csv(file) for file in trial_csv_files]
    return data_frames


class IMU_data:
    def __init__(self, name, datas, sampling = 60,  cutoff_freq = 20, threshold = 2, thred_rasio = 0.2, num_roll_back = 10):
        #datas (試行数, 時間軸, ６軸)のデータ
        self.name = name
        self.law_datas = datas
        self.num_atempt = datas.shape[0]
        self.num_axis =   datas.shape[2]
        self.cutoff = cutoff_freq
        self.sampling = sampling
        self.threshold = threshold
        self.thred_rasio = thred_rasio
        self.num_roll_back = num_roll_back

        #データの加工
        self.filterd_datas = self.filtering_data(self.law_datas)
        self.norm_datas    = self.culc_acc_norm(self.filterd_datas)
        self.cliped_datas  = self.clip_by_threshold(self.filterd_datas)

    def filtering_data(self, data):
        #3次のバターワースフィルター
        num_atempt = data.shape[0]
        num_sensor = data.shape[2]
        filtered_data = np.zeros_like(data)
        for i in range(num_atempt):
            for m in range(num_sensor):
                b, a = signal.butter(3, self.cutoff, btype="low", analog=False, fs = self.sampling)
                filtered_data[i, :, m] = signal.filtfilt(b, a, data[i, :, m])
        return filtered_data
    
    def culc_acc_norm(self, data):
        #加速度のノルムを計算
        accs = np.sqrt(np.sum(data[:, :, :3]**2, axis=2))
        return accs
    
    def clip_by_threshold(self, data):
        #加速度が閾値よりも早くなるデータインデックスを取得
        indices = []
        for norm in self.norm_datas:
            threshold = np.max(norm)*self.thred_rasio
            if np.sum(norm > threshold):
                #閾値よりも大きくなるインデックスを取得
                indices.append(np.where(norm > threshold)[0])
            else:
                indices.append(np.array([0]))
        
        cliped_datas = np.zeros((self.num_atempt, 40, self.num_axis))
        for i in range(self.num_atempt):
            frame = indices[i][0]
            #閾値を超えたフレームから少し巻き戻す
            frame = frame - self.num_roll_back if frame >= self.num_roll_back else 0
            cliped_datas[i, :, :] = data[i, frame:frame + 40, :]
        return cliped_datas
    
    def get_filterd_data(self):
        return self.filterd_datas



