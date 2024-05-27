import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os
import scipy.signal as signal
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier

#下記は決定木可視化のためのツール
import graphviz
#import pydotplus
from IPython.display import Image
from io import StringIO
#=====================================================
#関数
#=====================================================
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

def plotter(datas, title, columns_list):
    #加工後のデータの表示
    fig, axes = plt.subplots(6, 1, figsize=(8, 16)) 
    for i, ax in enumerate(axes):
        #ax.plot(np.mean(imu.normalized_datas, axis = 0)[:, i], label = columns_list[i])
        ax.plot(datas[:, i].T)
        ax.set_title(columns_list[i])
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig("results/figure/mean_{}.png".format(title))
    plt.close()

def dataframes2ndarray(data_frames, columns_list, num_dim):
    #取り合えず200フレームで統一残りはゼロパディング
    data_array = np.zeros((len(data_frames), 200, num_dim))
    for i, df in enumerate(data_frames):
        data = df[columns_list].to_numpy()
        data_len = data.shape[0]
        data_array[i, :data_len, :] = data
    return data_array

#=====================================================
#クラス
#=====================================================  

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
        self.normalized_datas = self.get_normalize_data(self.cliped_datas)

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
            if np.sum(norm >= threshold):
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
    
    def get_normalize_data(self, data):
        #最大値を求める
        max = np.max(data, axis=1, keepdims=True)
        min = np.min(data, axis=1, keepdims=True)
        #正規化
        normalize_datas = (data - min) / (max - min)
        return normalize_datas
    
    def get_filterd_data(self):
        return self.filterd_datas


class Classifier:
    def __init__(self, test_split_rasio = 0.2, cv = 10, columns_list = [], save_path = None):
        self.datas = None
        self.labels = None
        self.train_datas = None
        self.train_labels = None
        self.test_datas = None
        self.train_labels = None
        self.test_split_rasio = test_split_rasio
        self.cv = cv
        self.culumns_list = columns_list
        self.clossval_scores = {}
        self.save_path = save_path
        self.label_names = []

    def append_data_array(self, data, class_name):
        if not class_name in self.label_names:
            self.label_names.append(class_name)
        num_sample = data.shape[0]
        data = data.reshape(num_sample, -1) #平坦化
        labels = np.full(num_sample, class_name)
        train_datas, test_datas, train_labels, test_labels = train_test_split(data, labels, test_size = self.test_split_rasio)
        if np.any(self.train_datas) == None:
            self.datas = data
            self.labels = labels
            self.train_datas, self.test_datas = train_datas, test_datas
            self.train_labels, self.test_labels = train_labels, test_labels
        else:
            self.datas = np.concatenate((self.datas, data))
            self.labels = np.concatenate((self.labels, labels))
            self.train_datas = np.concatenate((self.train_datas, train_datas), axis=0)
            self.test_datas = np.concatenate((self.test_datas, test_datas), axis=0)
            self.train_labels = np.concatenate((self.train_labels, train_labels))
            self.test_labels = np.concatenate((self.test_labels, test_labels))

    def validate_decision_tree(self):
        model_name = "DecisionTree"
        clf = DecisionTreeClassifier()
        #交差検証
        self.cross_validate(clf, model_name)

        # クロスバリデーションを実行して各分割の重要度を取得
        cv_results = cross_validate(clf, self.datas, self.labels, cv=self.cv, return_estimator=True)
        importances = []
        for estimator in cv_results['estimator']:
            importances.append(estimator.feature_importances_)

        # 重要度の平均を計算
        feature = np.mean(importances, axis=0)
        clf = clf.fit(self.train_datas, self.train_labels)
        feature = np.array(feature).reshape(6, -1)
        feature = np.sum(feature, axis=1)
        plt.barh(range(len(feature)), feature)
        plt.yticks(range(len(feature)), self.culumns_list, fontsize=14)
        plt.savefig("{}/figure/decision_tree_mean_feature.png".format(self.save_path))

        #分類木の保存
        tree.plot_tree(clf)
        plt.savefig("{}/figure/decision_tree.png".format(self.save_path))
        plt.close()

    def validate_svm(self):
        model_name = "SVM"
        clf = SVC(kernel='linear')
        self.cross_validate(clf, model_name)

    def validate_random_forest(self):
        model_name = "random_forest"
        clf = RandomForestClassifier(n_estimators=100)
        self.cross_validate(clf, model_name)

    def validate_xgboost(self):
        model_name = "xgboost"
        clf = xgb.XGBClassifier(objective='multi:softmax', num_class=3)
        self.cross_validate(clf, model_name)

    def validate_mlp(self):
        model_name = "MLP"
        clf = MLPClassifier(hidden_layer_sizes=(400,400,300), max_iter=1000)
        self.cross_validate(clf, model_name)

    def cross_validate(self, clf, model_name):
        score = cross_val_score(clf, self.datas, self.labels, cv=self.cv)
        y_pred = cross_val_predict(clf, self.datas, self.labels, cv=self.cv)
        self.make_confusion_matrix(self.labels, y_pred, model_name=model_name)
        self.clossval_scores[model_name] = score.mean() 

    def make_confusion_matrix(self, y, y_pred, model_name):
        #print(y)
        labels = np.unique(y)
        #print(labels)
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_names, yticklabels=self.label_names)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix : {}'.format(model_name))
        plt.savefig("{}/figure/confusion_matrix/{}_confusion_matrix.png".format(self.save_path, model_name))
        plt.close()
