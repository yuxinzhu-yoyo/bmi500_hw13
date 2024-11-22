import sys
import os
import numpy as np
from collections import defaultdict
from itertools import product
from pandas import Series
import warnings
warnings.filterwarnings("ignore")
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def preprocess(data, kernel_size):
    proc_data = np.zeros(data.shape)
    ''' preprocess '''
    for ch in range(data.shape[1]):
        kps_seq_ch = data[:, ch]
        kps_seq_ch = Series(kps_seq_ch).rolling(kernel_size, min_periods=1, center=True).mean().to_numpy()
        proc_data[:, ch] = kps_seq_ch
    return proc_data

def segmentation(data, win_size):
    ''' Sliding window parameters '''
    win_len = int(30*win_size) # 1 sec x 30 Hz
    win_step = int(30*0.5) # 0.5 sec x 30 Hz
    sample_windows = []
    for start_time in range(0, data.shape[0], win_step):
        end_time = start_time + win_len
        if end_time > data.shape[0]:
            end_time = data.shape[0]
            start_time = end_time - win_len
        frame = data[start_time:end_time]
        assert frame.shape[0] == win_len, (start_time, end_time, data.shape[0])
        sample_windows.append(frame)
    sample_windows = np.array(sample_windows)
    return sample_windows

def feature_extraction(sample_windows):
    ''' extract mean and std from each frame'''
    N, T, D = sample_windows.shape
    feats = []
    for i in range(N):
        frame = sample_windows[i]
        feat = []
        for ch in range(D):
            frame_ch = frame[:,ch]
            # mean feature
            mean_ch = np.mean(frame_ch)
            feat.append(mean_ch)
            # std feature
            std_ch = np.std(frame_ch)
            feat.append(std_ch)
            # min feature
            min_ch = np.min(frame_ch)
            feat.append(min_ch)
            # max feature
            max_ch = np.max(frame_ch)
            feat.append(max_ch)
        feats.append(feat)
    feats = np.array(feats)
    return feats

file_names = os.listdir('pose')
dict = defaultdict(list)
for file_name in file_names:
    subject_number = int(file_name[5:7])
    # creating file directory
    directory = 'pose/' + file_name
    label_act = int(file_name[1:3]) - 1
    # data load
    data3D = np.load(directory)
    # 3D to 2D
    data = data3D.reshape(data3D.shape[0], -1)
    kernel = 20
    data_prep = preprocess(data, kernel)
    win_len = 1.5
    data_seg = segmentation(data_prep, win_len)
    N = data_seg.shape[0]
    features = feature_extraction(data_seg)
    # storing features
    dict[subject_number].append((features, [label_act] * N))


trainx_list = []
train_labels = []

for i in range(1, 6):
    for j in range(32):
        trainx_list.append(dict[i][j][0])
        train_labels.append(dict[i][j][1])
        
trainx = np.vstack(trainx_list)
trainy = np.hstack(train_labels)

valx_list = []
val_labels = []

for i in range(6, 8):
    for j in range(32):
        valx_list.append(dict[i][j][0])
        val_labels.append(dict[i][j][1])
        
valx = np.vstack(valx_list)
valy = np.hstack(val_labels)

testx_list = []
test_labels = []
for i in range(8, 11):
    for j in range(32):
        testx_list.append(dict[i][j][0])
        test_labels.append(dict[i][j][1])
        
testx = np.vstack(testx_list)
testy = np.hstack(test_labels)

# hyperparameter grid
param_grid = {
    'hidden_layer_sizes': [(100, 50), (40, 20), (20, )],   
    'alpha': [0.001, 0.01], 
    'max_iter': [100, 200], 
    'solver': ['adam', 'sgd'], 
    'learning_rate': ['adaptive']
}

best_score = 0
best_params = {}

for params in product(*param_grid.values()):
    param_dict = {key: value for key, value in zip(param_grid.keys(), params)}
    print(params)
    
    model = MLPClassifier(**param_dict)
    model.fit(trainx, trainy)
    
    val_predictions = model.predict(valx)
    val_accuracy = accuracy_score(valy, val_predictions)
    if val_accuracy > best_score:
        best_score = val_accuracy
        best_params = param_dict

# reporting
best_model = MLPClassifier(**best_params)
trainx_all = np.vstack((trainx, valx))
trainy_all = np.hstack((trainy, valy))
best_model.fit(trainx_all, trainy_all)

test_predictions = best_model.predict(testx)
test_accuracy = accuracy_score(testy, test_predictions)

print("Best Parameters:", best_params)
print("Validation Set Accuracy with Best Parameters:", best_score)
print("Test Set Accuracy with Best Parameters:", test_accuracy)

