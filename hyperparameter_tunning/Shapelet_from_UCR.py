import numpy as np
import pickle
from scipy.stats import chi2_contingency
from glob import glob
import pandas as pd
import os
import sys
import math
import torch


def old_distance(s, t):
    if len(s) > len(t):
        s, t = t, s

    n = len(s)
    m = len(t)

    d = sys.float_info.max
    for i in range(0, m - n + 1):
        S = s
        T = t[i:i + n]

        d = min(d, math.sqrt(sum((a - b)**2 for a, b in zip(S, T))))

    return d/(2*np.sqrt(len(t)))


def extract_candidates(data, m, M):
    shapelets = []
    for series in data:
        n = len(series)
        print(m,M,n)
        for l in range(m, M + 1):
            for i in range(0, n - l + 1):
                shapelet = series[i:i + l]

                shapelets.append(shapelet)

    return shapelets


def confusion_matrix_fast(y_true, y_pred, mat):
    for elem in [0, 1]:
        idx = (y_true == elem)
        # Determine if it's TP or TN
        if elem > 0:
            # TP
            mat[0, 0] = sum(y_pred[idx] == elem) + 1
            # FN
            mat[1, 0] = sum(y_pred[idx] != elem) + 1
        else:
            # TN
            mat[1, 1] = sum(y_pred[idx] == elem) + 1
            # FP
            mat[0, 1] = sum(y_pred[idx] != elem) + 1
    return mat.astype(int)



import numpy as np
from multiprocessing import Pool, Manager


import time


def get_p(y_true, distances, thresholds):
    best_p = 1
    best_t = 0
    for threshold in thresholds:
        y_pred = (distances < threshold).astype(int)
        mat = np.ones((2, 2))
        p_val = chi2_contingency(confusion_matrix_fast(y_true, y_pred, mat))[1]
        if p_val < best_p:
            best_p = p_val
            best_t = threshold
    return best_p, best_t


def from_data_p(data, y_true, candidate, dist):
    distances = np.asarray(range(len(data))).astype(np.float32)
    for line, ts in enumerate(data):
        distances[line] = dist(ts, candidate)
    #all_dist = np.log10(np.linspace(1.**0, 10.1**(np.max(distances)), num=100))
    all_dist = np.unique(distances)  # np.array_split(np.unique(distances), processes)
    # after_unique = time.time()

    r_s = sum(y_true)

    best_p = 1
    best_t = 0
    for threshold in all_dist:
        y_pred = (distances < threshold).astype(int)
        n_1 = sum(y_pred)
        n_0 = len(y_pred) - n_1
        mat = np.ones((2, 2))
        mat = confusion_matrix_fast(y_true, y_pred, mat)
        p_val = chi2_contingency(mat)[1]
        if p_val < best_p:
            best_p = p_val
            best_t = threshold
    return best_p


def load_ucr(dir = './UCRArchive_2018/ShapeletSim'):
    files = glob(dir + '/*.tsv')
    print(files)
    full_train_data = pd.read_csv(files[1], header=None, sep='\t')
    full_test_data = pd.read_csv(files[0], header=None, sep='\t')

    train_dataset = full_train_data.iloc[:, range(1, full_train_data.shape[1])]  # traindatawithoutlabel
    label_train = full_train_data.iloc[:, 0]  # trainlabel

    test_dataset = full_test_data.iloc[:, range(1, full_test_data.shape[1])]  # testdatawithoutlabel
    label_test = full_test_data.iloc[:, 0]  # testlabel

    train_dataset = train_dataset.values
    label_train = label_train.values

    test_dataset = test_dataset.values
    label_test = label_test.values
    return train_dataset, label_train, test_dataset, label_test



def Shapelet_random(data, y_true, dist=old_distance, length=2,number_search=200, top_number=10, seed=31):#42
    y_true = np.asarray(y_true)

    lengths = []
    for ts in data:
        lengths.append(len(ts))
    #m = int(np.log(max(lengths))) + 1
    #M = int(np.log(max(lengths))) + 1
    m=length
    M=length
    print('shaplet_random',m,M)
    candidates = np.asarray(extract_candidates(data, m, M))
    print('candidates',candidates)
    r = np.random.RandomState(seed)
    idx_candidates = r.choice(range(len(candidates)), size=number_search)
    candidates = candidates[idx_candidates]

    p_vals = []
    while len(p_vals) < len(candidates):
        p_vals.append(1.)

    pool = Pool(processes=5)
    for candidate_i in range(number_search):
        candidate = candidates[candidate_i]
        result = pool.apply_async(from_data_p, [data, y_true, candidate, dist])
        p = result.get()
        p_vals[candidate_i] = p
    pool.close()

    indexes = range(len(p_vals))
    indexes = sorted(indexes, key=p_vals.__getitem__)
    p_vals = list(map(p_vals.__getitem__, indexes))
    candidates = list(map(candidates.__getitem__, indexes))


    return candidates[:top_number]


if __name__ == "__main__":
    #loads shapeletsim
    #put path to folder in argument for other datasets
    
    train_dataset, label_train, test_dataset, label_test = load_ucr()
    #Run ultra fast method top 10 shapelets
    shapelets = Shapelet_random(train_dataset, label_train)
    print(shapelets)
    #MAKE SURE TO GET SAME RESULT!!!
    #[array([0.91964757, 1.0954732 , 1.2712988 , 1.4471244 , 1.62295   , 1.7987756 , 1.62295   ]),
    #array([0.34915086, 0.52115057, 0.69315028, 0.86515   , 1.0371497 , 1.2091494 , 1.3811491 ]),
    #array([-0.52825751, -0.70076981, -0.87328212, -1.0457944 , -1.2183067 , -1.390819  , -1.5633314 ]),
    #array([0.29288519, 0.46823758, 0.64358997, 0.81894236, 0.99429476, 1.1696471 , 1.3449995 ]),
    #array([0.55560277, 0.81575785, 0.26979255, 0.18415007, 1.1064722 , 1.7507941 , 1.5381662 ]),
    #array([-0.52976422, -1.7148457 , -1.542846  , -1.3708462 , -1.1988465 , -1.0268468 , -0.85484712]),
    #array([ 0.13018051, -1.0953535 , -0.32684645, -1.142311  , -0.97074255, -1.4543239 , -1.4940207 ]),
    #array([-0.20655456,  0.21298111,  0.08323205,  0.11157481, -0.11991579, -1.3809006 , -0.56146972]),
    #array([-0.53034245, -1.3449982 , -1.2572928 , -1.243106  ,  1.1349048 ,  -1.4709271 , -1.141203  ]),
    #array([-0.30212465,  0.77747803,  0.09334634,  0.77370372,  1.1821334 , 1.2559379 ,  0.96367283])]

