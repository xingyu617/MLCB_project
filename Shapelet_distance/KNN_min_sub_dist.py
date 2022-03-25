import logging
logging.basicConfig(level=logging.DEBUG)
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import utilities
import os
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import paired_distances
import numpy as np
import itertools
from itertools import permutations

datasets_small = ['BeetleFly', 'BirdChicken', 'Coffee', 'Computers', 'DistalPhalanxOutlineCorrect', 'Earthquakes', 'ECG200', 
                  'ECGFiveDays', 'GunPoint', 'Ham', 'Herring', 'Lightning2', 'MiddlePhalanxOutlineCorrect', 'ProximalPhalanxOutlineCorrect', 
                  'ShapeletSim', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'Strawberry', 'ToeSegmentation1', 'ToeSegmentation2', 'Wine', 
                  'WormsTwoClass', 'Chinatown', 'DodgerLoopGame', 'DodgerLoopWeekend', 'GunPointAgeSpan', 'GunPointMaleVersusFemale', 
                  'GunPointOldVersusYoung', 'HouseTwenty', 'PowerCons', 'SemgHandGenderCh2']
#K(T1,T2)=K_s((s11,...,s1U), (s21,...,s2V))=min_{i,j}( min_dist(s1i,s2j) )
def pairwise_min_shapelet(T1,T2,k=None):
    #T1,T2 should be lists of subsequences
    T1=subsequences1d(T1,k)
    T2=subsequences1d(T2,k)
    assert isinstance(T1,np.ndarray)
    assert isinstance(T2,np.ndarray) 
    assert isinstance(T1[0],np.ndarray)
    assert isinstance(T2[0],np.ndarray)
    assert isinstance(T1[0][0],float)
    assert isinstance(T2[0][0],float)
    
    c = list(itertools.product(T1, T2))
    c=np.array(c)
    dist_list=[]
    dist_list = [np.linalg.norm(grp[0]-grp[1]) for grp in c]
   
    return min(dist_list)   



def subsequences1d(arr, m):
    assert isinstance(arr,np.ndarray)
    assert isinstance(arr[0],float)
    if m==None:
        m=int(np.log(arr.shape[0]+1))+1
    n = arr.shape[0] - m + 1
    s = arr.itemsize
    return np.lib.stride_tricks.as_strided(arr, shape=(n,m), strides=(s,s))    
#train and test all small datasets

for dataset in datasets_small:
    dataset_list=[]
    auroc_list=[]
    auprc_list=[]
    X_train, y_train, X_test, y_test = utilities.get_ucr_dataset('../UCRArchive_2018/',dataset)
    #clf = GridSearchCV(KNeighborsClassifier(metric=pairwise_min_shapelet),parameters, cv=2, verbose=1)
    clf = KNeighborsClassifier(n_neighbors=1,metric=pairwise_min_shapelet)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict_proba(X_test)
    auroc=roc_auc_score(y_test, y_pred[:, 1])
    auprc=average_precision_score(y_test, y_pred[:,1])
    print(dataset, " AUROC is: ", auroc," AUPRC is: ", auprc)
    dataset_list.append(dataset)
    auroc_list.append(auroc)
    auprc_list.append(auprc)
    title='KNN_min_sub_dist_'+dataset+'.csv'
    np.savetxt(title, [p for p in zip(dataset_list, auroc_list,auprc_list)], delimiter=',',fmt="%s")   
    