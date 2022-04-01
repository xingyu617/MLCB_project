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
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import itertools
from itertools import permutations
from itertools import repeat
from scipy.spatial import distance

datasets_small = ['BeetleFly', 'BirdChicken', 'Coffee', 'Computers', 'DistalPhalanxOutlineCorrect', 'Earthquakes', 'ECG200', 
                  'ECGFiveDays', 'GunPoint', 'Ham', 'Herring', 'Lightning2', 'MiddlePhalanxOutlineCorrect', 'ProximalPhalanxOutlineCorrect', 
                  'ShapeletSim', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'Strawberry', 'ToeSegmentation1', 'ToeSegmentation2', 'Wine', 
                  'WormsTwoClass', 'Chinatown', 'DodgerLoopGame', 'DodgerLoopWeekend', 'GunPointAgeSpan', 'GunPointMaleVersusFemale', 
                  'GunPointOldVersusYoung', 'HouseTwenty', 'PowerCons', 'SemgHandGenderCh2']
def min_distance(S,T):
    
    if S.all()==None or T.all()==None:
        return np.inf
    assert isinstance(S,np.ndarray)
    assert isinstance(T,np.ndarray) 
    assert isinstance(S[0],float)
    assert isinstance(T[0],float)
    
    if len(S)>len(T):
        aux_S=T
        T=S
        S=aux_S
    
    dist_list=[]
    m=len(T)
    w=len(S)
        
    for i in range(m-w+1):
        dist=np.linalg.norm(T[i:i+w]-S)
        dist_list.append(dist)
    return min(dist_list)

#K(T1,T2)=K_s((s11,...,s1U), (s21,...,s2V))=min_{i,j}( min_dist(s1i,s2j) )
def min_sub_distance(T1,T2,k=None):
    #T1,T2 should be lists of subsequences
    T1=subsequences1d(T1,k)
    T2=subsequences1d(T2,k)
    assert isinstance(T1,np.ndarray)
    assert isinstance(T2,np.ndarray) 
    assert isinstance(T1[0],np.ndarray)
    assert isinstance(T2[0],np.ndarray)
    assert isinstance(T1[0][0],float)
    assert isinstance(T2[0][0],float)
    
    dist_mat=distance.cdist(T1, T2, 'euclidean') 
    return np.min(dist_mat)   

def subsequences2d(T, k=None): 
    assert isinstance(T,np.ndarray)
    assert isinstance(T[0],np.ndarray)
    assert isinstance(T[0][0],float)
    
    m,n = T.shape
    if k==None:
        k=int(np.log(n+1))+1
    # INPUTS :
    # a is array
    # L is length of array along axis=1 to be cut for forming each subarray

    # Length of 3D output array along its axis=1
    #print(T.shape)
    nd0 = T.shape[1] - k + 1

    # Store shape and strides info
    s0,s1 = T.strides
    
    
    # Finally use strides to get the 3D array view
    return np.lib.stride_tricks.as_strided(T, shape=(m,nd0,k), strides=(s0,s1,s1))

def subsequences1d(arr, m=None):
    assert isinstance(arr,np.ndarray)
    assert isinstance(arr[0],float)
    
    if m==None:
        m=int(np.log(arr.shape[0]+1))+1
    n = arr.shape[0] - m + 1
    s = arr.itemsize
    #print(m,arr.shape[0])
    return np.lib.stride_tricks.as_strided(arr, shape=(n,m), strides=(s,s))    

def separate_dataset(train,test):
    labels=np.unique(test)
    X1=train[test==labels[0]]
    X2=train[test==labels[1]]
    return X1,X2

def intersection(X,Y,m=None,epsilon=None):
    
    if m==None:
        m=int(np.log(X.shape[1]+1))+1
    if epsilon==None:
        epsilon=(np.amax(X)-np.amin(X))*0.01*m
    
    #split dataset
    X1,X2=separate_dataset(X,Y)
    
    #extract subsequences
    
    S1=subsequences2d(X1)
    S2=subsequences2d(X2)
    S1=np.unique(S1.reshape(S1.shape[0]*S1.shape[1],S1.shape[2]),axis=0)
    S2=np.unique(S2.reshape(S2.shape[0]*S2.shape[1],S2.shape[2]),axis=0)
    
    
    #intesection: build pairwise distance matrix and find subsequences that don't meet conditions 
    dist_mat=distance.cdist(S1,S2, 'euclidean') 
    index=np.argwhere(dist_mat<=epsilon)
    index=index.T
    
    #UNION-INTERSECTION
    full_index1=np.arange(0,S1.shape[0])
    full_index2=np.arange(0,S2.shape[0])
    shapelet1=S1[list(set(full_index1)-set(index[0]))]
    shapelet2=S2[list(set(full_index2)-set(index[1]))]
    shapelet=np.concatenate((shapelet1,shapelet2), axis=0)
    
    return shapelet

def intersect_2d(A,B):
    return np.array([x for x in set(tuple(x) for x in A) & set(tuple(x) for x in B)])

def dist_vectorize(T,shapelets):
    # check shapelets is a 2d array
    assert isinstance(shapelets,np.ndarray)
    assert isinstance(shapelets[0],np.ndarray)
    assert isinstance(shapelets[0][0],float)
    
    # check T is a 1d array
    assert isinstance(T,np.ndarray)
    assert isinstance(T[0], float)
    
    #calculate the s-T distance vector
    u=map(min_distance,shapelets,repeat(T))
    return np.array(list(u))

def min_shapelet_distance(T1,T2,shapelets,k=None):
    
    assert isinstance(T1,np.ndarray)
    assert isinstance(T2,np.ndarray) 
    assert isinstance(T1[0],float)
    assert isinstance(T2[0],float)
    assert isinstance(shapelets,np.ndarray)
    assert isinstance(shapelets[0],np.ndarray)
    assert isinstance(shapelets[0][0],float)
    
    #calculate the distance vector where each element is min_dist between shapelet and T
    
    V1=dist_vectorize(T1,shapelets)
    V2=dist_vectorize(T2,shapelets)
    
    
    return np.linalg.norm(V1-V2, ord=2)   

#train and test all 42 datasets

for dataset in datasets_small:
    X_train, y_train, X_test, y_test = utilities.get_ucr_dataset('../UCRArchive_2018/',dataset)
    # calculate shapelets
    shapelets=intersection(X_train,y_train)
    #KNN
    clf = KNeighborsClassifier(n_neighbors=1,metric=min_shapelet_distance,metric_params={'shapelets':shapelets})
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    auroc=roc_auc_score(y_test, y_pred[:, 1])
    auprc=average_precision_score(y_test, y_pred[:,1])
    print(dataset, " AUROC is: ", auroc," AUPRC is: ", auprc)
    f=open('KNN_min_shapelet_dist.csv','a')
    np.savetxt(f, np.array([dataset, auroc,auprc]).reshape(1,3), delimiter=',',fmt="%s")  
    f.write("\n")
    f.close()       

       