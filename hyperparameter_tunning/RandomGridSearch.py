import logging
logging.basicConfig(level=logging.DEBUG)
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import utilities
import os
import sys
import Shapelet_from_UCR

datasets_binary = ['BeetleFly','BirdChicken','Coffee','Computers','DistalPhalanxOutlineCorrect','Earthquakes','ECG200',
                   'ECGFiveDays','FordA','FordB','GunPoint','Ham','HandOutlines','Herring','ItalyPowerDemand','Lightning2',
                   'MiddlePhalanxOutlineCorrect', 'MoteStrain','PhalangesOutlinesCorrect','ProximalPhalanxOutlineCorrect',
                   'ShapeletSim','SonyAIBORobotSurface1','SonyAIBORobotSurface2','Strawberry','ToeSegmentation1','ToeSegmentation2',
                   'TwoLeadECG','Wafer','Wine','WormsTwoClass','Yoga','Chinatown','DodgerLoopGame','DodgerLoopWeekend',
                   'FreezerRegularTrain','FreezerSmallTrain','GunPointAgeSpan','GunPointMaleVersusFemale','GunPointOldVersusYoung',
                   'HouseTwenty','PowerCons','SemgHandGenderCh2']
datasets_small = ['BeetleFly', 'BirdChicken', 'Coffee', 'Computers', 'DistalPhalanxOutlineCorrect', 'Earthquakes', 'ECG200', 
                  'ECGFiveDays', 'GunPoint', 'Ham', 'Herring', 'Lightning2', 'MiddlePhalanxOutlineCorrect', 'ProximalPhalanxOutlineCorrect', 
                  'ShapeletSim', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'Strawberry', 'ToeSegmentation1', 'ToeSegmentation2', 'Wine', 
                  'WormsTwoClass', 'Chinatown', 'DodgerLoopGame', 'DodgerLoopWeekend', 'GunPointAgeSpan', 'GunPointMaleVersusFemale', 
                  'GunPointOldVersusYoung', 'HouseTwenty', 'PowerCons', 'SemgHandGenderCh2']

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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone
from sklearn.metrics import make_scorer, accuracy_score
from scipy.stats import uniform
import itertools
import random

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
def dist_vectorize(T,shapelets):
    # check shapelets is a 2d array
    assert isinstance(shapelets,np.ndarray)
    assert isinstance(shapelets[0],np.ndarray)
    #assert isinstance(shapelets[0][0],float)
    
    # check T is a 1d array
    assert isinstance(T,np.ndarray)
    assert isinstance(T[0], float)
    
    dist_list=[]
    m=len(T)
    w=len(shapelets[0])
    subsequences=subsequences1d(T,w) 
    #print(subsequences)
    dist_mat=distance.cdist(shapelets,subsequences)
    
    return dist_mat.min(axis=1)


def subsequences1d(arr, m=None):
    assert isinstance(arr,np.ndarray)
    assert isinstance(arr[0],float)
    
    if m==None:
        m=int(np.log(arr.shape[0]+1))+1
    n = arr.shape[0] - m + 1
    s = arr.itemsize
    #print(m,arr.shape[0])
    return np.lib.stride_tricks.as_strided(arr, shape=(n,m), strides=(s,s))    
def min_sub_distance(T1,T2,k=None):
    #T1,T2 should be lists of subsequences
    S1=subsequences1d(T1,k)
    S2=subsequences1d(T2,k)
    assert isinstance(S1,np.ndarray)
    assert isinstance(S2,np.ndarray) 
    assert isinstance(S1[0],np.ndarray)
    assert isinstance(S2[0],np.ndarray)
    assert isinstance(S1[0][0],float)
    assert isinstance(S2[0][0],float)
    
    #union
    S=np.unique(np.concatenate((S1,S2), axis=0),axis=0)
    
   
    v1=dist_vectorize(T1,S)
    v2=dist_vectorize(T2,S)
    
    return np.linalg.norm(v1-v2, ord=2)  
def min_shapelet_distance(T1,T2,shapelets,k=None):
    
    assert isinstance(T1,np.ndarray)
    assert isinstance(T2,np.ndarray) 
    assert isinstance(T1[0],float)
    assert isinstance(T2[0],float)
    assert isinstance(shapelets,np.ndarray)
    assert isinstance(shapelets[0],np.ndarray)
    #assert isinstance(shapelets[0][0],float)
    
    #calculate the distance vector where each element is min_dist between shapelet and T
    
    V1=dist_vectorize(T1,shapelets)
    V2=dist_vectorize(T2,shapelets)
    
    
    return np.linalg.norm(V1-V2, ord=2)    
def customized_random_search_cv(train_dataset,train_label, param_grid,shapelet_grid, X, y, cv=5,iteration=10):
    # Stratified K-fold
    cv = StratifiedKFold(n_splits=cv, shuffle=False)
    results = []
    #combine parameters at random: length & number & K(neighbour)
    w=shapelet_grid['length']
    n=shapelet_grid['number']
    k=param_grid['K']
    C=random.sample(list(itertools.product(w,n,k)),iteration)
    #C=list(itertools.product(w,n,k))
    print(C)
    for train_index, test_index in cv.split(X,y):
        split_results = []
        params = [] 
        for idx, comb in enumerate(C):
            print('comb:',comb)
            #####Should the shapelets be extracted from entire datasets or splitted training sets?#####
            shapelets = Shapelet_from_UCR.Shapelet_random(train_dataset[train_index],train_label[train_index],length=int(comb[0]),number_search=1000,top_number=comb[1])
            shapelets=np.asarray(shapelets)
            print('extracted shapelets')
            clf = KNeighborsClassifier(n_neighbors=comb[2],metric=min_shapelet_distance,metric_params={'shapelets':shapelets})
            clf.fit(train_dataset[train_index],train_label[train_index])
            sc=clf.score(train_dataset[test_index],train_label[test_index])        
            split_results.append(sc)
            params.append({'idx': idx, 'params': comb})
        results.append(split_results)
    # Collect results and average
    results = np.array(results)
    fin_results = results.mean(axis=0)
    # select the best results
    best_idx = np.argmax(fin_results)
    # Return the fitted model and the best_parameters
    best_shapelets=Shapelet_from_UCR.Shapelet_random(train_dataset, train_label,length=int(params[best_idx]['params'][0]),number_search=1000,top_number=params[best_idx]['params'][1])
    best_shapelets=np.asarray(best_shapelets)
    print(params[best_idx]['params'][0],params[best_idx]['params'][1],params[best_idx]['params'])
    #print('best_shapelets',best_shapelets,type(best_shapelets),type(best_shapelets[0]),type(best_shapelets[0][0]))
    best_model=KNeighborsClassifier(n_neighbors=params[best_idx]['params'][2],metric=min_shapelet_distance,metric_params={'shapelets':best_shapelets})
    
    return best_model.fit(X, y), params[best_idx]


if __name__ == "__main__":
    for dataset in datasets_small[:1]:
        X_train, y_train, X_test, y_test = utilities.get_ucr_dataset('../UCRArchive_2018/',dataset)
        L=len(X_train[0])
        #print(L)######L=max length of time series for this datasets?#####
        shapelet_grid={'length':np.logspace(np.log10(2),np.log10(np.log2(L)+1),num=5),'number':range(10,100)}
        model_grid={'K':range(2,6)}
        best_model,best_para=customized_random_search_cv(X_train,y_train, model_grid,shapelet_grid, X_train, y_train, cv=2)
        print(best_para)

        y_pred = best_model.predict_proba(X_test)
        auroc=roc_auc_score(y_test, y_pred[:, 1])
        auprc=average_precision_score(y_test, y_pred[:,1])
        print(dataset, " AUROC is: ", auroc," AUPRC is: ", auprc)
        f=open('randomGridCV.csv','a')
        np.savetxt(f, np.array([dataset, auroc,auprc,best_para]).reshape(1,4), delimiter=',',fmt="%s")  
        f.write("\n")
        f.close()        


