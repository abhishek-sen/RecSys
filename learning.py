





import numpy as np
from dataIO import *
from matrixCompletion import *
import cPickle as pickle



def learnModel_R1MP(rank, data, X, projMat,C, projMatC, biasMat, verbose=False, algorithm ='econ'):
    
    if algorithm == 'econ':
        Z, rmse_list = rankOneMatrixPursuit_econ(X, projMat, rank, C, projMatC, biasMat,verbose)
    else:
        Z, rmse_list = rankOneMatrixPursuit(X, projMat, rank, C, projMatC, biasMat,verbose)
        
    error, rmse = validateModel(data, Z, projMat, C, projMatC)
    print " rmse: " + str(rmse)
    
    return [Z, rmse_list]



def validateModel(data, Z, projMat, C, projMatC):
    
    errorVal = np.linalg.norm(np.multiply(projMatC -projMat, Z - C), 'fro')
    rmse = errorVal/np.sqrt(np.sum(projMatC - projMat))

    return [errorVal, rmse]


"""
experiment with various train/test splits
""" 

def runModelTuning():
    
    # read Data
    #dummy, sortedUserTuple = readUserData()
    
    print 'reading ratings data ...'
    
    ratingsData = readRatingsData()
    
    # rank of solution. In this context #iterations within the matrix completion
    # algorithm
    
    dataSplit_ratio = [0.01*el for el in range(10,60,10)]
    
    plot_points = list()
    
    for ratio in dataSplit_ratio:
        
        rank = 25
        
        # train-test split of 65/35
        Xu, projMat, C, projMatC, biasMat = dataSplit_naive(ratingsData, ratio)
    
        print 'initializing matrix completion ..'
        
        [Z, rmse_list] = learnModel_R1MP(rank, ratingsData, Xu, projMat,C, projMatC, biasMat, verbose=True)
        
        plot_points.append(rmse_list)
    
    pickle.dump(plot_points, open('rmse_data_full.p','w'))
    
        

    return


"""
sample wrapper for calling the completion code
"""

def runSampleMatCompletion():
    
    print 'reading ratings data ...'
    
    ratingsData = readRatingsData()
    rank = 10
    ratio = 0.5 # 50/50 train/test split
    Xu, projMat, C, projMatC, biasMat = dataSplit_naive(ratingsData, ratio)
    [Z, rmse_list] = learnModel_R1MP(rank, ratingsData, Xu, projMat,C, projMatC, biasMat, verbose=True)
    
    
    return

if __name__ =='__main__':
    runSampleMatCompletion()
    #runModelTuning()