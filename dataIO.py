
import pandas as pd
import numpy as np
from regrouping import *


def readUserData():
    
    
    userDataFile='/Users/abhisheksen/Kaggle/RecSys/data/ml-1m/users.dat'
    colNames = ['UserID','Gender','Age','Occupation','Zip-code']
    data = pd.read_table(userDataFile, sep='::', names = colNames);
    userId = data['UserID'].tolist()
    
    userLabels = clusterUsers(data)
    
    userTuples = zip(userId,userLabels)
    userTuples.sort(key= lambda x:x[1])

    
    return [data, userTuples]
    
    
def readMovieData():
    
    movieDataFile='/Users/abhisheksen/Kaggle/RecSys/data/ml-1m/movies.dat'
    colNames = ['MovieID','Title','Genres'];
    data = pd.read_table(movieDataFile, sep='::', names = colNames);
    #movieLabels = clusterMovies(data)
    
    return data
    
   
def readRatingsData():
    
    ratingsDataFile='/Users/abhisheksen/Kaggle/RecSys/data/ml-1m/ratings.dat'
    colNames = ['UserID','MovieID','Rating','Timestamp'];
    data = pd.read_table(ratingsDataFile, sep='::', names = colNames);
    
    
    return data
    
def dataSplit_naive(ratingsData, ratio):
    
   """
   Given the ratings data split into train and validation set
   
    # ratio : fraction of data withheld from the rating matrix for validation
    # Save the indices of the withheld data. This is test/validation set
    # Repeat the experiment with various set of matrix entries
    
    # Implementation details :
    # ratingsData is provided in form of list
    # save the indices of the data from this list as testset
    
    IMPORTANT NOTE : 
    --------------
        In order to get high-quality prediction it was important to
        de-bias the data set.
        Sources of bias :
        i) average rating on the entire dataset
        ii) average rating on each movie across all user
        iii) average rating provided by user across all movies
        
        So, for  each matrix entry (i,j)
            bias(u,i) = \mu + bu + bi
        
        These biases are stored in a bias matrix.
        
        Matrix completion is peformed on the de-biased rating matrix.
        So rating_d(u,i) = rating(u,i) - bias(u,i)
        
        After completing matrix the bias is added back before estimating
        the reprojection error
    
        
    Return : X =  rating matrix consisting of entries from the training set.
            Test set entries set to 0.
            C = complete rating matrix, this is used during validation
            projMat = boolean matrix of same shape as X. Entry corresponding to
            the training set are set to 1, rest are 0
            projMatC = boolean matrix of same shape as C. Entry corresponding to
            missing entries are set to 0, rest are 1.
            biasMat = matrix with biasData for each entry
    
   """
    
   numData = len(ratingsData)
   randInds = np.random.permutation(numData)
   numDataUsed = int((1-ratio)*numData)
   #unusedData = randInds[numDataUsed:]
   
   userID = ratingsData['UserID'].tolist()
   movieID = ratingsData['MovieID'].tolist()
   rating = ratingsData['Rating'].tolist()
   
   maxUserID = max(userID)
   maxMovieID = max(movieID)
   
   X = np.zeros((maxUserID, maxMovieID))
   
   C = np.zeros((maxUserID, maxMovieID))
   
   
   projMat = np.zeros((maxUserID, maxMovieID))
   projMatC = np.zeros((maxUserID, maxMovieID))
   
   for k in range(len(randInds)):
       ind = randInds[k]  
       mID = movieID[ind]
       uID = userID[ind]
       
       if (k < numDataUsed):
           projMat[uID-1, mID-1] = 1.0
           
           
       C[uID-1, mID-1] = rating[ind]
       projMatC[uID-1, mID-1] = 1.0
       
   X = np.multiply(C, projMat)
   
   
   """
   De-bias the rating matrix based on training set information
   """
   
   # bias for the dataset
   mu = np.mean(X)
   
   
   # bias for each user
   user_means = list()
   for userInfo in X:
        tmp_list = [el-mu if el>0 else 0 for el in userInfo]
        if(len(tmp_list) > 0):
            bu = np.mean(tmp_list)
        else:
            bu = 0
        user_means.append(bu)
    
   assert(len(user_means) == len(X)) #users
    
   # bias for each movie
   movie_means = list()
   tmpX = np.transpose(X)
   for movieInfo in tmpX:
        
        tmp_list = list()
        for k in range(len(movieInfo)):
            u_mean = user_means[k]
            if (movieInfo[k] > 0):
                tmp_list.append(movieInfo[k] - u_mean - mu)
            
        if(len(tmp_list) > 0):
            bi = np.mean(tmp_list)
        else:
            bi = 0
        
        movie_means.append(bi)
        
   assert (len(movie_means) == len(tmpX)) #movies


   # save the bias data in a matrix to avoid element-wise recomputation later
   biasMat = np.zeros(X.shape)
    
   for k1 in range(X.shape[0]):
        for k2 in range(X.shape[1]):
            
            biasMat[k1,k2] = mu + movie_means[k2] + user_means[k1]
            
    
   # de-bias the entries of X, followed by projection on the visible entries    
   X = X - biasMat
   X = np.multiply(X, projMat)
   
    
   return [X, projMat, C, projMatC, biasMat]
   

def dataSplit_sort(ratingsData, userTuples, ratio):
    
    """
    
    Tried to sort the rows and columns of the rating matrix in an informed manner,
    by clustering the users and ranking the movies by their average ratings.
    Eventually didn't get any significant performance gain, so was not used 
    for eventual recommendation.
    
    """
    userID = ratingsData['UserID'].tolist()
    movieID = ratingsData['MovieID'].tolist()
    ratings = ratingsData['Rating'].tolist()
    
    movieDict = dict()
    movieCount = dict()
    
    for k in range(len(userID)):
        if (movieDict.has_key(movieID[k])):
            movieDict[movieID[k]] = movieDict[movieID[k]] + ratings[k]
            movieCount[movieID[k]] = movieCount[movieID[k]] + 1
        else:
            movieDict[movieID[k]] = ratings[k]
            movieCount[movieID[k]] = 1
    
    
    
    for key in movieDict.keys():
        movieDict[key] = (1.0*movieDict[key])/movieCount[key]
    
    movieTuples = [(key, movieDict[key]) for key in movieDict.keys()]
    movieTuples.sort(key= lambda x:x[1])
    
    # Sort the columns of movies according to the rating
    # mappedMovieID tells us where the existing movie id will be mapped to
    # in the matrix
    
    
    mappedMovieID = dict()
    movieCount = 1
    for k in range(len(movieTuples)):
        tup = movieTuples[k]
        mappedMovieID[tup[0]] = k+1
        movieCount = movieCount+1
    
    
    # Add the movies for which any rating were absent
    for moviename in movieID:
        if not mappedMovieID.has_key(moviename):
            mappedMovieID[moviename]=movieCount
            movieCount=movieCount+1
    
     
     
    # Sort the rows of users
    # Maps the users
    mappedUserID = dict()
    userCount = 1
    for k in range(len(userTuples)):
        tup = userTuples[k]
        mappedUserID[tup[0]] = k+1
        userCount = userCount+1
    
    
    # Add the names whose data were absent in the users.dat
    for username in userID:
        if not mappedUserID.has_key(username):
            mappedUserID[username] = userCount
            userCount = userCount + 1
    
    numData = len(ratingsData)
    randInds = np.random.permutation(numData) 
    numDataUsed = int((1-ratio)*numData)       
    maxUserID = max(userID)
    maxMovieID = max(movieID)
    X = np.zeros((maxUserID, maxMovieID))
    C = np.zeros((maxUserID, maxMovieID))
    projMat = np.zeros((maxUserID, maxMovieID))
    projMatC = np.zeros((maxUserID, maxMovieID))
    
    
    for k in range(len(randInds)):
       ind = randInds[k]  
       mID = mappedMovieID[movieID[ind]]
       uID = mappedUserID[userID[ind]]
       if (k < numDataUsed):
           projMat[uID-1, mID-1] = 1.0
           
           
       C[uID-1, mID-1] = ratings[ind]
       projMatC[uID-1, mID-1] = 1.0
   
    X = np.multiply(C,projMat)
    mu = np.mean(X)
    
    # compute bias for each user. 
    # Note that this de-biasing is done only based on trainSet
    user_means = list()
    for userInfo in X:
        tmp_list = [el-mu if el>0 else 0 for el in userInfo]
        if(len(tmp_list) > 0):
            bu = np.mean(tmp_list)
        else:
            bu = 0
        user_means.append(bu)
    
    assert(len(user_means) == len(X)) #users
    
    movie_means = list()
    tmpX = np.transpose(X)
    for movieInfo in tmpX:
        
        tmp_list = list()
        for k in range(len(movieInfo)):
            u_mean = user_means[k]
            if (movieInfo[k] > 0):
                tmp_list.append(movieInfo[k] - u_mean - mu)
            
        if(len(tmp_list) > 0):
            bi = np.mean(tmp_list)
        else:
            bi = 0
        
        movie_means.append(bi)
        
    assert (len(movie_means) == len(tmpX)) #movies
    
    biasMat = np.zeros(X.shape)
    
    for k1 in range(X.shape[0]):
        for k2 in range(X.shape[1]):
            
            biasMat[k1,k2] = mu + movie_means[k2] + user_means[k1]
            
    
    # unbias the entries of X, followed by projection on the visible entries    
    X = X - biasMat
    X = np.multiply(X, projMat)
    
    
        
        
    
    return [X, projMat, C, projMatC, biasMat]
    

#readRatingsData()










