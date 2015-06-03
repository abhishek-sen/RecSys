# -*- coding: utf-8 -*-
"""
Created on Sat May 23 19:06:14 2015

@author: abhisheksen
"""

import numpy as np
import scipy as sc
import scipy.sparse as sp
import scipy.linalg as sl
import scipy.sparse.linalg as spl




"""

An implementation of Orthogonal rank-one matrix pursuit as appears in 
http://arxiv.org/pdf/1404.1377v2.pdf

Goal : Complete the ratings matrix as a linear combination of rank-one basis 
matrices. In each step,k, a new rank-one matrix is estimated 
based on previous (k-1) rank-one basis matrices, while minimizing the 
reconstruction error.

Each step  involves :

1. Solving low rank SVD on the residual matrix to estimate the left and right
singular-vectors corresponding to the highest singular value. This generates 
a new rank-1 matrix in each step. [Residual matrix = Difference between
observed and reconstructed matrix. Entries of this matrix are projected on to
the set of observed entries.]

2. Solving a set of linear-equations to iteratively re-estimate the 
coefficients of each basis matrix, so that linear combination of the bases
match the observed entries in the ratings matrix.

Reconstruction of the complete matrix at the end of k-th iteration is given by

        X(k) = theta_0*M(0) + theta_1*M(1) + ... + theta_k*M(k) 


"""


def rankOneMatrixPursuit(Y, projMat, solRank, C, projMatC, biasMat, verbose = False):
    
    
    M = list()
    
    rec_X = 0 # matrix reconstructed using basis matrices 
    
    projMat_c = sp.coo_matrix(projMat)
    row_coo, col_coo, dummy = sp.find(projMat_c != 0)
    proj_coo = zip(row_coo, col_coo)
    proj_basis_set = list()
    
    
    # projection
    Y_proj = np.multiply(Y, projMat)
    
    # get observed entries as a list for solving linear equations to
    # estimate coefficients
    Yp = np.transpose(np.matrix([Y_proj[el] for el in proj_coo]))
    
    # for rank-1 matrices of the form M = u*v', storing u and v are sufficient
    u_set = list()
    v_set = list()
    
    rmse_list = list()
    
    
    
    for maxIter in range(solRank):
        
        # residual
        Res_k = np.matrix(Y_proj - rec_X)

        # get left and right highest singular vectors of residual
        Res_k = sp.bsr_matrix(Res_k)
        [u,s,vt] = spl.svds(Res_k, k=1, which='LM')
        
        # get the rank-1 basis matrix        
        Mk = u*vt;
        proj_basis_set.append(np.multiply(Mk, projMat)) # projection on observed entries
        
        
        Mp = np.matrix([Mk[el] for el in proj_coo])
        Mp = np.transpose(Mp) # get the entries for solving the linear-equations
        
        if(len(M) == 0):
            M = Mp
            u_set = np.matrix(np.transpose(u))
            v_set = np.matrix(vt)
            
        else:
            M = np.hstack([M,Mp])
            u_set = np.vstack([u_set, np.transpose(u)])
            v_set = np.vstack([v_set, vt])
        
        # solving for coefficients of the basis matrices   
        theta_k = solve(M,Yp)
        
        # performing reconstruction on the projected set
        rec_X = reconstruct_proj(proj_basis_set, theta_k)
        
        if verbose :
            #print 'res norm: ' + str(np.linalg.norm(rec_X,'fro'))
            print "iter:" + str(maxIter) + " err: "+ str(np.linalg.norm(rec_X - Y_proj, 'fro'))
        
        #print "coeffs: " + str(theta_k)
        if verbose :
            Z_inter = reconstruct_full(u_set, v_set, theta_k, biasMat)
            err = np.linalg.norm(np.multiply(projMatC - projMat, Z_inter - C), 'fro')/np.sqrt(np.sum(projMatC - projMat))
            print "rmse full: "+ str(err)
            rmse_list.append(err)
        
    print theta_k    
    Z_final = reconstruct_full( u_set, v_set, theta_k, biasMat)   
    
    return [Z_final, rmse_list]


"""

Above algorithm is memory intensive because at the end of every step, we solve 
a large system of linear equation to iteratively re-estimate the coefficients of
basis matrices to minimize reconstruction error on the observed entries.

In the economic version of the algorithm we don't completely re-estimate the 
coefficients of previous (k-1) basis matrices during step k. Instead we look at
the following reconstruction at the end of k-th iteration.

        X(k) = alpha_1*X(k-1) + alpha_2*M(k)
        
Subsequently, we update the paramters as 
        theta_k = alpha_2
        theta_i = alpha_1 * theta_i, for i < k

Because of this the linear-equation system is considerably light-weight and 
easy to compute
        

"""    
def rankOneMatrixPursuit_econ(Y, projMat, solRank, C, projMatC, biasMat, verbose=False):
    
    M = list()
    
    rec_X = np.zeros((projMat.shape[0], projMat.shape[1])) # matrix reconstructed using basis matrices 
    
    projMat_c = sp.coo_matrix(projMat)
    row_coo, col_coo, dummy = sp.find(projMat_c != 0)
    proj_coo = zip(row_coo, col_coo)
    
    
    # projection
    Y_proj = np.multiply(Y.copy(), projMat)
    
    # get observed entries as a list for solving linear equations to
    # estimate coefficients
    Yp = np.transpose(np.matrix([Y_proj[el] for el in proj_coo]))
    
    u_set = list()
    v_set = list()
    
    rmse_list = list()
    
    count = -1
    
    for maxIter in range(solRank):
        
        count = count + 1
        
        
        # residual
        Res_k = np.matrix(Y_proj - rec_X)

        # left and right highest singular vectors of the residual matrix
        Res_k_sp = sp.bsr_matrix(Res_k)
        [u,s,vt] = spl.svds(Res_k_sp, k=1, which='LM')
        
        
        # rank-1 basis matrix
        Mk = u*vt; #u*np.transpose(v)
        proj_basis_set = list()
        
        # rec_X is 0, when count==0
        if count > 0:
            proj_basis_set.append(np.multiply(rec_X, projMat))
            
        proj_basis_set.append(np.multiply(Mk, projMat))
        
        # coordinate wise content of rec_X
        rec_Xp = np.matrix([rec_X[el] for el in proj_coo])
        rec_Xp = np.transpose(rec_Xp)
        
        
        Mp = np.matrix([Mk[el] for el in proj_coo])
        Mp = np.transpose(Mp)
        
        if count > 0:
            M = rec_Xp
            M = np.hstack([M, Mp])
        else:
            M = Mp
        
        if (count == 0):
            u_set = np.matrix(np.transpose(u))
            v_set = np.matrix(vt)
        else:
            u_set = np.vstack([u_set, np.transpose(u)])
            v_set = np.vstack([v_set, vt])

        #print 'solving for coefficients ...'
        # len(alpha_k) == 2    
        alpha_k = solve(M,Yp)
        
        # update coefficients
        if count == 0:
            theta_k = alpha_k
        elif count == 1:
            theta_k[0] = theta_k[0]*alpha_k[0]
            theta_k.extend([alpha_k[1]])
        else:
            for c in range(maxIter-1):
                theta_k[c] = theta_k[c]*alpha_k[0]
            
            theta_k.extend([alpha_k[1]])
        
        
            
            
        # reconstruction of the projected matrix
        rec_X = reconstruct_proj(proj_basis_set, alpha_k)
        if verbose:
            #print 'res norm: ' + str(np.linalg.norm(rec_X,'fro'))
            print "iter:" + str(maxIter) + " err: "+ str(np.linalg.norm(rec_X - Y_proj, 'fro'))
        
        #print "coeffs: " + str(theta_k)
        
        Z_inter = reconstruct_full(u_set, v_set, theta_k, biasMat)
        
        if verbose :
            err = np.linalg.norm(np.multiply(projMatC - projMat, Z_inter - C), 'fro')/np.sqrt(np.sum(projMatC - projMat))
            print "rmse full: "+ str(err)
            rmse_list.append(err)
        
    # full reconstruction   
    Z_final = reconstruct_full(u_set, v_set, theta_k, biasMat)
    
    return [Z_final, rmse_list]
    
    
    
    
def reconstruct_proj(M_set, theta):
    
    # reconstruction of hte projected matrix on the set of observed entries
    
    rec_X = 0
    count = 0
    for coeff in theta:
        rec_X = np.matrix(rec_X + coeff*M_set[count])
        count = count + 1
    
    return rec_X

def reconstruct_full(u_set, v_set, theta, biasMat):

    # reconstruction of complete matrix

    rec_Z = 0
    count = 0
    
    for coeff in theta:
        rec_Z = np.matrix(rec_Z + coeff*np.transpose(u_set[count])*v_set[count])
        count = count+1
    
    # perform biasing to get the final completed matrix
    if len(biasMat)>0:
        rec_Z = rec_Z + biasMat
                
        
    return rec_Z

    
    
def solve(M,y):
    theta = incrementalInverse(M)*np.transpose(M)*y
    theta = np.transpose(theta)
    theta_list = theta.tolist()[0]
    return theta_list




"""
Compute inv(M'*M) using an iterative, block-matrix based technique.

See : http://en.wikipedia.org/wiki/Block_matrix_pseudoinverse

"""
def incrementalInverse(M):
    

    
    ncols = M.shape[1]

    A = 1.0/(np.transpose(M[:,0])*M[:,0])

    #incremental updates using blockwise inverse
    
    for count in range(ncols-1):
        
        D = np.transpose(M[:,count+1])*M[:,count+1]
        
        B = np.transpose(M[:,0:count+1])*M[:,count+1]
        C = np.transpose(B)
        
        tmpInv = np.linalg.inv(D-C*A*B)
        
        A = np.bmat([[A+ (A*B*tmpInv*C*A), -A*B*tmpInv ],[-tmpInv*C*A , tmpInv]])
 
    return A
    
    
"""
testing for incrementalInverse:
"""    
"""
def testWrapper():
    
    M1 = np.matrix([[1,0,12,33],[3,12,1,1],[21,1,4,2],[1,0,0,1]])
    
    print M1
    
    print np.linalg.inv(np.transpose(M1) * M1)
    print incrementalInverse(M1)
    
    return
    
testWrapper()  

"""


    























    
    
