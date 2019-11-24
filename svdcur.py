import numpy as np
from collections import defaultdict
import time

ratings = defaultdict(dict)
MAX_USERS = 6040
MAX_ITEMS = 3952

with open('./dataset/ratings.dat', 'r') as file:
    for line in file:
        user_id,movie_id,rating_value,timestamp = [*map(int, line.split('::'))]
        ratings[user_id][movie_id] = rating_value

def normalise(ratings):
    for user,movie_ratings in ratings.items():
        mean = sum(movie_ratings.values()) / len(movie_ratings.values())
        for movie_id in movie_ratings:
            if(movie_ratings[movie_id] > 0):
                movie_ratings[movie_id] -= mean
    return ratings

ratings = normalise(ratings)

def RMSE(pred, truth):
    '''
    Calculate Root Mean Square Error (RMSE).
    Inputs:
    pred (1D numpy array): numpy array containing predicted values.
    truth (1D numpy array): numpy array containing the ground truth values.
    Returns:
    rmse (float): The Root Mean Square Error.
    '''
    return np.sqrt(np.sum(np.square(pred-truth)/float(pred.shape[0])))

def RMSE_mat(matA, matB):
    '''
    Calculate Root Mean Square Error (RMSE) between two matrices. Mainly used
    to find error original and reconstructed matrices while working with
    matrix decompositions.
    Inputs:
    matA (2D numpy array): Matrix A
    matB (2D numpy array): Matrix B

    Returns:
    rmse (float): Root Mean Square Error.
    '''
    return np.sqrt(np.sum(np.square(matA-matB))/(matA.shape[0]*matA.shape[1]))

def MAE(matA, matB):
    return np.sum(np.abs(matA-matB)) /(matA.shape[0]*matA.shape[1])


#SVD


class SVD_topk:
    """
    A Recommender System model based on the Singular Value Decomposition concepts.

    Normalizes the Utility matrix consisting of users, movies and their ratings by subtracting values in a row
    by their row mean , which handles the problem of strict and generous raters.
    SVD factorizes the utility matrix into U(m x m), Sigma(m X n) and V* (V-transpose) (n X n).
    The dot product U.Sigma.V* in the reduced form gives the prediction matrix.
    """
    def __init__(self, matrix, K):
        self.K = K
        self.matrix = matrix
        self.MAX_USERS = len(self.matrix)
        self.MAX_ITEMS = len(self.matrix[0])
        self.generate_svd_matrices()
        self.tryDimReduction()
        self.multiply()

    def generate_svd_matrices(self):
        """ Performs SVD on the normalized utility matrix and factorizes it into U, S and VT
        """
        A = np.array(self.matrix)
        AAt = np.dot(A,A.T)
        self.eigen_values_AAt, self.eigen_vectors_AAt = np.linalg.eigh(AAt)
        self.rank_AAt = np.linalg.matrix_rank(AAt)
        self.index_AAt = sorted(range(len(self.eigen_values_AAt)),
                             key=lambda k: self.eigen_values_AAt[k], reverse=True)[:self.rank_AAt]
        self.U = np.zeros(shape=(self.MAX_USERS,self.rank_AAt))
        self.eigen_values_AAt = self.eigen_values_AAt[::-1]
        self.eigen_values_AAt = self.eigen_values_AAt[:self.rank_AAt]

        for i in range(self.rank_AAt):
            self.U[:,i] = self.eigen_vectors_AAt[:,self.index_AAt[i]]

        self.sigma = np.zeros(shape=(self.rank_AAt, self.rank_AAt))
        for i in range(self.rank_AAt):
            self.sigma[i,i] = self.eigen_values_AAt[i] ** 0.5

        AtA = (A.T).dot(A)
        self.eigen_values_AtA, self.eigen_vectors_AtA = np.linalg.eigh(AtA)
        self.rank_AtA = np.linalg.matrix_rank(AtA)
        self.index_AtA = sorted(range(len(self.eigen_values_AtA)), key=lambda k: self.eigen_values_AtA[k], reverse=True)[:self.rank_AtA]
        self.V = np.zeros(shape=(self.MAX_ITEMS,self.rank_AtA))
        self.eigen_values_AtA = self.eigen_values_AtA[::-1]
        self.eigen_values_AtA = self.eigen_values_AtA[:self.rank_AtA]

        for i in range(self.rank_AtA):
            self.V[:,i] = self.eigen_vectors_AtA[:,self.index_AtA[i]]
        self.V = self.V.T

    def tryDimReduction(self):
        dk = len(self.sigma) - self.K
        self.sigma = self.sigma[:self.K, :self.K]
        self.U = self.U[:,:-dk]
        self.V = self.V[:-dk,:]

    def multiply(self):
        """ Multiplies the U, V and sigma matrices to get the matrix of predicted ratings.
        """
        self.result = np.dot((np.dot(self.U, self.sigma)), self.V) * (-1)

matrix = np.zeros(shape=(3000, 3000))
for i in range(3000):
    for j in range(3000):
        if i in ratings and j in ratings[i]:
            matrix[i-1][j-1] = ratings[i][j]

start = time.time()
svd = SVD_topk(matrix,7)
pred = svd.result
end = time.time()
print("SVD took : ", end-start, "secs to complete")
print("RMSE for SVD: ", RMSE_mat(pred,matrix))
print("MAE for SVD: ", MAE(pred,matrix))

#svd 90%

class SVD_90:
    """
    A Recommender System model based on the Singular Value Decomposition concepts.

    Normalizes the Utility matrix consisting of users, movies and their ratings by subtracting values in a row
    by their row mean , which handles the problem of strict and generous raters.
    SVD factorizes the utility matrix into U(m x m), Sigma(m X n) and V* (V-transpose) (n X n).
    The dot product U.Sigma.V* in the reduced form gives the prediction matrix.
    """
    def __init__(self, matrix):
        self.matrix = matrix
        self.MAX_USERS = len(self.matrix)
        self.MAX_ITEMS = len(self.matrix[0])
        self.generate_svd_matrices()
        self.tryDimReduction()
        self.multiply()

    def generate_svd_matrices(self):
        """ Performs SVD on the normalized utility matrix and factorizes it into U, S and VT
        """
        A = np.array(self.matrix)
        AAt = np.dot(A,A.T)
        self.eigen_values_AAt, self.eigen_vectors_AAt = np.linalg.eigh(AAt)
        self.rank_AAt = np.linalg.matrix_rank(AAt)
        self.index_AAt = sorted(range(len(self.eigen_values_AAt)),
                             key=lambda k: self.eigen_values_AAt[k], reverse=True)[:self.rank_AAt]
        self.U = np.zeros(shape=(self.MAX_USERS,self.rank_AAt))
        self.eigen_values_AAt = self.eigen_values_AAt[::-1]
        self.eigen_values_AAt = self.eigen_values_AAt[:self.rank_AAt]

        for i in range(self.rank_AAt):
            self.U[:,i] = self.eigen_vectors_AAt[:,self.index_AAt[i]]


        self.sigma = np.zeros(shape=(self.rank_AAt, self.rank_AAt))
        for i in range(self.rank_AAt):
            self.sigma[i,i] = self.eigen_values_AAt[i] ** 0.5

        AtA = (A.T).dot(A)
        self.eigen_values_AtA, self.eigen_vectors_AtA = np.linalg.eigh(AtA)
        self.rank_AtA = np.linalg.matrix_rank(AtA)
        self.index_AtA = sorted(range(len(self.eigen_values_AtA)), key=lambda k: self.eigen_values_AtA[k], reverse=True)[:self.rank_AtA]
        self.V = np.zeros(shape=(self.MAX_ITEMS,self.rank_AtA))
        self.eigen_values_AtA = self.eigen_values_AtA[::-1]
        self.eigen_values_AtA = self.eigen_values_AtA[:self.rank_AtA]

        for i in range(self.rank_AtA):
            self.V[:,i] = self.eigen_vectors_AtA[:,self.index_AtA[i]]
        self.V = self.V.T

    def tryDimReduction(self):
        """ Tries to reduce the dimension of the matrics by following 90 %
            retain energy rule.
        """
        while True:
            total_E = 0
            size = np.shape(self.sigma)[0]
            for i in range(size):
                total_E += self.sigma[i,i]**2
            retained_E = 0
            if size > 0:
                retained_E = total_E - self.sigma[size-1,size-1]**2
            if total_E == 0 or retained_E/total_E < 0.9:
                break
            else:
                self.U = self.U[:,:-1:]
                self.V = self.V[:-1,:]
                self.sigma = self.sigma[:,:-1]
                self.sigma = self.sigma[:-1,:]

    def multiply(self):
        """ Multiplies the U, V and sigma matrices to get the matrix of predicted ratings.
        """
        self.result = np.dot((np.dot(self.U, self.sigma)), self.V) * (-1)

matrix = np.zeros(shape=(3000, 3000))
for i in range(3000):
    for j in range(3000):
        if i in ratings and j in ratings[i]:
            matrix[i-1][j-1] = ratings[i][j]

start = time.time()
svd = SVD_90(matrix)
pred = svd.result
end = time.time()
print("SVD with 90% energy took : ", end-start, "secs to complete")
print("RMSE for SVD with 90% energy: ", RMSE_mat(pred,matrix))
print("MAE for SVD with 90% energy: ", MAE(pred,matrix))
#cur
class CUR_topk:
    """
    A Recommender System model based on CUR approximation.

    Normalizes the Utility matrix consisting of users, movies and their ratings by subtracting values in a row
    by their row mean , which handles the problem of strict and generous raters.
    CUR factorizes the utility matrix into C(m x k), U(k X k) and R(k X n)..
    The dot product C.U.R in the reduced form gives the prediction matrix.

    """
    def __init__(self, matrix, K):
        self.K = K
        self.matrix = np.array(matrix)
        self.numRows = np.shape(self.matrix)[0]
        self.numCols = np.shape(self.matrix)[1]
        self.getProbDistribution()
        self.find_C_U_R(1,50)
        self.multiply()

    def getProbDistribution(self):
        """ Gets the probability distribution for the rows and columns to be used in
            random list generation.This list enables us to randomly select rows and columns.
        """
        self.totalSum = sum(sum(self.matrix ** 2))
        self.rowSum = (sum((self.matrix.T)**2))/self.totalSum
        self.colSum = (sum(self.matrix ** 2))/self.totalSum

    def generateRandomNos(self, probDist, size, sampleSize, choice):
        """ The method generates sampleSize number of random numbers which are in the range of
            size. It samples out them using the given probability distribution
        """
        if choice == 0: return np.random.choice(np.arange(0,size), sampleSize, p=probDist)
        else: return np.random.choice(np.arange(0,size), sampleSize, replace=False,p=probDist)

    def find_C_U_R(self, choice, sampleSize):
        """ Method to compute C,U and R matrix in the CUR decomposition
        """
        rand_no = self.generateRandomNos(self.colSum, self.numCols, 50, choice)
        self.c_indices = rand_no
        self.C = (self.matrix[:,rand_no]).astype(float)
        colIdx, idx = 0, 0
        while colIdx < sampleSize:
            for rowIdx in range(0, self.numRows):
                self.C[rowIdx, colIdx] /= (sampleSize*self.colSum[rand_no[idx]])**0.5
            idx += 1
            colIdx += 1


        rand_no = self.generateRandomNos(self.rowSum, self.numRows, 50, choice)
        self.R = (self.matrix[rand_no,:]).astype(float)
        rowIdx, idx = 0, 0
        while rowIdx < sampleSize:
            for colIdx in range(0, self.numCols):
                self.R[rowIdx, colIdx] /= (sampleSize*self.rowSum[rand_no[idx]])**0.5
            idx += 1
            rowIdx += 1

        self.U = self.R[:,self.c_indices]
        svd = SVD_topk(self.U, self.K)
        Y = svd.V.T
        sigma_sq_plus = 1/(svd.sigma ** 2)
        sigma_sq_plus[sigma_sq_plus == np.inf] = 0
        sigma_sq_plus[sigma_sq_plus == -np.inf] = 0
        sigma_sq_plus[sigma_sq_plus == np.nan] = 0
        X = svd.U.T
        self.U = np.dot(Y,np.dot(sigma_sq_plus,X))
        self.U[self.U == np.inf] = 0
        self.U[self.U == -np.inf] = 0
        self.U[self.U == np.nan] = 0

    def multiply(self):
        """ Multiplies the C, U and R matrices to get the matrix of predicted ratings.
        """
        self.result = np.dot((np.dot(self.C, self.U)), self.R)
        self.result[self.result == np.inf] = 0
        self.result[self.result == -np.inf] = 0
        self.result[self.result == np.nan] = 0

matrix = np.zeros(shape=(3000, 3000))
for i in range(3000):
    for j in range(3000):
        if i in ratings and j in ratings[i]:
            matrix[i-1][j-1] = ratings[i][j]

start = time.time()
cur = CUR_topk(matrix,6)
pred = cur.result
end = time.time()
print("CUR took : ", end-start, "secs to complete")
print("RMSE for CUR : ", RMSE_mat(pred,matrix))
print("MAE for CUR: ", MAE(pred,matrix))

#cur 90%
class CUR_90:
    """
    A Recommender System model based on CUR approximation with 90% energy.

    Normalizes the Utility matrix consisting of users, movies and their ratings by subtracting values in a row
    by their row mean , which handles the problem of strict and generous raters.
    CUR factorizes the utility matrix into C(m x k), U(k X k) and R(k X n)..
    The dot product C.U.R in the reduced form gives the prediction matrix.

    """
    def __init__(self, matrix):
        self.matrix = np.array(matrix)
        self.numRows = np.shape(self.matrix)[0]
        self.numCols = np.shape(self.matrix)[1]
        self.getProbDistribution()
        self.find_C_U_R(1,50)
        self.multiply()

    def getProbDistribution(self):
        """ Gets the probability distribution for the rows and columns to be used in
            random list generation.
        """
        self.totalSum = sum(sum(self.matrix ** 2))
        self.rowSum = (sum((self.matrix.T)**2))/self.totalSum
        self.colSum = (sum(self.matrix ** 2))/self.totalSum

    def generateRandomNos(self, probDist, size, sampleSize, choice):
        """ The method generates sampleSsize number of random numbers which are in the range of
            size. It samples out them using the given probability distribution
        """
        if choice == 0: return np.random.choice(np.arange(0,size), sampleSize, p=probDist)
        else: return np.random.choice(np.arange(0,size), sampleSize, replace=False,p=probDist)

    def find_C_U_R(self, choice, sampleSize):
        """ Method to compute C,U and R matrix in the CUR decomposition
        """
        rand_no = self.generateRandomNos(self.colSum, self.numCols, 50, choice)
        self.c_indices = rand_no
        self.C = (self.matrix[:,rand_no]).astype(float)
        colIdx, idx = 0, 0
        while colIdx < sampleSize:
            for rowIdx in range(0, self.numRows):
                self.C[rowIdx, colIdx] /= (sampleSize*self.colSum[rand_no[idx]])**0.5
            idx += 1
            colIdx += 1

        rand_no = self.generateRandomNos(self.rowSum, self.numRows, 50, choice)
        self.R = (self.matrix[rand_no,:]).astype(float)
        rowIdx, idx = 0, 0
        while rowIdx < sampleSize:
            for colIdx in range(0, self.numCols):
                self.R[rowIdx, colIdx] /= (sampleSize*self.rowSum[rand_no[idx]])**0.5
            idx += 1
            rowIdx += 1

        self.U = self.R[:,self.c_indices]
        svd = SVD_90(self.U)
        Y = svd.V.T
        sigma_sq_plus = 1/(svd.sigma ** 2)
        sigma_sq_plus[sigma_sq_plus == np.inf] = 0
        sigma_sq_plus[sigma_sq_plus == -np.inf] = 0
        sigma_sq_plus[sigma_sq_plus == np.nan] = 0
        X = svd.U.T
        self.U = np.dot(Y,np.dot(sigma_sq_plus,X))
        self.U[self.U == np.inf] = 0
        self.U[self.U == -np.inf] = 0
        self.U[self.U == np.nan] = 0

    def multiply(self):
        """ Multiplies the C, U and R matrices to get the matrix of predicted ratings.
        """
        self.result = np.dot((np.dot(self.C, self.U)), self.R)
        self.result[self.result == np.inf] = 0
        self.result[self.result == -np.inf] = 0
        self.result[self.result == np.nan] = 0

matrix = np.zeros(shape=(3000, 3000))
for i in range(3000):
    for j in range(3000):
        if i in ratings and j in ratings[i]:
            matrix[i-1][j-1] = ratings[i][j]

start = time.time()
cur = CUR_90(matrix)
pred = cur.result
end = time.time()
print(" CUR with 90% energy took : ", end-start, "secs to complete")
print("RMSE for CUR with 90% energy: ", RMSE_mat(pred,matrix))
print("MAE for CUR with 90% energy: ", MAE(pred,matrix))
