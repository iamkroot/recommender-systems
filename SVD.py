import pandas as pd
from scipy.linalg import svd
import numpy as np
import math
import time
from dataset_handler import DatasetHandler
from utility_matrix import UtilityMatrix


class SVD:
    """
    A Recommender System model based on the Singular Value Decomposition concepts.

    The 0 values in each user row are replaced by the mean rating of each user.
    SVD factorizes the utility matrix into U(m x m), Sigma(m X n) and V-transpose(n X n)
    Dimensionality reduction reduces the dimensions of each matrix to k dimensions.
    The dot product U.Sigma.V* in the reduced form gives the prediction matrix.
    U is an m X m unitary matrix.
    Sigma is an m X n rectangular diagonal matrix, with each diagonal element as the
    singular values of the utility matrix.
    V is an n X n unitary matrix.
    """

    def __init__(self):
        self.um = UtilityMatrix().utility_mat.values
        self.movies = list(self.umDf.columns)

    def generate_svd_matrices(self):
        """
        Normalizes the Utility matrix consisting of users, movies and their ratings by
        replacing 0s in a row by their row mean.
        Performs SVD on the normalized utility matrix and factorizes it into U, S and V*

        Returns:
            U (np.ndarray)  : An m X m unitary matrix
            S (list)        : List if singular values of the utility matrix
            V* (np.ndarray) : An n X n unitary matrix
        """

        for i in range(self.um.shape[0]):
            sum = 0
            count = 0
            for j in range(self.um.shape[1]):
                if not math.isnan(self.um[i][j]):
                    sum += self.um[i][j]
                    count += 1
            for j in range(self.um.shape[1]):
                if math.isnan(self.um[i][j]):
                    self.um[i][j] = sum / count

        U, S, VT = svd(self.um)

        return U, S, VT

    def svd_with_k_dimensions(self, U, S, VT, k):
        """
        Reduces the matrices U, Sigma, V* to k dimensions

        Args:
            U (np.ndarray)  : An m X m unitary matrix
            S (list)        : List if singular values of the utility matrix
            V* (np.ndarray) : An n X n unitary matrix
            k (int)         : The dimension value to be reduced to

        Returns:
            A(np.ndarray) : The prediction matrix of the utility matrix.
        """
        U = U[:, :k]
        Sigma = np.zeros((k, k))
        Sigma[:k, :k] = np.diag(S[:k])
        VT = VT[:k, :]

        A = U.dot(Sigma.dot(VT))

        return A

    def get_dimensions_for_x_energy(self, S, fraction, initial):
        """
        Finds the number of the dimensions to which Sigma matrix can be reduced to,
        so as to preserve (fraction * 100)% of the energy.

        Args:
            S (list)        : List if singular values of the utility matrix
            fraction(float) : The value to preserve (fraction * 100)% of the energy
            initial (int)   : The initial dimension number of dimensions

        Returns:
            dim (int) : The number of dimension that will preserve atleast
                (fraction * 100)% of the energy.
        """
        S = S[:initial]
        sq_sum, red_sum, dim = 0, 0, 0
        for x in S:
            sq_sum += x * x
        for x in S:
            red_sum += x * x
            dim += 1
            if red_sum / sq_sum >= fraction:
                return dim

    def predict_and_find_error(self, test_ratings):
        """Predicts the matrix equla to the utility matrix.
        Has two prediction components:
        1. Prediction using complete SVD reduced to 20 dimensions.
        2. Prediction using SVD with 90% energy
        Also displays the Root Mean Square Error, Mean Absolute Error values and
        the prediction time for each component.

        Args:
            test_ratings (np.ndarray): An array of <user_id, item_id, rating> tuples
        """
        start = time.time()
        U, S, VT = self.generate_svd_matrices()
        print("Generation of SVD took " + str(time.time() - start) + " secs")

        next_part = time.time()

        # A = self.svd_with_energy_k(U, S, VT, k=len(self.movies)-1)
        A = self.svd_with_k_dimensions(U, S, VT, k=20)
        rmse, mae = self.error(A, test_ratings)
        print("For complete SVD:")
        print("RMSE: ", rmse)
        print("MAE: ", mae)
        print(
            "Prediction and error calculation of complete SVD took "
            + str(time.time() - next_part)
            + " secs"
        )

        next_part = time.time()

        dim_90 = self.get_dimensions_for_x_energy(S, fraction=0.9, initial=20)
        # A = self.svd_with_energy_k(U, S, VT, k=int((len(self.movies)-1)*0.9))
        A = self.svd_with_k_dimensions(U, S, VT, k=dim_90)
        rmse, mae = self.error(A, test_ratings)
        print("\nFor SVD with 90% energy:")
        print("RMSE: ", rmse)
        print("MAE: ", mae)
        print(
            "Prediction and error calculation of SVD with 90% energy took "
            + str(time.time() - next_part)
            + " secs"
        )
        print("\n\nOverall process: " + str(time.time() - start) + " secs")

    def error(self, A, test_ratings):
        """
        Computes the error of the input ratings vs predicted values from model.

        Args:
            ratings (np.ndarray): An array of <user_id, item_id, true_rating> tuples

        Returns:
            The Root Mean Square Error and Mean Absolute Error values.
        """
        sq_err, abs_err = 0, 0
        for user_id, item_id, rating in test_ratings:
            predicted = A[user_id - 1][item_id - 1]
            diff = predicted - rating
            abs_err += abs(diff)
            sq_err += diff * diff

        rmse = np.sqrt(sq_err / len(test_ratings))
        mae = abs_err / len(test_ratings)
        return rmse, mae


if __name__ == "__main__":
    s = SVD()
    dh = DatasetHandler()
    s.predict_and_find_error(dh.test_ratings.values)
