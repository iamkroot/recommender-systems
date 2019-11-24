import numpy as np
import math
from utility_matrix import UtilityMatrix
from dataset_handler import DatasetHandler


class CUR:
    def mysvd(self, matrix, k):
        """
        Performs the SVD decomposition on the input matrix

        Args:
            matrix (np.ndarray) : The user rating matrix
            k (int) : the reduced dimensionality after decomposition

        Returns:
            The three SVD matrices U,Sigma and V_T

        """
        m = matrix.shape[0]
        n = matrix.shape[1]

        if (k > m) or (k > n):
            print("error: k greater than matrix dimensions.\n")
            return

        matrix_t = matrix.T

        A = np.dot(matrix, matrix_t)  # calculate matrix multiplied by its transpose
        values1, v1 = np.linalg.eigh(A)  # get eigenvalues and eigenvectors
        v1_t = v1.T
        # discarding negative eigenvalues and corresponding eigenvectors (they are anyway tending to zero)
        v1_t[values1 < 0] = 0
        v1 = v1_t.T
        values1[values1 < 0] = 0
        # values1 = np.absolute(values1)

        values1 = np.sqrt(values1)  # finding singular values.
        # sort eigenvalues and eigenvectors in decreasing order
        idx = np.argsort(values1)
        idx = idx[::-1]
        values1 = values1[idx]
        v1 = v1[:, idx]

        U = v1

        A = np.dot(matrix_t, matrix)  # calculate matrix transpose multiplied by matrix.
        values2, v2 = np.linalg.eigh(A)  # get eigenvalues and eigenvectors
        # values2 = np.absolute(values2)
        # discarding negative eigenvalues and corresponding eigenvectors(they are anyway tending to zero)
        v2_t = v2.T
        v2_t[values2 < 0] = 0
        v2 = v2_t.T
        values2[values2 < 0] = 0

        values2 = np.sqrt(values2)  # finding singular values.
        # sort eigenvalues and eigenvectors in decreasing order.
        idx = np.argsort(values2)
        idx = idx[::-1]
        values2 = values2[idx]
        v2 = v2[:, idx]

        V = v2
        V_t = V.T  # taking V transpose.

        sigma = np.zeros((m, n))

        if m > n:  # setting the dimensions of sigma matrix.

            sigma[:n, :] = np.diag(values2)

        elif n > m:
            sigma[:, :m] = np.diag(values1)

        else:
            sigma[:, :] = np.diag(values1)

        if m > k:  # slicing the matrices according to the k value.
            U = U[:, :k]
            sigma = sigma[:k, :]

        if n > k:
            V_t = V_t[:k, :]
            sigma = sigma[:, :k]

        check = np.dot(matrix, V_t.T)
        # case = np.divide(check, values2[:k])

        s1 = np.sign(check)
        s2 = np.sign(U)
        c = s1 == s2
        # choosing the correct signs of eigenvectors in the U matrix.
        for i in range(U.shape[1]):
            if c[0, i] is False:
                U[:, i] = U[:, i] * -1

        return U, sigma, V_t

    def mycur(self, matrix, k):
        """
        Performs the CUR decomposition on the input matrix

        Args:
            matrix (np.ndarray) : The user rating matrix
            k (int) : the reduced dimensionality after decomposition

        Returns:
            The three CUR matrices C,U and R

        """

        m = matrix.shape[0]
        n = matrix.shape[1]

        if (k > m) or (k > n):
            print("error: k greater than matrix dimensions.\n")
            return

        C = np.zeros((m, k))
        R = np.zeros((k, n))

        matrix_sq = matrix ** 2
        sum_sq = np.sum(matrix_sq)

        frob_col = np.sum(matrix_sq, axis=0)
        probs_col = frob_col / sum_sq  # probability of each column.

        count = 0
        count1 = 0
        temp = 0
        idx = np.arange(n)  # array of column indexes.
        taken_c = []
        dup_c = []

        while count < k:
            # choosing column index based on probability.
            i = np.random.choice(idx, p=probs_col)
            count1 = count1 + 1
            if i not in taken_c:
                # taking column after dividing it with root of k*probability.
                C[:, count] = matrix[:, i] / np.sqrt(probs_col[i] * k)
                count = count + 1
                # np.sqrt(probs_col[i])
                taken_c.append(i)
                dup_c.append(1)
            # discarding the duplicate column and increasing its count of duplicates.
            else:
                temp = taken_c.index(i)
                dup_c[temp] = dup_c[temp] + 1
        # multiply columns by root of number of duplicates.
        C = np.multiply(C, np.sqrt(dup_c))

        frob_row = np.sum(matrix_sq, axis=1)
        probs_row = frob_row / sum_sq
        # probability of each row.

        count = 0
        count1 = 0
        idx = np.arange(m)  # array of row indexes.
        taken_r = []
        dup_r = []

        while count < k:
            # choosing row index based on probability.
            i = np.random.choice(idx, p=probs_row)
            count1 = count1 + 1
            if i not in taken_r:
                # taking row after dividing it with root of k*probability.
                R[count, :] = matrix[i, :] / np.sqrt(probs_row[i] * k)
                count = count + 1
                taken_r.append(i)
                dup_r.append(1)
            else:
                # discarding the duplicate row and increasing its count of duplicates.
                temp = taken_r.index(i)
                dup_r[temp] = dup_r[temp] + 1

        # multiply rows by root of number of duplicates.
        R = np.multiply(R.T, np.sqrt(dup_r))
        R = R.T

        W = np.zeros((k, k))

        for i, I in enumerate(taken_r):
            for j, J in enumerate(taken_c):  # forming the intersection matrix W.
                W[i, j] = matrix[I, J]

        X, sigma, Y_t = self.mysvd(W, k)  # svd decomposition of W.

        for i in range(k):
            if sigma[i, i] >= 1:  # taking pseudo-inverse of sigma.
                sigma[i, i] = 1 / sigma[i, i]
            else:
                sigma[i, i] = 0

        U = np.dot(Y_t.T, np.dot(np.dot(sigma, sigma), X.T))  # finding U.

        return C, U, R

    def calcError(self, A, test):
        """Calculates the error between the predicted ratings and actual ratings

        Args:
            A (np.ndarray) : The predicted CUR Matrix
            test (np.ndarray) : The actual test ratings

        Returns:
            The rmse and mae values
        """
        sq_err, abs_err = 0, 0
        for user_id, item_id, rating in test:
            predicted = A[user_id - 1][item_id - 1]
            diff = abs(predicted) - rating
            abs_err += abs(diff)
            sq_err += diff * diff

        rmse = np.sqrt(sq_err / len(test))
        mae = abs_err / len(test)
        return rmse, mae


if __name__ == "__main__":
    t = CUR()
    path1 = "data/test_ratings.csv"
    M = UtilityMatrix().utility_mat.values
    Test_Set = DatasetHandler().test_ratings.values

    for i in range(M.shape[0]):
        sum = 0
        count = 0
        for j in range(M.shape[1]):
            if not math.isnan(M[i][j]):
                sum += M[i][j]
                count += 1
        for j in range(M.shape[1]):
            if math.isnan(M[i][j]):
                M[i][j] = sum / count

    C, U, R = t.mycur(M, 1000)

    A = np.dot(C, np.dot(U, R))

    err1, err2 = t.calcError(A, Test_Set)

    print(err1, err2)
