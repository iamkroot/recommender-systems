import numpy as np
import pandas as pd
import math
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt


class Collab_Filter:
    """
    A Recommender System model based on the Collaborative filtering concepts.

    An Item-Item based collaborative filtering is used to find similar items which then
    is used to predict rating a user might give to a movie/item based on the ratings he
    gave to similar items. Also calculates rating deviations of users to the form of
    the mean of ratings to handle strict and generous raters.
    """

    def __init__(self, path):
        """
        Normalizes the Utility matrix consisting of users, movies and their ratings
        by subtracting values in a row by their row mean, which handles the problem
        of strict and generous raters.

        Args:
            path (string) : The path to the csv which stores the utility matrix.
        """
        self.path = path
        self.umDf = pd.read_csv(path, low_memory=False)
        self.user_mean = [0 for j in range(6041)]
        self.um = self.umDf.values
        self.movies = list(self.umDf.columns)
        # print(len(self.movies))
        for i in range(1, 6041):
            m = 0
            cnt = 0
            for j in range(1, len(self.movies)):
                if not math.isnan(self.um[i][j]):
                    m = m + self.um[i][j]
                    cnt += 1
            self.user_mean[i] = m / cnt
            for j in range(1, len(self.movies)):
                if not math.isnan(self.um[i][j]):
                    self.um[i][j] = self.um[i][j] - self.user_mean[i]
                else:
                    self.um[i][j] = 0
        # print(self.um[1][:],self.user_mean[1])
        self.um = np.array(self.um)

    def item_sim(self, i, j):
        """
        Calculates similarity between two items/movies

        Args:
            i (int) : Column number of first movie
            j (int) : Column number of second movie

        Returns:
            The similarity value between the two items
        """
        return np.dot(self.um[1:, i], self.um[1:, j]) / (
            norm(self.um[1:, i]) * norm(self.um[1:, j])
        )

    def top_sim_items(self, u, i):
        """
        Finds the items most similar to given item , which are rated by the user

        Args:
            u (int) : User's ID
            i (int) : Column number/movie_id of required item

        Returns:
            list : A list of movie_ids of movies similar to given movie and their
            similarity values
        """
        ti = []
        for j in range(1, len(self.movies)):
            if j != i and self.um[u][j] != 0:
                ti.append((self.item_sim(i, j), j))
        ti.sort(reverse=True)
        return ti[:15]

    def predict_rating(self, u, m):
        """
        Predicts the rating a user might give to a movie

        Args:
            u (int) : User's ID
            m (int) : movie_id of the required movie/item

        Returns:
            The predicted rating user u might give to movie m
        """
        m = self.movies.index(str(m))
        ti = self.top_sim_items(u, m)
        num, den = 0, 0
        for x in ti:
            num += x[0] * self.um[u][x[1]]
            den += x[0]
        r = num / den
        r = r + self.user_mean[u]
        return r


if __name__ == "__main__":
    cf = Collab_Filter("./data/utility_matrix.csv")
    # for x in cf.top_sim_items(1,1):
    #     print(x,cf.movies[x[1]])
    print("Rating: ", cf.predict_rating(970, 2194))
    
