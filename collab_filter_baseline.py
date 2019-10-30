import numpy as np
import pandas as pd
import math
from numpy.linalg import norm


class CF_Baseline:
    def __init__(self, path):
        self.path = path
        self.umDf = pd.read_csv(path, low_memory=False)
        self.user_mean = [0 for j in range(6041)]
        self.um = self.umDf.values
        self.movies = list(self.umDf.columns)
        # print(len(self.movies))
        self.movie_mean = [0 for j in range(len(self.movies))]
        self.global_mean = 0
        self.global_count = 0
        for i in range(1, 6041):
            m = 0
            cnt = 0
            for j in range(1, len(self.movies)):
                if not math.isnan(self.um[i][j]):
                    m = m + self.um[i][j]
                    cnt += 1
            self.user_mean[i] = m / cnt
            self.global_mean += m
            self.global_count += cnt
            for j in range(1, len(self.movies)):
                if not math.isnan(self.um[i][j]):
                    self.um[i][j] = self.um[i][j]  # -self.user_mean[i]
                else:
                    self.um[i][j] = 0
        self.global_mean = self.global_mean / self.global_count
        for j in range(1, len(self.movies)):
            m = 0
            cnt = 0
            for i in range(1, 6041):
                if self.um[i][j] != 0:
                    m = m + self.um[i][j]
                    cnt = cnt + 1
            if cnt != 0:
                self.movie_mean[j] = m / cnt
        # print(self.um[1][:],self.user_mean[1])
        self.um = np.array(self.um)

    def item_sim(self, i, j):
        return np.dot(self.um[1:, i], self.um[1:, j]) / (
            norm(self.um[1:, i]) * norm(self.um[1:, j])
        )

    def top_sim_items(self, u, i):
        ti = []
        for j in range(1, len(self.movies)):
            if j != i and self.um[u][j] != 0:
                ti.append((self.item_sim(i, j), j))
        ti.sort(reverse=True)
        return ti[:15]

    def baseline(self, u, m):
        return self.user_mean[u] + self.movie_mean[m] - self.global_mean

    def predict_rating(self, u, m):
        m = self.movies.index(str(m))
        ti = self.top_sim_items(u, m)
        num, den = 0, 0
        for x in ti:
            num += x[0] * (self.um[u][x[1]] - self.baseline(u, x[1]))
            den += x[0]
        r = num / den
        r = r + self.baseline(u, m)
        return r


if __name__ == "__main__":
    cfb = CF_Baseline("./data/utility_matrix.csv")
    # for x in cf.top_sim_items(1,1):
    #     print(x,cf.movies[x[1]])
    print("Rating: ", cfb.predict_rating(3589, 1562))
