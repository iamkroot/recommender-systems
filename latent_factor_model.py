import time
import numpy as np
from dataset_handler import DatasetHandler


class LatentFactorModel:
    """
    A Recommender System model based on the Latent Factor Modelling of concepts.

    Gradient descent is applied to solve the matrix factorization to optimize for
    least RMSE. Also calculates rating deviations of users to handle strict and
    generous raters.
    """

    def __init__(self, num_items, num_users, mu, num_factors, init_sd=0.1):
        """
        Args:
            num_items (int): The number of items (max_item + 1 due to missing values)
            num_users (int): The number of users (max_user + 1 due to missing values)
            mu (float): The global average rating over all items and users
            num_factors (int): The number of dimensions of the latent space
            init_sd (float): Standard deviation of the initial values of P and Q
        """
        self.mu = mu
        self.num_factors = num_factors

        # initialize the rating deviations per user/items
        self.b_u = np.zeros((num_users,), dtype=np.float32)
        self.b_i = np.zeros((num_items,), dtype=np.float32)

        self.P = np.random.normal(scale=init_sd, size=(num_factors, num_users))
        self.Q = np.random.normal(scale=init_sd, size=(num_factors, num_items))

    def predict(self, user_id: int, item_id: int) -> float:
        """Returns the predicted value of rating for given item_id-user_id pair"""
        return (
            self.mu
            + self.b_i[item_id]
            + self.b_u[user_id]
            + sum(self.Q[:, item_id] * self.P[:, user_id])
        )

    def error(self, ratings):
        sq_err, abs_err = 0, 0
        for user_id, item_id, rating in ratings:
            predicted = self.predict(user_id, item_id)
            diff = predicted - rating
            abs_err += abs(diff)
            sq_err += diff * diff
        rmse = np.sqrt(sq_err / len(ratings))
        mae = abs_err / len(ratings)
        return rmse, mae

    def step(self, user_id, item_id, real_rating, gamma, lambda_):
        """Perform a gradient descent step."""
        err_ui = real_rating - self.predict(user_id, item_id)
        self.b_i[item_id] += gamma * (err_ui - lambda_ * self.b_u[user_id])
        self.b_u[user_id] += gamma * (err_ui - lambda_ * self.b_i[item_id])
        self.P[:, user_id] += gamma * (
            err_ui * self.Q[:, item_id] - lambda_ * self.P[:, user_id]
        )
        self.Q[:, item_id] += gamma * (
            err_ui * self.P[:, user_id] - lambda_ * self.Q[:, item_id]
        )

    def train(self, ratings, num_epochs=15, gamma=0.005, lambda_=0.02):
        for epoch in range(num_epochs):
            start, done = time.time(), 0
            for i, idx in enumerate(np.random.permutation(len(ratings))):
                self.step(*ratings[idx], gamma, lambda_)
                if i and not i % 20000:
                    done += 20000
                    rate = done / (time.time() - start)
                    print(f"\rRate={rate:.0f} ratings/s", end="")
            print()
            yield epoch + 1, time.time() - start


def run_lfm(num_factors, num_epochs, gamma, lambda_):
    dh = DatasetHandler()
    lf = LatentFactorModel(
        dh.max_movie + 1, dh.max_user + 1, dh.global_test_avg, num_factors
    )
    train_ratings, test_ratings = dh.train_ratings.values, dh.test_ratings.values
    start = time.time()
    for epoch_num, epoch_time in lf.train(train_ratings, num_epochs, gamma, lambda_):
        print(f"Epoch {epoch_num}: Time taken: {epoch_time:.0f} seconds")
        print("\tTrain RMSE={:.3f} MAE={:.3f}".format(*lf.error(train_ratings)))
        print("\tTest RMSE={:.3f} MAE={:.3f}".format(*lf.error(test_ratings)))
    print("Run time: {:.0f}s".format(time.time() - start))


def main():
    from collections import namedtuple

    Params = namedtuple("Params", ["num_factors", "num_epochs", "gamma", "lambda_"])
    configs = (
        Params(10, 20, 0.005, 0.02),  # RMSE: 0.877 MAE: 0.690
        Params(40, 20, 0.005, 0.02),  # RMSE: 0.871 MAE: 0.684
        Params(100, 20, 0.005, 0.02),  # RMSE: 0.876 MAE: 0.687
        Params(1000, 20, 0.005, 0.02),  # RMSE: 0.892 MAE: 0.702
    )
    for params in configs:
        print(params)
        run_lfm(params)


if __name__ == "__main__":
    main()
