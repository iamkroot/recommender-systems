import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


class DatasetHandler:
    """Class to read the dataset and handle train/test splitting"""

    TEST_FRAC = 0.2
    RANDOM_STATE = 9
    FULL_RATINGS_PATH = Path("dataset", "ratings.dat")
    TRAIN_RATINGS_PATH = Path("data", "train_ratings.csv")
    TEST_RATINGS_PATH = Path("data", "test_ratings.csv")

    def __init__(self):
        Path("data").mkdir(exist_ok=True)
        self._test_ratings = None
        self._train_ratings = None
        self._full_ratings = None
        self._max_movie = None
        self._max_user = None
        self._global_test_avg = None

    def read_ratings(self) -> pd.DataFrame:
        self._full_ratings = pd.read_csv(
            self.FULL_RATINGS_PATH,
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating"],
            usecols=[0, 1, 2],
        )
        self._max_user = self.full_ratings.user_id.max()
        self._max_movie = self.full_ratings.movie_id.max()

    def split_and_store(self, store_only_test=True):
        self._train_ratings, self._test_ratings = train_test_split(
            self.full_ratings, test_size=self.TEST_FRAC, random_state=self.RANDOM_STATE
        )
        self._global_test_avg = self._train_ratings.rating.mean()
        if not store_only_test:
            self._train_ratings.to_csv(self.TRAIN_RATINGS_PATH, index=False)
        self._test_ratings.to_csv(self.TEST_RATINGS_PATH, index=False)

    @property
    def full_ratings(self):
        if self._full_ratings is None:
            self.read_ratings()
        return self._full_ratings

    @property
    def max_movie(self):
        if self._max_movie is None:
            self.read_ratings()
        return self._max_movie

    @property
    def max_user(self):
        if self._max_user is None:
            self.read_ratings()
        return self._max_user

    @property
    def global_test_avg(self):
        if self._global_test_avg is None:
            self.split_and_store()
        return self._global_test_avg

    @property
    def test_ratings(self) -> pd.DataFrame:
        if self._test_ratings is None:
            if self.TEST_RATINGS_PATH.exists():  # csv already stored earlier
                self._test_ratings = pd.read_csv(self.TEST_RATINGS_PATH)
            else:
                self.split_and_store()
        return self._test_ratings

    @property
    def train_ratings(self) -> pd.DataFrame:
        if self._train_ratings is None:
            if self.TRAIN_RATINGS_PATH.exists():  # csv already stored earlier
                self._train_ratings = pd.read_csv(self.TRAIN_RATINGS_PATH)
            else:
                self.split_and_store()
        return self._train_ratings
