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
        self._global_mean = None

    def read_ratings(self) -> pd.DataFrame:
        """Read the raw Movielens Dataset into a Dataframe"""
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
        """Split the raw dataset into train and test sets, and store them as csv."""
        self._train_ratings, self._test_ratings = train_test_split(
            self.full_ratings, test_size=self.TEST_FRAC, random_state=self.RANDOM_STATE
        )
        self._global_mean = self._train_ratings.rating.mean()
        if not store_only_test:
            self._train_ratings.to_csv(self.TRAIN_RATINGS_PATH, index=False)
        self._test_ratings.to_csv(self.TEST_RATINGS_PATH, index=False)

    @property
    def full_ratings(self) -> pd.DataFrame:
        """The full ratings dataset."""
        if self._full_ratings is None:
            self.read_ratings()
        return self._full_ratings

    @property
    def max_movie(self) -> int:
        """The largest movie_id in the raw dataset."""
        if self._max_movie is None:
            self.read_ratings()
        return self._max_movie

    @property
    def max_user(self) -> int:
        """The largest user_id in the raw dataset."""
        if self._max_user is None:
            self.read_ratings()
        return self._max_user

    @property
    def global_mean(self) -> float:
        """The mean rating in the entire train dataset."""
        if self._global_mean is None:
            self.split_and_store()
        return self._global_mean

    @property
    def test_ratings(self) -> pd.DataFrame:
        """The test dataset."""
        if self._test_ratings is None:
            if self.TEST_RATINGS_PATH.exists():  # csv already stored earlier
                self._test_ratings = pd.read_csv(self.TEST_RATINGS_PATH)
            else:
                self.split_and_store()
        return self._test_ratings

    @property
    def train_ratings(self) -> pd.DataFrame:
        """The train dataset."""
        if self._train_ratings is None:
            if self.TRAIN_RATINGS_PATH.exists():  # csv already stored earlier
                self._train_ratings = pd.read_csv(self.TRAIN_RATINGS_PATH)
            else:
                self.split_and_store()
        return self._train_ratings
