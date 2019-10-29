import pandas as pd
from pathlib import Path
from dataset_handler import DatasetHandler
from utils import read_pivot_csv, pivot_to_csv


class UtilityMatrix:
    """Represents the utility matrix of Users x Movies"""

    UTILITY_MAT_PATH = Path("data", "utility_matrix.csv")

    def __init__(self):
        self.dh = DatasetHandler()
        self._utility_mat = None

    @property
    def utility_mat(self) -> pd.DataFrame:
        if self._utility_mat is None:
            if self.UTILITY_MAT_PATH.exists():
                self._utility_mat = read_pivot_csv(self.UTILITY_MAT_PATH)
            else:
                self.generate_matrix()
        return self._utility_mat

    def generate_matrix(self):
        print("Generating utility matrix")
        self._utility_mat = self.dh.train_ratings.pivot(
            index="user_id", columns="movie_id", values="rating"
        ).reindex(
            index=range(1, self.dh.max_user + 1),
            columns=range(1, self.dh.max_movie + 1),
        )
        pivot_to_csv(self._utility_mat, self.UTILITY_MAT_PATH)


if __name__ == "__main__":
    um = UtilityMatrix()
    print(um.utility_mat)
