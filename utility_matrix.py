import pandas as pd
from pathlib import Path
from dataset_handler import DatasetHandler


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
                self._utility_mat = pd.read_csv(self.UTILITY_MAT_PATH)
            else:
                self._utility_mat = self.generate_matrix()
        return self._utility_mat

    def generate_matrix(self) -> pd.DataFrame:
        print("Generating utility matrix")
        self._utility_mat = self.dh.train_ratings.pivot(
            index="user_id", columns="movie_id", values="rating"
        ).reindex(columns=sorted(self.dh.full_ratings.movie_id.unique()))
        self._utility_mat.to_csv(self.UTILITY_MAT_PATH, index=False)
        return self._utility_mat


if __name__ == "__main__":
    um = UtilityMatrix()
    print(um.utility_mat)
