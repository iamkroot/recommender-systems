# Recommender Systems

Implementation and comparision of various techniques of building recommender systems, such as:
* Collaborative Filtering
* SVD (Using Dense Matrices)
* CUR (Using Sparse Matrices)
* Latent Factor Model - SVD using Gradient Descent

## Dataset
We use the [Movielens 1M](https://grouplens.org/datasets/movielens/1m/) movie ratings dataset to train and test the various models. The datasets contain around 1 million anonymous ratings of approximately 3,900 movies made by 6,000 MovieLens users who joined MovieLens in 2000.

## How to run
1. Clone this repo / click "Download as Zip" and extract the files.
2. Rename the `sample_config.toml` to `config.toml` and set the required values.
3. Ensure Python 3.7 is installed, and in your system `PATH`.
4. Install pipenv using `pip install -U pipenv`.
5. In the project folder, run `pipenv install` to install all python dependencies.
6. Download and extract the dataset (see [Dataset](#dataset) section) into a new folder called `dataset`.
7. To run the recommender for the specific technique, run its module using `pipenv run python <module_name>.py`. 

## Contributors
* [Krut Patel](https://github.com/iamkroot)
* [P Yedhu Tilak](https://github.com/pyt243)
* [Smit Sheth](https://github.com/Sheth-Smit)
* [Akhil Agrawal](https://github.com/KaNeKi2298)
