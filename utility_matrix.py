# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 20:28:49 2019

@author: Sheth_Smit
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class UtilityMatrix:
    
    def __init__(self):
        
        ratings_file = open('./ml-1m/ratings.dat', 'r')
        users_file = open('./ml-1m/users.dat', 'r')
        movies_file = open('./ml-1m/movies.dat', 'r')

        self.ratings = list(ratings_file)
        self.users = list(users_file)
        self.movies = list(movies_file)
        
        self.ratings_train, self.ratings_test = train_test_split(self.ratings, test_size = 0.2, random_state=9)
        
        ratings_file.close()
        users_file.close()
        movies_file.close()
    
    def generate_matrix(self, ratings_model):
        mat = [[0 for j in range(3952)] for i in range(6041)]
        
        cnt = 0
        for movie in self.movies:
            movie_name = movie.split("::")[1]
            mat[0][cnt] = movie_name
            cnt = cnt+1
            
        for line in ratings_model:
            row = line.split("::")
            mat[int(row[0])][int(row[1])-1] = int(row[2])

        return mat
    
    def generate_csv(self):
        
        mat_train = self.generate_matrix(self.ratings_train)
        pd.DataFrame(mat_train).to_csv("train_utility_matrix.csv")
        print("Created train csv")
        
        mat_test = self.generate_matrix(self.ratings_test)
        pd.DataFrame(mat_test).to_csv("test_utility_matrix.csv")
        print("Created test csv")
        
um = UtilityMatrix()
um.generate_csv()
    
