"""
 * CS 422 - 1002, Final Group Project
 * Group Member Names: Charles Ballesteros, Christopher Liscano, and
 *                     Ethan Zambrano
 * Input: .csv file: student_spending.csv
 * Output: Table showing coefficients of the seven independent variables
 *         and RMSE, for each fold (10-fold). ### CHANGE THIS TO BETTER SUIT OUR DATASET
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# -- load the data file --
# Loads the CSV file/dataset
data = pd.read_csv('student_spending.csv')  # 1000 rows x 18 columns
data = data.rename(columns={'Unnamed: 0': 'Student'})

# -- Preprocess the data as needed --

# change un-numerical data into numerical data in features: gender, year_in_school, major, preferred_payment_method

# 'y' contains the unnamed number labels 0 - 999
y = data.iloc[:, 0]  # 999 rows x 1 col

# 'x' contains the features (label 'Student' is separated)
x = data.drop('Student', axis=1)  # 1000 rows x 17 col

# -- machine learning implementations --

# -- main ? --

# -- output section --
