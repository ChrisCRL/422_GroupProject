"""
 * CS 422 - 1002, Final Group Project
 * Group Member Names: Charles Ballesteros, Christopher Liscano, and
 *                     Ethan Zambrano
 * Input: a .csv file: student_spending.csv
 * Output: Table showing coefficients of the seven independent variables
 *         and RMSE, for each fold (10-fold). ### CHANGE THIS TO BETTER SUIT OUR DATASET
"""
import pandas as pd
import numpy as np

# -- load the data file --
# Loads the CSV file/dataset
data = pd.read_csv('student_spending.csv')  # 1000 rows x 18 columns
data = data.rename(columns={'Unnamed: 0': 'Student'})
# print(data)
# -- Preprocess the data as needed --

# change un-numerical data into numerical data in features: gender, year_in_school, major, preferred_payment_method
dict_gender = {'Female': 0, 'Male': 1, 'Non-binary': 3}
dict_year_in_school = {'Freshman': 1, 'Sophomore': 2, 'Junior': 3, 'Senior': 4}
dict_major = {'Biology': 1, 'Economics': 2, 'Computer Science': 3, 'Engineering': 4, 'Pyschology': 5}
dict_preferred_payment_method = {'Cash': 1, 'Credit/Debit Card': 2, 'Mobile Payment App': 3}

data['gender'] = data['gender'].map(dict_gender)
data['year_in_school'] = data['year_in_school'].map(dict_year_in_school)
data['major'] = data['major'].map(dict_major)
data['preferred_payment_method'] = data['preferred_payment_method'].map(dict_preferred_payment_method)

# 'y' contains the unnamed number labels 0 - 999
y = data.iloc[:, 0]  # 999 rows x 1 col
# 'X' contains the features (label 'Student' is separated)
X = data.drop('Student', axis=1)  # 1000 rows x 17 col

# -- machine learning implementations (KNN and linear regression)--


# - use cross-validation to compare between KNN and linear reg -

# -- main --

# -- output section --
