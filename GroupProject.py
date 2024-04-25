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
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# -- load the data file --
# Loads the CSV file/dataset
data = pd.read_csv('student_spending.csv')  # 1000 rows x 18 columns
data = data.rename(columns={'Unnamed: 0': 'Student'})  # naming unlabelled column

# converting to pandas dataframe
df = pd.DataFrame(data)


# -- Preprocess the data as needed --

# -- Data analysis before training --
# creates a countPlot based on the categorical data within the dataframe
def countPlot(category, df):
    sns.set(style='darkgrid')

    plt.figure(figsize=(9, 9))
    plot = sns.countplot(x=category, data=df, palette='mako', hue=category, legend=False)  # plotting data

    # Add count to each bar
    for p in plot.patches:
        s = format(int(p.get_height()))
        xy = (p.get_x() + p.get_width() / 2, p.get_height())
        plot.annotate(s, xy, ha='center', va='center', xytext=(0, 6), textcoords='offset points')


cat_columns = df.select_dtypes(include=['object']).columns.tolist()  # obtaining categorical data

# creating count plots
for categories in cat_columns:
    countPlot(categories, df)
    plt.title(f'Distribution of {categories[0].upper() + categories[1:]}')  # plot title
    plt.show()  # display plot

# honestly there are so many graphs you can do, you guys can pick any and code it lol. maybe create a heatmap????

# -- Encode categorical data into multiple labels which are used for training --
# 1. gender (Male, Female)
# 2. year_in_school(Freshman, Sophomore, Junior, Senior)
# 3. major (Biology, Economics, Computer Science, Engineering, Psychology)
# 4. preferred_payment_method (Cash, Credit/Debit Card, Mobile Payment App
one_hot_encoder = OneHotEncoder(sparse_output=False)  # initializing encoder
encoded = one_hot_encoder.fit_transform(df[cat_columns])  # applying one hot encoding

# creating a dataframe with encoded data and converts data into int (it originally converts to float for some reason)
one_hot_df = pd.DataFrame(encoded, columns=one_hot_encoder.get_feature_names_out(cat_columns)).astype(int)

# renaming labels as the encoder concatenates the original column name with the encoded names
for categories in cat_columns:
    categories = categories + '_'
    one_hot_df.columns = one_hot_df.columns.str.replace(categories, '')

# 'y' contains the unnamed number labels 0 - 999
y = data.iloc[:, 0]  # 999 rows x 1 col
# 'X' contains the features (label 'Student' is separated)
X = data.drop('Student', axis=1)  # 1000 rows x 17 col

# -- machine learning implementations (KNN and linear regression)--


# - use cross-validation to compare between KNN and linear reg -

# -- main --

# -- output section --
