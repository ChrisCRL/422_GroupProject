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

# -- Preprocess the data as needed --
data = data.rename(columns={'Unnamed: 0': 'Student'})  # naming unlabelled column

# converting to pandas dataframe
df = pd.DataFrame(data)


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

# TODO:
# honestly there are so many graphs you can do, you guys can pick any and code it lol. maybe create a heatmap??? - Chris

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

# -- Creating new dataframe
drop_columns = cat_columns  # columns we are dropping
drop_columns.append('Student')
df_copy = df

# dropping Student since it's useless. dropping categorical data to replace with encoded data
df_copy.drop(columns=drop_columns, inplace=True)

# merging the two dataframes
concatenated_df = pd.concat([one_hot_df, df_copy], axis=1)
df = concatenated_df
df.to_csv('new_student_spending.csv', index=False)  # can remove if want, used for debugging purposes - Chris

# TODO:
# -- Creating training data
# 'y' contains labels (first 15 columns)
# when testing labels, do it one by one. Be sure to temporarily drop other labels when testing them individually - Chris

# 'X' contains the features

# -- machine learning implementations (KNN and linear regression)--


# - use cross-validation to compare between KNN and linear reg - (if possible)

# -- main --

# -- output/print section --
