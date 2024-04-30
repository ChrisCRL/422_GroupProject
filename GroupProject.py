"""
 * CS 422 - 1002, Final Group Project
 * Group Member Names: Charles Ballesteros, Christopher Liscano, and
 *                     Ethan Zambrano
 * Input: a .csv file: student_spending.csv
 * Output: Table showing coefficients of the seven independent variables
 *         and RMSE, for each fold (10-fold). ### CHANGE THIS TO BETTER SUIT OUR DATASET/WORk
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn import svm

# -- load the data file --
# Loads the CSV file/dataset
data = pd.read_csv('student_spending.csv')  # 1000 rows x 18 columns

# -- Preprocess the data as needed --
data = data.rename(columns={'Unnamed: 0': 'Student'})  # naming unlabeled column

# converting to pandas dataframe
df = pd.DataFrame(data)


# -- Data analysis before training --
# creates a count_plot based on the categorical data within the dataframe
def count_plot(category, df):
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
# TODO: uncomment after testing ML algo's
# for categories in cat_columns:
#     count_plot(categories, df)
#     plt.title(f'Distribution of {categories[0].upper() + categories[1:]}')  # plot title
#     plt.show()  # display plot

# -- Encode categorical data into multiple labels which are used for training --
# 1. gender (Male, Female, Non-binary)
# 2. year_in_school(Freshman, Sophomore, Junior, Senior)
# 3. major (Biology, Economics, Computer Science, Engi neering, Psychology)
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

# -- Creating training data
# when testing labels, do it one by one. Be sure to temporarily drop other labels when testing them individually - Chris

# LINEAR REGRESSION: Uses LR to analyze the relationship between gender/year_in_school and student spending habits
#                   - Model to predict student spending based on gender/year_in_school
# Define the features (gender / year in school)
labels = ['Female', 'Male', 'Non-binary', 'Freshman', 'Junior', 'Senior', 'Sophomore',
          'Biology', 'Computer Science', 'Economics', 'Engineering', 'Psychology',
          'Cash', 'Credit/Debit Card', 'Mobile Payment App']

# Initialize StandardScaler
scaler = StandardScaler()

# features (skipping financial aid, monthly_income, and age since they are not spending habits)
X = df.iloc[:, 18:]

X_scaled = scaler.fit_transform(X)  # Fit and transform the features

# dictionaries to store the expense results
LR_expense_results = {}
SVM_expense_results = {}

# -- machine learning implementation (linear regression) --

# Iterate over each label
for col in labels:
    y = df[col]  # y contains target variable

    # Split the data into training and testing sets
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=10)

    model = LinearRegression()  # Create and train the linear regression model
    model.fit(X_train_scaled, y_train)

    predict_y = model.predict(X_test_scaled)  # Make predictions on the test set

    # Evaluate RMSE and R-squared
    # - calculate RMSE mean and sum (MAYBE remove RMSE_sum ?)
    rmse_mean = np.sqrt(np.mean((y_test - predict_y) ** 2))
    rmse_sum = np.sqrt(np.sum((y_test - predict_y) ** 2))

    r2 = r2_score(y_test, predict_y)

    # Get the number of observations and features
    n = X_scaled.shape[0]
    p = X_scaled.shape[1]

    # - Calculate adjusted R-squared
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # Store the results in the specified dictionary
    LR_expense_results[col] = {'rmse_mean': rmse_mean, 'rmse_sum': rmse_sum, 'adj_r2': adj_r2}

# -- machine learning implementation (KNN) --

for col in labels:
    y = df[labels]  # y contains target variable

    # Split the data into training and testing sets
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=10)

    # TODO: maybe test multiple neighbors? idk. unless 4 is the best
    knn = KNeighborsRegressor(n_neighbors=4)
    knn.fit(X_train_scaled, y_train)

    y_pred = knn.predict(X_test_scaled)

    # TODO: format output

    print(r2_score(y_test, y_pred))

# -- machine learning implementation (SVM) --

for col in labels:
    y = df[col]

    clf_linear = svm.SVC(kernel='linear', C=1, random_state=10)
    linear_scores = cross_val_score(clf_linear, X_scaled, y, cv=5)

    # Store the results in the specified dictionary
    SVM_expense_results[col] = {'linear_svm_mean': linear_scores.mean(), 'linear_svm_std': linear_scores.std()}

# -- output/print section --
# Linear Regression
print('*****LINEAR REGRESSION*****')
print('---------------------')
for print_category, result in LR_expense_results.items():
    print(f'Category: {print_category}')
    print(f'RMSE (mean): {result["rmse_mean"]}')
    print(f'RMSE (sum): {result["rmse_sum"]}')
    print(f'R-squared: {result["adj_r2"]}')
    print('---------------------')

# KNN



# Linear SVM
print('*****LINEAR SVM*****')
for print_category, result in SVM_expense_results.items():
    print(f'Category: {print_category}')
    print(f'Linear SVM (mean): {result["linear_svm_mean"]}')
    print(f'Linear SVM (std): {result["linear_svm_std"]}')
    print('---------------------')
