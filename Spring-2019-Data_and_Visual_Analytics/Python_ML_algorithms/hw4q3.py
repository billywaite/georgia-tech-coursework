## Data and Visual Analytics - Homework 4
## Georgia Institute of Technology
## Applying ML algorithms to detect eye state

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

######################################### Reading and Splitting the Data ###############################################
# XXX
# TODO: Read in all the data. Replace the 'xxx' with the path to the data set.
# XXX
data = pd.read_csv('eeg_dataset.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]

# The random state to use while splitting the data.
random_state = 100

# XXX
# TODO: Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
# Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 100.
# XXX

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = random_state, shuffle = True)

# ############################################### Linear Regression ###################################################
# XXX
# TODO: Create a LinearRegression classifier and train it.
# XXX

lin_reg = LinearRegression().fit(x_train, y_train)

# XXX
# TODO: Test its accuracy (on the training set) using the accuracy_score method.
# TODO: Test its accuracy (on the testing set) using the accuracy_score method.
# Note: Round the output values greater than or equal to 0.5 to 1 and those less than 0.5 to 0. You can use y_predict.round() or any other method.
# XXX

# Use model to predict on x_train
y_train_predict = lin_reg.predict(x_train).round()
train_accuracy = accuracy_score(y_train, y_train_predict)

# Use model to predict on x_test
y_test_predict = lin_reg.predict(x_test).round()
test_accuracy = accuracy_score(y_test, y_test_predict)

print(train_accuracy.round(2), test_accuracy.round(2))

# ############################################### Random Forest Classifier ##############################################
# XXX
# TODO: Create a RandomForestClassifier and train it.
# XXX

# Train random forest using training sets
rf_classifier = RandomForestClassifier().fit(x_train, y_train)

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX

# Check accuracy on training set
rf_y_train_predict = rf_classifier.predict(x_train)
rf_train_accuracy = accuracy_score(y_train, rf_y_train_predict)

# Check accuracy on test set
rf_y_test_predict = rf_classifier.predict(x_test)
rf_test_accuracy = accuracy_score(y_test, rf_y_test_predict)

print(rf_train_accuracy.round(2), rf_test_accuracy.round(2))

# XXX
# TODO: Determine the feature importance as evaluated by the Random Forest Classifier.
#       Sort them in the descending order and print the feature numbers. The report the most important and the least important feature.
#       Mention the features with the exact names, e.g. X11, X1, etc.
#       Hint: There is a direct function available in sklearn to achieve this. Also checkout argsort() function in Python.
# XXX

importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_data.shape[1]):
    print("feature %d, %s (%f)" % (indices[f], x_data.columns[indices[f]], importances[indices[f]]))

# Print Most Important factor
print("Most Important Feature:")
print("feature %d, %s (%f)" % (indices[0], x_data.columns[indices[0]], importances[indices[0]]))

# Print Least Important factor
print("Least Important Feature:")
print("feature %d, %s (%f)" % (indices[-1], x_data.columns[indices[-1]], importances[indices[-1]]))

# XXX
# TODO: Tune the hyper-parameters 'n_estimators' and 'max_depth'.
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# Get the training and test set accuracy values after hyperparameter tuning.
# XXX

# Number of trees in random forest
n_estimators = [100, 200, 300]

# Maximum number of levels in tree
max_depth = [10,20,30]

# Create the  grid
grid = {'n_estimators': n_estimators,
               'max_depth': max_depth
               }
# Use the  grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier(random_state=random_state)

# Grid search of parameters, using 10 fold cross validation
rf_grid = GridSearchCV(estimator = rf, param_grid = grid, cv = 10, verbose=2)

# Fit the random search model
rf_grid.fit(x_train, y_train)

rf_grid.best_params_

# Check accuracy on test set
rf_grid_predict = rf_grid.predict(x_test)
rf_grid_accuracy = accuracy_score(y_test, rf_grid_predict)
print(rf_grid_accuracy.round(2))

# ############################################ Support Vector Machine ###################################################
# XXX
# TODO: Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
# TODO: Create a SVC classifier and train it.
# XXX

# normalize the data
normalized_x_train = normalize(x_train)
normalized_x_test = normalize(x_test)

svclassifier = SVC()  
svclassifier.fit(normalized_x_train, y_train)

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX

# Accuracy on training set
svc_y_train_pred = svclassifier.predict(normalized_x_train)
svc_train_accuracy = accuracy_score(y_train, svc_y_train_pred)


# Accuracy on test set
svc_y__test_pred = svclassifier.predict(normalized_x_test)
svc_test_accuracy = accuracy_score(y_test, svc_y__test_pred)

print(svc_train_accuracy.round(2), svc_test_accuracy.round(2))

# XXX
# TODO: Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear).
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# Get the training and test set accuracy values after hyperparameter tuning.
# XXX

Cs = [0.001, 0.01, 0.1, 1, 10]
kernels = ['rbf', 'linear']
param_grid = {'C': Cs, 'kernel' : kernels}
grid_search = GridSearchCV(SVC(), param_grid, cv=10)
grid_search.fit(normalized_x_train, y_train)
grid_search.best_params_

# Accuracy on train set
grid_search_train_predict = grid_search.predict(normalized_x_train)
grid_train_accuracy = accuracy_score(y_train, grid_search_train_predict)

# Accuracy on test set
grid_search_test_predict = grid_search.predict(normalized_x_test)
grid_test_accuracy = accuracy_score(y_test, grid_search_test_predict)

print(grid_train_accuracy.round(2), grid_test_accuracy.round(2))

# XXX
# TODO: Calculate the mean training score, mean testing score and mean fit time for the 
# best combination of hyperparameter values that you obtained in Q3.2. The GridSearchCV 
# class holds a  ‘cv_results_’ dictionary that should help you report these metrics easily.
# XXX

index = grid_search.cv_results_['params'].index(grid_search.best_params_)
mean_train_score = grid_search.cv_results_['mean_train_score'][index]
mean_test_score = grid_search.cv_results_['mean_test_score'][index]
mean_fit_time = grid_search.cv_results_['mean_fit_time'][0]
print(mean_train_score.round(2), mean_test_score.round(2), mean_fit_time.round(2))

# ######################################### Principal Component Analysis #################################################
# XXX
# TODO: Perform dimensionality reduction of the data using PCA.
#       Set parameters n_component to 10 and svd_solver to 'full'. Keep other parameters at their default value.
#       Print the following arrays:
#       - Percentage of variance explained by each of the selected components
#       - The singular values corresponding to each of the selected components.
# XXX

pca = PCA(n_components=10, svd_solver = 'full')
pca.fit(x_data)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)

