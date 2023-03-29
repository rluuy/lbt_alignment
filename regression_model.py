import torch
import torch.nn as nn
from utils import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from utils import *

'''
random forest 
'''
def random_forest(df):
    model = RandomForestRegressor(n_estimators=100, max_depth=5, max_features=1, min_samples_leaf=4, min_samples_split=2, random_state=42)
    X = df[['p_x', 'p_y']]  # Input Features
    y = df[['d_x', 'd_y', 'd_z', 't_x', 't_y']] # Output Features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # evaluate the model performance
    score = model.score(X_test, y_test)
    print(f'Random Forest R^2 score: {score}')

'''
grid_search is a helper algorithm to help find best parameters for random forest.
'''
def grid_search(df):
    # define the hyperparameters to search over
    param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'max_features':[1,3,5,7],
    'min_samples_split': [2,3, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4]
    }

    # perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # print the best hyperparameters
    print(f"Best parameters: {grid_search.best_params_}")

    # use the best model to make predictions on the testing data
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # evaluate the performance of the model
    score = best_model.score(X_test, y_test)
    print(f'R^2 score: {score}')

'''
Basic Linear Classifier
'''
def linear_classifier(df):
    model = LinearRegression()
    X = df[['p_x', 'p_y']]  # Input Features
    y = df[['d_x', 'd_y', 'd_z', 't_x', 't_y']] # Output Features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    print(f'Linear Regression R^2 score: {score}')

if __name__ == '__main__':
    test_df = load_dataframe('10_Data.pt')
    linear_classifier(test_df)
    random_forest(test_df)

