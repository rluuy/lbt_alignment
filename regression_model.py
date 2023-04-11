import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from utils import load_dataframe



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


'''
KNN
'''
def knn_regressor(df):
    model = KNeighborsRegressor(n_neighbors=6)
    X = df[['p_x', 'p_y']] # Input Features
    y = df[['d_x', 'd_y', 'd_z', 't_x', 't_y']] # Output Features


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    print(f'KNN R^2 score: {score}')

'''
Neural network
'''

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 5)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


def neural_network(df):
    # Convert the input and output features to PyTorch tensors
    X = torch.tensor(df[['p_x', 'p_y']].values).float()  # Input Features
    y = torch.tensor(df[['d_x', 'd_y', 'd_z', 't_x', 't_y']].values).float() # Output Features

    # Normalize the input features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

    # Create an instance of the neural network model
    model = Net()

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(1000):
        optimizer.zero_grad()
        y_pred = model(torch.tensor(X_train).float())
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

#        if epoch % 100 == 0:
#            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    # Evaluate the performance of the model
    y_pred = model(torch.tensor(X_test).float())
    score = r2_score(y_test.detach().numpy(), y_pred.detach().numpy())
    print(f'Neural Network R^2 score: {score}')



def gradient_boosting(df):
    model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, max_depth=5, random_state=42))
    X = df[['p_x', 'p_y']]  # Input Features
    y = df[['d_x', 'd_y', 'd_z', 't_x', 't_y']] # Output Features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    print(f'Gradient Boosting R^2 score: {score}')



if __name__ == '__main__':
    test_df = load_dataframe('processed_data/10_Data.pt')
    linear_classifier(test_df)
    random_forest(test_df)
    knn_regressor(test_df)
    #neural_network(test_df)
    gradient_boosting(test_df)



