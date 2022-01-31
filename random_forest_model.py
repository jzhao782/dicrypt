import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dicrypt.utils import get_scikit_metrics
from sklearn.ensemble import RandomForestRegressor

data_dir = 'data/'
output_dir = 'output/random_forest/'

X = pd.read_csv(data_dir + 'X.csv', index_col = 'Unnamed: 0') # read in data
y = pd.read_csv(data_dir + 'y.csv', index_col = 'Unnamed: 0') # read in data
y_mean = pd.read_csv(data_dir + 'y.csv', index_col = 'Unnamed: 0') # read in data
# Y mean is made functionally the same as y temporarily since current implementation of y mean would cause data leakage

X_train, X_valid, y_train, y_valid, y_mean_train, y_mean_valid = train_test_split(X, y, y_mean, train_size=0.9, test_size=0.1) # split into train/test

model = RandomForestRegressor(n_estimators = 400, n_jobs = -1, verbose=3)
model.fit(X_train, y_mean_train)

train_pred = model.predict(X_train)
train_rmse, train_mape, train_r2score = get_scikit_metrics(train_pred, y_train)

test_pred = model.predict(X_valid)
test_rmse, test_mape, test_r2score = get_scikit_metrics(test_pred, y_train)

pred = pd.Series(test_pred)
pred.to_csv(output_dir + "predictions.csv")
y_valid.to_csv(output_dir + "y_valid.csv")
# model.save_model("model_random_forest.json")

print(f'train_rmse={train_rmse}')
print(f'train_mape={train_mape}')
print(f'train_r2={train_r2score}')

print(f'test_rmse={test_rmse}')
print(f'test_mape={test_mape}')
print(f'test_r2={test_r2score}')