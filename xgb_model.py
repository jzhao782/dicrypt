import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dicrypt.utils import get_scikit_metrics
from xgboost import XGBRegressor

data_dir = 'data/'
output_dir = 'output/xgb/'

X = pd.read_csv(data_dir + 'X.csv', index_col = 'id') # read in data
y = pd.read_csv(data_dir + 'y.csv', index_col = 'id') # read in data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9, test_size=0.1) # split into train/test

# X_train_standardized, lmbdas, mins = boxcox_standardize(X_train)
# X_valid_standardized, _, _ = boxcox_standardize(X_valid, df_lmbdas=lmbdas, df_min=mins)

model = XGBRegressor(n_estimators = 2000, learning_rate = 0.05)
model.fit(X_train, y_train,
            early_stopping_rounds=50,
            eval_set=[(X_valid, y_valid)],
            verbose=3)

train_pred = model.predict(X_train)
train_rmse, train_mape, train_r2score = get_scikit_metrics(train_pred, y_train)

test_pred = model.predict(X_valid)
test_rmse, test_mape, test_r2score = get_scikit_metrics(test_pred, y_valid)

pred = pd.Series(test_pred)
pred.to_csv(output_dir + "predictions.csv")
y_valid.to_csv(output_dir + "y_valid.csv")
model.save_model(output_dir + "model_xgboost.json")

print(f'train_rmse={train_rmse}')
print(f'train_mape={train_mape}')
print(f'train_r2={train_r2score}')

print(f'test_rmse={test_rmse}')
print(f'test_mape={test_mape}')
print(f'test_r2={test_r2score}')