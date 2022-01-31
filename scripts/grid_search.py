import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

params = {
        'n_estimators': [1000, 2000, 4000, 8000],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.05, 0.1]
         }
folds = 5

dataset = pd.read_csv('magpiedownloaded.csv') # read in data
X, y = dataset.iloc[:,4:], dataset.iloc[:,2]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=0)

xgb = XGBRegressor(random_state=0)

grid_search = GridSearchCV(xgb, param_grid=params, scoring='neg_mean_absolute_percentage_error', n_jobs=-1, cv=folds, verbose=3)

grid_search.fit(X_train, y_train)

results = grid_search.cv_results_
results_df = pd.DataFrame(results)
results_df.to_csv("xgb_grid_search_results.csv")