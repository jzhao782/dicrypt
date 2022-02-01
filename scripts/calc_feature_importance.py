import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score

data_dir = 'data/'
output_dir = 'output/xgb/'

def predict(model, X, y):
    pred = model.predict(X)
    score = r2_score(pred, y)
    return score

def calc_rel_diffs(model, X, y) :
    score = predict(model, X, y)
    print(score)

    X_shuff = X.copy()
    rel_diffs = {}
    for i in X_shuff.columns:
        X_shuff[i] = np.random.permutation(X[i].values)
        shuff_score = predict(model, X_shuff, y)
        rel_diff = shuff_score - score
        rel_diffs[i] = rel_diff
        X_shuff[i] = X[i].values
        print(f'calculated {i}')

    rel_diffs_df = pd.DataFrame(data=rel_diffs, index=[0])
    rel_diffs_df_sorted = rel_diffs_df.sort_values(by = 0, axis=1)
    
    return rel_diffs_df_sorted

X = pd.read_csv(data_dir + 'X.csv', index_col = 'id') # read in data
y = pd.read_csv(data_dir + 'y.csv', index_col = 'id') # read in data

model = XGBRegressor()
model.load_model(output_dir + "model_xgboost.json")

df = calc_rel_diffs(model, X, y)
df.to_csv(output_dir + "rel_diffs.csv")