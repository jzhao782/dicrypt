import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score

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

    rel_diffs_pd = pd.DataFrame(data=rel_diffs, index=[0])
    rel_diffs_pd_sorted = rel_diffs_pd.sort_values(by = 0, axis=1)
    rel_diffs_pd_sorted.to_pickle("rel_diffs.pkl")
    rel_diffs_pd_sorted.to_csv("rel_diffs.csv")

dataset = pd.read_csv('addedfeatures.csv')
X, y = dataset.iloc[:,4:], dataset.iloc[:,2]

model = XGBRegressor()
model.load_model("model_xgboost.json")

calc_rel_diffs(model, X, y)