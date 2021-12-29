import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error

def predict(model, X, y):
    pred = model.predict(X)
    mape = mean_absolute_percentage_error(pred, y)
    return mape

dataset = pd.read_csv('/Users/jesse/projects/Python/Element Densities/materials_H-Rn/magpiedownloaded.csv')
X, y = dataset.iloc[:,4:], dataset.iloc[:,2]

model = XGBRegressor()
model.load_model("model_xgboost.json")

mape = predict(model, X, y)

X_shuff = X.copy()
rel_diffs = {}
for i in X_shuff.columns:
    X_shuff[i] = np.random.permutation(X[i].values)
    shuff_mape = predict(model, X_shuff, y)
    rel_diff = mape - shuff_mape
    rel_diffs[i] = rel_diff
    X_shuff[i] = X[i].values

rel_diffs_pd = pd.DataFrame(data=rel_diffs, index=[0])
rel_diffs_pd.to_pickle("rel_diffs.pkl")
rel_diffs_pd.to_csv("rel_diffs.csv")
