import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def boxcox_standardize(df_num, df_lmbdas=None, df_min=None):
    """
    Use boxcox to normalize dataframe.
    
    Args:
        df_num: must be all numerical
        df_lmbdas: Optional series of lambdas. If none, 
                    function will solve for it
        df_min: Optional series of column minimums. If none, 
                function will solve for it
        
    Returns:
        1) Transformed df_num
        2) Series of lambdas
        3) Series of min
    """
    
    if df_min is None:
        df_min = df_num.min(axis=0)
    df_num1 = df_num.sub(df_min, axis=1)
    df_num1[df_num1 < 0] = 0
    eps = 0.000001
    df_num2 = df_num1 + eps
    
    
    lmbdas = []
    df_num3 = df_num2.copy()
    for i in range(df_num2.shape[1]):
        if df_lmbdas is None:
            lmbda = None
            df_num3.iloc[:,i], lmbda = stats.boxcox(df_num2.iloc[:,i])
            lmbdas.append(lmbda)
        else:
            lmbda = df_lmbdas[i]
            df_num3.iloc[:,i] = stats.boxcox(df_num2.iloc[:,i], lmbda=lmbda)
    
    if df_lmbdas is None:
        df_lmbdas = pd.Series(lmbdas, index=df_num.columns)
    
    return df_num3, df_lmbdas, df_min


dataset = pd.read_csv('/Users/jesse/projects/Python/Element Densities/materials_H-Rn/magpiedownloaded.csv') # read in data
X, y = dataset.iloc[:,4:], dataset.iloc[:,2]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9, test_size=0.1) # split into train/test

X_train_standardized = X_train
X_valid_standardized = X_valid

# X_train_standardized, lmbdas, mins = boxcox_standardize(X_train)
# X_valid_standardized, _, _ = boxcox_standardize(X_valid, df_lmbdas=lmbdas, df_min=mins)

model = XGBRegressor(n_estimators = 1000, learning_rate = 0.05)
model.fit(X_train_standardized, y_train,
            early_stopping_rounds=5,
            eval_set=[(X_valid_standardized, y_valid)],
            verbose=False)

train_pred = model.predict(X_train_standardized)
train_mae = mean_absolute_error(train_pred, y_train)
train_mape = mean_absolute_percentage_error(train_pred, y_train)

test_pred = model.predict(X_valid_standardized)
test_mae = mean_absolute_error(test_pred, y_valid)
test_mape = mean_absolute_percentage_error(test_pred, y_valid)

pred = pd.Series(test_pred)
pred.to_csv("predictions_1.csv")
y_valid.to_csv("y_valid.csv")
model.save_model("model_xgboost.json")

print(train_mae)
print(train_mape)

print(test_mae)
print(test_mape)

# X_train.drop(['Unnamed: 0', 'is_mp'], axis=1, inplace=True) # possible data leakage values imo
# X_valid.drop(['Unnamed: 0', 'is_mp'], axis=1, inplace=True)

# model_2 = XGBRegressor(n_estimators = 1000, learning_rate = 0.05)
# model_2.fit(X_train, y_train,
#             early_stopping_rounds=5,
#             eval_set=[(X_valid, y_valid)],
#             verbose=False)

# predictions_2 = model_2.predict(X_valid)
# mae_2 = mean_absolute_error(predictions_2, y_valid)

# print(mae_2)