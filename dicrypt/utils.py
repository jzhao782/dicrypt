import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
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

def get_scikit_metrics(pred, y):
    rmse = mean_squared_error(pred, y) ** 0.5
    mape = mean_absolute_percentage_error(pred, y)
    r2 = r2_score(pred, y)

    return rmse, mape, r2