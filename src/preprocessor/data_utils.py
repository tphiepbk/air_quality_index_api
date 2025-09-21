from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge

import pandas as pd
import numpy as np

def mice(df, method=None):
    if method is not None:
        assert method in ["random_forest", "extra_trees"]
    df_columns = df.columns
    if method == "random_forest":
        estimator = RandomForestRegressor()
        max_iter = 10
    elif method == "extra_trees":
        estimator = ExtraTreesRegressor()
        max_iter = 10
    else:
        estimator = BayesianRidge()
        max_iter = 100
    imputer = IterativeImputer(estimator=estimator, random_state=100, max_iter=max_iter, keep_empty_features=True, min_value=0)
    imputer.fit(df[df_columns])
    imputed_value = imputer.transform(df[df_columns])
    mice_df = df.copy()
    mice_df.loc[:, df_columns] = imputed_value
    return mice_df

def reframePastFuture(df, n_past=1, n_future=1, keep_label_only=False):
    assert isinstance(df, pd.DataFrame), "df should be a DataFrame"

    total_len = len(df)
    print("total_len = ", total_len)
    ret_X, ret_y = [], []

    for window_start in range(total_len):
        print(window_start)
        past_end = window_start + n_past
        future_end = past_end + n_future
        print("past_end = ", past_end)
        print("future_end = ", future_end)

        # If this case happens, it means the length of input data is exactly n_past
        if future_end > total_len:
              break

        ret_X.append(df.iloc[window_start:past_end, :])
        if keep_label_only:
            ret_y.append(df.iloc[past_end:future_end, -1])
        else:
            ret_y.append(df.iloc[past_end:future_end, :])

    if keep_label_only:
        ret_y = np.expand_dims(ret_y, axis=-1)

    return np.array(ret_X), np.array(ret_y)

