import pandas as pd
import numpy as np
import copy
import os

from src.preprocessor.data_utils import mice, reframePastFuture
from sklearn.preprocessing import MinMaxScaler

class Preprocessor:
    def __init__(self, n_past=7, n_future=1):
        self.__n_past = n_past
        self.__n_future = n_future

    def execute(self, input_data: pd.DataFrame):
        # Fill missing values
        filled_data = self.__fill_missing(input_data)
        print(filled_data.columns)

        # Define label and features
        label = "pm25"
        features = list(filled_data.columns)
        features.remove(label)
        X = filled_data.loc[:, features]
        y = filled_data.loc[:, [label]]
        print("X", X)
        print("y", y)

        # Scale data
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = pd.DataFrame(feature_scaler.fit_transform(X), columns=X.columns)
        label_scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaled = pd.DataFrame(label_scaler.fit_transform(y), columns=y.columns)

        # Reframe data
        combined_df = pd.concat((X_scaled, y_scaled), axis=1)
        return combined_df, label_scaler
        print("combined_df = ")
        print(combined_df)
        X_scaled_reframed, y_scaled_reframed = reframePastFuture(combined_df, self.__n_past, self.__n_future, keep_label_only=True)
        print("X_scaled_reframed = ")
        print(X_scaled_reframed)
        return X_scaled_reframed, y_scaled_reframed

    def __fill_missing(self, input_data: pd.DataFrame):
        all_stations_df = []

        df = input_data.replace(-1, np.nan)
        print(df)

        for lat in sorted(df["lat"].unique()):
            print(f"\nHandling for lat {lat}\n")

            # Get dataframe of current station
            df_current_station = df[df["lat"] == lat]

            # Fill "aod" column
            df_current_station_imputed = copy.deepcopy(df_current_station)
            df_current_station_imputed.loc[:, "aod"] = mice(df_current_station_imputed)["aod"]

            # Reappend the location to dataset
            all_stations_df.append(df_current_station_imputed)

        # Merge all stations
        df = pd.concat(all_stations_df)
        # Store file for debugging after filling missing values
        df.to_csv(os.path.join("temp", "filled_missing.csv"), index=False)

        return df


