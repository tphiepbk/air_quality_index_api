import pandas as pd
import numpy as np
import copy
import os
import joblib

from src.preprocessor.data_utils import mice, reframePastFuture
from sklearn.preprocessing import MinMaxScaler
from src.logger.logger import info

class Preprocessor:
    def __init__(self, data_type="aod"):
        func_name = "Preprocessor.__init__()"
        info("{}: is called", func_name)

        self.__data_type = data_type
        self.__feature_scaler = joblib.load(os.path.join("models", "scaler", f"{self.__data_type}_feature_scaler.pkl"))
        self.__label_scaler = joblib.load(os.path.join("models", "scaler", f"{self.__data_type}_label_scaler.pkl"))

    # Split dataset into features and label
    def __split_feature_label(self, filled_data: pd.DataFrame):
        func_name = "Preprocessor.__split_feature_label()"
        info("{}: is called", func_name)

        # Choose the label based on data type
        if self.__data_type == "aod":
            label = "pm25"
        else:
            label = "no"

        features = list(filled_data.columns)
        features.remove(label)
        features.remove("station")

        feature_data = filled_data.loc[:, features]
        label_data = filled_data.loc[:, [label]]

        return feature_data, label_data

    def __fill_missing(self, input_data: pd.DataFrame):
        func_name = "Preprocessor.__fill_missing()"
        info("{}: is called", func_name)

        # Get missing feature
        missing_feature = input_data.isnull().sum()

        all_stations_df = []

        df = input_data.replace(-1, np.nan)
        print(df)

        for station in sorted(df["station"].unique()):
            print(f"\nHandling for station {station}\n")

            # Get dataframe of current station
            df_current_station = df[df["station"] == station]

            # Fill "aod" column
            df_current_station_imputed = copy.deepcopy(df_current_station)
            df_current_station_imputed.loc[:, "aod"] = mice(df_current_station_imputed.drop(columns=["station"]))["aod"]

            # Reappend the location to dataset
            all_stations_df.append(df_current_station_imputed)

        # Merge all stations
        df = pd.concat(all_stations_df)
        # Store file for debugging after filling missing values
        df.to_csv(os.path.join("temp", "filled_missing.csv"), index=False)

        return df

    def execute(self, input_data: pd.DataFrame):
        func_name = "Preprocessor.execute()"
        info("{}: is called", func_name)

        # Fill missing values
        filled_data = self.__fill_missing(input_data)
        print(filled_data.columns)

        # Split into feature and label
        X, y = self.__split_feature_label(filled_data=filled_data)

        # Scale data
        # Because the order of columns are different, so we have to reorder
        feature_names_order = self.__feature_scaler.get_feature_names_out()
        print("feature_names_order = ", feature_names_order)
        X = X.loc[:, feature_names_order]
        X_scaled = pd.DataFrame(self.__feature_scaler.transform(X), columns=X.columns)
        y_scaled = pd.DataFrame(self.__label_scaler.transform(y), columns=y.columns)

        return X_scaled, y_scaled


