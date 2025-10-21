import pandas as pd
import numpy as np
import os
import joblib

from src.preprocessor.data_utils import mice
from src.logger.logger import info

class Preprocessor:
    def __init__(self, data_type="aod"):
        func_name = "Preprocessor.__init__()"
        info("{}: is called", func_name)

        self.__data_type = data_type
        self.__feature_scaler = joblib.load(os.path.join("models", "scaler", f"{self.__data_type}_features_scaler.pkl"))
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

        feature_data = filled_data.loc[:, features]
        label_data = filled_data.loc[:, [label]]

        return feature_data, label_data

    def __fill_missing(self, input_data: pd.DataFrame):
        func_name = "Preprocessor.__fill_missing()"
        info("{}: is called", func_name)

        input_data = input_data.replace(-1, np.nan)

        input_data_imputed = mice(input_data)

        info("{}: filled missing data = \n{}", func_name, input_data_imputed)

        return input_data_imputed

    def inverse_transform(self, input_data):
        func_name = "Preprocessor.inverse_transform()"
        info("{}: is called", func_name)

        return self.__label_scaler.inverse_transform(input_data)

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


