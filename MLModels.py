from typing import Tuple
from DataHandler import DataPreProcess
from sklearn.linear_model import LinearRegression
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.metrics import mean_squared_error , r2_score

logging.basicConfig(level=logging.INFO)


"""
create class to build and train the models
all the data get its data from DataPrePricess class
"""

class TrainModel:

### create and train and evaluate context based models ###
#in each function we train the model make prediction and evaluate the prediction 

    #linear regression model 
    @staticmethod
    def context_based_linear_regression_model(x_train:pd.DataFrame, x_test:pd.DataFrame, y_train:pd.Series):
        model = LinearRegression()
        model.fit(x_train,y_train)
        logging.info("model training complete")
        prediction = model.predict(x_test)
        logging.info("created prediction")

        return model, prediction
    
    #random forest tree regression
    @staticmethod
    def context_based_radom_tree_regression(x_train:pd.DataFrame, x_test:pd.DataFrame, y_train:pd.Series):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)
        logging.info("training context based random forest regression model")
        prediction = model.predict(x_test)
        logging.info("create a prediction")
        return model, prediction
    
    #extrime gradient boosting model
    @staticmethod
    def XBG_gradient_boosting_model(x_train:pd.DataFrame, x_test:pd.DataFrame, y_train:pd.Series):
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(x_train, y_train)
        logging.info("successfully trained the model")
        prediction = model.predict(x_test)
        logging.info("prediction has been made")
        return model, prediction
    
    #light gradient boosting model
    @staticmethod
    def Light_gradient_boosting_model(x_train:pd.DataFrame, x_test:pd.DataFrame, y_train:pd.Series):
        model = LGBMRegressor(n_estimators=100,learning_rate=0.1, random_state=42)
        model.fit(x_train, y_train)
        logging.info("create and train model")
        prediction = model.predict(x_test)
        logging.info("successfully created a prediction")
        return model, prediction
    
    #models evaluation 
    @staticmethod
    def models_evaluetion(prediction:np.ndarray, y_test:pd.Series):
        mse = mean_squared_error(y_test, prediction)
        logging.info(f"the mse evaluetion is: {mse}")
        rmse = np.sqrt(mse)
        logging.info(f"the rmse evaluetion is: {rmse}")
        r2 = r2_score(y_test, prediction)
        logging.info(f"the r2 evaluetion is: {r2}")

        evaluation_scores = {
        'r2': r2,
        'mse': mse,
        'rmse': rmse
        }
        return evaluation_scores


    