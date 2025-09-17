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
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from typing import Any, Dict, Tuple
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

logging.basicConfig(level=logging.INFO)


"""
create class to build and train the models
all the data get its data from DataPrePricess class
"""

class TrainModel:

### create and train and evaluate context based models ###
#in each function we train the model make prediction and evaluate the prediction 
#they all get the data from context_based_data_preprocessing_pipeline in DataPreProcessing class

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
    def context_based_models_evaluetion(prediction:np.ndarray, y_test:pd.Series):
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


### create train and evaluate users based model ###
#get user-item train matrix from split_user_item_matrix in DataPreProcessing

    #KNN model
    @staticmethod
    def knn_user_based_model(train_matrix):
        model = NearestNeighbors(metric="cosine", algorithm="brute")
        model.fit(train_matrix)
        logging.info("successfully created user based model with KNN")
        
        return model
    
    
    #ALS model
    @staticmethod
    def als_user_based_model(train_matrix):
        model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=20)
        model.fit(train_matrix.T)
        logging.info("ALS model creates and trained successfully")
        
        return model
    
    ## important!! SVD algorithm get Ratins dataframe not the item user matrix

    #svd model
    #return the model the train set and the test set
    @staticmethod
    def svd_user_based_model(rating_df:pd.DataFrame):
        reader = Reader(rating_scale=(1,10))    #rating scale
        data = Dataset.load_from_df(rating_df[["user_id", "isbn", "book_rating"]], reader)   #surprise data object
        train_set, testset = train_test_split(data, test_size=0.2, random_state=42)   #split the data into 2 matrixes
        model = SVD(n_factors=50, random_state=42)
        model.fit(train_set)
        logging.info("SVD model create and trained successfully")
        return model, train_set, testset

    #create user based predictions
      
    #KNN prediction
    @staticmethod
    def user_based_knn_prediction(user_id: int, model: NearestNeighbors, user_item_matrix: pd.DataFrame, train_matrix: csr_matrix, k=5) -> list:
        user_index = user_item_matrix.index.get_loc(user_id)
        user_vector = train_matrix[user_index]
        distances, indices = model.kneighbors(user_vector, n_neighbors=k + 1)
        neighbor_indices = indices.flatten()[1:]  
        neighbor_vectors = train_matrix[neighbor_indices]    
        recommendation_scores_array = neighbor_vectors.mean(axis=0)    
        recommendation_scores = pd.Series(recommendation_scores_array.A1, index=user_item_matrix.columns)
        user_rated_books = user_item_matrix.iloc[user_index]    
        recommendation_scores = recommendation_scores[user_rated_books == 0]    
        top_recommends = recommendation_scores.sort_values(ascending=False).head(k)    
        logging.info(f"Generated {k} recommendations for user {user_id} using KNN.")
        return list(top_recommends.items())

    #ALS prediction
    @staticmethod
    def user_based_als_prediction(user_id: int, model: AlternatingLeastSquares, user_item_matrix: pd.DataFrame, train_matrix_T: csr_matrix, num_recommendation: int = 5) -> list:
        user_index = user_item_matrix.index.get_loc(user_id)        
        item_indices, scores = model.recommend(user_index, train_matrix_T, N=num_recommendation)  
        recommendations = []
        for item_index, score in zip(item_indices, scores):
            isbn = user_item_matrix.columns[item_index]
            recommendations.append((isbn, score))
            
        logging.info(f"Generated {num_recommendation} recommendations for user {user_id} using ALS.")
        return recommendations

    #SVD prediction
    @staticmethod
    def user_based_svd_prediction(user_id: int, model: SVD, train_df: pd.DataFrame, num_recommendation=5) -> list:
        all_isbns = set(train_df['isbn'].unique())
        rated_isbns = set(train_df[train_df['user_id'] == user_id]['isbn'].unique())
        unrated_books = all_isbns - rated_isbns
        predictions = []
        for isbn in unrated_books:
            pred_obj = model.predict(uid=user_id, iid=isbn)
            predictions.append((pred_obj.iid, pred_obj.est))
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        logging.info(f"Generated {num_recommendation} recommendations for user {user_id} using SVD.")
        return predictions[:num_recommendation]

    
    
"""
create class with functions to improve the models
"""

class ModelsHyperparametersImprovment:

##improve context base models##

    # random forest regression hyperparameters improvement
    @staticmethod
    def context_based_radom_forest_hyperparameters_improvement( x_train: pd.DataFrame, y_train:pd.Series):
        param_grid = {
            "n_estimators":[100,200,300,400,500,600,700,800],
            'max_depth': [10, 20, 30, None],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10]
        }
        model = RandomForestRegressor(random_state=42)
        randon_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=8, 
                                   cv=3, scoring='neg_mean_squared_error', verbose=2, random_state=42, n_jobs=-1)
        randon_search.fit(x_train,y_train)
        logging.info(f"the best n_estimator for the model is: {randon_search.best_params_}")
        return randon_search.best_estimator_
    
    # extrem gradient boosting hyperparameters improvment
    @staticmethod
    def context_based_XBGgradient_boosting_hyperparameters_improvment( x_train:pd.DataFrame, y_train:pd.Series):
        param_grid = {
                "n_estimators": [100, 200, 300, 400, 500],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9]
        }
        model = XGBRegressor(random_state=42)
        random_search_XGBoost = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=20,
                                                   cv=3, scoring="neg_mean_squared_error", verbose=2, random_state=42, n_jobs=-1)
        random_search_XGBoost.fit(x_train, y_train)
        logging.info(f"model trained successfully, best parameters {random_search_XGBoost.best_params_}")
        return random_search_XGBoost.best_estimator_
    
    #light gradient boosting hyperparameters improvment
    @staticmethod
    def context_based_LGBgradient_boostin_hyperparameters_improvment(x_train:pd.DataFrame, y_train:pd.Series):
        param_grid = {
                "n_estimators": [100, 200, 300, 400, 500],
                "learning_rate": [0.05, 0.1, 0.2],
                "num_leaves": [20, 31, 40, 50],
                "max_depth": [3, 5, 7],
                "colsample_bytree": [0.7, 0.8, 0.9]
        }
        model = LGBMRegressor(random_state=42)
        random_search_LBGoost = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=20,
                                                   cv=3, scoring="neg_mean_squared_error", verbose=2, random_state=42, n_jobs=-1)
        random_search_LBGoost.fit(x_train, y_train)
        logging.info(f"model trained successfully, best parameters {random_search_LBGoost.best_params_}")
        return random_search_LBGoost.best_estimator_

    ### user based models hyperparameters improvment ###

    #knn hyperparameters improvment 
    @staticmethod
    def knn_user_based_hyperparameters_improvment(train_matrix, test_matrix):
        param_to_check = [5,10,20,30,40]
        best_rmse = float(int)
        best_k = -1
        
        for k in param_to_check:
            model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k)
            model.fit(train_matrix)
            


    #svd model hyperparameters improvment
    @staticmethod
    def svd_user_based_hyperparameters_improvment():
        pass


"""
create a class for features engineering
"""
class FeaturesEngineer:

    ### context based models feature engeneering ###

    #get data frame from context_based_df 
    #add to the data frame the rows: 
    @staticmethod
    def context_base_models_features_engineer(x_train:pd.DataFrame, x_test:pd.DataFrame, y_train:pd.DataFrame):
        
        book_mean_rate_map = x_train.groupby("isbn")["book_rating"].mean()
        rating_count_map = x_train.groupby("isbn")["book_rating"].count()

        x_train["book_mean_rate"] = x_train["isbn"].map(book_mean_rate_map)
        x_test["book_mean_rate"] = x_test["isbn"].map(book_mean_rate_map)
        logging.info("added the book mean rating to the train and test sets")

        x_train["rating_count"] = x_train["isbn"].map(rating_count_map)
        x_test["rating_count"] = x_test["isbn"].map(rating_count_map)
        logging.info("added the rating count to the train and test sets")

        