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
    def user_based_knn_prediction(user_id:int, model:NearestNeighbors, train_matrix:csr_matrix, k=5)->list:
        user_index = train_matrix.index.get_loc(user_id)   #get the user_index in the user item matrix
        user_vector = train_matrix.iloc[user_index].values.reshape(1,-1)
        distances, indices = model.kneighbors(user_vector, n_neighbors=k+1)
        logging.info("trained the model successfully")
        logging.info(f"Found {k} nearest neighbors for user {user_id}.")
        neighbor_indices = indices.flatten()[1:]
        neighbor_rating = train_matrix.iloc[neighbor_indices]
        recommendation_score = neighbor_rating.mean(axis=0)
        user_rated_book = train_matrix.iloc[user_index]
        recommendation_score = recommendation_score[user_rated_book == 0]
        top_recommends = recommendation_score.sort_values(ascending=False).head(k)
        return list(top_recommends.items())

        return distances, indices

    #ALS prediction
    @staticmethod
    def user_based_als_prediction(user_id:int, model:AlternatingLeastSquares, train_matrix:csr_matrix, num_recommendation: int=5):
        user_index = train_matrix.index.get_loc(user_id)
        item_indices, scores = model.recommend(user_index, train_matrix, N=num_recommendation)
        logging.info("ALS model created andtrained successfully")
        recommendations = []
        for i, item_index in enumerate(item_indices):
            isbn = train_matrix.columns[item_index]
            score = scores[i]
            recommendations.append((isbn, score))
            logging.info(f"Book ISBN: {isbn}, Score: {score:.2f}")
        return recommendations

    #SVD prediction
    @staticmethod
    def user_based_svd_prediction(user_id:int, model:SVD, test_df:pd.DataFrame, num_recommendation = 5):
        books_isbn = set(test_df["isbn"].unique())
        rated_isbn = set(test_df[test_df["user_id"] == user_id]["isbn"].unique())
        unrated_books = books_isbn - rated_isbn
        prediction = []
        for isbn in unrated_books:
                prediction_obj = model.predict(uid=user_id, iid=isbn)
                prediction.append(prediction_obj)
        logging.info("created svd prediction")
        prediction.sort(key=lambda x: x.est, reverse=True)
        return prediction[:10]
    

    ##creating evaluation for each user based model ##

    #knn evaluation
    def knn_model_evaluation(user_is:int, model:BaseEstimator, test_matrix:csr_matrix):
        pass