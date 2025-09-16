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
        neighbor_ratings = user_item_matrix.iloc[neighbor_indices]
        recommendation_scores = neighbor_ratings.mean(axis=0)
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
    

def evaluate_model(
    model: Any,
    model_type: str,
    test_df: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
    train_data: Any,
    k: int = 10
) -> Dict[str, float]:
    """
    Computes Precision@k and Recall@k averaged over users appearing in test_df.
    Assumes each model_*_prediction returns an Iterable[(isbn, score)] sorted by score desc.
    """

    def _predict(user_id: Any, k: int) -> List[Tuple[Any, float]]:
        if model_type == 'knn':
            return list(TrainModel.user_based_knn_prediction(user_id, model, user_item_matrix, train_data, k))
        elif model_type == 'als':
            return list(TrainModel.user_based_als_prediction(user_id, model, user_item_matrix, train_data, k))
        elif model_type == 'svd':
            return list(TrainModel.user_based_svd_prediction(user_id, model, train_data, k))
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    test_users = test_df['user_id'].unique()
    precisions: List[float] = []
    recalls: List[float] = []

    for user_id in test_users:
        # Ground truth: items the user actually interacted with in the test set
        gt_items = set(test_df.loc[test_df['user_id'] == user_id, 'isbn'])
        if len(gt_items) == 0:
            # nothing to evaluate for this user
            continue

        try:
            recs_with_scores = _predict(user_id, k)
        except KeyError:
            # e.g., cold-start user not in training data
            continue

        recommended_items = [isbn for isbn, _ in recs_with_scores]
        if len(recommended_items) == 0:
            continue

        # Use top-k (truncate if more returned)
        topk = recommended_items[:k]
        hits = len(set(topk) & gt_items)

        # If model returns fewer than k, use the actual number for precision denominator
        denom_prec = max(1, min(k, len(topk)))
        precision_at_k = hits / denom_prec
        recall_at_k = hits / len(gt_items)

        precisions.append(precision_at_k)
        recalls.append(recall_at_k)

    avg_precision = float(np.mean(precisions)) if len(precisions) > 0 else 0.0
    avg_recall = float(np.mean(recalls)) if len(recalls) > 0 else 0.0

    return {
        'precision_at_k': avg_precision,
        'recall_at_k': avg_recall,
        'evaluated_users': len(precisions)  # handy sanity check
    }

    
"""
create class with functions to improve the models
"""

class ModelsImprovment:

    ##improve context base models##

    pass
